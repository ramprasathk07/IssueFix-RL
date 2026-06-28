import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from accelerate import Accelerator
from pathlib import Path
import sys
import json
import time
import random
import zipfile
import subprocess
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data import create_sft_dataloader
from trainers.utils import dft_loss, entropy_from_logits
from configs import Config

_TRAINING_STATE_FILE = "training_state.pt"


class SFTTrainer:
    def __init__(self, cfg: Config):
        self.train_cfg = cfg.training_params
        self.model_cfg = cfg.model_params
        self.data_cfg = cfg.dataloader_params
        self.scheduler = None
        self._wandb = None
        self._mlflow = None
        self._mlflow_run = None

        mixed_precision = (
            "bf16" if self.train_cfg.bf16 else ("fp16" if self.train_cfg.fp16 else "no")
        )
        self.accelerator = Accelerator(mixed_precision=mixed_precision)

        self._load_model()
        self._setup_optimizer()

    # ── model loading ─────────────────────────────────────────────────────────

    def add_special_tokens(self, specials=["<think>", "</think>", "<answer>", "</answer>"]):
        self.tokenizer.add_special_tokens({"additional_special_tokens": specials})
        if self.model is not None:
            self.model.resize_token_embeddings(len(self.tokenizer))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_model(self, checkpoint_dir: Path | None = None):
        is_distributed = self.accelerator.num_processes > 1
        model_name = str(checkpoint_dir) if checkpoint_dir else self.model_cfg.base_model
        trust_remote = self.model_cfg.trust_remote_code

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote)

        bnb_config = None
        torch_dtype = torch.bfloat16 if self.train_cfg.bf16 else torch.float16

        if is_distributed and (self.model_cfg.load_in_4bit or self.model_cfg.load_in_8bit):
            # bitsandbytes quantization is incompatible with DDP wrapping
            self.accelerator.print(
                "WARNING: quantization disabled for multi-GPU DDP — loading in bf16."
            )
        elif self.model_cfg.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if self.train_cfg.bf16 else torch.float16,
                bnb_4bit_quant_type=self.model_cfg.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.model_cfg.bnb_4bit_use_double_quant,
            )
        elif self.model_cfg.load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype if bnb_config is None else None,
            # device_map="auto" conflicts with DDP — each process owns its GPU via accelerate
            device_map=None if is_distributed else "auto",
            trust_remote_code=trust_remote,
        )

        if self.train_cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if not checkpoint_dir:
            self.add_special_tokens()

    # ── optimizer / scheduler ─────────────────────────────────────────────────

    def _setup_optimizer(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
        )

    def _setup_scheduler(self, total_steps: int, completed_steps: int = 0):
        warmup = self.train_cfg.warmup_steps
        if self.train_cfg.lr_scheduler == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup, num_training_steps=total_steps
            )
        elif self.train_cfg.lr_scheduler == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup, num_training_steps=total_steps
            )
        if self.scheduler and completed_steps > 0:
            for _ in range(completed_steps):
                self.scheduler.step()

    # ── experiment tracking ───────────────────────────────────────────────────

    def _init_wandb(self):
        if not self.train_cfg.wandb_project or not self.accelerator.is_main_process:
            return
        try:
            import wandb
            wandb.init(
                project=self.train_cfg.wandb_project,
                name=self.train_cfg.wandb_run_name,
                config={
                    **self.train_cfg.model_dump(),
                    **self.model_cfg.model_dump(),
                    **self.data_cfg.model_dump(),
                    "num_gpus": self.accelerator.num_processes,
                },
            )
            self._wandb = wandb
        except ImportError:
            print("wandb not installed — skipping wandb logging.")

    def _init_mlflow(self):
        if not self.train_cfg.mlflow_tracking_uri or not self.accelerator.is_main_process:
            return
        try:
            import mlflow
            mlflow.set_tracking_uri(self.train_cfg.mlflow_tracking_uri)
            mlflow.set_experiment(self.train_cfg.mlflow_experiment)
            self._mlflow_run = mlflow.start_run(run_name=self.train_cfg.wandb_run_name)
            mlflow.log_params({
                k: v for k, v in {
                    **self.train_cfg.model_dump(),
                    **self.model_cfg.model_dump(),
                    "num_gpus": self.accelerator.num_processes,
                }.items() if v is not None
            })
            self._mlflow = mlflow
            print(f"MLflow run: {self._mlflow_run.info.run_id}")
        except ImportError:
            print("mlflow not installed — skipping mlflow logging.")

    def _log(self, metrics: dict, step: int):
        if not self.accelerator.is_main_process:
            return
        # use run.log() so we're explicit about the active run; guard against finished/absent run
        if self._wandb and self._wandb.run is not None:
            self._wandb.run.log(metrics, step=step)
        if self._mlflow:
            self._mlflow.log_metrics(metrics, step=step)

    def _get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    # ── metrics ───────────────────────────────────────────────────────────────

    def compute_entropy(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        entropy = entropy_from_logits(logits)
        mask = labels != -100
        return (entropy * mask).sum() / mask.sum().clamp(min=1)

    # ── checkpointing ─────────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, global_step: int) -> Path | None:
        if not self.accelerator.is_main_process:
            return None

        out = Path(self.train_cfg.output_dir) / f"checkpoint-epoch{epoch+1}-step{global_step}"
        out.mkdir(parents=True, exist_ok=True)

        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.save_pretrained(out)
        self.tokenizer.save_pretrained(out)

        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            },
            out / _TRAINING_STATE_FILE,
        )

        if self._mlflow:
            self._mlflow.log_artifacts(
                str(out), artifact_path=f"checkpoint-epoch{epoch+1}-step{global_step}"
            )

        print(f"Checkpoint saved → {out}")
        return out

    def _zip_best_checkpoint(self, checkpoint_path: Path | None):
        if not self.accelerator.is_main_process or checkpoint_path is None:
            return
        run_name = self.train_cfg.wandb_run_name or "best"
        kaggle_out = Path("/kaggle/working")
        out_dir = kaggle_out if kaggle_out.exists() else Path(self.train_cfg.output_dir)
        zip_path = out_dir / f"{run_name}_best.zip"

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file in checkpoint_path.rglob("*"):
                if file.is_file():
                    zf.write(file, arcname=file.relative_to(checkpoint_path))

        print(f"Best checkpoint zipped → {zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB)")
        return zip_path

    def _push_to_kaggle_models(self, checkpoint_path: Path | None):
        if not self.accelerator.is_main_process or checkpoint_path is None:
            return
        if not self.train_cfg.push_to_kaggle or not self.train_cfg.kaggle_model_handle:
            return

        handle = self.train_cfg.kaggle_model_handle  # "owner/model/framework/variation"
        parts = handle.split("/")
        if len(parts) != 4:
            print(f"[kaggle] invalid handle '{handle}' — expected owner/model/framework/variation")
            return

        # Verify kaggle CLI is available
        if subprocess.run(["kaggle", "--version"], capture_output=True).returncode != 0:
            print("[kaggle] kaggle CLI not found — skipping model push")
            return

        owner, model_slug, framework, variation = parts
        run_name = self.train_cfg.wandb_run_name or "sft_run"

        # Create model if not exists (ignore errors — likely already exists)
        subprocess.run(
            ["kaggle", "models", "create",
             "--owner", owner, "--name", model_slug,
             "--framework", framework,
             "--license", self.train_cfg.kaggle_model_license],
            capture_output=True,
        )

        # Push new version — creates model instance automatically if missing
        result = subprocess.run(
            ["kaggle", "models", "instances", "versions", "create", handle,
             "--path", str(checkpoint_path),
             "--version-notes", f"Best checkpoint — run: {run_name}"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"[kaggle] pushed → https://www.kaggle.com/models/{owner}/{model_slug}")
        else:
            print(f"[kaggle] push failed:\n{result.stderr.strip()}")

    def _load_training_state(self, checkpoint_dir: Path) -> tuple[int, int]:
        state_file = checkpoint_dir / _TRAINING_STATE_FILE
        if not state_file.exists():
            self.accelerator.print(
                f"No training_state.pt in {checkpoint_dir} — fresh optimizer/scheduler."
            )
            return 0, 0

        state = torch.load(state_file, map_location="cpu", weights_only=False)
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        epoch = state["epoch"]
        global_step = state["global_step"]
        self.accelerator.print(f"Resumed from epoch {epoch+1}, global_step {global_step}")
        return epoch + 1, global_step

    # ── validation ────────────────────────────────────────────────────────────

    def validate(self, epoch: int, val_loader: DataLoader, global_step: int) -> float:
        self.model.eval()
        running_loss = 0.0
        running_entropy = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                labels = batch["labels"]
                model_inputs = {k: v for k, v in batch.items() if k != "labels"}
                results = self.model(**model_inputs)
                running_loss += dft_loss(results.logits, labels).item()
                running_entropy += self.compute_entropy(results.logits, labels).item()

        # gather across GPUs so all processes have the same avg
        avg_loss_t = self.accelerator.gather(
            torch.tensor(running_loss / len(val_loader), device=self.accelerator.device)
        ).mean().item()
        avg_entropy_t = self.accelerator.gather(
            torch.tensor(running_entropy / len(val_loader), device=self.accelerator.device)
        ).mean().item()

        self._log(
            {"val/loss": avg_loss_t, "val/entropy": avg_entropy_t, "epoch": epoch + 1},
            step=global_step,
        )
        self.accelerator.print(
            f"[val] epoch {epoch+1} | loss {avg_loss_t:.4f} | entropy {avg_entropy_t:.4f}"
        )
        return avg_loss_t

    # ── training loop ─────────────────────────────────────────────────────────

    def train_epoch(
        self, epoch: int, train_loader: DataLoader, global_step: int
    ) -> tuple[float, int]:
        grad_accumulation = self.train_cfg.gradient_accumulation_steps
        grad_clip = self.train_cfg.max_grad_norm
        log_interval = self.train_cfg.logging_steps
        save_steps = self.train_cfg.save_steps

        self.model.train()
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{self.train_cfg.num_epochs}",
            leave=True,
            disable=not self.accelerator.is_main_process,
        )

        running_loss = 0.0
        last_grad_norm = 0.0
        last_entropy = 0.0
        start_time = time.time()
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
            labels = batch["labels"]
            model_inputs = {k: v for k, v in batch.items() if k != "labels"}

            with self.accelerator.autocast():
                results = self.model(**model_inputs)
                loss = dft_loss(results.logits, labels) / grad_accumulation

            self.accelerator.backward(loss)
            running_loss += loss.item() * grad_accumulation

            is_accum_step = (batch_idx + 1) % grad_accumulation == 0
            is_last_batch = batch_idx == len(train_loader) - 1

            if is_accum_step or is_last_batch:
                if grad_clip:
                    last_grad_norm = self.accelerator.clip_grad_norm_(
                        self.model.parameters(), grad_clip
                    ).item()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # compute entropy at optimizer step so it's always fresh
                with torch.no_grad():
                    last_entropy = self.compute_entropy(results.logits, labels).item()

                if global_step % save_steps == 0:
                    self.save_checkpoint(epoch, global_step)

                # log at optimizer-step granularity — guarantees monotonically increasing
                # wandb step and avoids duplicate step values that silently stop chart updates
                if global_step % log_interval == 0 and self.accelerator.is_main_process:
                    elapsed = time.time() - start_time
                    current_loss = running_loss / (batch_idx + 1)
                    self.accelerator.print(
                        f"[train] epoch {epoch+1} | step {global_step} | "
                        f"loss {current_loss:.4f} | entropy {last_entropy:.4f} | "
                        f"grad_norm {last_grad_norm:.3f} | lr {self._get_lr():.2e} | "
                        f"elapsed {elapsed:.1f}s"
                    )
                    self._log(
                        {
                            "train/loss": current_loss,
                            "train/entropy": last_entropy,
                            "train/grad_norm": last_grad_norm,
                            "train/lr": self._get_lr(),
                            "epoch": epoch + 1,
                        },
                        step=global_step,
                    )

            if self.accelerator.is_main_process:
                pbar.set_postfix(
                    {"loss": f"{running_loss / (batch_idx + 1):.4f}", "lr": f"{self._get_lr():.6f}"}
                )

        # gather avg train loss across all ranks for accurate epoch summary
        avg_loss_t = self.accelerator.gather(
            torch.tensor(running_loss / len(train_loader), device=self.accelerator.device)
        ).mean().item()

        return avg_loss_t, global_step

    # ── entrypoint ────────────────────────────────────────────────────────────

    def __call__(self, data_path: str, resume_from: str | None = None):
        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))

        random.seed(42)
        random.shuffle(data)

        split = int(0.8 * len(data))
        train_data, val_data = data[:split], data[split:]

        is_distributed = self.accelerator.num_processes > 1
        train_loader = create_sft_dataloader(
            data=train_data, tokenizer=self.tokenizer, data_config=self.data_cfg,
            drop_last=is_distributed,  # avoid uneven last-batch across ranks
        )
        val_loader = create_sft_dataloader(
            data=val_data, tokenizer=self.tokenizer, data_config=self.data_cfg,
            drop_last=is_distributed,
        )

        # resume: reload model+optimizer from checkpoint before prepare
        start_epoch = 0
        global_step = 0
        if resume_from:
            ckpt_dir = Path(resume_from)
            self._load_model(ckpt_dir)
            self._setup_optimizer()
            start_epoch, global_step = self._load_training_state(ckpt_dir)

        # prepare before computing total steps so len(train_loader) reflects
        # per-process batch count (DDP splits dataset across ranks)
        (
            self.model,
            self.optimizer,
            train_loader,
            val_loader,
        ) = self.accelerator.prepare(self.model, self.optimizer, train_loader, val_loader)

        # correct total steps: per-process batches / grad_acc * epochs
        total_optimizer_steps = (
            len(train_loader) // self.train_cfg.gradient_accumulation_steps
        ) * self.train_cfg.num_epochs

        self._setup_scheduler(total_optimizer_steps, completed_steps=global_step)
        if self.scheduler:
            self.scheduler = self.accelerator.prepare(self.scheduler)

        self._init_wandb()
        self._init_mlflow()

        self.accelerator.print(
            f"Training on {self.accelerator.num_processes} GPU(s) | "
            f"per-GPU batch {self.data_cfg.batch_size} | "
            f"grad_acc {self.train_cfg.gradient_accumulation_steps} | "
            f"effective batch {self.data_cfg.batch_size * self.accelerator.num_processes * self.train_cfg.gradient_accumulation_steps}"
        )

        best_val_loss = float("inf")
        best_checkpoint_path: Path | None = None

        for epoch in range(start_epoch, self.train_cfg.num_epochs):
            avg_train_loss, global_step = self.train_epoch(epoch, train_loader, global_step)
            avg_val_loss = self.validate(epoch, val_loader, global_step)
            ckpt = self.save_checkpoint(epoch, global_step)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_checkpoint_path = ckpt
                self.accelerator.print(f"New best val_loss {best_val_loss:.4f} → {ckpt}")

            self.accelerator.print(
                f"Epoch {epoch+1} complete | "
                f"train_loss {avg_train_loss:.4f} | val_loss {avg_val_loss:.4f}"
            )

        self._zip_best_checkpoint(best_checkpoint_path)
        self._push_to_kaggle_models(best_checkpoint_path)

        # register final model in MLflow registry (main process only)
        if self._mlflow and self._mlflow_run:
            model_name = self.model_cfg.base_model.split("/")[-1]
            model_uri = f"runs:/{self._mlflow_run.info.run_id}/checkpoint-epoch{self.train_cfg.num_epochs}-step{global_step}"
            try:
                self._mlflow.register_model(model_uri, model_name)
                print(f"Model registered in MLflow registry as '{model_name}'")
            except Exception as e:
                print(f"MLflow model registration skipped: {e}")
            self._mlflow.end_run()

        if self._wandb:
            self._wandb.finish()

        self.accelerator.end_training()
