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
import math
import time
import random
import shutil
import zipfile
import subprocess
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data import create_sft_dataloader
from trainers.utils import ce_loss, entropy_from_logits, token_accuracy
from configs import Config

_MAX_PPL_LOSS = 20.0  # clamp before exp() — avoids inf spikes early in training (post-resize embeddings)

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
        self._resume_wandb_id: str | None = None
        # checkpoint retention bookkeeping (main process only)
        self._epoch_ckpts: list[tuple[float, Path]] = []  # (val_loss, path), best-k retained
        self._last_mid_ckpt: Path | None = None           # rolling mid-epoch resume point

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
            dtype=torch_dtype if bnb_config is None else None,
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
        if self.train_cfg.optimizer == "adamw_8bit":
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(
                    self.model.parameters(),
                    lr=self.train_cfg.learning_rate,
                    weight_decay=self.train_cfg.weight_decay,
                )
                return
            except ImportError:
                self.accelerator.print(
                    "[optim] bitsandbytes not installed — falling back to torch AdamW"
                )
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
                # resume the SAME run across Kaggle sessions — Kaggle's 12h wall
                # forces multi-session runs, and without id=/resume= every
                # resumed session opened a brand-new run, so anything logged
                # after the first session (including most val/* points) landed
                # on a run the user never had open.
                id=self._resume_wandb_id,
                resume="allow" if self._resume_wandb_id else None,
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

    def compute_token_accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return token_accuracy(logits, labels)

    @staticmethod
    def _perplexity(loss: float) -> float:
        return math.exp(min(loss, _MAX_PPL_LOSS))

    def _gpu_mem_stats(self) -> tuple[float, float]:
        """Peak alloc/reserved GB since the last reset, gathered as worst-case across ranks."""
        if not torch.cuda.is_available():
            return 0.0, 0.0
        device = self.accelerator.device
        alloc = torch.tensor(torch.cuda.max_memory_allocated(device) / 1e9, device=device)
        reserved = torch.tensor(torch.cuda.max_memory_reserved(device) / 1e9, device=device)
        alloc = self.accelerator.gather(alloc).max().item()
        reserved = self.accelerator.gather(reserved).max().item()
        torch.cuda.reset_peak_memory_stats(device)
        return alloc, reserved

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
                "wandb_run_id": self._wandb.run.id if self._wandb and self._wandb.run else None,
            },
            out / _TRAINING_STATE_FILE,
        )

        if self._mlflow:
            self._mlflow.log_artifacts(
                str(out), artifact_path=f"checkpoint-epoch{epoch+1}-step{global_step}"
            )

        print(f"Checkpoint saved → {out}")
        return out

    # ── checkpoint retention ──────────────────────────────────────────────────

    def _safe_rmtree(self, path: Path, protected: set[Path]) -> bool:
        """
        Delete a checkpoint dir, refusing anything that isn't provably one of ours.
        Multi-GB deletes on a shared Kaggle disk — every guard here is load-bearing.
        """
        out_root = Path(self.train_cfg.output_dir).resolve()
        try:
            p = Path(path).resolve()
        except OSError:
            return False

        if p in {q.resolve() for q in protected if q is not None}:
            return False                                   # never drop best / in-use
        if not p.is_dir():
            return False
        if out_root not in p.parents:
            print(f"[prune] refusing to delete outside output_dir: {p}")
            return False
        if not p.name.startswith("checkpoint-"):
            print(f"[prune] refusing to delete non-checkpoint dir: {p}")
            return False

        shutil.rmtree(p, ignore_errors=True)
        print(f"[prune] removed {p.name}")
        return True

    def _retain_mid_epoch_checkpoint(self, ckpt: Path | None):
        """
        Mid-epoch saves are rolling resume points — keep only the newest.
        Protects the epoch (best-k) checkpoints, never the one being replaced.
        """
        if not self.accelerator.is_main_process or ckpt is None:
            return
        prev = self._last_mid_ckpt
        self._last_mid_ckpt = ckpt
        if prev is not None and prev != ckpt:
            self._safe_rmtree(prev, protected={p for _, p in self._epoch_ckpts})

    def _retain_best_k_checkpoints(
        self, ckpt: Path | None, val_loss: float, best_path: Path | None
    ):
        """
        Keep the save_best_k epoch checkpoints with the lowest val loss.
        Protects the live resume point and best_path — the latter is what gets
        zipped and pushed after the loop. (best_path always sorts into `keep`
        for k>=1 since it is the lowest loss seen; the guard is belt-and-braces.)
        """
        if not self.accelerator.is_main_process or ckpt is None:
            return
        self._epoch_ckpts.append((val_loss, ckpt))
        k = max(self.train_cfg.save_best_k, 1)
        if len(self._epoch_ckpts) <= k:
            return

        protected: set[Path] = set()
        if best_path is not None:
            protected.add(best_path)
        if self._last_mid_ckpt is not None:
            protected.add(self._last_mid_ckpt)

        ranked = sorted(self._epoch_ckpts, key=lambda pair: pair[0])  # ascending loss
        keep, drop = ranked[:k], ranked[k:]
        for _, path in drop:
            self._safe_rmtree(path, protected)
        self._epoch_ckpts = keep

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
        self._resume_wandb_id = state.get("wandb_run_id")
        self.accelerator.print(f"Resumed from epoch {epoch+1}, global_step {global_step}")
        return epoch + 1, global_step

    # ── data splitting ───────────────────────────────────────────────────────

    def _group_split(
        self, data: list[dict], val_frac: float = 0.2, seed: int = 42
    ) -> tuple[list[dict], list[dict]]:
        """
        Split by unique prompt, not by row. OpenCodeReasoning ships multiple
        solutions per problem (10k rows / ~3.7k unique problems); a row-level
        split let 87.8% of val rows share a problem with train, so every
        val-loss number in this project's history measured memorization, not
        generalization. Whole problems go to one side only.
        """
        groups: dict[str, list[dict]] = {}
        for item in data:
            groups.setdefault(item.get("prompt", ""), []).append(item)

        keys = list(groups.keys())
        rng = random.Random(seed)
        rng.shuffle(keys)

        target_val = int(val_frac * len(data))
        val_data: list[dict] = []
        train_data: list[dict] = []
        for key in keys:
            group = groups[key]
            if len(val_data) < target_val:
                val_data.extend(group)
            else:
                train_data.extend(group)

        train_prompts = {item.get("prompt", "") for item in train_data}
        val_prompts = {item.get("prompt", "") for item in val_data}
        overlap = train_prompts & val_prompts
        assert not overlap, f"group split leaked {len(overlap)} prompts across train/val"

        self.accelerator.print(
            f"[split] {len(groups)} unique prompts -> "
            f"{len(train_data)} train rows / {len(val_data)} val rows | zero prompt overlap"
        )
        return train_data, val_data

    # ── validation ────────────────────────────────────────────────────────────

    def validate(
        self,
        epoch: int,
        val_loader: DataLoader,
        global_step: int,
        max_batches: int = 0,
    ) -> float:
        """
        max_batches > 0 caps the number of val batches (used for cheap mid-epoch
        checks); 0 runs the full set. Every rank must run the SAME count or the
        gather below deadlocks, so the cap is a plain constant, never rank-derived.
        """
        was_training = self.model.training
        self.model.eval()
        running_loss = 0.0
        running_entropy = 0.0
        running_token_acc = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                if max_batches and n_batches >= max_batches:
                    break
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                labels = batch["labels"]
                model_inputs = {k: v for k, v in batch.items() if k != "labels"}
                results = self.model(**model_inputs)
                running_loss += ce_loss(results.logits, labels).item()
                running_entropy += self.compute_entropy(results.logits, labels).item()
                running_token_acc += self.compute_token_accuracy(results.logits, labels).item()
                n_batches += 1

        denom = max(n_batches, 1)
        # gather across GPUs so all processes have the same avg
        avg_loss_t = self.accelerator.gather(
            torch.tensor(running_loss / denom, device=self.accelerator.device)
        ).mean().item()
        avg_entropy_t = self.accelerator.gather(
            torch.tensor(running_entropy / denom, device=self.accelerator.device)
        ).mean().item()
        avg_token_acc_t = self.accelerator.gather(
            torch.tensor(running_token_acc / denom, device=self.accelerator.device)
        ).mean().item()

        self._log(
            {
                "val/loss": avg_loss_t,
                # ce_loss is already token-mean NLL — alias kept for readability
                "val/nll": avg_loss_t,
                "val/perplexity": self._perplexity(avg_loss_t),
                "val/entropy": avg_entropy_t,
                "val/token_acc": avg_token_acc_t,
                # lets you tell a capped mid-epoch point from a full epoch-end one
                "val/n_batches": n_batches,
                "epoch": epoch + 1,
            },
            step=global_step,
        )
        scope = f"{n_batches} batches" if max_batches else "full"
        self.accelerator.print(
            f"[val] epoch {epoch+1} | step {global_step} | {scope} | "
            f"loss {avg_loss_t:.4f} | ppl {self._perplexity(avg_loss_t):.2f} | "
            f"entropy {avg_entropy_t:.4f} | token_acc {avg_token_acc_t:.4f}"
        )

        # restore train mode — without this a mid-epoch call would leave the
        # model in eval() (dropout off) for the rest of the epoch
        if was_training:
            self.model.train()
        return avg_loss_t

    # ── training loop ─────────────────────────────────────────────────────────

    def train_epoch(
        self,
        epoch: int,
        train_loader: DataLoader,
        global_step: int,
        val_loader: DataLoader | None = None,
    ) -> tuple[float, int]:
        grad_accumulation = self.train_cfg.gradient_accumulation_steps
        grad_clip = self.train_cfg.max_grad_norm
        log_interval = self.train_cfg.logging_steps
        save_steps = self.train_cfg.save_steps
        eval_steps = self.train_cfg.eval_steps

        self.model.train()
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{self.train_cfg.num_epochs}",
            leave=True,
            disable=not self.accelerator.is_main_process,
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.accelerator.device)

        running_loss = 0.0
        last_grad_norm = 0.0
        last_entropy = 0.0
        last_token_acc = 0.0
        tokens_since_log = 0
        start_time = time.time()
        last_log_time = start_time
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
            labels = batch["labels"]
            model_inputs = {k: v for k, v in batch.items() if k != "labels"}
            tokens_since_log += batch["attention_mask"].sum().item()

            with self.accelerator.autocast():
                results = self.model(**model_inputs)
                loss = ce_loss(results.logits, labels) / grad_accumulation

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

                # compute entropy/accuracy at optimizer step so they're always fresh
                with torch.no_grad():
                    last_entropy = self.compute_entropy(results.logits, labels).item()
                    last_token_acc = self.compute_token_accuracy(results.logits, labels).item()

                if global_step % save_steps == 0:
                    mid_ckpt = self.save_checkpoint(epoch, global_step)
                    # rolling resume point — drop the previous one so mid-epoch
                    # saves don't accumulate (~2GB each) and fill the Kaggle disk
                    self._retain_mid_epoch_checkpoint(mid_ckpt)

                # mid-epoch validation — without this, val/* is logged only at
                # epoch boundaries (2 points for a 2-epoch run, and just 1 if the
                # session hits the Kaggle wall mid-epoch), which is invisible on
                # a chart. Capped by eval_max_batches: a full pass costs ~4
                # training steps. All ranks hit this together — global_step is
                # rank-invariant — so the gather inside validate() stays aligned.
                if (
                    eval_steps
                    and val_loader is not None
                    and global_step % eval_steps == 0
                ):
                    self.validate(
                        epoch,
                        val_loader,
                        global_step,
                        max_batches=self.train_cfg.eval_max_batches,
                    )

                # log at optimizer-step granularity — guarantees monotonically increasing
                # wandb step and avoids duplicate step values that silently stop chart updates
                if global_step % log_interval == 0:
                    # gpu-mem and token-throughput are collective (accelerator.gather) —
                    # every rank must call these, so they sit OUTSIDE the
                    # is_main_process gate below or the gather deadlocks on multi-GPU
                    gpu_alloc_gb, gpu_reserved_gb = self._gpu_mem_stats()
                    tokens_t = self.accelerator.gather(
                        torch.tensor(float(tokens_since_log), device=self.accelerator.device)
                    ).sum().item()

                    if self.accelerator.is_main_process:
                        elapsed = time.time() - start_time
                        window = max(time.time() - last_log_time, 1e-6)
                        current_loss = running_loss / (batch_idx + 1)
                        tokens_per_sec = tokens_t / window
                        self.accelerator.print(
                            f"[train] epoch {epoch+1} | step {global_step} | "
                            f"loss {current_loss:.4f} | ppl {self._perplexity(current_loss):.2f} | "
                            f"entropy {last_entropy:.4f} | token_acc {last_token_acc:.4f} | "
                            f"grad_norm {last_grad_norm:.3f} | lr {self._get_lr():.2e} | "
                            f"tok/s {tokens_per_sec:.0f} | gpu_mem {gpu_alloc_gb:.1f}GB | "
                            f"elapsed {elapsed:.1f}s"
                        )
                        self._log(
                            {
                                "train/loss": current_loss,
                                # ce_loss is already token-mean NLL — alias kept for readability
                                "train/nll": current_loss,
                                "train/perplexity": self._perplexity(current_loss),
                                "train/entropy": last_entropy,
                                "train/token_acc": last_token_acc,
                                "train/grad_norm": last_grad_norm,
                                "train/lr": self._get_lr(),
                                "train/tokens_per_sec": tokens_per_sec,
                                "train/gpu_mem_alloc_gb": gpu_alloc_gb,
                                "train/gpu_mem_reserved_gb": gpu_reserved_gb,
                                "epoch": epoch + 1,
                            },
                            step=global_step,
                        )

                    # reset on every rank — tokens_since_log feeds the next window's gather
                    tokens_since_log = 0
                    last_log_time = time.time()

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

        train_data, val_data = self._group_split(data, val_frac=0.2, seed=42)

        # data sanity: catch wrong-file mistakes (missing tags, truncated responses)
        # before spending GPU-hours
        probe = train_data[:100]
        tagged = sum(1 for d in probe if "<answer>" in d["response"] and "</answer>" in d["response"])
        sample = train_data[0]["response"]
        self.accelerator.print(
            f"[data] {data_path} | {len(train_data)} train / {len(val_data)} val\n"
            f"[data] <answer> tag coverage (first {len(probe)}): {tagged}/{len(probe)}\n"
            f"[data] sample response head: {sample[:120]!r}\n"
            f"[data] sample response tail: {sample[-120:]!r}"
        )

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

        # correct total steps: per-process batches / grad_acc (ceil — the loop also
        # steps on the final partial accumulation window) * epochs
        total_optimizer_steps = -(
            -len(train_loader) // self.train_cfg.gradient_accumulation_steps
        ) * self.train_cfg.num_epochs

        self._setup_scheduler(total_optimizer_steps, completed_steps=global_step)
        # NOTE: deliberately NOT passed through accelerator.prepare() — the prepared
        # wrapper steps the scheduler num_processes times per .step() call, which
        # compressed the cosine schedule 2x on 2 GPUs (it hit zero at mid-training,
        # then rebounded to ~max LR). Each rank steps its identical local scheduler
        # once per optimizer step instead.

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
            avg_train_loss, global_step = self.train_epoch(
                epoch, train_loader, global_step, val_loader=val_loader
            )
            # epoch-boundary pass is always FULL (uncapped) — this is the number
            # best-checkpoint selection uses
            avg_val_loss = self.validate(epoch, val_loader, global_step)
            ckpt = self.save_checkpoint(epoch, global_step)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_checkpoint_path = ckpt
                self.accelerator.print(f"New best val_loss {best_val_loss:.4f} → {ckpt}")

            # prune AFTER best-tracking updates, and protect the best explicitly:
            # best_checkpoint_path is what gets zipped and pushed to Kaggle Models
            # once the loop ends, so it must survive pruning.
            self._retain_best_k_checkpoints(ckpt, avg_val_loss, best_checkpoint_path)

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
