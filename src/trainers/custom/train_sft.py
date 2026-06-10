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
from contextlib import nullcontext
from pathlib import Path
import sys
import json
import time
import random
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data import create_sft_dataloader
from trainers.utils import dft_loss, entropy_from_logits
from configs import Config

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SFTTrainer:
    def __init__(self, cfg: Config):
        self.train_cfg = cfg.training_params
        self.model_cfg = cfg.model_params
        self.data_cfg = cfg.dataloader_params
        self.scheduler = None
        self._wandb = None

        self._load_model()
        self._setup_optimizer()

    def add_special_tokens(self, specials=["<think>", "</think>", "<answer>", "</answer>"]):
        self.tokenizer.add_special_tokens({"additional_special_tokens": specials})
        if self.model is not None:
            self.model.resize_token_embeddings(len(self.tokenizer))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_model(self):
        model_name = self.model_cfg.base_model
        trust_remote = self.model_cfg.trust_remote_code
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote)

        bnb_config = None
        torch_dtype = torch.bfloat16 if self.train_cfg.bf16 else torch.float16

        if self.model_cfg.load_in_4bit:
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
            device_map="auto",
            trust_remote_code=trust_remote,
        )

        if self.train_cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.add_special_tokens()

    def _setup_optimizer(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
        )

    def _setup_scheduler(self, total_steps: int):
        warmup = self.train_cfg.warmup_steps
        if self.train_cfg.lr_scheduler == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup, num_training_steps=total_steps
            )
        elif self.train_cfg.lr_scheduler == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup, num_training_steps=total_steps
            )
        # "constant" → scheduler stays None, lr unchanged

    def _init_wandb(self):
        if not self.train_cfg.wandb_project:
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
                },
            )
            self._wandb = wandb
        except ImportError:
            print("wandb not installed — skipping wandb logging.")

    def _log(self, metrics: dict, step: int):
        if self._wandb:
            self._wandb.log(metrics, step=step)

    def _get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def compute_entropy(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        entropy = entropy_from_logits(logits)       # [B, L]
        mask = labels != -100                       # response tokens only
        return (entropy * mask).sum() / mask.sum().clamp(min=1)

    def save_checkpoint(self, epoch: int, global_step: int):
        out = Path(self.train_cfg.output_dir) / f"checkpoint-epoch{epoch+1}-step{global_step}"
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out)
        self.tokenizer.save_pretrained(out)
        print(f"Checkpoint saved → {out}")

    def validate(self, epoch: int, val_loader: DataLoader, global_step: int) -> float:
        self.model.eval()
        running_loss = 0.0
        running_entropy = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(_DEVICE) for k, v in batch.items()}
                labels = batch["labels"]
                model_inputs = {k: v for k, v in batch.items() if k != "labels"}
                results = self.model(**model_inputs)
                running_loss += dft_loss(results.logits, labels).item()
                running_entropy += self.compute_entropy(results.logits, labels).item()

        avg_loss = running_loss / len(val_loader)
        avg_entropy = running_entropy / len(val_loader)

        self._log(
            {"val/loss": avg_loss, "val/entropy": avg_entropy, "epoch": epoch + 1},
            step=global_step,
        )
        print(f"[val] epoch {epoch+1} | loss {avg_loss:.4f} | entropy {avg_entropy:.4f}")
        return avg_loss

    def train_epoch(
        self, epoch: int, train_loader: DataLoader, global_step: int
    ) -> tuple[float, int]:
        dtype = torch.bfloat16 if self.train_cfg.bf16 else torch.float16
        autocast_ctx = (
            nullcontext()
            if _DEVICE.type == "cpu"
            else torch.cuda.amp.autocast(dtype=dtype)
        )
        scaler = torch.cuda.amp.GradScaler(enabled=(not self.train_cfg.bf16))

        grad_accumulation = self.train_cfg.gradient_accumulation_steps
        grad_clip = self.train_cfg.max_grad_norm
        log_interval = self.train_cfg.logging_steps
        save_steps = self.train_cfg.save_steps

        self.model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.train_cfg.num_epochs}", leave=True)

        running_loss = 0.0
        last_grad_norm = 0.0
        start_time = time.time()
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(_DEVICE) for k, v in batch.items()}
            labels = batch["labels"]
            model_inputs = {k: v for k, v in batch.items() if k != "labels"}

            with autocast_ctx:
                results = self.model(**model_inputs)
                loss = dft_loss(results.logits, labels) / grad_accumulation

            scaler.scale(loss).backward()

            is_accum_step = (batch_idx + 1) % grad_accumulation == 0
            is_last_batch = batch_idx == len(train_loader) - 1

            if is_accum_step or is_last_batch:
                scaler.unscale_(self.optimizer)
                if grad_clip:
                    last_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), grad_clip
                    ).item()
                scaler.step(self.optimizer)
                scaler.update()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % save_steps == 0:
                    self.save_checkpoint(epoch, global_step)

            running_loss += loss.item() * grad_accumulation

            if batch_idx % log_interval == 0:
                elapsed = time.time() - start_time
                with torch.no_grad():
                    mean_entropy = self.compute_entropy(results.logits, labels).item()
                current_loss = running_loss / (batch_idx + 1)

                print(
                    f"[train] epoch {epoch+1} | step {global_step} | "
                    f"loss {current_loss:.4f} | entropy {mean_entropy:.4f} | "
                    f"grad_norm {last_grad_norm:.3f} | lr {self._get_lr():.2e} | "
                    f"elapsed {elapsed:.1f}s"
                )
                self._log(
                    {
                        "train/loss": current_loss,
                        "train/entropy": mean_entropy,
                        "train/grad_norm": last_grad_norm,
                        "train/lr": self._get_lr(),
                        "epoch": epoch + 1,
                    },
                    step=global_step,
                )

            pbar.set_postfix(
                {"loss": f"{running_loss / (batch_idx + 1):.4f}", "lr": f"{self._get_lr():.6f}"}
            )

        return running_loss / len(train_loader), global_step

    def __call__(self, data_path: str):
        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))

        random.seed(42)
        random.shuffle(data)

        split = int(0.8 * len(data))
        train_data, val_data = data[:split], data[split:]

        train_loader = create_sft_dataloader(
            data=train_data, tokenizer=self.tokenizer, data_config=self.data_cfg
        )
        val_loader = create_sft_dataloader(
            data=val_data, tokenizer=self.tokenizer, data_config=self.data_cfg
        )

        total_optimizer_steps = (
            len(train_loader) // self.train_cfg.gradient_accumulation_steps
        ) * self.train_cfg.num_epochs
        self._setup_scheduler(total_optimizer_steps)
        self._init_wandb()

        global_step = 0
        for epoch in range(self.train_cfg.num_epochs):
            avg_train_loss, global_step = self.train_epoch(epoch, train_loader, global_step)
            avg_val_loss = self.validate(epoch, val_loader, global_step)
            self.save_checkpoint(epoch, global_step)
            print(
                f"Epoch {epoch+1} complete | "
                f"train_loss {avg_train_loss:.4f} | val_loss {avg_val_loss:.4f}"
            )

        if self._wandb:
            self._wandb.finish()
