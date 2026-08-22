"""Two-GPU On-Policy Self-Distillation (OPSD) trainer.

Student rollouts are produced on GPU 0. A frozen copy of the same base model
uses a privileged verified-solution context on GPU 1 to score those exact
tokens. Only top-k distributions cross devices, keeping T4 memory manageable.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import (
    LoraConfig, get_peft_model, prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup,
)

from configs import Config
from data import create_opsd_dataloader


class OPSDTrainer:
    """Train a LoRA student against a fixed privileged-context self-teacher."""

    def __init__(self, cfg: Config):
        if cfg.opsd_params is None:
            raise ValueError("The OPSD config needs an opsd_params section.")
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("Dual-device OPSD requires two CUDA GPUs.")
        if cfg.training_params.bf16 and not torch.cuda.is_bf16_supported():
            raise ValueError("T4 does not support bf16; use bf16=false and fp16=true.")
        if not cfg.model_params.use_lora and cfg.model_params.load_in_4bit:
            raise ValueError(
                "Full fine-tuning cannot update 4-bit weights; set load_in_4bit=false."
            )
        self.cfg, self.model_cfg = cfg, cfg.model_params
        self.train_cfg, self.opsd_cfg = cfg.training_params, cfg.opsd_params
        self.student_device = torch.device(self.opsd_cfg.student_device)
        self.teacher_device = torch.device(self.opsd_cfg.teacher_device)
        if self.student_device == self.teacher_device:
            raise ValueError("student_device and teacher_device must differ.")
        self.dtype = torch.bfloat16 if self.train_cfg.bf16 else torch.float16
        torch.backends.cuda.matmul.allow_tf32 = self.train_cfg.tf32

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_cfg.base_model, trust_remote_code=self.model_cfg.trust_remote_code,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.student = self._load_student()
        self.teacher = self._load_teacher()
        self.optimizer = self._make_optimizer()
        self.scheduler = None
        self._wandb = None

    def _quant_config(self, enabled: bool):
        if not enabled:
            return None
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=self.dtype,
            bnb_4bit_quant_type=self.model_cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.model_cfg.bnb_4bit_use_double_quant,
        )

    def _load_student(self):
        quantized = self.model_cfg.load_in_4bit
        model = AutoModelForCausalLM.from_pretrained(
            self.model_cfg.base_model, trust_remote_code=self.model_cfg.trust_remote_code,
            quantization_config=self._quant_config(quantized),
            dtype=None if quantized else self.dtype,
            device_map={"": self.student_device.index},
        )
        if quantized:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=self.train_cfg.gradient_checkpointing
            )
        elif self.train_cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        model.config.use_cache = False
        if self.model_cfg.use_lora:
            model = get_peft_model(model, LoraConfig(
                r=self.model_cfg.lora_r or 16, lora_alpha=self.model_cfg.lora_alpha or 32,
                lora_dropout=self.model_cfg.lora_dropout,
                target_modules=self.model_cfg.lora_target_modules,
                bias="none", task_type="CAUSAL_LM",
            ))
            model.print_trainable_parameters()
        else:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Full student fine-tuning: {trainable:,} trainable parameters")
        return model

    def _load_teacher(self):
        quantized = self.opsd_cfg.teacher_load_in_4bit
        model = AutoModelForCausalLM.from_pretrained(
            self.model_cfg.base_model, trust_remote_code=self.model_cfg.trust_remote_code,
            quantization_config=self._quant_config(quantized),
            dtype=None if quantized else self.dtype,
            device_map={"": self.teacher_device.index},
        )
        model.requires_grad_(False)
        model.eval()
        model.config.use_cache = True
        return model

    def _make_optimizer(self):
        params = [p for p in self.student.parameters() if p.requires_grad]
        if self.train_cfg.optimizer == "adamw_8bit":
            try:
                import bitsandbytes as bnb
                return bnb.optim.AdamW8bit(params, lr=self.train_cfg.learning_rate,
                                            weight_decay=self.train_cfg.weight_decay)
            except ImportError:
                print("[optim] bitsandbytes unavailable; using torch AdamW")
        return torch.optim.AdamW(params, lr=self.train_cfg.learning_rate,
                                 weight_decay=self.train_cfg.weight_decay)

    def _chat_prompt(self, content: str, thinking: bool) -> str:
        messages = []
        if self.opsd_cfg.system_prompt:
            messages.append({"role": "system", "content": self.opsd_cfg.system_prompt})
        messages.append({"role": "user", "content": content})
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=thinking, **kwargs)
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _prompts(self, row: dict[str, str]) -> tuple[str, str]:
        problem, solution = row["problem"], row["solution"]
        student = self._chat_prompt(
            f"Problem:\n{problem}\n\nReason step by step and provide the final answer.",
            self.opsd_cfg.student_thinking)
        teacher = self._chat_prompt(
            f"Problem:\n{problem}\n\nVerified reference solution:\n"
            f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n\n"
            "Use the reference to understand the correct approach. Now independently continue "
            "with a correct step-by-step answer to the original problem.",
            self.opsd_cfg.teacher_thinking)
        return student, teacher

    def _encode(self, text: str, device: torch.device):
        return self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.opsd_cfg.max_prompt_length, add_special_tokens=False,
        ).to(device)

    @torch.no_grad()
    def _rollout(self, inputs) -> torch.Tensor:
        self.student.eval()
        old_cache = self.student.config.use_cache
        self.student.config.use_cache = True
        generated = self.student.generate(
            **inputs, max_new_tokens=self.opsd_cfg.max_completion_length,
            do_sample=True, temperature=self.opsd_cfg.temperature,
            top_p=self.opsd_cfg.top_p, pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id, use_cache=True,
        )
        self.student.config.use_cache = old_cache
        self.student.train()
        return generated[:, inputs["input_ids"].shape[1]:]

    def _loss(self, row: dict[str, str]) -> tuple[torch.Tensor, int]:
        student_text, teacher_text = self._prompts(row)
        student_inputs = self._encode(student_text, self.student_device)
        completion = self._rollout(student_inputs)
        if completion.shape[1] == 0:
            raise RuntimeError("Student produced an empty completion.")
        student_ids = torch.cat([student_inputs["input_ids"], completion], dim=1)
        student_prompt_len = student_inputs["input_ids"].shape[1]

        teacher_inputs = self._encode(teacher_text, self.teacher_device)
        teacher_ids = torch.cat(
            [teacher_inputs["input_ids"], completion.to(self.teacher_device)], dim=1)
        teacher_prompt_len = teacher_inputs["input_ids"].shape[1]
        with torch.no_grad(), torch.autocast("cuda", dtype=self.dtype):
            logits = self.teacher(input_ids=teacher_ids,
                                  attention_mask=torch.ones_like(teacher_ids)).logits
            logits = logits[:, teacher_prompt_len - 1:-1]
            k = min(self.opsd_cfg.top_k_loss, logits.shape[-1])
            values, indices = torch.topk(logits, k=k, dim=-1)
            teacher_logp = F.log_softmax(
                values / self.opsd_cfg.temperature, dim=-1).to(self.student_device)
            indices = indices.to(self.student_device)
        del logits, values

        with torch.autocast("cuda", dtype=self.dtype):
            logits = self.student(input_ids=student_ids,
                                  attention_mask=torch.ones_like(student_ids)).logits
            logits = logits[:, student_prompt_len - 1:-1]
            student_logp = F.log_softmax(
                torch.gather(logits, -1, indices) / self.opsd_cfg.temperature, dim=-1)
            beta = self.opsd_cfg.jsd_beta
            if 0 < beta < 1:
                mixture = torch.logsumexp(torch.stack([
                    student_logp + math.log(1 - beta),
                    teacher_logp + math.log(beta)]), dim=0)
                per_token = beta * F.kl_div(
                    mixture, teacher_logp, reduction="none", log_target=True).sum(-1)
                per_token += (1 - beta) * F.kl_div(
                    mixture, student_logp, reduction="none", log_target=True).sum(-1)
            else:
                input_p, target_p = ((student_logp, teacher_logp) if beta == 0
                                     else (teacher_logp, student_logp))
                per_token = F.kl_div(
                    input_p, target_p, reduction="none", log_target=True).sum(-1)
            if self.opsd_cfg.token_clip > 0:
                per_token = per_token.clamp(max=self.opsd_cfg.token_clip)
            valid = completion.ne(self.tokenizer.pad_token_id)
            loss = (per_token * valid).sum() / valid.sum().clamp_min(1)
        return loss, int(valid.sum())

    def _save(self, epoch: int, step: int):
        out = Path(self.train_cfg.output_dir) / f"checkpoint-epoch{epoch + 1}-step{step}"
        out.mkdir(parents=True, exist_ok=True)
        self.student.save_pretrained(out)
        self.tokenizer.save_pretrained(out)
        torch.save({
            "epoch": epoch, "global_step": step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
        }, out / "training_state.pt")
        print(f"Checkpoint saved -> {out}")

    def _init_wandb(self):
        if not self.train_cfg.wandb_project:
            return
        try:
            import wandb
            wandb.init(project=self.train_cfg.wandb_project,
                       name=self.train_cfg.wandb_run_name, config=self.cfg.model_dump())
            self._wandb = wandb
        except ImportError:
            print("wandb unavailable; continuing without it")

    def __call__(self, data_path: str, resume_from: str | None = None):
        with open(data_path, "r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        loader = create_opsd_dataloader(rows, self.cfg.dataloader_params)
        updates = math.ceil(len(loader) / self.train_cfg.gradient_accumulation_steps)
        total_steps = updates * self.train_cfg.num_epochs
        scheduler_fn = (get_linear_schedule_with_warmup
                        if self.train_cfg.lr_scheduler == "linear"
                        else get_cosine_schedule_with_warmup)
        start_epoch = global_step = 0
        if resume_from:
            state = torch.load(Path(resume_from) / "training_state.pt",
                               map_location="cpu", weights_only=False)
            weights_file = Path(resume_from) / (
                "adapter_model.safetensors" if self.model_cfg.use_lora else "model.safetensors"
            )
            if not weights_file.exists():
                raise FileNotFoundError(f"Missing student weights: {weights_file}")
            weights = load_file(str(weights_file))
            if self.model_cfg.use_lora:
                set_peft_model_state_dict(self.student, weights)
            else:
                missing, unexpected = self.student.load_state_dict(weights, strict=False)
                # save_pretrained may omit one side of tied input/output embeddings.
                real_missing = [
                    key for key in missing
                    if not key.endswith(("lm_head.weight", "embed_tokens.weight"))
                ]
                if real_missing or unexpected:
                    raise RuntimeError(
                        f"Full-model checkpoint mismatch: missing={real_missing}, "
                        f"unexpected={unexpected}"
                    )
            self.optimizer = self._make_optimizer()
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            start_epoch, global_step = state["epoch"] + 1, state["global_step"]
        self.scheduler = scheduler_fn(self.optimizer, self.train_cfg.warmup_steps, total_steps)
        if resume_from and state.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(state["scheduler_state_dict"])

        self._init_wandb()
        mode = "LoRA" if self.model_cfg.use_lora else "full"
        print(f"OPSD | student={self.student_device} teacher={self.teacher_device} | "
              f"{len(loader)} rows | {mode} fine-tuning | top-k={self.opsd_cfg.top_k_loss}")
        self.student.train()
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(start_epoch, self.train_cfg.num_epochs):
            progress = tqdm(loader, desc=f"OPSD epoch {epoch + 1}")
            for batch_idx, row in enumerate(progress):
                loss, tokens = self._loss(row)
                (loss / self.train_cfg.gradient_accumulation_steps).backward()
                should_step = ((batch_idx + 1) % self.train_cfg.gradient_accumulation_steps == 0
                               or batch_idx + 1 == len(loader))
                if should_step:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.student.parameters() if p.requires_grad],
                        self.train_cfg.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    metrics = {"train/opsd_loss": loss.item(),
                               "train/completion_tokens": tokens,
                               "train/lr": self.scheduler.get_last_lr()[0]}
                    progress.set_postfix(loss=f"{loss.item():.4f}", tokens=tokens)
                    if self._wandb:
                        self._wandb.log(metrics, step=global_step)
                    if global_step % self.train_cfg.logging_steps == 0:
                        print(f"[opsd] step={global_step} {metrics}")
                    if global_step % self.train_cfg.save_steps == 0:
                        self._save(epoch, global_step)
            self._save(epoch, global_step)
        if self._wandb:
            self._wandb.finish()
