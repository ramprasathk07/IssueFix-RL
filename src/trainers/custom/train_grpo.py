"""Repository-owned GRPO training for IssueFix code-repair data.

The optimization loop is implemented directly with PyTorch. Transformers owns
model/tokenizer loading, PEFT owns LoRA adapters, and Accelerate supplies mixed
precision and DDP plumbing; no high-level RL trainer is used.

Generated code is parsed for a syntax reward but is never executed. Reference
similarity is deliberately only one component: exact string matching alone is
too sparse for semantically equivalent patches.
"""
from __future__ import annotations

import ast
import json
import math
import re
from contextlib import nullcontext
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from configs import Config
from data.loader import SYSTEM_PROMPT


_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_FORMAT_RE = re.compile(
    r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$",
    re.DOTALL | re.IGNORECASE,
)
_FENCE_RE = re.compile(r"^\s*```(?:python)?\s*|\s*```\s*$", re.IGNORECASE)
_TRAINING_STATE_FILE = "training_state.pt"


def _completion_text(completion: Any) -> str:
    """Normalize plain-text or conversational completion representations."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        item = completion[0]
        if isinstance(item, dict):
            return str(item.get("content", ""))
    return str(completion or "")


def _answer(text: str) -> str:
    match = _ANSWER_RE.search(text)
    value = match.group(1) if match else text
    return _FENCE_RE.sub("", value).strip()


def format_reward(completions, **kwargs) -> list[float]:
    """Reward the required think/answer response envelope."""
    return [1.0 if _FORMAT_RE.fullmatch(_completion_text(c)) else 0.0 for c in completions]


def python_syntax_reward(completions, **kwargs) -> list[float]:
    """Parse answers without executing them; non-Python answers receive zero."""
    rewards = []
    for completion in completions:
        code = _answer(_completion_text(completion))
        try:
            ast.parse(code)
            rewards.append(1.0 if code else 0.0)
        except (SyntaxError, ValueError, TypeError, MemoryError):
            rewards.append(0.0)
    return rewards


def reference_similarity_reward(completions, solution, **kwargs) -> list[float]:
    """Give a dense lexical signal toward the verified patch, bounded to [0, 1]."""
    rewards = []
    for completion, reference in zip(completions, solution, strict=True):
        candidate = " ".join(_answer(_completion_text(completion)).split())
        target = " ".join(str(reference).split())
        rewards.append(SequenceMatcher(None, candidate, target, autojunk=False).ratio())
    return rewards


def group_normalized_advantages(
    rewards: torch.Tensor,
    num_generations: int,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Normalize rewards independently within each prompt's completion group."""
    if rewards.ndim != 1:
        raise ValueError(f"rewards must be one-dimensional, got shape {tuple(rewards.shape)}")
    if num_generations < 2:
        raise ValueError("GRPO requires at least two generations per prompt.")
    if rewards.numel() % num_generations:
        raise ValueError(
            f"{rewards.numel()} rewards cannot be grouped into sets of {num_generations}."
        )
    grouped = rewards.reshape(-1, num_generations)
    means = grouped.mean(dim=1, keepdim=True)
    stds = grouped.std(dim=1, keepdim=True, unbiased=False)
    return ((grouped - means) / (stds + epsilon)).reshape_as(rewards)


def build_completion_mask(
    completion_ids: torch.Tensor,
    eos_token_id: int | list[int] | tuple[int, ...] | None,
    pad_token_id: int | None,
    mask_truncated: bool,
) -> torch.Tensor:
    """Keep tokens through the first EOS and optionally reject max-length rollouts."""
    if completion_ids.ndim != 2:
        raise ValueError("completion_ids must have shape [batch, completion_length].")

    mask = torch.ones_like(completion_ids, dtype=torch.bool)
    if pad_token_id is not None:
        mask &= completion_ids.ne(pad_token_id)

    if eos_token_id is None:
        return torch.zeros_like(mask) if mask_truncated else mask

    eos_ids = (
        list(eos_token_id)
        if isinstance(eos_token_id, (list, tuple))
        else [eos_token_id]
    )
    eos_hits = torch.zeros_like(completion_ids, dtype=torch.bool)
    for token_id in eos_ids:
        eos_hits |= completion_ids.eq(token_id)

    has_eos = eos_hits.any(dim=1)
    first_eos = eos_hits.to(torch.int64).argmax(dim=1)
    positions = torch.arange(completion_ids.shape[1], device=completion_ids.device)
    through_eos = positions.unsqueeze(0) <= first_eos.unsqueeze(1)
    mask &= torch.where(has_eos.unsqueeze(1), through_eos, torch.ones_like(through_eos))
    if mask_truncated:
        mask &= has_eos.unsqueeze(1)
    return mask


def _grpo_token_terms(
    current_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float,
    reference_logps: torch.Tensor | None,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-token loss, policy KL estimate, clipping, and reference KL."""
    if current_logps.shape != old_logps.shape:
        raise ValueError("current_logps and old_logps must have identical shapes.")
    if advantages.ndim != 1 or advantages.shape[0] != current_logps.shape[0]:
        raise ValueError("advantages must have one value per completion.")
    if reference_logps is not None and reference_logps.shape != current_logps.shape:
        raise ValueError("reference_logps must match current_logps when provided.")

    log_ratio = (current_logps - old_logps).clamp(min=-20.0, max=20.0)
    ratio = log_ratio.exp()
    expanded_advantages = advantages.unsqueeze(1)
    unclipped = ratio * expanded_advantages
    clipped_ratio = ratio.clamp(1.0 - epsilon, 1.0 + epsilon)
    clipped = clipped_ratio * expanded_advantages
    objective = torch.minimum(unclipped, clipped)

    old_delta = old_logps - current_logps
    policy_kl = old_delta.exp() - old_delta - 1.0
    clip_indicator = ratio.ne(clipped_ratio).to(current_logps.dtype)

    reference_kl = torch.zeros_like(current_logps)
    if reference_logps is not None and beta:
        reference_delta = reference_logps - current_logps
        reference_kl = reference_delta.exp() - reference_delta - 1.0
        objective = objective - beta * reference_kl

    return -objective, policy_kl, clip_indicator, reference_kl


def grpo_policy_loss(
    current_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    epsilon: float,
    loss_type: str = "dapo",
    reference_logps: torch.Tensor | None = None,
    beta: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the clipped GRPO loss and masked policy diagnostics.

    ``dapo`` uses a token-level denominator. ``grpo`` first averages each
    completion and then averages the valid completions.
    """
    if completion_mask.shape != current_logps.shape:
        raise ValueError("completion_mask must match the log-probability tensors.")
    token_loss, policy_kl, clipped, _ = _grpo_token_terms(
        current_logps,
        old_logps,
        advantages,
        epsilon,
        reference_logps,
        beta,
    )
    mask = completion_mask.to(current_logps.dtype)
    valid_tokens = mask.sum()

    if loss_type == "dapo":
        loss = (token_loss * mask).sum() / valid_tokens.clamp_min(1.0)
    elif loss_type == "grpo":
        lengths = mask.sum(dim=1)
        valid_sequences = lengths.gt(0)
        sequence_losses = (token_loss * mask).sum(dim=1) / lengths.clamp_min(1.0)
        loss = (
            sequence_losses[valid_sequences].mean()
            if valid_sequences.any()
            else token_loss.sum() * 0
        )
    else:
        raise ValueError(f"Unsupported GRPO loss_type: {loss_type!r}")

    approx_kl = (policy_kl * mask).sum() / valid_tokens.clamp_min(1.0)
    clip_fraction = (clipped * mask).sum() / valid_tokens.clamp_min(1.0)
    return loss, approx_kl, clip_fraction


class GRPOIssueFixTrainer:
    """Custom PyTorch GRPO trainer with LoRA, DDP, AMP, and checkpoint support."""

    def __init__(self, cfg: Config):
        if cfg.grpo_params is None:
            raise ValueError("The GRPO config needs a grpo_params section.")
        if cfg.model_params.load_in_4bit or cfg.model_params.load_in_8bit:
            raise ValueError(
                "Quantized model placement is not supported by this multi-GPU GRPO path. "
                "Use LoRA with fp16/bf16 weights instead."
            )
        if cfg.grpo_params.beta > 0 and not cfg.model_params.use_lora:
            raise ValueError(
                "beta > 0 requires LoRA so the frozen base policy can be used as the "
                "reference model without allocating a second full model."
            )
        reward_weight = (
            cfg.grpo_params.format_reward_weight
            + cfg.grpo_params.syntax_reward_weight
            + cfg.grpo_params.reference_reward_weight
        )
        if reward_weight <= 0:
            raise ValueError("At least one GRPO reward weight must be greater than zero.")

        self.cfg = cfg
        self.model_cfg = cfg.model_params
        self.data_cfg = cfg.dataloader_params
        self.train_cfg = cfg.training_params
        self.grpo_cfg = cfg.grpo_params
        mixed_precision = (
            "bf16" if self.train_cfg.bf16 else ("fp16" if self.train_cfg.fp16 else "no")
        )
        self.accelerator = Accelerator(mixed_precision=mixed_precision)
        set_seed(self.grpo_cfg.seed, device_specific=True)

        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self._wandb = None
        self._resume_wandb_id: str | None = None

    def _load_model(self, resume_dir: Path | None) -> None:
        tokenizer_source = (
            str(resume_dir)
            if resume_dir is not None and (resume_dir / "tokenizer_config.json").exists()
            else self.model_cfg.base_model
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=self.model_cfg.trust_remote_code,
            padding_side="left",
            truncation_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = (
            torch.bfloat16
            if self.train_cfg.bf16
            else (torch.float16 if self.train_cfg.fp16 else torch.float32)
        )
        model_source = (
            str(resume_dir)
            if resume_dir is not None and not self.model_cfg.use_lora
            else self.model_cfg.base_model
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            trust_remote_code=self.model_cfg.trust_remote_code,
            dtype=dtype,
            device_map=None,
        )

        if self.model_cfg.use_lora:
            from peft import LoraConfig, PeftModel, get_peft_model

            if resume_dir is not None:
                model = PeftModel.from_pretrained(model, str(resume_dir), is_trainable=True)
            else:
                model = get_peft_model(
                    model,
                    LoraConfig(
                        r=self.model_cfg.lora_r or 16,
                        lora_alpha=self.model_cfg.lora_alpha or 32,
                        lora_dropout=self.model_cfg.lora_dropout,
                        target_modules=self.model_cfg.lora_target_modules,
                        bias="none",
                        task_type="CAUSAL_LM",
                    ),
                )
            if self.accelerator.is_main_process:
                model.print_trainable_parameters()

        if self.train_cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        # Policy ratios assume the old and current policy are deterministic for
        # the same sequence. Keep dropout disabled while still using train mode
        # so gradient checkpointing remains active during policy updates.
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        model.config.use_cache = False
        self.model = model

    def _make_optimizer(self) -> None:
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if self.train_cfg.optimizer == "adamw_8bit":
            try:
                import bitsandbytes as bnb

                self.optimizer = bnb.optim.AdamW8bit(
                    parameters,
                    lr=self.train_cfg.learning_rate,
                    weight_decay=self.train_cfg.weight_decay,
                )
                return
            except ImportError:
                self.accelerator.print(
                    "[optim] bitsandbytes unavailable; falling back to torch.optim.AdamW"
                )
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
        )

    def _make_scheduler(self, total_steps: int):
        scheduler_factory = (
            get_linear_schedule_with_warmup
            if self.train_cfg.lr_scheduler == "linear"
            else get_cosine_schedule_with_warmup
        )
        return scheduler_factory(
            self.optimizer,
            num_warmup_steps=self.train_cfg.warmup_steps,
            num_training_steps=total_steps,
        )

    @staticmethod
    def _collate(rows: list[dict[str, str]]) -> dict[str, list[str]]:
        return {
            "problem": [row["problem"] for row in rows],
            "solution": [row["solution"] for row in rows],
        }

    def _load_rows(self, data_path: str) -> list[dict[str, str]]:
        records = []
        with open(data_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                problem = row.get("problem") or row.get("prompt")
                solution = row.get("solution")
                if not solution:
                    solution = _answer(row.get("response", ""))
                if problem and solution:
                    records.append({"problem": str(problem), "solution": str(solution)})
        if not records:
            raise ValueError(
                "No usable GRPO rows; expected problem/prompt and solution/response fields."
            )
        return records

    def _prompt_text(self, problem: str) -> str:
        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    @staticmethod
    def _selective_log_softmax(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute selected-token log probabilities without materializing log_softmax."""
        selected = logits.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
        return selected - torch.logsumexp(logits, dim=-1)

    def _forward_logps(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_width: int,
        completion_ids: torch.Tensor,
    ) -> torch.Tensor:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        completion_logits = outputs.logits[:, prompt_width - 1 : -1]
        return self._selective_log_softmax(completion_logits, completion_ids).float()

    @torch.no_grad()
    def _batched_logps(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_width: int,
        completion_ids: torch.Tensor,
    ) -> torch.Tensor:
        batches = []
        micro_batch = self.grpo_cfg.forward_batch_size
        for start in range(0, input_ids.shape[0], micro_batch):
            stop = start + micro_batch
            with self.accelerator.autocast():
                logps = self._forward_logps(
                    model,
                    input_ids[start:stop],
                    attention_mask[start:stop],
                    prompt_width,
                    completion_ids[start:stop],
                )
            batches.append(logps.detach())
        return torch.cat(batches, dim=0)

    def _reward_rollouts(
        self,
        completion_texts: list[str],
        solutions: list[str],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = self.accelerator.device
        components = {
            "format": torch.tensor(format_reward(completion_texts), device=device),
            "syntax": torch.tensor(python_syntax_reward(completion_texts), device=device),
            "reference": torch.tensor(
                reference_similarity_reward(completion_texts, solution=solutions),
                device=device,
            ),
        }
        reward = (
            self.grpo_cfg.format_reward_weight * components["format"]
            + self.grpo_cfg.syntax_reward_weight * components["syntax"]
            + self.grpo_cfg.reference_reward_weight * components["reference"]
        )
        return reward, components

    @torch.no_grad()
    def _rollout(self, batch: dict[str, list[str]]) -> dict[str, Any]:
        prompt_texts = [self._prompt_text(problem) for problem in batch["problem"]]
        prompt_inputs = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.grpo_cfg.max_prompt_length,
            add_special_tokens=False,
        ).to(self.accelerator.device)

        raw_model = self.accelerator.unwrap_model(self.model)
        raw_model.eval()
        previous_cache_setting = raw_model.config.use_cache
        raw_model.config.use_cache = True
        try:
            with self.accelerator.autocast():
                generated_ids = raw_model.generate(
                    **prompt_inputs,
                    max_new_tokens=self.grpo_cfg.max_completion_length,
                    do_sample=True,
                    temperature=self.grpo_cfg.temperature,
                    top_p=self.grpo_cfg.top_p,
                    num_return_sequences=self.grpo_cfg.num_generations,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    synced_gpus=self.accelerator.num_processes > 1,
                )
        finally:
            raw_model.config.use_cache = previous_cache_setting

        generations = self.grpo_cfg.num_generations
        prompt_width = prompt_inputs["input_ids"].shape[1]
        completion_ids = generated_ids[:, prompt_width:]
        repeated_prompt_ids = prompt_inputs["input_ids"].repeat_interleave(generations, dim=0)
        repeated_prompt_mask = prompt_inputs["attention_mask"].repeat_interleave(
            generations, dim=0
        )
        full_input_ids = torch.cat([repeated_prompt_ids, completion_ids], dim=1)
        completion_mask = build_completion_mask(
            completion_ids,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id != self.tokenizer.eos_token_id
            else None,
            self.grpo_cfg.mask_truncated_completions,
        )
        full_attention_mask = torch.cat(
            [repeated_prompt_mask, completion_mask.to(repeated_prompt_mask.dtype)], dim=1
        )

        completion_texts = self.tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        solutions = [
            solution
            for solution in batch["solution"]
            for _ in range(self.grpo_cfg.num_generations)
        ]
        rewards, reward_components = self._reward_rollouts(completion_texts, solutions)
        advantages = group_normalized_advantages(
            rewards,
            num_generations=self.grpo_cfg.num_generations,
            epsilon=self.grpo_cfg.advantage_epsilon,
        )
        old_logps = self._batched_logps(
            raw_model,
            full_input_ids,
            full_attention_mask,
            prompt_width,
            completion_ids,
        )

        reference_logps = None
        if self.grpo_cfg.beta > 0:
            with raw_model.disable_adapter():
                reference_logps = self._batched_logps(
                    raw_model,
                    full_input_ids,
                    full_attention_mask,
                    prompt_width,
                    completion_ids,
                )

        return {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
            "prompt_width": prompt_width,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_logps": old_logps,
            "reference_logps": reference_logps,
            "advantages": advantages,
            "rewards": rewards,
            "reward_components": reward_components,
            "completion_texts": completion_texts,
        }

    def _update_policy(
        self,
        rollout: dict[str, Any],
        accumulation_divisor: int,
        synchronize_gradients: bool,
    ) -> dict[str, float]:
        self.model.train()
        mask = rollout["completion_mask"]
        mask_float = mask.float()
        total_tokens = mask_float.sum().clamp_min(1.0)
        sequence_lengths = mask_float.sum(dim=1)
        valid_sequence_count = sequence_lengths.gt(0).sum().clamp_min(1)
        micro_batch = self.grpo_cfg.forward_batch_size

        loss_total = torch.zeros((), device=self.accelerator.device)
        policy_kl_total = torch.zeros((), device=self.accelerator.device)
        reference_kl_total = torch.zeros((), device=self.accelerator.device)
        clipped_total = torch.zeros((), device=self.accelerator.device)
        sample_count = rollout["input_ids"].shape[0]

        for start in range(0, sample_count, micro_batch):
            stop = min(start + micro_batch, sample_count)
            is_last_micro_batch = stop == sample_count
            should_sync_now = synchronize_gradients and is_last_micro_batch
            sync_context = (
                nullcontext()
                if should_sync_now
                else self.accelerator.no_sync(self.model)
            )

            with sync_context:
                with self.accelerator.autocast():
                    current_logps = self._forward_logps(
                        self.model,
                        rollout["input_ids"][start:stop],
                        rollout["attention_mask"][start:stop],
                        rollout["prompt_width"],
                        rollout["completion_ids"][start:stop],
                    )
                    token_loss, policy_kl, clipped, reference_kl = _grpo_token_terms(
                        current_logps,
                        rollout["old_logps"][start:stop],
                        rollout["advantages"][start:stop],
                        self.grpo_cfg.epsilon,
                        None
                        if rollout["reference_logps"] is None
                        else rollout["reference_logps"][start:stop],
                        self.grpo_cfg.beta,
                    )
                    chunk_mask = mask_float[start:stop]
                    if self.grpo_cfg.loss_type == "dapo":
                        weights = chunk_mask / total_tokens
                    else:
                        weights = (
                            chunk_mask
                            / sequence_lengths[start:stop].clamp_min(1.0).unsqueeze(1)
                            / valid_sequence_count
                        )
                    chunk_loss = (token_loss * weights).sum()

                self.accelerator.backward(chunk_loss / accumulation_divisor)

            loss_total += chunk_loss.detach()
            policy_kl_total += (policy_kl.detach() * chunk_mask).sum()
            reference_kl_total += (reference_kl.detach() * chunk_mask).sum()
            clipped_total += (clipped.detach() * chunk_mask).sum()

        return {
            "loss": loss_total.item(),
            "approx_kl": (policy_kl_total / total_tokens).item(),
            "reference_kl": (reference_kl_total / total_tokens).item(),
            "clip_fraction": (clipped_total / total_tokens).item(),
            "completion_tokens": mask_float.sum(dim=1).mean().item(),
            "valid_completion_fraction": sequence_lengths.gt(0).float().mean().item(),
        }

    def _load_training_state(self, resume_dir: Path) -> dict[str, Any]:
        state_path = resume_dir / _TRAINING_STATE_FILE
        if not state_path.exists():
            raise FileNotFoundError(f"Missing GRPO training state: {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self._resume_wandb_id = state.get("wandb_run_id")
        return state

    def _save_checkpoint(self, epoch: int, batch_idx: int, global_step: int) -> Path | None:
        self.accelerator.wait_for_everyone()
        output = None
        if self.accelerator.is_main_process:
            output = Path(self.train_cfg.output_dir) / (
                f"checkpoint-epoch{epoch + 1}-step{global_step}"
            )
            output.mkdir(parents=True, exist_ok=True)
            unwrapped = self.accelerator.unwrap_model(self.model)
            unwrapped.save_pretrained(output)
            self.tokenizer.save_pretrained(output)
            torch.save(
                {
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "global_step": global_step,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "wandb_run_id": (
                        self._wandb.run.id if self._wandb and self._wandb.run else None
                    ),
                },
                output / _TRAINING_STATE_FILE,
            )
            print(f"GRPO checkpoint saved -> {output}")
        self.accelerator.wait_for_everyone()
        return output

    def _init_wandb(self) -> None:
        if not self.train_cfg.wandb_project or not self.accelerator.is_main_process:
            return
        try:
            import wandb

            wandb.init(
                project=self.train_cfg.wandb_project,
                name=self.train_cfg.wandb_run_name,
                config=self.cfg.model_dump(),
                id=self._resume_wandb_id,
                resume="allow" if self._resume_wandb_id else None,
            )
            self._wandb = wandb
        except ImportError:
            print("wandb unavailable; continuing without it")

    def _mean_across_processes(self, value: float) -> float:
        value_tensor = torch.tensor(value, device=self.accelerator.device)
        return self.accelerator.reduce(value_tensor, reduction="mean").item()

    def _log_step(
        self,
        metrics: dict[str, float],
        rollout: dict[str, Any],
        global_step: int,
    ) -> None:
        reduced = {key: self._mean_across_processes(value) for key, value in metrics.items()}
        reward_components = rollout["reward_components"]
        reduced.update(
            {
                "reward": self._mean_across_processes(rollout["rewards"].mean().item()),
                "reward_std": self._mean_across_processes(
                    rollout["rewards"].std(unbiased=False).item()
                ),
                "reward_format": self._mean_across_processes(
                    reward_components["format"].mean().item()
                ),
                "reward_syntax": self._mean_across_processes(
                    reward_components["syntax"].mean().item()
                ),
                "reward_reference": self._mean_across_processes(
                    reward_components["reference"].mean().item()
                ),
                "lr": self.optimizer.param_groups[0]["lr"],
            }
        )
        if not self.accelerator.is_main_process:
            return

        printable = " | ".join(
            [
                f"loss {reduced['loss']:.4f}",
                f"reward {reduced['reward']:.4f}",
                f"kl {reduced['approx_kl']:.5f}",
                f"clip {reduced['clip_fraction']:.3f}",
                f"tokens {reduced['completion_tokens']:.1f}",
                f"lr {reduced['lr']:.2e}",
            ]
        )
        self.accelerator.print(f"[grpo] step {global_step} | {printable}")
        if self._wandb and self._wandb.run is not None:
            self._wandb.run.log(
                {f"train/{key}": value for key, value in reduced.items()},
                step=global_step,
            )
        if self.grpo_cfg.log_completions:
            for index, text in enumerate(rollout["completion_texts"][:2], start=1):
                reward = rollout["rewards"][index - 1].item()
                print(f"[grpo completion {index} | reward={reward:.3f}]\n{text[:1000]}")

    def __call__(self, data_path: str, resume_from: str | None = None):
        resume_dir = Path(resume_from) if resume_from else None
        self._load_model(resume_dir)
        self._make_optimizer()
        state = self._load_training_state(resume_dir) if resume_dir else None

        rows = self._load_rows(data_path)
        loader_kwargs: dict[str, Any] = {
            "dataset": rows,
            "batch_size": self.data_cfg.batch_size,
            "shuffle": self.data_cfg.shuffle,
            "num_workers": self.data_cfg.num_workers,
            "pin_memory": self.data_cfg.pin_memory,
            "collate_fn": self._collate,
        }
        if self.data_cfg.num_workers:
            loader_kwargs["prefetch_factor"] = self.data_cfg.prefetch_factor
        loader = DataLoader(**loader_kwargs)

        self.model, self.optimizer, loader = self.accelerator.prepare(
            self.model, self.optimizer, loader
        )
        steps_per_epoch = math.ceil(
            len(loader) / self.train_cfg.gradient_accumulation_steps
        )
        total_steps = steps_per_epoch * self.train_cfg.num_epochs
        self.scheduler = self._make_scheduler(total_steps)
        if state and state.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(state["scheduler_state_dict"])

        start_epoch = int(state["epoch"]) if state else 0
        start_batch = int(state.get("batch_idx", -1)) + 1 if state else 0
        global_step = int(state["global_step"]) if state else 0
        if start_batch >= len(loader):
            start_epoch += 1
            start_batch = 0

        self._init_wandb()
        self.accelerator.print(
            "Custom PyTorch GRPO | "
            f"{self.accelerator.num_processes} GPU(s) | {len(rows)} prompts | "
            f"{self.grpo_cfg.num_generations} generations/prompt | "
            f"forward micro-batch {self.grpo_cfg.forward_batch_size} | "
            f"loss={self.grpo_cfg.loss_type}"
        )

        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(start_epoch, self.train_cfg.num_epochs):
            if hasattr(loader, "set_epoch"):
                loader.set_epoch(epoch)
            progress = tqdm(
                loader,
                desc=f"GRPO epoch {epoch + 1}/{self.train_cfg.num_epochs}",
                disable=not self.accelerator.is_main_process,
            )
            for batch_idx, batch in enumerate(progress):
                if epoch == start_epoch and batch_idx < start_batch:
                    continue

                rollout = self._rollout(batch)
                accumulation = self.train_cfg.gradient_accumulation_steps
                window_start = (batch_idx // accumulation) * accumulation
                accumulation_divisor = min(accumulation, len(loader) - window_start)
                should_step = (
                    (batch_idx + 1) % accumulation == 0
                    or batch_idx + 1 == len(loader)
                )
                metrics = self._update_policy(
                    rollout,
                    accumulation_divisor=accumulation_divisor,
                    synchronize_gradients=should_step,
                )

                if should_step:
                    grad_norm = self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.train_cfg.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    metrics["grad_norm"] = float(grad_norm)
                    progress.set_postfix(
                        loss=f"{metrics['loss']:.4f}",
                        reward=f"{rollout['rewards'].mean().item():.3f}",
                    )

                    if global_step % self.train_cfg.logging_steps == 0:
                        self._log_step(metrics, rollout, global_step)
                    if global_step % self.train_cfg.save_steps == 0:
                        self._save_checkpoint(epoch, batch_idx, global_step)

            self._save_checkpoint(epoch, len(loader) - 1, global_step)
            start_batch = 0

        if self._wandb:
            self._wandb.finish()
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()
