"""GRPO training for IssueFix code-repair data using TRL.

Generated code is parsed for a syntax reward but is never executed. Reference
similarity is deliberately only one component: exact string matching alone is
too sparse for semantically equivalent patches.
"""
from __future__ import annotations

import ast
import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer

from configs import Config
from data.loader import SYSTEM_PROMPT


_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_FORMAT_RE = re.compile(
    r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$",
    re.DOTALL | re.IGNORECASE,
)
_FENCE_RE = re.compile(r"^\s*```(?:python)?\s*|\s*```\s*$", re.IGNORECASE)


def _completion_text(completion: Any) -> str:
    """Normalize TRL standard or conversational completion representations."""
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


class GRPOIssueFixTrainer:
    """Repository adapter around TRL's GRPOTrainer."""

    def __init__(self, cfg: Config):
        if cfg.grpo_params is None:
            raise ValueError("The GRPO config needs a grpo_params section.")
        if cfg.model_params.load_in_4bit or cfg.model_params.load_in_8bit:
            raise ValueError(
                "Quantized model placement is not supported by this multi-GPU GRPO path. "
                "Use LoRA with fp16/bf16 weights instead."
            )
        self.cfg = cfg
        self.model_cfg = cfg.model_params
        self.data_cfg = cfg.dataloader_params
        self.train_cfg = cfg.training_params
        self.grpo_cfg = cfg.grpo_params
        self.trainer = None

    def _load_rows(self, data_path: str, tokenizer) -> Dataset:
        records = []
        with open(data_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                problem = row.get("problem") or row.get("prompt")
                solution = row.get("solution")
                if not solution:
                    response = row.get("response", "")
                    solution = _answer(response)
                if problem and solution:
                    problem_ids = tokenizer.encode(
                        str(problem),
                        add_special_tokens=False,
                        truncation=True,
                        max_length=self.grpo_cfg.max_prompt_length,
                    )
                    problem = tokenizer.decode(problem_ids, skip_special_tokens=False)
                    records.append({
                        "prompt": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": str(problem)},
                        ],
                        "solution": str(solution),
                    })
        if not records:
            raise ValueError(
                "No usable GRPO rows; expected problem/prompt and solution/response fields."
            )
        return Dataset.from_list(records)

    def _peft_config(self):
        if not self.model_cfg.use_lora:
            return None
        from peft import LoraConfig

        return LoraConfig(
            r=self.model_cfg.lora_r or 16,
            lora_alpha=self.model_cfg.lora_alpha or 32,
            lora_dropout=self.model_cfg.lora_dropout,
            target_modules=self.model_cfg.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

    def _training_args(self):
        from trl import GRPOConfig

        report_to = ["wandb"] if self.train_cfg.wandb_project else []
        return GRPOConfig(
            output_dir=self.train_cfg.output_dir,
            run_name=self.train_cfg.wandb_run_name,
            learning_rate=self.train_cfg.learning_rate,
            num_train_epochs=self.train_cfg.num_epochs,
            per_device_train_batch_size=self.data_cfg.batch_size,
            gradient_accumulation_steps=self.train_cfg.gradient_accumulation_steps,
            max_grad_norm=self.train_cfg.max_grad_norm,
            warmup_steps=self.train_cfg.warmup_steps,
            weight_decay=self.train_cfg.weight_decay,
            lr_scheduler_type=self.train_cfg.lr_scheduler,
            optim=self.train_cfg.optimizer,
            logging_steps=self.train_cfg.logging_steps,
            save_steps=self.train_cfg.save_steps,
            save_strategy="steps",
            bf16=self.train_cfg.bf16,
            fp16=self.train_cfg.fp16,
            tf32=self.train_cfg.tf32,
            gradient_checkpointing=self.train_cfg.gradient_checkpointing,
            report_to=report_to,
            remove_unused_columns=False,
            model_init_kwargs={
                "trust_remote_code": self.model_cfg.trust_remote_code,
                "dtype": "bfloat16" if self.train_cfg.bf16 else "float16",
                "use_cache": False,
            },
            num_generations=self.grpo_cfg.num_generations,
            max_completion_length=self.grpo_cfg.max_completion_length,
            temperature=self.grpo_cfg.temperature,
            top_p=self.grpo_cfg.top_p,
            beta=self.grpo_cfg.beta,
            epsilon=self.grpo_cfg.epsilon,
            loss_type=self.grpo_cfg.loss_type,
            mask_truncated_completions=self.grpo_cfg.mask_truncated_completions,
            reward_weights=[
                self.grpo_cfg.format_reward_weight,
                self.grpo_cfg.syntax_reward_weight,
                self.grpo_cfg.reference_reward_weight,
            ],
            log_completions=self.grpo_cfg.log_completions,
            num_completions_to_print=2,
            use_vllm=False,
        )

    def __call__(self, data_path: str, resume_from: str | None = None):
        from trl import GRPOTrainer

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        effective_batch = (
            self.data_cfg.batch_size
            * world_size
            * self.train_cfg.gradient_accumulation_steps
        )
        if effective_batch % self.grpo_cfg.num_generations:
            raise ValueError(
                "GRPO requires batch_size * world_size * gradient_accumulation_steps "
                f"({effective_batch}) to be divisible by num_generations "
                f"({self.grpo_cfg.num_generations})."
            )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_cfg.base_model,
            trust_remote_code=self.model_cfg.trust_remote_code,
            padding_side="left",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dataset = self._load_rows(data_path, tokenizer)
        if self.train_cfg.wandb_project:
            os.environ["WANDB_PROJECT"] = self.train_cfg.wandb_project
        self.trainer = GRPOTrainer(
            model=self.model_cfg.base_model,
            args=self._training_args(),
            reward_funcs=[format_reward, python_syntax_reward, reference_similarity_reward],
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=self._peft_config(),
        )
        self.trainer.train(resume_from_checkpoint=resume_from)
        self.trainer.save_model(self.train_cfg.output_dir)
