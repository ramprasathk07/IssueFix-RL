from pydantic import BaseModel, Field, PositiveInt, ConfigDict
from typing import List, Literal, Optional


class DataloaderParams(BaseModel):
    model_config = ConfigDict(extra="ignore")

    batch_size: PositiveInt = Field(default=4)
    max_length: PositiveInt = Field(default=8192, le=10240)
    shuffle: bool = Field(default=True)
    num_workers: int = Field(default=4, ge=0)
    prefetch_factor: int = Field(default=2, ge=1)
    pin_memory: bool = Field(default=True)
    pre_tokenize: bool = Field(default=True)
    # drop samples longer than max_length instead of truncating them —
    # truncation strips the EOS target, teaching the model to never stop.
    # Requires pre_tokenize.
    drop_overlong: bool = Field(default=False)


class ModelParams(BaseModel):
    model_config = ConfigDict(extra="ignore")

    base_model: str
    model_type: str = "AutoModelForCausalLM"
    trust_remote_code: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    use_lora: bool = False
    lora_r: Optional[PositiveInt] = None
    lora_alpha: Optional[int] = None
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None


class TrainingParams(BaseModel):
    model_config = ConfigDict(extra="ignore")

    learning_rate: float = Field(gt=0, lt=1)
    num_epochs: PositiveInt
    bf16: bool = False
    fp16: bool = False
    tf32: bool = False
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    warmup_steps: int = 0
    lr_scheduler: str = "cosine"
    # "adamw_torch" or "adamw_8bit" (bitsandbytes; optimizer-state quantization is
    # DDP-safe, unlike weight quantization — saves ~1.5GB on a 0.5B full finetune)
    optimizer: str = "adamw_torch"
    use_amp: bool = False
    gradient_checkpointing: bool = False
    logging_steps: int = 10
    save_steps: int = 500
    # run validation every N optimizer steps mid-epoch (0 disables mid-epoch
    # validation, leaving only the full pass at each epoch boundary)
    eval_steps: int = 100
    # cap batches for MID-EPOCH validation only; the epoch-boundary pass is
    # always full. A full pass costs ~14 min on 2xT4 (~4 training steps), so
    # uncapped mid-epoch eval would add ~14% overhead at eval_steps=100.
    # 0 = uncapped.
    eval_max_batches: int = Field(default=100, ge=0)
    output_dir: str = "./outputs"
    save_best_k: int = Field(default=2, ge=1)  # keep only top-k epoch checkpoints by val loss
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment: str = "sft_training"
    num_gpus: int = Field(default=1, ge=1)  # informational; actual count set by accelerate launch

    # ── Kaggle Model Hub upload (best checkpoint, after training) ──
    push_to_kaggle: bool = False
    # handle format: "<owner-slug>/<model-slug>/<framework>/<variation-slug>"
    kaggle_model_handle: Optional[str] = None
    kaggle_model_license: str = "apache-2.0"


class OPSDParams(BaseModel):
    """Settings for a student on GPU 0 and frozen self-teacher on GPU 1."""

    model_config = ConfigDict(extra="ignore")
    teacher_device: str = "cuda:1"
    student_device: str = "cuda:0"
    max_prompt_length: PositiveInt = 2048
    max_completion_length: PositiveInt = 512
    temperature: float = Field(default=1.0, gt=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k_loss: PositiveInt = 32
    jsd_beta: float = Field(default=0.5, ge=0, le=1)
    token_clip: float = Field(default=0.05, ge=0)
    teacher_load_in_4bit: bool = True
    student_thinking: bool = False
    teacher_thinking: bool = True
    system_prompt: Optional[str] = None


class GRPOParams(BaseModel):
    """Custom PyTorch GRPO rollout, objective, and reward settings."""

    model_config = ConfigDict(extra="ignore")
    num_generations: int = Field(default=4, ge=2)
    max_prompt_length: PositiveInt = 2048
    max_completion_length: PositiveInt = 512
    forward_batch_size: PositiveInt = 1
    temperature: float = Field(default=1.0, gt=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    beta: float = Field(default=0.0, ge=0)
    epsilon: float = Field(default=0.2, gt=0)
    advantage_epsilon: float = Field(default=1e-4, gt=0)
    loss_type: Literal["grpo", "dapo"] = "dapo"
    mask_truncated_completions: bool = True
    format_reward_weight: float = Field(default=0.2, ge=0)
    syntax_reward_weight: float = Field(default=0.2, ge=0)
    reference_reward_weight: float = Field(default=0.6, ge=0)
    log_completions: bool = True
    seed: int = 42


class Config(BaseModel):
    model_params: ModelParams
    dataloader_params: DataloaderParams
    training_params: TrainingParams
    opsd_params: Optional[OPSDParams] = None
    grpo_params: Optional[GRPOParams] = None
