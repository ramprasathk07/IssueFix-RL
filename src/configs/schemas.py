from pydantic import BaseModel, Field, PositiveInt, ConfigDict
from typing import List, Optional


class DataloaderParams(BaseModel):
    model_config = ConfigDict(extra="ignore")

    batch_size: PositiveInt = Field(default=4)
    max_length: PositiveInt = Field(default=8192, le=10240)
    shuffle: bool = Field(default=True)
    num_workers: int = Field(default=4, ge=0)
    prefetch_factor: int = Field(default=2, ge=1)
    pin_memory: bool = Field(default=True)
    pre_tokenize: bool = Field(default=True)


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
    use_amp: bool = False
    gradient_checkpointing: bool = False
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    output_dir: str = "./outputs"
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


class Config(BaseModel):
    model_params: ModelParams
    dataloader_params: DataloaderParams
    training_params: TrainingParams