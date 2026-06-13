# IssueFix-RL

SFT training pipeline for fine-tuning LLMs on code issue-fixing tasks. Uses [DFT loss](https://huggingface.co/papers/2508.05629) (reinforcement-learning-inspired SFT objective), cosine LR scheduling, and multi-GPU DDP via `accelerate`.

Default model: `Qwen/Qwen2.5-0.5B-Instruct`. Config tuned for **Kaggle 2× T4 (15 GB each)**.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# torch is pre-installed on Kaggle; locally: pip install torch>=2.1.0
```

---

## Training

**Single GPU**
```bash
python train.py
```

**Multi-GPU (2× T4 on Kaggle)**
```bash
accelerate launch --num_processes 2 train.py
```

**Resume from checkpoint**
```bash
accelerate launch --num_processes 2 train.py \
  --resume outputs/sft_run1/checkpoint-epoch1-step150
```

**Custom config / data**
```bash
python train.py --config configs/sft.yaml \
                --data datasets/processed/opencode_sft_filtered.jsonl
```

At end of training the best checkpoint (lowest val loss) is automatically zipped to `/kaggle/working/<run_name>_best.zip` (or `output_dir` if not on Kaggle).

---

## Config — `configs/sft.yaml`

| Section | Key fields |
|---|---|
| `model_params` | `base_model`, `load_in_4bit`\*, `use_lora`, `lora_r` |
| `training_params` | `learning_rate`, `num_epochs`, `gradient_accumulation_steps`, `bf16`, `lr_scheduler`, `warmup_steps`, `num_gpus` |
| `dataloader_params` | `batch_size` (per GPU), `max_length`, `num_workers` |

\* `load_in_4bit` must be `false` for multi-GPU DDP — bitsandbytes quantization is incompatible with DDP wrapping. Re-enable for single-GPU only.

**Kaggle 2× T4 defaults** (effective batch = `4 × 2 GPUs × 8 grad_acc = 64`):
```yaml
dataloader_params:
  batch_size: 4
  max_length: 2048

training_params:
  gradient_accumulation_steps: 8
  bf16: true
  gradient_checkpointing: true
```

**Experiment tracking** — set in `training_params`:
```yaml
wandb_project: "my_project"       # enables wandb
wandb_run_name: "run1"
mlflow_tracking_uri: "./mlruns"   # enables mlflow (or http://host:port)
mlflow_experiment: "sft_training"
```

---

## Project structure

```
IssueFix-RL/
├── train.py                          # entry point (argparse CLI)
├── configs/
│   └── sft.yaml                      # full training config
├── datasets/
│   ├── prepare_sft.py
│   └── processed/                    # .jsonl training data
└── src/
    ├── configs/
    │   └── schemas.py                # Pydantic config schemas (Config, ModelParams, ...)
    ├── data/
    │   └── loader.py                 # SFTDataset, collate_fn, create_sft_dataloader
    └── trainers/
        ├── custom/
        │   └── train_sft.py          # SFTTrainer (DDP, wandb, mlflow, checkpointing)
        └── utils/
            └── loss_helper.py        # dft_loss, entropy_from_logits
```

---

## Logged metrics

| Metric | Description |
|---|---|
| `train/loss` | DFT loss, response tokens only, logged at optimizer steps |
| `train/entropy` | Mean Shannon entropy over response tokens |
| `train/grad_norm` | Gradient L2 norm after clipping |
| `train/lr` | Learning rate (cosine schedule) |
| `val/loss` | Validation DFT loss, gathered across GPUs |
| `val/entropy` | Validation entropy |

Metrics logged to both **wandb** and **mlflow** if configured. Step counter is tied to optimizer steps (not batch steps) — wandb charts stay contiguous across grad accumulation.

---

## Checkpointing

Checkpoints saved to `output_dir/checkpoint-epochN-stepM/`, containing:
- `model.safetensors` + `config.json` — model weights
- `tokenizer.json` + `tokenizer_config.json` — tokenizer with `<think>/<answer>` tags
- `training_state.pt` — optimizer, scheduler, epoch, global_step (for resume)

Best checkpoint (lowest val loss) is zipped at end of training.

MLflow model registry: final model registered under `base_model` name if `mlflow_tracking_uri` is set.

---

## Checkpoint verification

```bash
# verify latest checkpoint
pytest src/tests/test_checkpoint.py -v

# verify specific checkpoint + print generations (requires CUDA)
pytest src/tests/test_checkpoint.py -v -s \
  --ckpt outputs/sft_run1/checkpoint-epoch1-step150
```

Tests cover: required files, config/tokenizer validity, special tokens, training state integrity, safetensors readability, and sample generation output.

---

## Data format

Each line in the `.jsonl` file must have `prompt` and `response` keys:

```json
{"prompt": "Fix the off-by-one error in ...", "response": "<think>reasoning</think><answer>code</answer>"}
```

The model is trained to produce responses wrapped in `<think>` (reasoning) and `<answer>` (code) tags.
