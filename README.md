# IssueFix-RL

SFT training pipeline for fine-tuning LLMs on code issue-fixing tasks.
Uses DFT loss ([paper](https://huggingface.co/papers/2508.05629)), 4-bit QLoRA, and cosine LR scheduling.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

Custom config or data path:

```bash
python train.py --config configs/sft.yaml --data datasets/processed/opencode_sft_filtered.jsonl
```

## Config

Edit `configs/sft.yaml`. Key sections:

| Section | Key fields |
|---|---|
| `model_params` | `base_model`, `load_in_4bit`, `use_lora`, `lora_r` |
| `training_params` | `learning_rate`, `num_epochs`, `gradient_accumulation_steps`, `bf16`, `lr_scheduler`, `warmup_steps` |
| `dataloader_params` | `batch_size`, `max_length`, `num_workers` |

Set `wandb_project` in `training_params` to enable wandb logging.

## Project structure

```
IssueFix-RL/
├── train.py                        # entry point
├── configs/
│   └── sft.yaml                    # training config
├── datasets/
│   ├── prepare_sft.py
│   └── processed/                  # .jsonl training data
└── src/
    ├── configs/schemas.py          # Pydantic config schemas
    ├── data/loader.py              # SFTDataset + DataLoader
    └── trainers/
        ├── custom/train_sft.py     # SFTTrainer
        └── utils/loss_helper.py    # dft_loss, entropy_from_logits
```

## Logged metrics

| Metric | Description |
|---|---|
| `train/loss` | DFT loss (response tokens only) |
| `train/entropy` | Mean Shannon entropy over response tokens |
| `train/grad_norm` | Gradient norm after clipping |
| `train/lr` | Current learning rate |
| `val/loss` | Validation DFT loss |
| `val/entropy` | Validation entropy |

## Data format

Each line in the `.jsonl` file must be a JSON object with `prompt` and `response` keys:

```json
{"prompt": "Fix the off-by-one error in ...", "response": "<think>...</think><answer>...</answer>"}
```
