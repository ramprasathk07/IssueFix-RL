"""
Single-GPU:
    python train.py

Multi-GPU (Kaggle 2x T4 notebook cell):
    python train.py                     # auto-uses notebook_launcher when num_gpus > 1

Multi-GPU (terminal / accelerate launch):
    accelerate launch --num_processes 2 train.py

Resume:
    accelerate launch --num_processes 2 train.py --resume outputs/sft_run1/checkpoint-epoch1-step150
"""
import argparse
import os
import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from configs import Config
from trainers.custom.train_sft import SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT training launcher")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sft.yaml",
        help="Path to YAML config (default: configs/sft.yaml)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/processed/opencode_sft_filtered.jsonl",
        help="Path to training data .jsonl",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint dir to resume from",
    )
    return parser.parse_args()


def train(config_path: str, data_path: str, resume: str | None = None):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    config = Config(**cfg_dict)
    trainer = SFTTrainer(config)
    trainer(data_path, resume_from=resume)


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    num_gpus = cfg_dict.get("training_params", {}).get("num_gpus", 1)

    if os.environ.get("WORLD_SIZE"):
        # already launched via `accelerate launch` — run directly, accelerate owns the processes
        train(args.config, args.data, args.resume)
    elif num_gpus > 1:
        # Kaggle notebook / Jupyter: notebook_launcher spawns num_gpus processes internally
        from accelerate import notebook_launcher
        notebook_launcher(
            train,
            args=(args.config, args.data, args.resume),
            num_processes=num_gpus,
        )
    else:
        train(args.config, args.data, args.resume)


if __name__ == "__main__":
    main()
