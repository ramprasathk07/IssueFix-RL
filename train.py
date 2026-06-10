import argparse
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    config = Config(**cfg_dict)
    trainer = SFTTrainer(config)
    trainer(args.data)


if __name__ == "__main__":
    main()
