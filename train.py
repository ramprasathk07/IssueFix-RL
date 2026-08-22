"""
Single-GPU:
    python train.py

Multi-GPU (Kaggle notebook cell or terminal):
    python train.py                     # mp.spawn when num_gpus > 1 in config

Multi-GPU (accelerate launch):
    accelerate launch --num_processes 2 train.py

Resume:
    accelerate launch --num_processes 2 train.py --resume outputs/sft_run1/checkpoint-epoch1-step150

Override wandb project/run name without editing configs/sft.yaml:
    python train.py --wandb_run_name qwen0.5_sft_run2 --wandb_project my_sft_project_v5

OPSD on Kaggle dual T4 (one process; student GPU 0, teacher GPU 1):
    python train.py --method opsd --config configs/opsd.yaml --data datasets/processed/data.jsonl
"""
import argparse
import os
import yaml
import sys
from pathlib import Path

import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent / "src"))

from configs import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IssueFix SFT/OPSD/GRPO training launcher")
    parser.add_argument(
        "--method", choices=("sft", "opsd", "grpo"), default="sft",
        help="Training method. OPSD uses one process and two GPUs.",
    )
    parser.add_argument(
        "--finetuning", choices=("lora", "full"), default=None,
        help="OPSD/GRPO: train LoRA adapters or all policy weights (overrides YAML).",
    )
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
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Override training_params.wandb_project from the config",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Override training_params.wandb_run_name from the config",
    )
    parser.add_argument("--student_device", default=None,
                        help="OPSD student device override, e.g. cuda:0")
    parser.add_argument("--teacher_device", default=None,
                        help="OPSD teacher device override, e.g. cuda:1")
    parser.add_argument("--max_completion_length", type=int, default=None,
                        help="OPSD/GRPO rollout token limit override")
    parser.add_argument("--top_k_loss", type=int, default=None,
                        help="OPSD teacher distribution width override")
    return parser.parse_args()


def train(
    method: str,
    config_path: str,
    data_path: str,
    resume: str | None = None,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    student_device: str | None = None,
    teacher_device: str | None = None,
    max_completion_length: int | None = None,
    top_k_loss: int | None = None,
    finetuning: str | None = None,
):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    if wandb_project is not None:
        cfg_dict["training_params"]["wandb_project"] = wandb_project
    if wandb_run_name is not None:
        cfg_dict["training_params"]["wandb_run_name"] = wandb_run_name
    if finetuning is not None and method == "sft":
        raise ValueError("--finetuning is supported with --method opsd or grpo")
    if finetuning is not None:
        cfg_dict["model_params"]["use_lora"] = finetuning == "lora"
    if method == "opsd":
        if "opsd_params" not in cfg_dict:
            raise ValueError("--method opsd requires an opsd_params section in the config")
        overrides = {
            "student_device": student_device,
            "teacher_device": teacher_device,
            "max_completion_length": max_completion_length,
            "top_k_loss": top_k_loss,
        }
        cfg_dict["opsd_params"].update({k: v for k, v in overrides.items() if v is not None})
    elif method == "grpo" and max_completion_length is not None:
        if "grpo_params" not in cfg_dict:
            raise ValueError("--method grpo requires a grpo_params section in the config")
        cfg_dict["grpo_params"]["max_completion_length"] = max_completion_length
    config = Config(**cfg_dict)
    if method == "opsd":
        from trainers.custom.train_opsd import OPSDTrainer
        trainer = OPSDTrainer(config)
    elif method == "grpo":
        from trainers.custom.train_grpo import GRPOIssueFixTrainer
        trainer = GRPOIssueFixTrainer(config)
    else:
        from trainers.custom.train_sft import SFTTrainer
        trainer = SFTTrainer(config)
    trainer(data_path, resume_from=resume)


def _spawn_worker(
    rank: int,
    method: str,
    world_size: int,
    config_path: str,
    data_path: str,
    resume: str | None,
    wandb_project: str | None,
    wandb_run_name: str | None,
    finetuning: str | None,
    max_completion_length: int | None,
):
    # Set distributed env vars before Accelerator is created — it reads these to init process group
    os.environ.update({
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    })
    train(method, config_path, data_path, resume, wandb_project, wandb_run_name,
          finetuning=finetuning, max_completion_length=max_completion_length)


def main() -> None:
    args = parse_args()

    if args.method == "opsd":
        if os.environ.get("WORLD_SIZE") not in (None, "", "1"):
            raise RuntimeError(
                "Do not use accelerate/torchrun for OPSD. Run `python train.py --method opsd`; "
                "the single process places student and teacher on separate GPUs."
            )
        train(
            args.method, args.config, args.data, args.resume,
            args.wandb_project, args.wandb_run_name,
            args.student_device, args.teacher_device,
            args.max_completion_length, args.top_k_loss,
            args.finetuning,
        )
        return

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    num_gpus = cfg_dict.get("training_params", {}).get("num_gpus", 1)

    if os.environ.get("WORLD_SIZE"):
        # already launched via `accelerate launch` — env vars already set, run directly
        train(args.method, args.config, args.data, args.resume, args.wandb_project,
              args.wandb_run_name, finetuning=args.finetuning,
              max_completion_length=args.max_completion_length)
    elif num_gpus > 1:
        # mp.spawn uses 'spawn' start method — safe with CUDA, works from notebook cells
        # notebook_launcher uses 'fork' which raises RuntimeError with CUDA
        mp.spawn(
            _spawn_worker,
            args=(args.method, num_gpus, args.config, args.data, args.resume, args.wandb_project,
                  args.wandb_run_name, args.finetuning, args.max_completion_length),
            nprocs=num_gpus,
            join=True,
        )
    else:
        train(args.method, args.config, args.data, args.resume, args.wandb_project,
              args.wandb_run_name, finetuning=args.finetuning,
              max_completion_length=args.max_completion_length)


if __name__ == "__main__":
    main()
