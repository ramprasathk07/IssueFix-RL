#!/usr/bin/env python3
"""
Export OpenCodeReasoning dataset to JSONL format with token length filtering.
Keeps only samples where the 'solution' token length is between min_len and max_len.
Saves up to max_samples filtered records.
"""

import os
import json
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

os.environ["HF_DATASETS_CACHE"] = "./datasets/raw"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def export_opencode_filtered(
    output_path: str = "./datasets/processed/opencode_sft_filtered.jsonl",
    max_samples: int = 10_000,
    config_name: str = "split_0",
    cache_dir: str = "./datasets/raw",
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    min_len: int = 1024,
    max_len: int = 8192,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print(f"Loading dataset: nvidia/OpenCodeReasoning/{config_name} (streaming)")
    dataset = load_dataset(
        "nvidia/OpenCodeReasoning",
        config_name,
        split=config_name,
        streaming=True,
        cache_dir=cache_dir,
    )

    count_written = 0
    total_checked = 0

    # Progress bar: we don't know total, so leave total=None (infinite)
    pbar = tqdm(desc="Filtering samples", unit=" samples", dynamic_ncols=True)

    output_path = f'{output_path.split(".jsonl")[0]}_sl{max_len}_{max_samples}.jsonl'
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            total_checked += 1
            pbar.update(1)
            pbar.set_postfix(kept=count_written, refresh=True)

            response = item.get("output", "")
            if not response:
                continue

            tokens = tokenizer.encode(response, add_special_tokens=False)
            token_len = len(tokens)

            if min_len <= token_len <= max_len:
                record = {
                    "prompt": item.get("input", ""),
                    "response": response,
                    "solution": item.get("solution", ""),
                    "difficulty": item.get("difficulty", ""),
                    "source": item.get("source", ""),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count_written += 1
                # Optional: show the token length of the accepted sample
                pbar.set_description(f"✓ Kept sample {count_written} (len={token_len})")

                if count_written >= max_samples:
                    break

    pbar.close()
    print(f"\n✅ Exported {count_written} samples (filtered from {total_checked} total) to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="Number of filtered samples to export")
    parser.add_argument("--output", type=str, default="./datasets/processed/opencode_sft_filtered.jsonl")
    parser.add_argument("--config", type=str, default="split_0", choices=["split_0", "split_1"])
    parser.add_argument("--cache_dir", type=str, default="./datasets/raw")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace tokenizer for length filtering")
    parser.add_argument("--min_len", type=int, default=1024,
                        help="Minimum token length of solution (inclusive)")
    parser.add_argument("--max_len", type=int, default=4096,
                        help="Maximum token length of solution (inclusive)")
    args = parser.parse_args()

    export_opencode_filtered(
        output_path=args.output,
        max_samples=args.max_samples,
        config_name=args.config,
        cache_dir=args.cache_dir,
        tokenizer_name=args.tokenizer_name,
        min_len=args.min_len,
        max_len=args.max_len,
    )