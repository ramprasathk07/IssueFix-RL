#!/usr/bin/env python3
"""
Export OpenCodeReasoning dataset to JSONL with correct <think>/<answer> format.

Each record response is structured as:
    <think>
    {reasoning extracted from model output}
    </think>
    <answer>
    {solution code}
    </answer>

Filters by total formatted-response token length [min_len, max_len].

Usage:
    python datasets/prepare_sft.py --max_samples 10000 --max_len 4096
"""

import os
import re
import json
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

os.environ["HF_DATASETS_CACHE"] = "./datasets/raw"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def extract_think(output: str) -> str:
    """Pull content between <think>...</think>; fall back to full output."""
    m = re.search(r"<think>(.*?)</think>", output, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Some outputs start reasoning without the tag — strip code fences at end
    # and return the prose as the think trace
    cleaned = re.sub(r"```[\s\S]*?```", "", output).strip()
    return cleaned if cleaned else output.strip()


def format_response(think: str, solution_code: str) -> str:
    return f"<think>\n{think}\n</think>\n<answer>\n{solution_code}\n</answer>"


def export_opencode_filtered(
    output_path: str = "./datasets/processed/opencode_sft_filtered.jsonl",
    max_samples: int = 10_000,
    config_name: str = "split_0",
    cache_dir: str = "./datasets/raw",
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    min_len: int = 128,
    max_len: int = 4096,
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
    skipped_no_solution = 0
    skipped_len = 0

    out_path = f'{output_path.split(".jsonl")[0]}_sl{max_len}_{max_samples}.jsonl'
    pbar = tqdm(desc="Filtering samples", unit=" samples", dynamic_ncols=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for item in dataset:
            total_checked += 1
            pbar.update(1)
            pbar.set_postfix(kept=count_written, skip_nosol=skipped_no_solution,
                             skip_len=skipped_len, refresh=True)

            raw_output = item.get("output", "")
            solution_code = item.get("solution", "")

            if not raw_output or not solution_code:
                skipped_no_solution += 1
                continue

            think_trace = extract_think(raw_output)
            response = format_response(think_trace, solution_code)

            tokens = tokenizer.encode(response, add_special_tokens=False)
            token_len = len(tokens)

            if not (min_len <= token_len <= max_len):
                skipped_len += 1
                continue

            record = {
                "prompt": item.get("input", ""),
                "response": response,
                "solution": solution_code,
                "difficulty": item.get("difficulty", ""),
                "source": item.get("source", ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count_written += 1

            if count_written >= max_samples:
                break

    pbar.close()
    print(f"\nChecked : {total_checked}")
    print(f"Kept    : {count_written}")
    print(f"Skipped (no solution)  : {skipped_no_solution}")
    print(f"Skipped (token length) : {skipped_len}")
    print(f"Output  : {out_path}")

    # Sanity-check first record format
    if count_written > 0:
        with open(out_path, "r", encoding="utf-8") as f:
            first = json.loads(f.readline())
        resp = first["response"]
        ok = "<think>" in resp and "</think>" in resp and "<answer>" in resp and "</answer>" in resp
        print(f"Format check (first record): {'OK' if ok else 'FAIL — missing tags!'}")
        if not ok:
            print(f"  Preview: {resp[:200]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=10_000)
    parser.add_argument("--output", type=str,
                        default="./datasets/processed/opencode_sft_filtered.jsonl")
    parser.add_argument("--config", type=str, default="split_0",
                        choices=["split_0", "split_1"])
    parser.add_argument("--cache_dir", type=str, default="./datasets/raw")
    parser.add_argument("--tokenizer_name", type=str,
                        default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--min_len", type=int, default=128,
                        help="Min token length of formatted response (inclusive)")
    parser.add_argument("--max_len", type=int, default=4096,
                        help="Max token length of formatted response (inclusive)")
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
