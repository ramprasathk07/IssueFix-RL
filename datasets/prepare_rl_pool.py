"""Build test-backed RL prompt pools and a held-out repair eval set.

Downloads MBPP and KodCode-Light-RL-10K (training) and HumanEvalPack-Python
(HumanEvalFix, eval only), normalizes them to one JSONL schema, and keeps a row
only if its gold solution passes its own tests. That gate is what makes an
execution reward trustworthy: a prompt whose reference fails its tests can
never yield a correct positive reward.

Every row carries `problem` + `solution` (what the GRPO loader reads today) plus
`test_code`, a pytest module that imports the candidate from `solution.py`.

Usage:
    python datasets/prepare_rl_pool.py --out datasets/processed/rl
    python datasets/prepare_rl_pool.py --sources humanevalfix     # rebuild one source
    python datasets/prepare_rl_pool.py --limit 50                 # quick check
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from data.loader import SYSTEM_PROMPT  # noqa: E402

SFT_TOKENIZER = REPO / "out/outputs/sft_run1/checkpoint-epoch2-step150"
FALLBACK_TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_PROMPT_TOKENS = 2048  # grpo_params.max_prompt_length
TEST_TIMEOUT_S = 20.0
CONTAMINATION_JACCARD = 0.7
SEED = 42

SOURCES = {
    "humanevalfix": "bigcode/humanevalpack",
    "mbpp": "google-research-datasets/mbpp",
    "kodcode": "KodCode/KodCode-Light-RL-10K",
}
TRAIN_SOURCES = ("mbpp", "kodcode")


def run_pytest(solution: str, test_code: str) -> dict:
    """Run `test_code` against `solution` in a scratch directory."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        Path(tmp, "solution.py").write_text(solution, encoding="utf-8")
        Path(tmp, "test_solution.py").write_text(test_code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--tb=no", "-p", "no:cacheprovider",
                 "test_solution.py"],
                cwd=tmp, capture_output=True, text=True, timeout=TEST_TIMEOUT_S,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired:
            return {"passed": 0, "failed": 0, "ok": False, "status": "timeout"}
    lines = proc.stdout.strip().splitlines()
    summary = lines[-1] if lines else ""
    passed = sum(int(n) for n in re.findall(r"(\d+) passed", summary))
    failed = sum(int(n) for n in re.findall(r"(\d+) (?:failed|errors?)", summary))
    return {
        "passed": passed,
        "failed": failed,
        "ok": proc.returncode == 0 and passed > 0 and failed == 0,
        "status": f"rc={proc.returncode}",
    }


def run_all(pairs: list[tuple[str, str]], workers: int, desc: str) -> list[dict]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(tqdm(pool.map(lambda p: run_pytest(*p), pairs), total=len(pairs), desc=desc))


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", text.lower()))


def build_mbpp(limit: int) -> tuple[list[dict], dict]:
    rows, skipped = [], 0
    for split, data in load_dataset(SOURCES["mbpp"], "full").items():
        for record in data:
            code = record["code"].replace("\r\n", "\n").replace("\t", "    ").strip() + "\n"
            tests = [t.strip() for t in record["test_list"] if t.strip()]
            try:
                defs = [n.name for n in ast.parse(code).body if isinstance(n, ast.FunctionDef)]
            except SyntaxError:
                defs = []
            if not tests or not defs:
                skipped += 1
                continue
            entry = next((d for d in defs if re.search(rf"\b{re.escape(d)}\s*\(", tests[0])), defs[-1])
            setup = (record["test_setup_code"] or "").strip()
            test_code = "from solution import *\n" + (setup + "\n" if setup else "")
            test_code += "".join(f"\n\ndef test_{i}():\n    {t}\n" for i, t in enumerate(tests))
            rows.append({
                "id": f"mbpp/{record['task_id']}",
                "source": "mbpp",
                "problem": f"{record['text'].strip()}\n\nYour code should pass this test:\n{tests[0]}",
                "solution": code,
                "entry_point": entry,
                "test_code": test_code,
                "difficulty": None,
                "split": split,
            })
    rows.sort(key=lambda r: int(r["id"].split("/")[1]))
    return rows[:limit] if limit else rows, {"skipped_unparseable": skipped}


def build_kodcode(limit: int) -> tuple[list[dict], dict]:
    rows, skipped = [], 0
    for record in load_dataset(SOURCES["kodcode"], split="train"):
        infos = record["test_info"]
        if isinstance(infos, str):
            infos = ast.literal_eval(infos or "[]")
        decls = [i["function_declaration"].strip() for i in infos or [] if i.get("function_declaration")]
        names = [i["function_name"] for i in infos or [] if i.get("function_name")]
        if not decls or not names or not record["test"]:
            skipped += 1
            continue
        noun = "signature" if len(decls) == 1 else "signatures"
        rows.append({
            "id": f"kodcode/{record['question_id']}",
            "source": "kodcode",
            "problem": f"{record['question'].strip()}\n\nImplement it in Python with this {noun}:\n"
                       + "\n".join(decls),
            "solution": record["solution"].strip() + "\n",
            "entry_point": names[0],
            "test_code": record["test"],
            "difficulty": record["gpt_difficulty"],
            "gpt_pass_percentage": float(record["gpt_pass_percentage"]),
        })
        if limit and len(rows) >= limit:
            break
    return rows, {"skipped_no_signature_or_tests": skipped}


def build_humanevalfix(limit: int) -> list[dict]:
    rows = []
    for record in load_dataset(SOURCES["humanevalfix"], "python", split="test"):
        # `prompt` is imports + signature + docstring. The docstring is the only
        # statement of intended behavior, so the buggy function must carry it.
        head = record["prompt"]
        entry = record["entry_point"]
        buggy = head + record["buggy_solution"]
        test_code = ("from solution import *\n" + (record["test_setup"] or "") + "\n"
                     + record["test"] + f"\n\n\ndef test_check():\n    check({entry})\n")
        rows.append({
            "id": f"humanevalfix/{record['task_id']}",
            "source": "humanevalfix",
            "problem": f"Fix the bug in `{entry}` so it behaves as its docstring describes. "
                       f"Output the complete corrected function.\n\n```python\n{buggy.rstrip()}\n```",
            "solution": head + record["canonical_solution"],
            "buggy_code": buggy,
            "bug_type": record["bug_type"],
            "entry_point": entry,
            "test_code": test_code,
            "docstring": record["docstring"],
        })
    return rows[:limit] if limit else rows


def prompt_token_counter():
    source = str(SFT_TOKENIZER) if SFT_TOKENIZER.is_dir() else FALLBACK_TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(source)

    def count(problem: str) -> int:
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": problem}],
            tokenize=False, add_generation_prompt=True,
        )
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    return count, source


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(rows)} rows, {path.stat().st_size / 2**20:.1f} MiB)")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def license_of(dataset: str) -> str | None:
    try:
        card = HfApi().dataset_info(dataset).card_data
        value = getattr(card, "license", None) if card else None
        return ", ".join(value) if isinstance(value, list) else value
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(REPO / "datasets/processed/rl"))
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--limit", type=int, default=0, help="rows per source; 0 = all")
    parser.add_argument("--sources", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    count_tokens, tokenizer_source = prompt_token_counter()

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.update({
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tokenizer_for_prompt_tokens": tokenizer_source,
        "system_prompt_included_in_prompt_tokens": True,
        "gold_gate": "row kept only if its gold solution passes every test in test_code",
        "execution": "write the candidate to solution.py and test_code to test_solution.py, "
                     "then run `python -m pytest -q test_solution.py` in that directory",
        "licenses": {name: license_of(repo) for name, repo in SOURCES.items()},
    })
    manifest.setdefault("sources", {})

    # HumanEvalFix docstrings drive the contamination filter even when the eval
    # file itself is not being rebuilt.
    heval = build_humanevalfix(args.limit)
    eval_word_sets = [_words(r["docstring"]) for r in heval]

    if "humanevalfix" in args.sources:
        gold = run_all([(r["solution"], r["test_code"]) for r in heval], args.workers,
                       "humanevalfix gold")
        bug = run_all([(r["buggy_code"], r["test_code"]) for r in heval], args.workers,
                      "humanevalfix buggy")
        kept = []
        for row, g, b in zip(heval, gold, bug):
            if g["ok"] and not b["ok"]:
                row["num_tests"] = g["passed"]
                row["prompt_tokens"] = count_tokens(row["problem"])
                kept.append(row)
        write_jsonl(out / "humanevalfix_python.eval.jsonl", kept)
        manifest["sources"]["humanevalfix"] = {
            "role": "EVAL ONLY - never train on it", "rows": len(heval),
            "kept_gold_passes_and_buggy_fails": len(kept),
        }

    for name, builder in (("mbpp", build_mbpp), ("kodcode", build_kodcode)):
        if name not in args.sources:
            continue
        rows, notes = builder(args.limit)
        results = run_all([(r["solution"], r["test_code"]) for r in rows], args.workers, f"{name} gold")
        stats = {"rows": len(rows), **notes, "gold_failed": 0, "contaminated": 0, "too_long": 0}
        kept = []
        for row, result in zip(rows, results):
            if not result["ok"]:
                stats["gold_failed"] += 1
                continue
            words = _words(row["problem"])
            if any(len(words & e) / max(len(words | e), 1) >= CONTAMINATION_JACCARD
                   for e in eval_word_sets):
                stats["contaminated"] += 1
                continue
            row["prompt_tokens"] = count_tokens(row["problem"])
            if row["prompt_tokens"] > MAX_PROMPT_TOKENS:
                stats["too_long"] += 1
                continue
            row["num_tests"] = result["passed"]
            kept.append(row)
        stats["kept"] = len(kept)
        manifest["sources"][name] = {"role": "train", **stats}
        write_jsonl(out / f"{name}.verified.jsonl", kept)

    if any(name in args.sources for name in TRAIN_SOURCES):
        # Rebuild the mix from every verified training file on disk, so a
        # single-source rebuild never silently drops the other source.
        mix = [row for name in TRAIN_SOURCES if (out / f"{name}.verified.jsonl").exists()
               for row in read_jsonl(out / f"{name}.verified.jsonl")]
        random.Random(SEED).shuffle(mix)
        write_jsonl(out / "rl_train_mix.jsonl", mix)
        manifest["rl_train_mix_rows"] = len(mix)

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["sources"], indent=2))


if __name__ == "__main__":
    main()
