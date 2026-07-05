"""
Checkpoint performance evaluator.

Usage:
    python eval_checkpoint.py
    python eval_checkpoint.py --checkpoint checkpoints/checkpoint-kaggle-1
    python eval_checkpoint.py --checkpoint checkpoints/checkpoint-kaggle-1 --max_new_tokens 512
"""

import re
import sys
import time
import textwrap
import argparse
import subprocess
import tempfile
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Windows console defaults to cp1252 and dies on model output containing e.g. arrows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    name: str
    prompt: str
    expected_output: str          # exact stdout match (stripped)
    check_contains: list[str] = field(default_factory=list)  # substrings in output

TEST_CASES: list[TestCase] = [
    TestCase(
        name="factorial_recursive",
        prompt="Write a Python function `factorial(n)` that returns n! using recursion. Then call it with n=5 and print the result.",
        expected_output="120",
    ),
    TestCase(
        name="sum_list",
        prompt="Write Python code to compute the sum of [1, 2, 3, 4, 5] and print it.",
        expected_output="15",
    ),
    TestCase(
        name="reverse_string",
        prompt="Write Python code to reverse the string 'hello' and print the result.",
        expected_output="olleh",
    ),
    TestCase(
        name="fizzbuzz_15",
        prompt="Write Python code that prints FizzBuzz for numbers 1 to 15 (one per line). Rules: multiples of 3 print 'Fizz', multiples of 5 print 'Buzz', multiples of both print 'FizzBuzz', otherwise print the number.",
        expected_output="\n".join([
            "1","2","Fizz","4","Buzz","Fizz","7","8","Fizz","Buzz",
            "11","Fizz","13","14","FizzBuzz"
        ]),
    ),
    TestCase(
        name="list_comprehension_squares",
        prompt="Write Python code to print a list of squares of numbers from 1 to 5 using a list comprehension.",
        expected_output="[1, 4, 9, 16, 25]",
    ),
    TestCase(
        name="count_vowels",
        prompt="Write a Python function `count_vowels(s)` that counts vowels (a,e,i,o,u, case-insensitive) in a string. Call it with 'Hello World' and print the result.",
        expected_output="3",
    ),
    TestCase(
        name="bug_fix",
        prompt="Fix the bug in this Python function and print the result of calling it with a=3, b=4:\ndef add(a, b):\n    return a - b",
        expected_output="7",
    ),
    TestCase(
        name="flatten_list",
        prompt="Write Python code to flatten [[1,2],[3,4],[5]] into a single list and print it.",
        expected_output="[1, 2, 3, 4, 5]",
    ),
]

SYSTEM_PROMPT = """You are an expert software engineer.
For each coding problem:

1. Think through the solution step by step - place your reasoning inside <think> ... </think> tags.
2. Provide the final code solution inside <answer> ... </answer> tags.

Do not output anything outside these tags.
The answer must contain only the runnable code (no extra explanation after the tags)."""


# ---------------------------------------------------------------------------
# Format checking
# ---------------------------------------------------------------------------

def check_format(text: str) -> dict:
    has_think_open  = "<think>"   in text
    has_think_close = "</think>"  in text
    has_ans_open    = "<answer>"  in text
    has_ans_close   = "</answer>" in text

    # nothing before <think>
    junk_before = text.strip().startswith("<think>") if has_think_open else False

    # nothing after </answer>
    after_ans = ""
    if has_ans_close:
        after_ans = text.split("</answer>")[-1].strip()
    junk_after = bool(after_ans)

    # think comes before answer
    order_ok = False
    if has_think_open and has_ans_open:
        order_ok = text.index("<think>") < text.index("<answer>")

    return {
        "has_think":    has_think_open and has_think_close,
        "has_answer":   has_ans_open and has_ans_close,
        "clean_start":  not has_think_open or junk_before,   # True = OK
        "clean_end":    not junk_after,
        "order_ok":     order_ok,
        "fully_valid":  (has_think_open and has_think_close and
                         has_ans_open and has_ans_close and
                         not junk_after and order_ok),
    }


def extract_answer_code(text: str) -> tuple[Optional[str], str]:
    """Returns (code, source) where source is 'tag', 'markdown', or 'none'."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip(), "tag"
    # fallback: last ```python ... ``` block
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip(), "markdown"
    return None, "none"


# ---------------------------------------------------------------------------
# Code execution
# ---------------------------------------------------------------------------

def run_code(code: str, timeout: int = 10) -> tuple[str, str, bool]:
    """Returns (stdout, stderr, timed_out)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip(), result.stderr.strip(), False
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT", True
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(model, tokenizer, prompt: str, max_new_tokens: int, device: str) -> tuple[str, float]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[-1]

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0

    generated = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=False)
    return generated, elapsed


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

PASS  = "\033[92m PASS \033[0m"
FAIL  = "\033[91m FAIL \033[0m"
WARN  = "\033[93m WARN \033[0m"
SKIP  = "\033[90m SKIP \033[0m"

def _pf(ok: bool) -> str:
    return PASS if ok else FAIL

def sep(char="-", n=70):
    print(char * n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint-kaggle-1")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--exec_timeout", type=int, default=10,
                        help="Seconds before code execution is killed")
    parser.add_argument("--skip_exec", action="store_true",
                        help="Skip code execution (format check only)")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        if "/" in args.checkpoint and not args.checkpoint.startswith("."):
            ckpt = args.checkpoint  # HF hub id, e.g. Qwen/Qwen2.5-0.5B-Instruct
        else:
            sys.exit(f"Checkpoint not found: {ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nCheckpoint : {ckpt}")
    print(f"Device     : {device}")
    print(f"Max tokens : {args.max_new_tokens}")
    sep("=")

    # Load
    print("Loading tokenizer + model …")
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Loaded.\n")

    results = []

    for tc in TEST_CASES:
        sep()
        print(f"Test : {tc.name}")
        print(f"Q    : {tc.prompt[:100]}{'…' if len(tc.prompt) > 100 else ''}")

        # Generate
        output, elapsed = generate(model, tokenizer, tc.prompt, args.max_new_tokens, device)
        tokens_generated = len(tokenizer.encode(output, add_special_tokens=False))
        tok_per_sec = tokens_generated / elapsed if elapsed > 0 else 0

        print(f"Time : {elapsed:.2f}s  |  {tokens_generated} tokens  |  {tok_per_sec:.1f} tok/s")

        # Format
        fmt = check_format(output)
        print(f"Format  [{_pf(fmt['has_think'])}] <think>  "
              f"[{_pf(fmt['has_answer'])}] <answer>  "
              f"[{_pf(fmt['clean_end'])}] no_junk_after  "
              f"[{_pf(fmt['order_ok'])}] order")

        # Full raw output
        print(f"Full output:\n{textwrap.indent(output, '  ')}")

        # Code exec
        code, code_src = extract_answer_code(output)
        exec_pass = None
        stdout = stderr = ""

        if args.skip_exec or not code:
            if not code:
                print(f"Exec   [{SKIP}] no code block found (checked <answer> + markdown)")
        else:
            print(f"Code src : {code_src}")
            stdout, stderr, timed_out = run_code(code, timeout=args.exec_timeout)
            if timed_out:
                print(f"Exec   [{FAIL}] TIMEOUT after {args.exec_timeout}s")
                exec_pass = False
            else:
                exec_pass = stdout == tc.expected_output
                print(f"Exec   [{_pf(exec_pass)}] "
                      f"expected={repr(tc.expected_output[:60])}  got={repr(stdout[:60])}")
                if stderr:
                    print(f"Stderr : {stderr[:200]}")

        results.append({
            "name":       tc.name,
            "format_ok":  fmt["fully_valid"],
            "has_think":  fmt["has_think"],
            "has_answer": fmt["has_answer"],
            "code_src":   code_src,
            "exec_pass":  exec_pass,
            "elapsed":    elapsed,
            "tokens":     tokens_generated,
        })

    # Summary
    sep("=")
    print("SUMMARY")
    sep("=")

    n = len(results)
    fmt_ok    = sum(1 for r in results if r["format_ok"])
    think_ok  = sum(1 for r in results if r["has_think"])
    ans_ok    = sum(1 for r in results if r["has_answer"])
    tag_src   = sum(1 for r in results if r["code_src"] == "tag")
    md_src    = sum(1 for r in results if r["code_src"] == "markdown")
    exec_done = [r for r in results if r["exec_pass"] is not None]
    exec_pass_n = sum(1 for r in exec_done if r["exec_pass"])
    avg_time  = sum(r["elapsed"] for r in results) / n
    avg_tok   = sum(r["tokens"]  for r in results) / n

    print(f"Format fully valid : {fmt_ok}/{n}  ({100*fmt_ok/n:.0f}%)")
    print(f"Has <think>        : {think_ok}/{n}  ({100*think_ok/n:.0f}%)")
    print(f"Has <answer>       : {ans_ok}/{n}  ({100*ans_ok/n:.0f}%)")
    print(f"Code via <answer>  : {tag_src}/{n}")
    print(f"Code via markdown  : {md_src}/{n}  (fallback)")
    if exec_done:
        print(f"Code correctness   : {exec_pass_n}/{len(exec_done)}  ({100*exec_pass_n/len(exec_done):.0f}%)")
    print(f"Avg latency        : {avg_time:.2f}s")
    print(f"Avg tokens         : {avg_tok:.1f}")
    sep("=")

    print("\nPer-test breakdown:")
    header = f"{'Test':<30} {'Format':>6} {'Think':>5} {'Ans':>5} {'CodeSrc':>7} {'Exec':>6} {'Time':>7} {'Toks':>6}"
    print(header)
    sep("-", len(header))
    for r in results:
        exec_str = ("PASS" if r["exec_pass"] else "FAIL") if r["exec_pass"] is not None else "skip"
        src_str  = {"tag": "tag", "markdown": "md", "none": "-"}.get(r["code_src"], "-")
        print(f"{r['name']:<30} {'OK' if r['format_ok'] else 'FAIL':>6} "
              f"{'Y' if r['has_think'] else 'N':>5} {'Y' if r['has_answer'] else 'N':>5} "
              f"{src_str:>7} {exec_str:>6} {r['elapsed']:>6.1f}s {r['tokens']:>6}")


if __name__ == "__main__":
    main()
