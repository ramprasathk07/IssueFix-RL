"""
Checkpoint verification tests.

Run:  pytest src/tests/test_checkpoint.py -v
      pytest src/tests/test_checkpoint.py -v --ckpt outputs/sft_run1/checkpoint-epoch1-step150
"""
import json
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REQUIRED_FILES = {"config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"}


# ── file structure ────────────────────────────────────────────────────────────

def test_required_files_exist(checkpoint_dir):
    missing = REQUIRED_FILES - {f.name for f in checkpoint_dir.iterdir()}
    assert not missing, f"Missing files in checkpoint: {missing}"


def test_config_json_valid(checkpoint_dir):
    cfg_path = checkpoint_dir / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    assert "model_type" in cfg, "config.json missing 'model_type'"
    assert "vocab_size" in cfg, "config.json missing 'vocab_size'"


def test_tokenizer_config_valid(checkpoint_dir):
    tok_cfg_path = checkpoint_dir / "tokenizer_config.json"
    with open(tok_cfg_path) as f:
        tok_cfg = json.load(f)
    assert "tokenizer_class" in tok_cfg or "model_max_length" in tok_cfg, \
        "tokenizer_config.json looks malformed"


# ── tokenizer loading ─────────────────────────────────────────────────────────

def test_tokenizer_loads(checkpoint_dir):
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    assert tokenizer is not None
    assert tokenizer.vocab_size > 0


def test_tokenizer_has_special_tokens(checkpoint_dir):
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    added = set(tokenizer.get_added_vocab().keys())
    expected = {"<think>", "</think>", "<answer>", "</answer>"}
    assert expected.issubset(added), f"Missing special tokens: {expected - added}"


def test_tokenizer_encode_decode(checkpoint_dir):
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    text = "<think>test</think><answer>print('hello')</answer>"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    assert "<think>" in decoded and "<answer>" in decoded


# ── training state ────────────────────────────────────────────────────────────

def test_training_state_exists(checkpoint_dir):
    state_file = checkpoint_dir / "training_state.pt"
    if not state_file.exists():
        pytest.skip("training_state.pt not present (checkpoint predates resume support)")


def test_training_state_valid(checkpoint_dir):
    state_file = checkpoint_dir / "training_state.pt"
    if not state_file.exists():
        pytest.skip("training_state.pt not present")

    state = torch.load(state_file, map_location="cpu", weights_only=False)
    assert "epoch" in state, "training_state.pt missing 'epoch'"
    assert "global_step" in state, "training_state.pt missing 'global_step'"
    assert "optimizer_state_dict" in state, "training_state.pt missing 'optimizer_state_dict'"
    assert state["global_step"] >= 0
    assert state["epoch"] >= 0


def test_training_state_step_matches_dirname(checkpoint_dir):
    state_file = checkpoint_dir / "training_state.pt"
    if not state_file.exists():
        pytest.skip("training_state.pt not present")

    state = torch.load(state_file, map_location="cpu", weights_only=False)
    dir_name = checkpoint_dir.name  # e.g. checkpoint-epoch1-step150
    expected_step = int(dir_name.split("step")[-1])
    assert state["global_step"] == expected_step, (
        f"global_step in state ({state['global_step']}) != step in dirname ({expected_step})"
    )


# ── generation ───────────────────────────────────────────────────────────────

SAMPLE_PROMPTS = [
    "Write a Python function that returns the factorial of n using recursion.",
    "Fix the bug: def add(a, b): return a - b",
    "Write a Python one-liner to flatten a list of lists.",
]

SYSTEM_PROMPT = (
    "You are an expert software engineer.\n"
    "For each coding problem:\n"
    "1. Think through the solution step by step - place your reasoning inside <think> ... </think> tags.\n"
    "2. Provide the final code solution inside <answer> ... </answer> tags.\n"
    "Do not output anything outside these tags."
)


def _chat_input_ids(tokenizer, messages, device):
    # apply_chat_template returns BatchEncoding in newer transformers — extract tensor explicitly
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt")["input_ids"].to(device)


@pytest.fixture(scope="module")
def loaded_model(checkpoint_dir):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available — skipping generation test.")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def test_generate_samples(loaded_model, capsys):
    model, tokenizer = loaded_model

    print("\n" + "=" * 70)
    print("GENERATION SAMPLES")
    print("=" * 70)

    last_text = ""
    for i, prompt in enumerate(SAMPLE_PROMPTS):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        input_ids = _chat_input_ids(tokenizer, messages, model.device)
        prompt_len = input_ids.shape[-1]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=False)
        last_text = text

        print(f"\n--- Prompt {i+1} ---")
        print(f"Q: {prompt}")
        print(f"A: {text}")

    print("\n" + "=" * 70)

    assert "<answer>" in last_text or "<think>" in last_text, (
        "Model output missing <think>/<answer> tags — possible tokenizer or training issue."
    )


# ── safetensors integrity ─────────────────────────────────────────────────────

def test_model_safetensors_readable(checkpoint_dir):
    try:
        from safetensors import safe_open
    except ImportError:
        pytest.skip("safetensors not installed")

    sf_path = checkpoint_dir / "model.safetensors"
    with safe_open(str(sf_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
    assert len(keys) > 0, "model.safetensors has no tensors"
    print(f"\n  safetensors keys: {len(keys)} tensors")
