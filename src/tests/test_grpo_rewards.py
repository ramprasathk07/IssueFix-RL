import math
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from trainers.custom.train_grpo import (
    GRPOIssueFixTrainer,
    build_completion_mask,
    format_reward,
    grpo_policy_loss,
    group_normalized_advantages,
    python_syntax_reward,
    reference_similarity_reward,
    strip_control_tokens,
    _grpo_token_terms,
)


def test_format_reward_accepts_only_complete_envelope():
    completions = [
        "<think>reason</think><answer>print(1)</answer>",
        "print(1)",
    ]
    assert format_reward(completions) == [1.0, 0.0]


def test_syntax_reward_never_executes_code():
    completions = [
        "<answer>raise RuntimeError('must not execute')</answer>",
        "<answer>def broken(</answer>",
    ]
    assert python_syntax_reward(completions) == [1.0, 0.0]


def test_reference_similarity_is_dense_and_bounded():
    rewards = reference_similarity_reward(
        ["<answer>def add(a, b): return a + b</answer>", "<answer>x = 1</answer>"],
        solution=["def add(a, b): return a + b", "def add(a, b): return a + b"],
    )
    assert rewards[0] == 1.0
    assert 0.0 <= rewards[1] < rewards[0]


def test_advantages_are_normalized_within_each_prompt_group():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 7.0, 7.0, 7.0, 7.0])
    advantages = group_normalized_advantages(rewards, num_generations=4).reshape(2, 4)

    assert advantages[0].mean().item() == pytest.approx(0.0, abs=1e-6)
    assert advantages[0, 0] < 0 < advantages[0, -1]
    assert torch.equal(advantages[1], torch.zeros(4))


def test_completion_mask_stops_at_eos_and_can_reject_truncation():
    completion_ids = torch.tensor([[4, 2, 2], [5, 6, 7]])

    keep_truncated = build_completion_mask(
        completion_ids, eos_token_id=2, pad_token_id=None, mask_truncated=False
    )
    reject_truncated = build_completion_mask(
        completion_ids, eos_token_id=2, pad_token_id=None, mask_truncated=True
    )

    assert keep_truncated.tolist() == [[True, True, False], [True, True, True]]
    assert reject_truncated.tolist() == [[True, True, False], [False, False, False]]


def test_dapo_and_grpo_use_their_intended_denominators():
    current_logps = torch.zeros((2, 2), requires_grad=True)
    old_logps = torch.zeros_like(current_logps)
    advantages = torch.tensor([1.0, -1.0])
    mask = torch.tensor([[True, True], [True, False]])

    dapo_loss, approx_kl, clip_fraction = grpo_policy_loss(
        current_logps, old_logps, advantages, mask, epsilon=0.2, loss_type="dapo"
    )
    grpo_loss, _, _ = grpo_policy_loss(
        current_logps, old_logps, advantages, mask, epsilon=0.2, loss_type="grpo"
    )

    assert dapo_loss.item() == pytest.approx(-1.0 / 3.0)
    assert grpo_loss.item() == pytest.approx(0.0)
    assert approx_kl.item() == pytest.approx(0.0)
    assert clip_fraction.item() == pytest.approx(0.0)
    dapo_loss.backward()
    assert current_logps.grad is not None


def test_policy_ratio_is_clipped_for_positive_advantage():
    current_logps = torch.tensor([[math.log(1.5)]])
    old_logps = torch.zeros_like(current_logps)

    loss, _, clip_fraction = grpo_policy_loss(
        current_logps,
        old_logps,
        torch.tensor([1.0]),
        torch.ones_like(current_logps, dtype=torch.bool),
        epsilon=0.2,
    )

    assert loss.item() == pytest.approx(-1.2)
    assert clip_fraction.item() == pytest.approx(1.0)


def test_reference_policy_kl_adds_a_positive_penalty():
    current_logps = torch.zeros((1, 1))
    old_logps = torch.zeros_like(current_logps)
    reference_logps = torch.full_like(current_logps, -1.0)

    loss, _, _ = grpo_policy_loss(
        current_logps,
        old_logps,
        torch.zeros(1),
        torch.ones_like(current_logps, dtype=torch.bool),
        epsilon=0.2,
        reference_logps=reference_logps,
        beta=0.5,
    )

    assert loss.item() > 0


def test_selective_log_softmax_matches_full_log_softmax():
    logits = torch.tensor([[[1.0, 2.0, 3.0], [0.5, -0.5, 1.5]]])
    token_ids = torch.tensor([[2, 0]])

    selected = GRPOIssueFixTrainer._selective_log_softmax(logits, token_ids)
    expected = logits.log_softmax(dim=-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)

    assert torch.allclose(selected, expected)


def test_microbatched_policy_update_backpropagates():
    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.token_logits = torch.nn.Parameter(torch.tensor([0.2, -0.1, 0.0]))

        def forward(self, input_ids, attention_mask, use_cache=False):
            batch, length = input_ids.shape
            logits = self.token_logits.expand(batch, length, -1)
            return SimpleNamespace(logits=logits)

    class CpuAccelerator:
        device = torch.device("cpu")

        @staticmethod
        def autocast():
            return nullcontext()

        @staticmethod
        def no_sync(model):
            return nullcontext()

        @staticmethod
        def backward(loss):
            loss.backward()

    trainer = GRPOIssueFixTrainer.__new__(GRPOIssueFixTrainer)
    trainer.accelerator = CpuAccelerator()
    trainer.grpo_cfg = SimpleNamespace(
        forward_batch_size=1,
        epsilon=0.2,
        beta=0.0,
        loss_type="dapo",
    )
    trainer.model = TinyPolicy()
    input_ids = torch.tensor([[2, 2, 0, 0], [2, 2, 1, 1]])
    completion_ids = input_ids[:, 2:]
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        old_logps = trainer._forward_logps(
            trainer.model,
            input_ids,
            attention_mask,
            prompt_width=2,
            completion_ids=completion_ids,
        )
    rollout = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_width": 2,
        "completion_ids": completion_ids,
        "completion_mask": torch.ones_like(completion_ids, dtype=torch.bool),
        "old_logps": old_logps,
        "reference_logps": None,
        "advantages": torch.tensor([-1.0, 1.0]),
    }

    metrics = trainer._update_policy(
        rollout,
        accumulation_divisor=1,
        synchronize_gradients=True,
    )

    assert metrics["kl/old_policy"] == pytest.approx(0.0)
    assert metrics["policy/ratio_mean"] == pytest.approx(1.0)
    assert metrics["loss/kl_penalty"] == pytest.approx(0.0)
    assert metrics["loss/policy"] == pytest.approx(metrics["loss/total"])
    assert trainer.model.token_logits.grad is not None
    assert trainer.model.token_logits.grad.abs().sum().item() > 0


def test_decoded_control_tokens_are_stripped_but_reasoning_tags_kept():
    decoded = "<think>off by one</think><answer>print(1)</answer><|im_end|><|endoftext|>"
    assert format_reward([decoded]) == [0.0]

    cleaned = strip_control_tokens(decoded)

    assert cleaned == "<think>off by one</think><answer>print(1)</answer>"
    assert format_reward([cleaned]) == [1.0]
    assert python_syntax_reward([cleaned]) == [1.0]


def test_reference_kl_is_reported_even_without_penalty():
    current_logps = torch.zeros((1, 1))
    reference_logps = torch.full_like(current_logps, -1.0)

    token_loss, _, _, reference_kl = _grpo_token_terms(
        current_logps,
        torch.zeros_like(current_logps),
        torch.zeros(1),
        epsilon=0.2,
        reference_logps=reference_logps,
        beta=0.0,
    )

    assert reference_kl.item() > 0
    assert token_loss.item() == pytest.approx(0.0)


def test_rollout_metrics_cover_reward_kl_entropy_and_truncation():
    trainer = GRPOIssueFixTrainer.__new__(GRPOIssueFixTrainer)
    trainer.grpo_cfg = SimpleNamespace(num_generations=2)
    old_logps = torch.full((4, 3), math.log(0.5))
    rollout = {
        "rewards": torch.tensor([0.5, 0.5, 0.0, 1.0]),
        "advantages": torch.tensor([0.0, 0.0, -1.0, 1.0]),
        "reward_components": {"format": torch.tensor([1.0, 1.0, 0.0, 1.0])},
        "stopped": torch.tensor([True, True, False, True]),
        "completion_lengths": torch.tensor([1, 2, 3, 2]),
        "entropy": torch.full((4, 3), 0.7),
        "old_logps": old_logps,
        "reference_logps": old_logps - 1.0,
        "rollout_seconds": 2.0,
        "generation_seconds": 1.0,
    }

    metrics = trainer._rollout_metrics(rollout)

    assert metrics["reward/frac_zero_std_groups"] == pytest.approx(0.5)
    assert metrics["reward/format"] == pytest.approx(0.75)
    assert metrics["completion/truncated_fraction"] == pytest.approx(0.25)
    assert metrics["completion/length_mean"] == pytest.approx(2.0)
    assert metrics["completion/length_max"] == pytest.approx(3.0)
    assert metrics["policy/entropy"] == pytest.approx(0.7)
    assert metrics["policy/logp_mean"] == pytest.approx(math.log(0.5))
    assert metrics["kl/ref_k1"] == pytest.approx(1.0)
    assert metrics["kl/ref_k3"] == pytest.approx(math.exp(-1.0))
    assert metrics["perf/gen_tokens_per_s"] == pytest.approx(8.0)


def test_merged_adapter_restores_base_weights_exactly():
    from peft import LoraConfig, get_peft_model

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.proj(x)

    model = get_peft_model(Tiny(), LoraConfig(r=2, lora_alpha=4, target_modules=["proj"]))
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter)
    base = model.base_model.model.proj.base_layer.weight
    original = base.detach().clone()
    x = torch.randn(3, 4)
    expected = model(x).detach()

    trainer = GRPOIssueFixTrainer.__new__(GRPOIssueFixTrainer)
    trainer.model_cfg = SimpleNamespace(use_lora=True)
    trainer.grpo_cfg = SimpleNamespace(merge_lora_for_generation=True)
    with torch.no_grad(), trainer._merged_adapter(model) as merged:
        assert merged
        assert not torch.equal(base, original)
        assert torch.allclose(model(x), expected, atol=1e-5)

    assert torch.equal(base, original)
    assert torch.allclose(model(x), expected)
