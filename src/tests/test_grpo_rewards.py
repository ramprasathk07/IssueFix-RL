from trainers.custom.train_grpo import (
    format_reward,
    python_syntax_reward,
    reference_similarity_reward,
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
