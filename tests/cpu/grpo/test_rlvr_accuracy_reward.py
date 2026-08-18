#!/usr/bin/env python
"""rlvr.py ``accuracy_reward``: the \\boxed{} extraction must survive nested braces.

This reward IS the RL signal for RLVR online GRPO, and a truncated extraction scores a CORRECT
answer 0.0 with no error — a brace-unaware match turns every LaTeX answer carrying a nested group
(``\\frac{1}{2}``, ``\\text{...}``, ``\\sqrt{...}``) into silent negative reward. Plain-numeric
datasets (GSM8K) never trip it, so the defect is invisible without these cases.

Extraction is shared with ``src/environments/rewards.extract_last_boxed``; here it is pinned
end-to-end through the production reward, including the GSM8K ``####`` ground-truth split and the
comma/``$`` normalization on both sides.

    python tests/cpu/grpo/test_rlvr_accuracy_reward.py
"""

import sys
import time

import pytest

from tests.common.utils import load_script_module


@pytest.fixture(scope="module")
def accuracy_reward():
    return load_script_module("scripts/training/online_grpo/rlvr.py", "halo_test_rlvr").accuracy_reward


# --- Nested braces — the silent-zero case ---


def test_nested_braces_score_correct(accuracy_reward):
    """``\\boxed{\\frac{1}{2}}`` must extract ``\\frac{1}{2}``, not the truncated ``\\frac{1``."""
    completions = [r"So the value is \boxed{\frac{1}{2}}"]
    assert accuracy_reward(completions, [r"\frac{1}{2}"]) == [1.0]
    # The truncation a first-brace match would produce must NOT be accepted as the answer.
    assert accuracy_reward(completions, [r"\frac{1"]) == [0.0]


def test_deeply_nested_and_multi_group_answers(accuracy_reward):
    completions = [
        r"\boxed{\frac{\sqrt{3}}{2}}",
        r"\boxed{\text{Nov}}",
        r"\boxed{\{1, 2\}}",
    ]
    answers = [r"\frac{\sqrt{3}}{2}", r"\text{Nov}", r"\{1, 2\}"]
    assert accuracy_reward(completions, answers) == [1.0, 1.0, 1.0]


def test_wrong_nested_answer_still_scores_zero(accuracy_reward):
    """Grading stays strict: carrying nested braces through must not make a wrong answer match."""
    assert accuracy_reward([r"\boxed{\frac{1}{3}}"], [r"\frac{1}{2}"]) == [0.0]


# --- Which box wins ---


def test_last_of_several_boxes_is_graded(accuracy_reward):
    """Reasoning traces box intermediate results; the final box is the answer, nesting and all.

    The correcting box is deliberately the LaTeX one: last-box selection alone is easy to get right,
    and only carrying a nested answer through it distinguishes a brace-balanced scan.
    """
    content = r"First \boxed{5}, correcting: \boxed{\frac{1}{2}}"
    assert accuracy_reward([content], [r"\frac{1}{2}"]) == [1.0]
    assert accuracy_reward([content], ["5"]) == [0.0]


def test_unterminated_box_falls_back_to_the_last_closed_one(accuracy_reward):
    """A completion truncated mid-box must grade the last box that CLOSED, not the open fragment.

    The nested closed box is load-bearing: a first-``}`` scan reads ``\\boxed{\\frac{1}{2}}`` as the
    fragment ``\\frac{1`` and then finds no later closing box, so it grades the truncation as the
    answer instead of falling back.
    """
    truncated = r"Answer \boxed{\frac{1}{2}} then rambling \boxed{\frac{1"
    assert accuracy_reward([truncated], [r"\frac{1}{2}"]) == [1.0]
    assert accuracy_reward([truncated], [r"\frac{1"]) == [0.0]
    # No closed box at all -> empty extraction, reward 0, no exception.
    assert accuracy_reward([r"thinking... \boxed{\frac{1"], ["42"]) == [0.0]


def test_inner_box_wins_when_boxes_nest(accuracy_reward):
    """``\\boxed{A \\boxed{B}}`` grades as B — the rightmost opening wins.

    A deliberate tie-break in ``extract_last_boxed``: the innermost/latest box is the model's final
    commitment. Unpinned, a rewrite could return the outer ``A \\boxed{B}`` and stay green.
    """
    content = r"\boxed{7 \boxed{42}}"
    assert accuracy_reward([content], ["42"]) == [1.0]
    assert accuracy_reward([content], [r"7 \boxed{42}"]) == [0.0]


def test_degenerate_repetition_terminates(accuracy_reward):
    """Degenerate rollouts repeat ``\\boxed{`` thousands of times; extraction stays linear."""
    start = time.monotonic()
    assert accuracy_reward(["\\boxed{" * 20000], ["42"]) == [0.0]
    assert time.monotonic() - start < 5.0


def test_escaped_latex_boxed_is_still_graded(accuracy_reward):
    """Models emit JSON/markdown-escaped LaTeX; ``\\\\boxed{...}`` must not read as "no answer"."""
    assert accuracy_reward([r"The answer is \\boxed{42}"], ["42"]) == [1.0]
    assert accuracy_reward([r"\\boxed{\frac{1}{2}}"], [r"\frac{1}{2}"]) == [1.0]


def test_trailing_empty_box_does_not_erase_the_real_answer(accuracy_reward):
    """An empty ``\\boxed{}`` carries no answer, so an earlier real box is what gets graded."""
    assert accuracy_reward([r"\boxed{42} ... \boxed{}"], ["42"]) == [1.0]


# --- Plain, non-nested paths: brace balancing must not disturb them ---


def test_plain_numeric_answer_unchanged(accuracy_reward):
    completions = [r"The answer is \boxed{42}", r"\boxed{-3.5}", "no box at all"]
    assert accuracy_reward(completions, ["42", "-3.5", "42"]) == [1.0, 1.0, 0.0]


def test_conversational_completion_uses_last_message(accuracy_reward):
    completion = [
        {"role": "assistant", "content": r"Draft: \boxed{99}"},
        {"role": "assistant", "content": r"\boxed{\frac{1}{2}}"},
    ]
    assert accuracy_reward([completion], [r"\frac{1}{2}"]) == [1.0]
    assert accuracy_reward([completion], ["99"]) == [0.0]


def test_gsm8k_ground_truth_split_and_separator_stripping(accuracy_reward):
    """GSM8K ground truth is "<rationale> #### <final>"; only the final answer is compared,
    and commas / ``$`` are stripped from BOTH sides."""
    gt = "Janet sells 9 eggs.\n9 * 2 = 18\n#### 1,200"
    assert accuracy_reward([r"\boxed{1200}"], [gt]) == [1.0]
    assert accuracy_reward([r"\boxed{$1,200}"], [gt]) == [1.0]
    assert accuracy_reward([r"\boxed{1201}"], [gt]) == [0.0]
    # The rationale before #### must never be what gets compared.
    assert accuracy_reward([r"\boxed{9}"], [gt]) == [0.0]


def test_boxed_answer_against_empty_ground_truth_scores_zero(accuracy_reward):
    assert accuracy_reward([r"\boxed{42}"], [""]) == [0.0]


def test_no_answer_against_a_blank_ground_truth_is_not_rewarded(accuracy_reward):
    """A row whose ``answer`` column is blank must not pay full reward for answering nothing.

    Both sides normalize to ``""``, so a bare equality check scores 1.0 — and since the reward is the
    RL signal, "emit no \\boxed at all" is then the degenerate optimum the policy finds first. Blank
    prompts are filtered upstream; blank answers are not.
    """
    assert accuracy_reward(["I am not sure."], [""]) == [0.0]
    assert accuracy_reward([r"\boxed{}"], [""]) == [0.0]


def test_fewer_answers_than_completions_raises(accuracy_reward):
    """A short ``answer`` column must fail loud, not truncate.

    The reward owes one float per completion; a silently truncated zip hands the trainer a shorter
    reward list than the batch it generated, so the advantages line up against the wrong rows.
    """
    with pytest.raises(ValueError):
        accuracy_reward([r"\boxed{1}", r"\boxed{2}"], ["1"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
