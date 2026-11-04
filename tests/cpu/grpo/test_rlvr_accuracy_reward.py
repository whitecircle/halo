#!/usr/bin/env python
"""RLVR ``accuracy_reward``: the \\boxed{} extraction must survive nested braces. This reward is the RL signal, and a
truncated extraction scores a correct LaTeX answer 0.0 with no error; plain-numeric datasets (GSM8K) never trip it."""

import time

import pytest

from halo.rewards.graders.verifiable import accuracy_reward as _accuracy_reward


@pytest.fixture(scope="module")
def accuracy_reward():
    return _accuracy_reward


def test_nested_braces_score_correct(accuracy_reward):
    """``\\boxed{\\frac{1}{2}}`` must extract ``\\frac{1}{2}``, not the truncated ``\\frac{1``."""
    completions = [r"So the value is \boxed{\frac{1}{2}}"]
    assert accuracy_reward(completions, [r"\frac{1}{2}"]) == [1.0]
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
    assert accuracy_reward([r"\boxed{\frac{1}{3}}"], [r"\frac{1}{2}"]) == [0.0]


def test_last_of_several_boxes_is_graded(accuracy_reward):
    """The correcting box is deliberately the LaTeX one: only carrying a nested answer through it distinguishes a
    brace-balanced scan."""
    content = r"First \boxed{5}, correcting: \boxed{\frac{1}{2}}"
    assert accuracy_reward([content], [r"\frac{1}{2}"]) == [1.0]
    assert accuracy_reward([content], ["5"]) == [0.0]


def test_unterminated_box_falls_back_to_the_last_closed_one(accuracy_reward):
    """The nested closed box matters: a first-``}`` scan reads ``\\boxed{\\frac{1}{2}}`` as the fragment ``\\frac{1``,
    finds no later closing box, and grades the truncation."""
    truncated = r"Answer \boxed{\frac{1}{2}} then rambling \boxed{\frac{1"
    assert accuracy_reward([truncated], [r"\frac{1}{2}"]) == [1.0]
    assert accuracy_reward([truncated], [r"\frac{1"]) == [0.0]
    assert accuracy_reward([r"thinking... \boxed{\frac{1"], ["42"]) == [0.0]


def test_inner_box_wins_when_boxes_nest(accuracy_reward):
    """``\\boxed{A \\boxed{B}}`` grades as B. A deliberate tie-break in ``extract_last_boxed``: the innermost, latest
    box is the model's final commitment."""
    content = r"\boxed{7 \boxed{42}}"
    assert accuracy_reward([content], ["42"]) == [1.0]
    assert accuracy_reward([content], [r"7 \boxed{42}"]) == [0.0]


def test_degenerate_repetition_terminates(accuracy_reward):
    """Degenerate rollouts repeat ``\\boxed{`` thousands of times; extraction stays linear."""
    start = time.monotonic()
    assert accuracy_reward(["\\boxed{" * 20000], ["42"]) == [0.0]
    assert time.monotonic() - start < 5.0


def test_escaped_latex_boxed_is_still_graded(accuracy_reward):
    """Models emit JSON/markdown-escaped LaTeX."""
    assert accuracy_reward([r"The answer is \\boxed{42}"], ["42"]) == [1.0]
    assert accuracy_reward([r"\\boxed{\frac{1}{2}}"], [r"\frac{1}{2}"]) == [1.0]


def test_trailing_empty_box_does_not_erase_the_real_answer(accuracy_reward):
    assert accuracy_reward([r"\boxed{42} ... \boxed{}"], ["42"]) == [1.0]


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
    """GSM8K ground truth is "<rationale> #### <final>"; commas and ``$`` are stripped from both sides."""
    gt = "Janet sells 9 eggs.\n9 * 2 = 18\n#### 1,200"
    assert accuracy_reward([r"\boxed{1200}"], [gt]) == [1.0]
    assert accuracy_reward([r"\boxed{$1,200}"], [gt]) == [1.0]
    assert accuracy_reward([r"\boxed{1201}"], [gt]) == [0.0]
    # The rationale before #### must never be what gets compared.
    assert accuracy_reward([r"\boxed{9}"], [gt]) == [0.0]


def test_boxed_answer_against_empty_ground_truth_scores_zero(accuracy_reward):
    assert accuracy_reward([r"\boxed{42}"], [""]) == [0.0]


def test_no_answer_against_a_blank_ground_truth_is_not_rewarded(accuracy_reward):
    """Both sides normalize to ``""``, so a bare equality check scores 1.0 and "emit no \\boxed at all" is then the
    degenerate optimum the policy finds first. Blank prompts are filtered upstream; blank answers are not."""
    assert accuracy_reward(["I am not sure."], [""]) == [0.0]
    assert accuracy_reward([r"\boxed{}"], [""]) == [0.0]


def test_fewer_answers_than_completions_raises(accuracy_reward):
    """A silently truncated zip hands the trainer a shorter reward list than its batch, so the advantages line up
    against the wrong rows."""
    with pytest.raises(ValueError):
        accuracy_reward([r"\boxed{1}", r"\boxed{2}"], ["1"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
