#!/usr/bin/env python
"""Adversarial / edge-case tests for rule-based rewards and RLRR advantage shaping.

Covers three modules:
  * src/environments/rewards.py — numeric matching against crafted distractor inputs, and the
    empty-answer reward floor.
  * src/environments/envs/tasks/qa.py — multiple-choice letter extraction, which lives with the
    only environment that grades by letter.
  * src/trainers/grpo/objective/relative_rewards.py — RLRR degenerate-group handling
    (all-tied, single-element, empty) must produce finite, sane advantages.

Run: python tests/cpu/environments/test_rewards_adversarial.py  (or pytest)
"""

import re
import sys
import time

import numpy as np
import pytest

from src.args.mixins import RLRRConfig
from src.environments.envs.tasks.qa import (
    MULTIPLE_CHOICE_LETTERS,
    ExamQAEnvironment,
    multiple_choice_match,
)
from src.environments.rewards import (
    compute_answer_reward,
    extract_last_boxed,
    normalize_text,
    numeric_match,
)
from src.trainers.grpo.objective.relative_rewards import relative_advantages

# extract_last_boxed — brace-balanced, or every nested LaTeX answer scores 0


def test_boxed_extraction_survives_nested_braces():
    """A first-``}`` match truncates ``\\boxed{\\frac{1}{2}}`` to ``\\frac{1``, so a correct answer
    grades as wrong with no error. Depth matching is what keeps the reward signal honest."""
    assert extract_last_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_last_boxed(r"\boxed{\frac{\sqrt{3}}{2}}") == r"\frac{\sqrt{3}}{2}"
    assert normalize_text(r"the answer is \boxed{\text{Nov}}") == r"\text{nov}"


def test_boxed_extraction_takes_the_last_box():
    """Reasoning traces box intermediate results; the final box is the answer."""
    assert extract_last_boxed(r"first \boxed{5} then \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert normalize_text(r"\boxed{5} ... \boxed{7}") == "7"


def test_boxed_extraction_handles_unterminated_and_missing_boxes():
    """A completion truncated mid-box must not raise or return a fragment."""
    assert extract_last_boxed(r"\boxed{1+2") is None
    assert extract_last_boxed("no box here") is None
    assert extract_last_boxed(r"\boxed{42} then \boxed{\frac{1") == "42"


def test_boxed_extraction_ignores_escaped_braces():
    """``\\{``/``\\}`` are literal LaTeX braces — counting them as depth would end the scan early."""
    assert extract_last_boxed(r"\boxed{\{1,2\}}") == r"\{1,2\}"
    assert extract_last_boxed(r"\boxed{\}}") == r"\}"


def test_boxed_extraction_survives_a_doubled_backslash():
    """Escaped LaTeX (``\\\\boxed{...}``, routine from JSON/markdown-trained models) must still be
    found: skipping a backslash pair blindly would swallow the token's own backslash."""
    assert extract_last_boxed(r"\\boxed{42}") == "42"
    assert extract_last_boxed(r"\\\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert normalize_text(r"answer: \\boxed{paris}") == "paris"


def test_empty_box_is_not_an_answer():
    """An empty box carries no answer, so an earlier real box wins and a lone one extracts nothing."""
    assert extract_last_boxed(r"\boxed{42} \boxed{}") == "42"
    assert extract_last_boxed(r"\boxed{}") is None
    assert extract_last_boxed(r"\boxed{   }") is None
    # normalize_text keeps the surrounding text when nothing extractable is boxed.
    assert normalize_text(r"hello \boxed{}") == r"hello \boxed{}"


def test_boxed_extraction_is_linear_in_completion_length():
    """Degenerate rollouts repeat ``\\boxed{`` thousands of times. A rescan per candidate would be
    quadratic; this must stay bounded, not merely terminate eventually.

    The budget is ~300x the linear cost and ~150x below the quadratic one, so it separates the two
    without turning a loaded CI box into a failure.
    """
    start = time.monotonic()
    assert extract_last_boxed("\\boxed{" * 50000) is None
    assert extract_last_boxed("\\boxed{" * 50000 + "x}") == "x"
    assert time.monotonic() - start < 5.0


# numeric_match — first-number extraction is a known false-negative source


def test_numeric_match_leading_distractor_is_false_negative():
    """_NUMBER_RE.search picks the FIRST number in the prediction. When a distractor
    number precedes the real answer, the match FAILS even though the answer is correct.

    This pins the CURRENT (documented limitation) behavior so a change to last-number
    or boxed-answer extraction is a deliberate, test-visible decision. The prediction
    does not start with an answer-prefix, so normalize_text leaves "42" first.
    """
    assert numeric_match("I tried 42 but the answer is 7", "7") is False


def test_numeric_match_clean_answer_passes():
    """When the only number is the answer, numeric_match succeeds (and is not fooled
    by substring containment — "110" is matched as a number, not as containing "11")."""
    assert numeric_match("The answer is 110", "110") is True
    assert numeric_match("110", "11") is False  # not substring/containment


def test_numeric_match_percentage_and_tolerance():
    assert numeric_match("50%", "0.5") is True
    assert numeric_match("3.000001", "3.0") is True  # within rtol
    assert numeric_match("3.5", "3.0") is False


# multiple_choice_match — structured extraction, no startswith fallback


def test_multiple_choice_extracts_from_phrase():
    assert multiple_choice_match("Although A is wrong, answer is B", "B") is True


def test_multiple_choice_no_startswith_fallback():
    """A response that merely STARTS with the letter (here "Apple") must not score
    as choice "A" — the guard removed the startswith fallback."""
    assert multiple_choice_match("Apple", "A") is False


def test_multiple_choice_paren_and_invalid_expected():
    assert multiple_choice_match("The correct option is (C).", "C") is True
    assert multiple_choice_match("answer is A", "AB") is False


def test_multiple_choice_rejects_an_index_expected_answer():
    """MMLU's ``answer`` column is an int INDEX, and the matcher must keep refusing it.

    Coercing "1" → "B" here would guess at a choice ordering the matcher cannot see (``choices``
    lives in the trajectory context, not in either of its two string arguments) and would make a
    malformed answer column look like a working one. The conversion belongs in dataset prep, so
    ``run_env.py``'s example points at a letter-answer dataset instead.
    """
    for index in ("0", "1", 1, "3"):
        assert multiple_choice_match("The answer is B", index) is False


# ExamQAEnvironment prompt <-> grader agreement


def test_exam_prompt_states_the_full_graded_letter_range():
    """The instruction the model reads must not be narrower than what the grader scores.

    Naming "A, B, C, or D" on a 10-choice MMLU-Pro row steers the model off every E-J option, and
    each such answer is graded wrong even when the letter it never wrote was correct.
    """
    prompt = ExamQAEnvironment.EXAM_SYSTEM_PROMPT.format(search_instruction="")
    named = set(re.findall(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", prompt))

    assert MULTIPLE_CHOICE_LETTERS[0] in named, prompt
    assert MULTIPLE_CHOICE_LETTERS[-1] in named, prompt
    assert named <= set(MULTIPLE_CHOICE_LETTERS), sorted(named - set(MULTIPLE_CHOICE_LETTERS))


# compute_answer_reward — an empty answer must not grade as a match


def test_empty_prediction_gets_failure_reward():
    """An empty prediction matches nothing in the default chain, so it earns the failure reward."""
    r = compute_answer_reward("", "the expected answer", success_reward=1.0, failure_reward=0.0)
    assert r == 0.0


def test_exact_match_gets_success_reward():
    assert compute_answer_reward("42", "42", success_reward=2.0, failure_reward=-1.0) == 2.0


# RLRR — degenerate groups must yield finite, sensible advantages


def test_rlrr_all_tied_group_std_normalize_no_nan():
    """An all-correct, all-equal-reward group has zero shaped-reward variance.
    With std_normalize=True the divisor would be 0; the eps guard must keep the
    advantages finite (and ~0), never NaN/inf."""
    cfg = RLRRConfig(mode="hrr", std_normalize=True, correctness_clip=False, length_rerank=False)
    rewards = [1.0, 1.0, 1.0, 1.0]
    adv = relative_advantages(rewards, cfg)  # all above correctness_threshold -> all correct
    assert adv.shape == (4,)
    assert np.all(np.isfinite(adv)), f"non-finite advantages: {adv}"
    assert np.allclose(adv, 0.0, atol=1e-3), f"tied group should give ~0 advantage, got {adv}"


def test_rlrr_single_element_group_zero_advantage():
    """A group of one is its own mean -> advantage 0 (no relative signal)."""
    cfg = RLRRConfig(mode="prr", correctness_clip=False)
    adv = relative_advantages([0.7], cfg)
    assert adv.shape == (1,)
    assert np.all(np.isfinite(adv))
    assert float(adv[0]) == 0.0


def test_rlrr_empty_group_returns_empty():
    cfg = RLRRConfig()
    adv = relative_advantages([], cfg)
    assert adv.shape == (0,)


def test_rlrr_distinguishes_within_correct_group_by_length():
    """HRR keeps signal alive when every response is correct: with length re-rank,
    the SHORTER correct response gets the higher advantage (conciseness preference)."""
    cfg = RLRRConfig(mode="hrr", tau=0.1, lam=10.0, correctness_clip=False, length_rerank=True)
    adv = relative_advantages([1.0, 1.0], cfg, lengths=[5, 500])
    assert np.all(np.isfinite(adv))
    assert adv[0] > adv[1], f"shorter correct response should rank higher: {adv}"
    assert not np.allclose(adv, 0.0), "length re-rank must produce a non-zero relative signal"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
