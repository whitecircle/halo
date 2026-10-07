#!/usr/bin/env python
"""Adversarial / edge-case tests for rule-based rewards and RLRR advantage shaping.

Covers three modules:
  * src/rewards/graders/matching.py — numeric and exact matching against crafted distractor inputs, and
    the empty-answer floor of the validation chain.
  * src/environments/envs/tasks/qa.py — multiple-choice letter extraction, which lives with the
    only environment that grades by letter.
  * src/trainers/grpo/objective/relative_rewards.py — RLRR degenerate-group handling
    (all-tied, single-element, empty) must produce finite, sane advantages.

Run: python tests/cpu/environments/test_rewards_adversarial.py  (or pytest)
"""

import re
import time

import numpy as np
import pytest

from src.args.mixins import RLRRConfig
from src.environments.envs.tasks.qa import (
    MULTIPLE_CHOICE_LETTERS,
    ExamQAEnvironment,
    multiple_choice_match,
)
from src.rewards.graders.matching import (
    exact_match,
    extract_last_boxed,
    normalize_text,
    numeric_match,
    validate_answer,
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


# numeric_match — a prediction matches only when it states one value


def test_numeric_match_leading_distractor_names_two_values():
    """Every number in the prediction must equal the expected one. A distractor before the real
    answer names a second value, so the prediction commits to neither and grades as wrong even though
    its last number is right. It does not start with an answer prefix, so normalize_text keeps both.
    """
    assert numeric_match("I tried 42 but the answer is 7", "7") is False


@pytest.mark.parametrize(
    ("predicted", "expected"),
    [
        ("3,500", "3500"),  # thousands grouping reads as one number, on either side
        ("3500", "3,500"),
        ("1,000", "1000"),
        ("$1,000,000", "1000000"),
        ("1{,}000", "1000"),  # braces only group
        (r"10\,000", "10000"),  # LaTeX thin-space grouping
        ("The answer is 110", "110"),
        ("x = 5", "5"),
        ("5 apples", "5"),
        ("$5/hour", "5"),  # a word after the operator is a unit, not a variable
        ("9.8 m/s", "9.8"),
        ("50%", "0.5"),
        ("0.5", "50%"),
        ("50% (0.5)", "0.5"),  # one value, restated
        ("3.000001", "3.0"),  # within rtol
        ("1e3", "1000"),
        ("5.", "5"),
        ("9.8 m/s^2", "9.8"),  # a ^n after a letter is a unit exponent
        ("5 cm^2", "5"),
        ("5 m^{2}", "5"),
        ("5 m^(2)", "5"),
        ("5 m**2", "5"),
        ("5 m²", "5"),
        ("5 cm³", "5"),
        ("9.8 m s⁻²", "9.8"),
        ("5 m2", "5"),  # a number glued to a letter or digit is part of a token
        ("5 cm3", "5"),
        ("5 mol of H2O", "5"),
        ("5 mol of C12H22O11", "5"),
        ("x_1 = 5", "5"),
        ("答案是42", "42"),  # only a Latin letter glues, so CJK text needs no space
        ("x = -5", "-5"),
        ("-5", "-5"),
        ("answer -> 7", "7"),  # an arrow is no bound
        ("x => 7", "7"),
        ("5 log entries", "5"),  # a function name without an argument is a word
        ("The tan box weighs 5 kg", "5"),
        ("sin embargo, 5", "5"),
        (r"\boxed{42}", "42"),
        ("+5", "5"),
        ("\N{MINUS SIGN}5", "-5"),  # a dash-like sign is a minus
        ("\N{EN DASH}5", "-5"),
        ("\N{FULLWIDTH HYPHEN-MINUS}5", "-5"),
        (r"x \approx 3.14", "3.14"),  # a relation that states the value
        (r"90^\circ", "90"),
        (r"\displaystyle 5", "5"),  # formatting-only LaTeX
        (r"\textit{5}", "5"),
        (r"\textsf{5}", "5"),
        (r"\texttt{5}", "5"),
        (r"\mathit{5}", "5"),
        (r"\mathsf{5}", "5"),
        (r"\num{10000}", "10000"),
    ],
)
def test_numeric_match_accepts_one_stated_value(predicted, expected):
    assert numeric_match(predicted, expected) is True


@pytest.mark.parametrize(
    ("predicted", "expected"),
    [
        ("1/2", "1"),  # the leading number of an expression is not its value
        (r"\boxed{\frac{1}{2}}", "1"),
        (r"\frac{3}{4}", "3"),
        ("3,500", "3"),
        ("$1,000,000", "1"),
        ("1,2345", "12345"),  # grouping only in the strict form
        ("1,2345", "1234"),
        (r"1\,2345", "12345"),
        ("2x+1", "2"),
        ("1/x", "1"),  # an operand of a one-letter variable
        ("x/2", "2"),
        ("2^n", "2"),
        ("n+1", "1"),
        ("2x+2", "2"),
        ("10^3", "10"),
        ("2024-01-01", "2024"),
        ("3-4", "3"),
        ("2+2", "2"),  # operands, even when each one equals the expected value
        ("2 +2", "2"),  # a sign right after an operand is the operator
        (r"2 \times 2", "2"),
        ("2×2", "2"),
        ("2\N{DOT OPERATOR}2", "2"),
        ("2\N{ASTERISK OPERATOR}2", "2"),
        ("$5 + $5", "5"),
        ("10²", "10"),  # a power after a digit
        ("2³", "2"),
        ("10⁻³", "10"),
        (r"\sqrt{2}", "2"),
        ("√2", "2"),
        ("∛8", "8"),
        ("∜16", "16"),
        ("sqrt(2)", "2"),
        ("log 2", "2"),
        ("log_2 8", "8"),
        ("5 squared", "5"),
        ("2 cubed", "2"),
        (r"2\pi", "2"),
        ("±5", "5"),
        ("∓5", "5"),
        ("(5, ∞)", "5"),
        ("e^2", "2"),  # a letter's exponent is no value, and nothing else is left
        (r"x \le 3", "3"),  # a bound, not the value
        ("x < 3", "3"),
        ("x > 3", "3"),
        ("x <= 3", "3"),
        ("x >= 3", "3"),
        ("x != 3", "3"),
        ("x ≤ 3", "3"),
        ("≥ 3", "3"),
        ("x ≠ 3", "3"),
        ("v2", "2"),  # part of a token
        ("-5", "5"),
        ("\N{EN DASH}5", "5"),
        ("5\N{EN DASH}7", "5"),  # a range
        ("7 or 8", "7"),  # a hedge names two values
        ("**7** or **8**", "7"),  # several bold spans hedge too
        ("The answer is **7**, or possibly **8**.", "7"),
        ("between 3 and 4", "3"),
        ("1,2,3", "1"),
        ("7, since 3 + 4 = 7", "7"),  # a response must commit to one value, working included
        ("110", "11"),  # a number, not a substring
        ("3.5", "3.0"),
        ("3.5", "3,500"),
        ("no numbers here", "42"),
    ],
)
def test_numeric_match_rejects_expressions_and_hedges(predicted, expected):
    assert numeric_match(predicted, expected) is False


@pytest.mark.parametrize(
    ("predicted", "expected", "verdict"),
    [
        ("18", "$18", True),  # the expected answer is read like a prediction
        ("18", "18 dollars", True),
        ("5", "5 apples", True),
        ("-5", "\N{MINUS SIGN}5", True),
        ("2", "1+1", False),  # an expected expression or hedge states no one value
        ("7", "7 or 8", False),
    ],
)
def test_numeric_match_reads_the_expected_answer_like_a_prediction(predicted, expected, verdict):
    assert numeric_match(predicted, expected) is verdict


@pytest.mark.parametrize(
    ("predicted", "expected", "verdict"),
    [
        pytest.param("9" * 100_000, "5", False, id="digit-run"),
        pytest.param("0." + "3" * 100_000, "0.333", True, id="decimal-run"),
        pytest.param("1 " * 25_000, "1", True, id="spaced-ones"),
        pytest.param("1" + ",000" * 16_000 + "0", "5", False, id="group-chain-then-digit"),
        pytest.param("1" + ",000" * 16_000 + ",00", "5", False, id="group-chain-then-short-group"),
        pytest.param(",".join(str(100 + i % 900) for i in range(16_000)) + ",1000", "5", False, id="3-digit-list"),
        pytest.param("5", "9" * 16_000 + "x", False, id="expected-digit-run"),
    ],
)
def test_numeric_match_parses_degenerate_input_within_budget(predicted, expected, verdict):
    """Degenerate rollouts emit long digit runs and comma chains. Each input here parses in tens of
    milliseconds. A pattern that restarts a comma chain at every group, or splits a digit run many ways
    when a full match fails, takes 6-8 s on the chain and expected-side rows, so the 1 s budget fails
    such a regression in seconds instead of hanging the suite."""
    start = time.monotonic()
    assert numeric_match(predicted, expected) is verdict
    assert time.monotonic() - start < 1.0


@pytest.mark.parametrize(
    ("predicted", "verdict"),
    [
        ("The capital is **Paris**.", True),  # a lone bold span is the answer
        ("**Paris** or **London**", False),  # several are a hedge
        ("**Answer:** Paris", True),  # a lone label is not the answer
        ("**Final answer**: Paris", True),
    ],
)
def test_exact_match_reads_only_a_lone_bold_span_as_the_answer(predicted, verdict):
    assert exact_match(predicted, "Paris") is verdict
    assert numeric_match(predicted.replace("Paris", "5").replace("London", "6"), "5") is verdict


# multiple_choice_match — structured extraction, no startswith fallback


def test_multiple_choice_extracts_from_phrase():
    assert multiple_choice_match("Although A is wrong, answer is B", "B") is True


def test_multiple_choice_no_startswith_fallback():
    """A response that merely STARTS with the letter (here "Apple") must not score
    as choice "A": there is no startswith fallback."""
    assert multiple_choice_match("Apple", "A") is False


def test_multiple_choice_paren_and_invalid_expected():
    assert multiple_choice_match("The correct option is (C).", "C") is True
    assert multiple_choice_match("answer is A", "AB") is False


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("The answer is definitely C", "D"),
        ("The answer is clearly B", "C"),
        ("The answer is a tie, so B", "A"),
        ("I pick this option because it fits: C", "B"),
        ("The largest, i.e. the gas giant, is B", "I"),
        ("E.g. Jupiter, the largest planet", "E"),
    ],
)
def test_multiple_choice_reads_no_choice_out_of_a_word(response, expected):
    """Prose around a choice letter must not be read as one: a word after "the answer is" or "option" that
    starts with the expected letter, the article "a", "i.e." and "E.g." would each pay that choice."""
    assert multiple_choice_match(response, expected) is False


def test_multiple_choice_reads_a_marked_up_or_bracketed_letter():
    assert multiple_choice_match("The answer is Jupiter (B)", "B") is True
    assert multiple_choice_match("The answer is: **B**", "B") is True
    assert multiple_choice_match("(b)", "B") is True
    assert multiple_choice_match("c", "C") is True


def test_multiple_choice_rejects_an_index_expected_answer():
    """MMLU's ``answer`` column is an int INDEX, and the matcher must keep refusing it.

    Coercing "1" → "B" here would guess at a choice ordering the matcher cannot see (``choices``
    lives in the trajectory context, not in either of its two string arguments) and would make a
    malformed answer column look like a working one. The conversion is
    ``ExamQAEnvironment._expected_choice_letter``'s, which reads ``choices``.
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


# validate_answer — an empty answer must not grade as a match


def test_empty_prediction_does_not_validate():
    """An empty prediction is neither an exact nor a numeric match, so the objective grades 0."""
    assert validate_answer("", "the expected answer") is False


def test_exact_match_validates():
    assert validate_answer("42", "42") is True


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
    raise SystemExit(pytest.main([__file__, "-v"]))
