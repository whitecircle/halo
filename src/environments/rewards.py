"""Shared rule-based answer validation and reward utilities. Composable via validate_answer()."""

import logging
import re
from collections.abc import Callable
from typing import Any

from src.log import warn_once

logger = logging.getLogger(__name__)

_ANSWER_PREFIXES = [
    "the answer is",
    "the final answer is",
    "therefore",
    "thus",
    "so",
    "hence",
    "answer:",
    "final answer:",
    "result:",
]

_BOXED_TOKEN = "\\boxed{"

_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")

_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

_PERCENT_RE = re.compile(r"([-+]?\d*\.?\d+)\s*%")


def extract_last_boxed(text: str) -> str | None:
    """Content of the last balanced ``\\boxed{...}`` in ``text``, or ``None`` when none closes.

    Braces are matched by depth so nested groups survive: a first-``}`` match truncates
    ``\\boxed{\\frac{1}{2}}`` to ``\\frac{1`` and silently scores a correct LaTeX answer as wrong. A
    backslash escapes the next character, so ``\\{``/``\\}`` never shift the depth, while a doubled
    backslash still introduces the token (models emit escaped LaTeX). Neither an unterminated
    ``\\boxed{`` nor an empty ``\\boxed{}`` is a candidate — one would yield a truncated answer, the
    other no answer at all.

    One left-to-right pass: degenerate rollouts repeat ``\\boxed{`` thousands of times, and rescanning
    per candidate would be quadratic in the completion length.
    """
    open_braces: list[int | None] = []  # content start per open brace; None for a brace that opens no box
    best_start = -1
    best: str | None = None

    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\":
            if text.startswith(_BOXED_TOKEN, index):
                index += len(_BOXED_TOKEN)
                open_braces.append(index)
            elif text.startswith(_BOXED_TOKEN, index + 1):
                index += 1
            else:
                index += 2
            continue
        if char == "{":
            open_braces.append(None)
        elif char == "}" and open_braces:
            start = open_braces.pop()
            # Rightmost opening wins, so a box nested inside another reads as the inner one.
            if start is not None and start > best_start and text[start:index].strip():
                best_start, best = start, text[start:index]
        index += 1

    return best


def normalize_text(text: str) -> str:
    """Normalize text for answer comparison: lowercase, strip, extract from ``\\boxed{}`` /
    ``**bold**``, drop leading answer prefixes ("the answer is", …) and trailing period."""
    text = str(text).strip()

    boxed = extract_last_boxed(text)
    if boxed:
        text = boxed.strip()

    bold = _BOLD_RE.search(text)
    if bold:
        text = bold.group(1).strip()

    text = text.lower().strip()

    # Strip prefixes only at a word boundary, so "so"/"thus" don't eat "South Korea"/"Thusly".
    for prefix in _ANSWER_PREFIXES:
        if text.startswith(prefix):
            rest = text[len(prefix) :]
            if prefix.endswith(":") or rest == "" or rest[0].isspace() or rest[0] in ":,":
                text = rest.lstrip(":, \t\r\n").strip()
                break

    text = text.rstrip(".")

    return text.strip()


def exact_match(predicted: str, expected: str) -> bool:
    """Case-insensitive exact match after normalization."""
    return normalize_text(predicted) == normalize_text(expected)


def numeric_match(
    predicted: str,
    expected: str,
    rtol: float = 0.01,
    atol: float = 1e-6,
) -> bool:
    """Compare extracted numbers with tolerance. Handles ints, floats, negatives,
    scientific notation, and percentages (a ``%`` value is divided by 100 before comparison)."""
    pred_text = normalize_text(predicted)
    exp_text = normalize_text(expected)

    try:
        pct_match = _PERCENT_RE.search(pred_text)
        if pct_match:
            pred_num = float(pct_match.group(1)) / 100.0
        else:
            pred_match = _NUMBER_RE.search(pred_text)
            if not pred_match:
                return False
            pred_num = float(pred_match.group())

        pct_match_exp = _PERCENT_RE.search(exp_text)
        exp_num = float(pct_match_exp.group(1)) / 100.0 if pct_match_exp else float(exp_text)

        if abs(pred_num - exp_num) <= atol:
            return True
        return exp_num != 0 and abs(pred_num - exp_num) / abs(exp_num) <= rtol
    except (ValueError, TypeError, AttributeError):
        return False


# Substring containment is deliberately not a method here: it inflates rewards ("7" matches "17").
DEFAULT_METHODS: list[Callable[[str, str], bool]] = [
    exact_match,
    numeric_match,
]

# Validation methods already reported as raising. A broken matcher raises on every sample it grades, so
# warning per call would bury the run's logs in one repeated line — warn once per method instead.
_VALIDATION_FAILURE_WARNED: set[str] = set()


def _warn_validation_failure(method: Callable[[str, str], bool]) -> None:
    """Report a validation method that raised, once per method per process.

    A method that raises grades its answer as wrong, and a broken one does so for every sample — so
    the failure must be visible, but only once.
    """
    # Qualified so two matchers sharing a bare name stay distinct; a partial or callable object has
    # neither name and falls back to its repr.
    name = getattr(method, "__qualname__", None) or repr(method)
    warn_once(
        logger,
        _VALIDATION_FAILURE_WARNED,
        name,
        "answer-validation method %s raised; every answer it cannot process scores as wrong "
        "(further failures from it are not logged)",
        name,
        exc_info=True,
    )


def validate_answer(
    predicted: Any,
    expected: Any,
    methods: list[Callable[[str, str], bool]] | None = None,
) -> bool:
    """True as soon as one method of the chain accepts the answer. ``methods`` default to
    ``DEFAULT_METHODS`` (exact + numeric); a method that raises grades as no match.

    Grading is all-or-nothing: a near-miss scores zero rather than partial credit, because a
    similarity threshold rewards a wrong answer that merely reads like the right one
    ("Washington" vs "Washington DC")."""
    predicted_str = str(predicted)
    expected_str = str(expected)

    for method in methods if methods is not None else DEFAULT_METHODS:
        try:
            if method(predicted_str, expected_str):
                return True
        except Exception:
            _warn_validation_failure(method)

    return False


def compute_answer_reward(
    predicted: Any,
    expected: Any,
    success_reward: float = 1.0,
    failure_reward: float = 0.0,
    methods: list[Callable[[str, str], bool]] | None = None,
) -> float:
    """:func:`validate_answer` wrapper for ``_compute_reward``: ``success_reward`` on a match,
    ``failure_reward`` otherwise."""
    return success_reward if validate_answer(predicted, expected, methods=methods) else failure_reward
