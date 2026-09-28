"""Rule-based answer normalization and matching, composable via :func:`validate_answer`."""

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

# ``,`` groups thousands only in the strict ``1,234,567`` form, so ``1,2,3`` and ``3,5`` stay separate numbers.
# A chain never restarts at one of its own groups and a digit run splits one way, so the scan stays linear.
_NUMBER_RE = re.compile(
    r"(?P<sign>[-+]?)(?P<value>(?:(?<!\d,)\d{1,3}(?:,\d{3})+(?!,?\d)(?:\.\d*)?|\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)"
    r"(?P<percent>\s*%)?"
)

# Formatting-only LaTeX reads as a space, as does ``\approx``, which states the value (``\%``/``\$`` keep
# their symbol; ``\circ`` marks degrees). Braces only group, so dropping them reads ``1{,}000`` as one
# number and ``m^{2}`` as a unit exponent.
_LATEX_FORMATTING_RE = re.compile(
    r"\\(?:(?:text|textbf|textrm|mathrm|mathbf|mbox|left|right|quad|qquad|circ|approx)(?![a-z])|[,;:! ]|(?=[%$]))"
)
_PREDICTION_CHAR_MAP = str.maketrans({"{": None, "}": None, "\N{MINUS SIGN}": "-"})

# A ``^n`` right after a letter is a unit exponent (``m/s^2``), not a value.
_UNIT_EXPONENT_RE = re.compile(r"(?<=[^\W\d_])\^[-+]?\d+")

# Any LaTeX command left after formatting, a root, constant or function: the prediction is an expression.
_SYMBOLIC_RE = re.compile(r"\\[a-z]|[√π∞±∓]|(?<![a-z])(?:sqrt|pi|log|ln|exp|sin|cos|tan)(?![a-z])")

# Two numbers with only operators, spaces and brackets between them are operands: ``2+2``, ``2024-01-01``.
_OPERATOR_GAP_RE = re.compile(r"[\s()\[\]]*[-+*/^×÷·\N{EN DASH}][-+*/^×÷·\N{EN DASH}\s()\[\]]*")


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


def _number_value(match: re.Match[str]) -> float:
    """The value a ``_NUMBER_RE`` match spells: grouping commas dropped, a ``%`` value divided by 100."""
    value = float(match["sign"] + match["value"].replace(",", ""))
    return value / 100.0 if match["percent"] else value


def _stated_values(text: str) -> list[float]:
    """Every number a normalized prediction states, or none when one is an operand of an expression."""
    text = _LATEX_FORMATTING_RE.sub(" ", text).translate(_PREDICTION_CHAR_MAP)
    text = _UNIT_EXPONENT_RE.sub(" ", text)
    if _SYMBOLIC_RE.search(text):
        return []

    values: list[float] = []
    previous_end = None
    for match in _NUMBER_RE.finditer(text):
        # The gap runs up to the digits, so a sign right after an operand reads as the operator it is.
        if previous_end is not None and _OPERATOR_GAP_RE.fullmatch(text, previous_end, match.start("value")):
            return []
        values.append(_number_value(match))
        previous_end = match.end()
    return values


def numeric_match(
    predicted: str,
    expected: str,
    rtol: float = 0.01,
    atol: float = 1e-6,
) -> bool:
    """True when the prediction states one value and it equals the expected number within tolerance.

    Every number in the prediction must match, so a hedge (``7 or 8``) and working restated with other
    numbers (``7, since 3 + 4 = 7``) grade as wrong, and so does a number that is an operand of arithmetic
    or of a symbolic expression (``1/2``, ``2024-01-01``, ``\\sqrt{2}``, ``2\\pi``). ``,`` thousands
    grouping reads as one number, a ``%`` value is divided by 100, and a ``^n`` after a letter is a unit
    exponent (``9.8 m/s^2``). The expected answer must be one number as a whole.
    """
    expected_number = _NUMBER_RE.fullmatch(normalize_text(expected))
    if expected_number is None:
        return False
    target = _number_value(expected_number)

    values = _stated_values(normalize_text(predicted))
    return bool(values) and all(
        abs(value - target) <= atol or (target != 0 and abs(value - target) / abs(target) <= rtol) for value in values
    )


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
