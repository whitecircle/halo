"""Rule-based answer normalization and matching; :func:`validate_answer` is the check every environment
grades an answer through."""

import re
from typing import Any

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
# A bold span holding only one of these is a label (``**Answer:** 5``), not the answer.
_ANSWER_LABELS = frozenset(prefix.rstrip(":") for prefix in _ANSWER_PREFIXES)

_BOXED_TOKEN = "\\boxed{"

_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")

# ``,`` groups thousands only in the strict ``1,234,567`` form, so ``1,2,3`` and ``3,5`` stay separate numbers.
# A chain never restarts at one of its own groups and a digit run splits one way, so the scan stays linear.
# A number glued to a letter or digit is part of a token (``h2o``, ``v2``), not a value.
_NUMBER_RE = re.compile(
    r"(?<![a-z\d_])(?P<sign>[-+]?)"
    r"(?P<value>(?:(?<!\d,)\d{1,3}(?:,\d{3})+(?!,?\d)(?:\.\d*)?|\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)"
    r"(?P<percent>\s*%)?"
)

# Formatting-only LaTeX reads as a space, as does ``\approx``, which states the value (``\%``/``\$`` keep
# their symbol; ``\circ`` marks degrees).
_LATEX_FORMATTING_RE = re.compile(
    r"\\(?:(?:text|textbf|textit|textrm|textsf|texttt|mathrm|mathbf|mathit|mathsf|mbox|num|displaystyle"
    r"|left|right|quad|qquad|circ|approx)(?![a-z])|[,;:! ]|(?=[%$]))"
)
# Thin-space grouping (``10\,000``) reads as ``,`` grouping, under the same strict form.
_LATEX_GROUPING_RE = re.compile(r"(?<=\d)\\,(?=\d)")
# Braces only group, so dropping them reads ``1{,}000`` as one number and ``m^{2}`` as a unit exponent.
# Dash-like characters are the minus a model means (``−5``, ``–5``), so ``5–7`` reads as ``5-7``.
_DASHES = (
    "\N{HYPHEN}\N{NON-BREAKING HYPHEN}\N{FIGURE DASH}\N{EN DASH}\N{MINUS SIGN}"
    "\N{SMALL HYPHEN-MINUS}\N{FULLWIDTH HYPHEN-MINUS}"
)
_CHAR_MAP = str.maketrans({"{": None, "}": None} | dict.fromkeys(_DASHES, "-"))

_SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"

# A power right after a letter is a unit exponent (``m/s^2``, ``m^(2)``, ``m**2``, ``m²``), not a value.
_UNIT_EXPONENT_RE = re.compile(rf"(?<=[^\W\d_])(?:(?:\^|\*\*)(?:[-+]?\d+|\([-+]?\d+\))|[⁺⁻]?[{_SUPERSCRIPT_DIGITS}]+)")

# The text is an expression or a bound, not a value: a LaTeX command left after formatting, a root,
# constant, ``±``/``∞``, a power left after unit exponents (``10²``), a function applied to an argument,
# or a comparison outside an arrow (``->``, ``=>``, ``<-``).
_EXPRESSION_OR_BOUND_RE = re.compile(
    rf"\\[a-z]|[√∛∜π∞±∓{_SUPERSCRIPT_DIGITS}≤≥≠]|(?<![-=])>|<(?!-)|!="
    r"|(?<![a-z])(?:(?:sqrt|log|ln|exp|sin|cos|tan)\s*[\d(_]|(?:pi|squared|cubed)(?![a-z]))"
)

# Two numbers with only operators, spaces, brackets and ``$`` between them are operands: ``2+2``,
# ``2024-01-01``, ``$5 + $5``.
_NON_MINUS_OPERATORS = "+*/^×÷·\N{DOT OPERATOR}\N{ASTERISK OPERATOR}"
_OPERATORS = re.escape("-" + _NON_MINUS_OPERATORS)
_OPERATOR_GAP_RE = re.compile(rf"[\s()\[\]$]*[{_OPERATORS}][{_OPERATORS}\s()\[\]$]*")
# A number joined by an operator to a one-letter variable is an operand as well (``1/x``, ``2^n``,
# ``n+1``). Longer words stay units or names (``$5/hour``), and ``-`` is left out, since ``5 - a``
# reads as prose.
_VARIABLE_OPERAND_RE = re.compile(
    rf"\d\s*[{re.escape(_NON_MINUS_OPERATORS)}]\s*[a-z](?![a-z])"
    rf"|(?<![a-z])[a-z]\s*[{re.escape(_NON_MINUS_OPERATORS)}]\s*[-+]?\.?\d"
)


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
    """Normalize text for answer comparison: lowercase, strip, extract from ``\\boxed{}`` / a lone
    ``**bold**`` span, drop leading answer prefixes ("the answer is", …) and trailing period."""
    text = str(text).strip()

    boxed = extract_last_boxed(text)
    if boxed:
        text = boxed.strip()

    # Several bold spans (``**7** or **8**``) hedge between them, and a lone label names none, so in
    # both cases the whole text stays.
    bold_spans = _BOLD_RE.findall(text)
    if len(bold_spans) == 1 and bold_spans[0].strip().lower().rstrip(":") not in _ANSWER_LABELS:
        text = bold_spans[0].strip()
    elif bold_spans:
        text = text.replace("**", "")

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
    """Every number a normalized answer states, or none when it is an expression or a bound."""
    text = _LATEX_GROUPING_RE.sub(",", text)
    text = _LATEX_FORMATTING_RE.sub(" ", text).translate(_CHAR_MAP)
    text = _UNIT_EXPONENT_RE.sub(" ", text)
    if _EXPRESSION_OR_BOUND_RE.search(text) or _VARIABLE_OPERAND_RE.search(text):
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
    numbers (``7, since 3 + 4 = 7``) grade as wrong, and so do an operand of arithmetic, a power, root,
    constant or function (``1/2``, ``2024-01-01``, ``10²``, ``\\sqrt{2}``, ``2\\pi``, ``log 2``) and a
    stated bound (``x < 3``). ``,`` thousands grouping reads as one number, a ``%`` value is divided by
    100, a power after a letter is a unit exponent (``9.8 m/s^2``), and a number glued to a letter is part
    of a token (``h2o``). The expected answer is read the same way and must state exactly one value
    (``18``, ``$18``, ``18 dollars``); an expression or a hedge there gets no numeric match.
    """
    expected_values = _stated_values(normalize_text(expected))
    if len(expected_values) != 1:
        return False
    target = expected_values[0]

    values = _stated_values(normalize_text(predicted))
    return bool(values) and all(
        abs(value - target) <= atol or (target != 0 and abs(value - target) / abs(target) <= rtol) for value in values
    )


def validate_answer(predicted: Any, expected: Any) -> bool:
    """Whether ``predicted`` answers ``expected``: an :func:`exact_match`, else a :func:`numeric_match`.

    Grading is all-or-nothing: a near-miss scores zero rather than partial credit, because a
    similarity threshold rewards a wrong answer that merely reads like the right one
    ("Washington" vs "Washington DC"). Substring containment is no match either: it inflates rewards
    ("7" would match "17")."""
    predicted_str, expected_str = str(predicted), str(expected)
    return exact_match(predicted_str, expected_str) or numeric_match(predicted_str, expected_str)
