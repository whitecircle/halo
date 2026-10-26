"""How one test's output is judged: the output comparators with the float tolerance a statement states, the special
judge a problem ships (``checker.py input.txt correct_output.txt solution_output.txt`` printing ``1``/``0``, the
``open-r1/codeforces`` ``generated_checker`` contract), and the verdict a grade runs under. Every verdict shares
the ``(test_input, expected, actual) -> bool`` signature."""

import re
from collections.abc import Callable
from functools import partial
from typing import NamedTuple

from halo.environments.sandbox.base import SANDBOX_DEFAULT_TIMEOUT, SandboxExecutor
from halo.environments.sandbox.results import run_reporting_faults

# What a non-passing test's detail line shows the policy. ``outcome``, the default, states each failed
# test's verdict class and nothing beyond it: stderr, an exit status, an output size or a backend's
# error text can each carry the hidden input the program read. Which tests fail, and how, still
# reaches the policy; ``stop_on_first_failure`` narrows that to the first failing test, the Codeforces
# contract. ``full`` adds them, and the expected and produced output of a wrong answer, which turns
# every resubmission into a probe of the hidden tests.
VERDICT_DETAIL_FULL = "full"
VERDICT_DETAIL_OUTCOME = "outcome"
_VERDICT_DETAILS = (VERDICT_DETAIL_FULL, VERDICT_DETAIL_OUTCOME)

# The checker's argv contract: the test input, the reference output, the candidate output, in this order.
_CHECKER_FILES = ("input.txt", "correct_output.txt", "solution_output.txt")
# run() cannot pass argv, so a runpy shim supplies the Codeforces checker argv contract.
_CHECKER_DRIVER = (
    "import runpy, sys\n"
    f"sys.argv = {['checker.py', *_CHECKER_FILES]!r}\n"
    'runpy.run_path("checker.py", run_name="__main__")\n'
)

# A tolerance a statement states ("absolute or relative error does not exceed 10^{-4}"), in the spellings the
# scraped statements carry: 10^{-k}, 10^-k, 10^(-k), 1e-k, 0.0…01, and a lost superscript ("10 - k") only right
# after a bound word, where "|10 - 3|" in a worked example cannot pass for one.
_STATED_TOLERANCE = re.compile(
    r"10\s*\^\s*[{(]?\s*[-\u2212\u2013]\s*(\d{1,2})\s*[})]?(?!\d)"
    r"|(?:exceeds?|than|of|most|within|\u2264|<=|<)\s+10\s*[-\u2212\u2013]\s*(\d{1,2})(?!\d)"
    r"|\b1\s*[eE]\s*[-\u2212\u2013]\s*(\d{1,2})\b|\b0\.(0{1,11})1\b"
)
_TOLERANCE_CONTEXT = re.compile(r"\berror\b|\bprecision\b|\btolerance\b", re.IGNORECASE)
_TOLERANCE_CONTEXT_CHARS = 200
_TOLERANCE_KINDS = re.compile(r"\b(absolute|relative)\b", re.IGNORECASE)

VerdictFn = Callable[[str, str, str], bool]


class FloatTolerance(NamedTuple):
    """How far a float token may stray from the expected value: within ``absolute`` of it or within ``relative`` of
    its magnitude, each bound in its own kind; ``None`` where a statement states no bound of that kind."""

    absolute: float | None
    relative: float | None


DEFAULT_FLOAT_TOLERANCE = FloatTolerance(absolute=1e-6, relative=1e-6)


def find_stated_float_tolerance(statement: str) -> FloatTolerance | None:
    """The float tolerance a problem statement states, ``None`` when it states none: each ``10^-k`` near an "error",
    "precision" or "tolerance" bounds the kinds ("absolute", "relative") named since the previous one, loosest per
    kind; a bound naming no kind bounds both, only where no bound names one ("|a-o|/a <= 10^-8" restates a relative
    error).
    """
    named: dict[str, float] = {}
    unnamed = None
    previous_end = 0
    for match in _STATED_TOLERANCE.finditer(statement):
        lead = statement[max(previous_end, match.start() - _TOLERANCE_CONTEXT_CHARS) : match.start()]
        previous_end = match.end()
        window = statement[max(0, match.start() - _TOLERANCE_CONTEXT_CHARS) : match.end() + _TOLERANCE_CONTEXT_CHARS]
        if not _TOLERANCE_CONTEXT.search(window):
            continue
        power, lost_power, e_power, zeros = match.groups()
        digits = power or lost_power or e_power
        exponent = int(digits) if digits else len(zeros) + 1
        if not 1 <= exponent <= 12:
            continue
        bound = 10.0**-exponent
        kinds = {kind.lower() for kind in _TOLERANCE_KINDS.findall(lead)}
        if not kinds:
            unnamed = max(unnamed or 0.0, bound)
        for kind in kinds:
            named[kind] = max(named.get(kind, 0.0), bound)
    if named:
        return FloatTolerance(absolute=named.get("absolute"), relative=named.get("relative"))
    return None if unnamed is None else FloatTolerance(absolute=unnamed, relative=unnamed)


def float_tolerance(stated: FloatTolerance | None) -> FloatTolerance:
    """The tolerance a float token is compared under: per kind, a stated bound where looser than the default, never
    tighter, since the expected outputs are a reference solution's printed digits, not exact values."""
    if stated is None:
        return DEFAULT_FLOAT_TOLERANCE
    return FloatTolerance(
        absolute=max(DEFAULT_FLOAT_TOLERANCE.absolute, stated.absolute or 0.0),
        relative=max(DEFAULT_FLOAT_TOLERANCE.relative, stated.relative or 0.0),
    )


def _tokens_equal(expected: str, actual: str, tolerance: FloatTolerance = DEFAULT_FLOAT_TOLERANCE) -> bool:
    """Whether one output token matches: exactly, or — only for an *expected* token that looks like a float — within
    ``tolerance``, so integer problems stay byte-exact (``5`` != ``5.0000001``).
    """
    if expected == actual:
        return True
    if not any(c in expected for c in ".eE"):
        return False
    try:
        e, a = float(expected), float(actual)
    except ValueError:
        return False
    error = abs(e - a)
    return error <= tolerance.absolute or error <= tolerance.relative * abs(e)


def compare_tokens(expected: str, actual: str, tolerance: FloatTolerance = DEFAULT_FLOAT_TOLERANCE) -> bool:
    """Codeforces token comparison: same sequence of whitespace tokens (case-sensitive, float-tolerant within
    ``tolerance``), so line endings, trailing spaces and blank lines do not count."""
    et, at = expected.split(), actual.split()
    if len(et) != len(at):
        return False
    return all(_tokens_equal(e, a, tolerance) for e, a in zip(et, at, strict=False))


def _lf_lines(text: str) -> str:
    """``text`` stripped, with ``\r\n`` and ``\r`` read as ``\n``."""
    return text.strip().replace("\r\n", "\n").replace("\r", "\n")


def exact_output_match(expected: str, actual: str) -> bool:
    """Trimmed exact comparison of a program's output, ``\r\n`` and ``\r`` read as ``\n``: a judge compares lines,
    not line endings, whichever backend captured them. Not :func:`halo.rewards.graders.matching.exact_match`, which
    normalizes a free-text answer.
    """
    return _lf_lines(expected) == _lf_lines(actual)


# A GradingSpec's ``comparison`` (the env's ``output_comparison``) -> the comparator it names.
_OUTPUT_COMPARATORS: dict[str, Callable[[str, str], bool]] = {"exact": exact_output_match, "tokens": compare_tokens}


def require_comparator(comparison: str) -> Callable[[str, str], bool]:
    """The comparator ``comparison`` (the env's ``output_comparison``) names; an unknown name raises."""
    if comparison not in _OUTPUT_COMPARATORS:
        raise ValueError(
            f"comparison (output_comparison) must be one of {list(_OUTPUT_COMPARATORS)}, got {comparison!r}"
        )
    return _OUTPUT_COMPARATORS[comparison]


def require_verdict_detail(verdict_detail: str) -> str:
    """``verdict_detail`` when it is one of :data:`_VERDICT_DETAILS`; any other value raises."""
    if verdict_detail not in _VERDICT_DETAILS:
        raise ValueError(f"verdict_detail must be one of {_VERDICT_DETAILS}, got {verdict_detail!r}")
    return verdict_detail


def as_verdict(comparator: Callable[[str, str], bool]) -> VerdictFn:
    """Adapt an output-only ``(expected, actual)`` comparator to the ``(input, expected, actual)`` signature."""

    def _verdict(_test_input: str, expected: str, actual: str) -> bool:
        return comparator(expected, actual)

    return _verdict


class CheckerInfraError(RuntimeError):
    """The checker run was lost to a grading-backend failure, counted into ``infra_errors``: a plain ``False`` would
    score the outage as a wrong answer and hide it from the infra-outage guard.
    """


class CheckerVerdict:
    """Special judge: runs a problem's Python checker through a sandbox per test, accepting only a clean exit printing
    ``1``; a backend failure raises :class:`CheckerInfraError`. Its timeout bounds trusted problem-setter code, not the
    solution.
    """

    def __init__(
        self,
        checker_code: str,
        sandbox: SandboxExecutor,
        timeout: float = SANDBOX_DEFAULT_TIMEOUT,
    ):
        self.checker_code = checker_code
        self.sandbox = sandbox
        self.timeout = timeout

    def __call__(self, test_input: str, expected: str, actual: str) -> bool:
        result = run_reporting_faults(
            self.sandbox.run,
            _CHECKER_DRIVER,
            language="python",
            timeout=self.timeout,
            files={
                "checker.py": self.checker_code,
                **dict(zip(_CHECKER_FILES, (test_input, expected, actual), strict=True)),
            },
        )
        if result.error:
            raise CheckerInfraError(f"checker backend failure: {result.error}")
        if not result.ok:
            return False
        tokens = result.stdout.split()
        return bool(tokens) and tokens[-1] == "1"


def build_verdict(
    checker: str | None, comparison: str, sandbox: SandboxExecutor, *, stated_tolerance: FloatTolerance | None = None
) -> VerdictFn:
    """The per-test verdict: a special-judge ``checker`` when given, else ``comparison`` (``tokens``: whitespace
    tokens, floats within :func:`float_tolerance` of ``stated_tolerance``; ``exact``: trimmed equality). The checker
    runs at the infra default timeout: a tight C++-tuned limit must TLE the solution, not the judge.
    """
    if checker:
        return CheckerVerdict(checker, sandbox, timeout=SANDBOX_DEFAULT_TIMEOUT)
    comparator = require_comparator(comparison)
    if comparison == "tokens":
        comparator = partial(comparator, tolerance=float_tolerance(stated_tolerance))
    return as_verdict(comparator)
