"""Grading for competitive-programming solutions: code + tests -> pass count.

Shared by CodeContestsEnvironment and the offline eval runner, so online and offline scoring match.
Verdict primitives share the ``(test_input, expected, actual) -> bool`` signature.

``CheckerVerdict`` contract (``open-r1/codeforces`` ``generated_checker``):
``python checker.py input.txt correct_output.txt solution_output.txt`` printing ``1``/``0`` to stdout.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Any, NamedTuple

from src.environments.sandbox.base import SANDBOX_DEFAULT_TIMEOUT, SandboxExecutor, SandboxResult, resolve_language
from src.environments.sandbox.resolve import resolve_sandbox

logger = logging.getLogger(__name__)

VerdictFn = Callable[[str, str, str], bool]

_FLOAT_TOL = 1e-6

# Bytes of stdout compared per test; sized generously, since truncating below the real output would
# fail a correct solution.
DEFAULT_MAX_OUTPUT_SIZE = 1_000_000

_STDERR_EXCERPT_CHARS = 200
_OUTPUT_EXCERPT_CHARS = 100
# Verdict detail lists failures only and is capped: per-test PASS lines add nothing to the summary's
# pass count, and an uncapped failure list would run to hundreds of entries.
_MAX_FAILURE_DETAILS = 5

# run() cannot pass argv, so a runpy shim supplies the Codeforces checker argv contract.
_CHECKER_DRIVER = (
    "import runpy, sys\n"
    'sys.argv = ["checker.py", "input.txt", "correct_output.txt", "solution_output.txt"]\n'
    'runpy.run_path("checker.py", run_name="__main__")\n'
)


def _tokenize(text: str) -> list[str]:
    """Split into whitespace-separated tokens (handles ``\\r\\n``, trailing spaces, blank lines)."""
    return text.split()


def _tokens_equal(expected: str, actual: str) -> bool:
    """Compare one output token, tolerant of float rounding.

    Exact match first; only when the *expected* token looks like a float (``.``/``e``/``E``) do both
    parse as floats within :data:`_FLOAT_TOL`, so integer problems stay byte-exact
    (``5`` != ``5.0000001``).
    """
    if expected == actual:
        return True
    if not any(c in expected for c in ".eE"):
        return False
    try:
        e, a = float(expected), float(actual)
    except ValueError:
        return False
    return abs(e - a) <= _FLOAT_TOL * max(1.0, abs(e))


def compare_tokens(expected: str, actual: str) -> bool:
    """Codeforces token comparison: same sequence of whitespace tokens (case-sensitive, float-tolerant)."""
    et, at = _tokenize(expected), _tokenize(actual)
    if len(et) != len(at):
        return False
    return all(_tokens_equal(e, a) for e, a in zip(et, at, strict=False))


def exact_output_match(expected: str, actual: str) -> bool:
    """Compare program output exactly after stripping leading/trailing whitespace from both sides.

    Legacy CodeContests behavior. Distinct from :func:`src.environments.rewards.exact_match`, which
    normalizes a free-text answer.
    """
    return expected.strip() == actual.strip()


def as_verdict(comparator: Callable[[str, str], bool]) -> VerdictFn:
    """Adapt an output-only ``(expected, actual)`` comparator to the ``(input, expected, actual)`` signature."""

    def _verdict(_test_input: str, expected: str, actual: str) -> bool:
        return comparator(expected, actual)

    return _verdict


def _run_in_sandbox(sandbox: SandboxExecutor, code: str, **kwargs) -> SandboxResult:
    """Execute ``code`` through ``sandbox``, reporting an executor fault as ``SandboxResult(error=...)``.

    Grading requires that a run lost to the backend arrive as an ``error`` result rather than an
    exception. The remote backend already does this; a local one raises instead (no interpreter,
    missing ``bwrap``, fork exhaustion, ENOSPC writing the working dir). An escaping exception would
    leave ``submit_solution`` as an ordinary tool error, scoring ``failure_reward`` without triggering
    the infra-outage guard, so a host fault would enter the GRPO baseline as a wrong program.
    """
    try:
        return sandbox.run(code, **kwargs)
    except Exception as exc:  # anything raised on the grading side is infra, not a verdict
        logger.warning("Sandbox run failed during grading; scoring the test as an infra error", exc_info=True)
        return SandboxResult(error=f"sandbox backend failure: {type(exc).__name__}: {exc}")


class CheckerInfraError(RuntimeError):
    """The checker run was lost to a grading-backend failure (transport error, backend down).

    Raised by :class:`CheckerVerdict` and consumed by :func:`run_solution_against_tests` into
    ``infra_errors``. Returning ``False`` instead would score the outage as a wrong answer and hide it
    from the ``_grading_infra_outage`` guard.
    """


class CheckerVerdict:
    """Special-judge grader: runs a problem's ``generated_checker`` (always Python) in a sandbox per test.

    Verdict is ``True`` iff the checker exits cleanly and prints ``1``; a checker crash, timeout, or
    ``0`` all reject. A sandbox *backend* failure raises :class:`CheckerInfraError` instead, marking
    the test as lost rather than judged. The timeout bounds trusted problem-setter code and is not the
    solution's per-test limit.
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
        result = _run_in_sandbox(
            self.sandbox,
            _CHECKER_DRIVER,
            language="python",
            timeout=self.timeout,
            files={
                "checker.py": self.checker_code,
                "input.txt": test_input,
                "correct_output.txt": expected,
                "solution_output.txt": actual,
            },
        )
        if result.error:
            raise CheckerInfraError(f"checker backend failure: {result.error}")
        if not result.ok:
            return False
        tokens = result.stdout.split()
        return bool(tokens) and tokens[-1] == "1"


class GradeResult(NamedTuple):
    """Outcome of grading one solution against a test list.

    ``ran_ok`` = tests whose code ran to completion and produced output; it feeds the "runnable"
    reward term, so a no-output stub does not score alongside a real attempt. ``infra_errors`` = tests
    lost to a grading-backend failure; ``infra_errors == total`` means the grade carries no signal.
    ``graded`` = tests actually judged (below ``total`` once a budget stop or ``stop_on_first_failure``
    cuts the run short, while ``total`` stays the scoring denominator), and ``budget_hit`` says which
    of the two it was, since partial grading is otherwise indistinguishable from a wrong solution.
    ``graded`` has no default, since no value is right for every grade (0 would claim a fully judged
    run graded nothing), so each construction states what it judged.
    """

    passed: int
    total: int
    details: str
    ran_ok: int
    graded: int
    infra_errors: int = 0
    budget_hit: bool = False


def run_solution_against_tests(
    code: str,
    test_cases: list[dict[str, str]],
    timeout_per_test: float = SANDBOX_DEFAULT_TIMEOUT,
    max_output_size: int = DEFAULT_MAX_OUTPUT_SIZE,
    sandbox: SandboxExecutor | None = None,
    language: str = "python",
    verdict_fn: VerdictFn | None = None,
    stop_on_first_failure: bool = False,
    max_grading_seconds: float | None = None,
) -> GradeResult:
    """Run a solution against test cases through a :class:`SandboxExecutor` -> :class:`GradeResult`.

    Each test feeds ``input`` to stdin and compares stdout to expected ``output`` via ``verdict_fn``
    (default: trimmed exact match) in an independent sandbox run. Details list only non-passing tests,
    capped at ``_MAX_FAILURE_DETAILS``.

    ``max_grading_seconds`` bounds one grade's total wall clock, since tests run sequentially and a
    several-hundred-test problem would otherwise stall the whole rollout round. It is checked between
    tests (hard bound: the budget plus one per-test timeout) and at least one test always runs. A
    budget stop keeps the full pool as the denominator, so a solution too slow to reach its remaining
    tests cannot outscore one that ran them all; size the budget to the pool.
    """
    if not test_cases:
        return GradeResult(0, 0, "No test cases provided.", 0, graded=0)

    if sandbox is None:
        sandbox = resolve_sandbox()
    if verdict_fn is None:
        verdict_fn = as_verdict(exact_output_match)

    passed = 0
    ran_ok = 0  # clean exit and produced output (PASS or FAIL), not crash/TLE/overflow/backend error
    infra_errors = 0  # tests lost to a backend/transport failure, not the program's fault
    total = len(test_cases)
    details = []
    suppressed = 0
    deadline = None if max_grading_seconds is None else time.monotonic() + max_grading_seconds
    graded = 0
    budget_hit = False

    failure_details_shown = 0

    def add_detail(line: str) -> None:
        """Append a non-passing test's detail while under the cap; count it as suppressed past it."""
        nonlocal suppressed, failure_details_shown
        if failure_details_shown < _MAX_FAILURE_DETAILS:
            details.append(line)
            failure_details_shown += 1
        else:
            suppressed += 1

    for i, tc in enumerate(test_cases, 1):
        if deadline is not None and graded and time.monotonic() >= deadline:
            budget_hit = True
            break
        test_input = tc.get("input", "")
        expected_output = tc.get("output", "")

        result = _run_in_sandbox(sandbox, code, stdin=test_input, timeout=timeout_per_test, language=language)

        test_passed = False
        if result.timed_out:
            add_detail(f"Test {i}: TIME LIMIT EXCEEDED ({timeout_per_test:g}s)")
        elif result.error:
            # Backend/transport failure (not the program's stderr); bucketed as ERROR even with partial stdout.
            infra_errors += 1
            add_detail(f"Test {i}: ERROR -- {result.error}")
        elif result.returncode not in (0, None):
            # A non-zero exit is a runtime error on every judge, even when stdout matches.
            line = f"Test {i}: RUNTIME ERROR (exit {result.returncode})"
            if result.stderr:
                line += f"\n  Stderr: {result.stderr.strip()[:_STDERR_EXCERPT_CHARS]}"
            add_detail(line)
        elif len(result.stdout) > max_output_size:
            # Over-cap output gets its own verdict: truncating before comparison would fail a correct long answer.
            add_detail(f"Test {i}: OUTPUT LIMIT EXCEEDED ({len(result.stdout)} > {max_output_size} bytes)")
        else:
            actual_output = result.stdout
            try:
                test_passed = verdict_fn(test_input, expected_output, actual_output)
            except CheckerInfraError as e:
                # Verdict lost to infra: no ran_ok/pass credit, keeping an all-infra outage visible.
                infra_errors += 1
                add_detail(f"Test {i}: ERROR -- {e}")
            else:
                if test_passed or actual_output.strip() or not expected_output.strip():
                    # Requiring output stops a no-output stub from scoring on this term.
                    ran_ok += 1
                if test_passed:
                    passed += 1
                else:
                    line = (
                        f"Test {i}: FAIL\n  Expected: {expected_output.strip()[:_OUTPUT_EXCERPT_CHARS]}"
                        f"\n  Got:      {actual_output.strip()[:_OUTPUT_EXCERPT_CHARS]}"
                    )
                    if result.stderr:
                        line += f"\n  Stderr: {result.stderr.strip()[:_STDERR_EXCERPT_CHARS]}"
                    add_detail(line)

        graded = i
        if stop_on_first_failure and not test_passed:
            details.append(f"Stopped after first failing test ({total - i} not run).")
            break

    if suppressed:
        details.append(f"...and {suppressed} more non-passing tests (details omitted).")

    summary = f"Passed {passed}/{total} test cases."
    if budget_hit:
        # The graded prefix is reported for diagnosis, not for credit: the full pool stays the
        # scoring denominator.
        summary += f" (graded the first {graded} of {total}; {max_grading_seconds:g}s grading budget reached)"
    return GradeResult(passed, total, summary + "\n" + "\n".join(details), ran_ok, graded, infra_errors, budget_hit)


def select_verdict(checker: str | None, comparison: str, sandbox: SandboxExecutor) -> VerdictFn:
    """Pick the per-test verdict: a special-judge ``checker`` if given, else ``comparison``
    (``"tokens"`` = whitespace-token equality, ``"exact"`` = trimmed byte equality).

    The checker runs at the infra default timeout rather than the solution's clamped per-test limit,
    so a tight C++-tuned limit applies to the solution and not to the judge grading it."""
    if checker:
        return CheckerVerdict(checker, sandbox, timeout=SANDBOX_DEFAULT_TIMEOUT)
    if comparison == "tokens":
        return as_verdict(compare_tokens)
    if comparison == "exact":
        return as_verdict(exact_output_match)
    raise ValueError(f"unknown output comparison {comparison!r} (expected 'tokens' or 'exact')")


@dataclass(frozen=True)
class GradingSpec:
    """The grading settings a run is scored under: everything constant across its problems.

    Built once by the environment and passed to every :func:`grade_solution` call, so the offline
    re-grader reproduces a run's verdicts from the same object instead of re-threading each knob and
    defaulting any it misses. The two per-problem facts (``checker``, ``time_limit``) come from the
    problem payload and stay arguments.

    :meth:`to_meta` / :meth:`with_meta` carry these settings across a trajectory dump, derived from
    the field list so a field added here reaches the re-grade without a second edit.
    """

    sandbox: SandboxExecutor
    comparison: str = "exact"
    language: str = "python"
    max_output_size: int = DEFAULT_MAX_OUTPUT_SIZE
    stop_on_first_failure: bool = False
    default_timeout: float = SANDBOX_DEFAULT_TIMEOUT
    max_time_limit: float = SANDBOX_DEFAULT_TIMEOUT
    max_grading_seconds: float | None = None

    # The live executor: rebuilt from the run's env kwargs offline, not carried through a JSON dump.
    _META_EXCLUDED = frozenset({"sandbox"})

    def to_meta(self) -> dict[str, Any]:
        """These settings as a JSON-able block for a trajectory meta line."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in self._META_EXCLUDED}

    def with_meta(self, meta: dict[str, Any], **overrides: Any) -> "GradingSpec":
        """These settings with a dumped :meth:`to_meta` block applied over them, then ``overrides``.

        A key naming no field raises rather than being ignored, so a trajectory written under a
        retired spelling is rejected instead of re-graded under current defaults.
        """
        unknown = sorted(set(meta) - set(self.to_meta()))
        if unknown:
            raise ValueError(
                f"grading meta carries {unknown}, which name no GradingSpec field; the trajectory was "
                f"written by an older run and cannot be re-graded under its own contract."
            )
        return replace(self, **{**meta, **overrides})


def grade_solution(
    code: str,
    tests: list[dict[str, str]],
    spec: GradingSpec,
    *,
    checker: str | None = None,
    time_limit: float | None = None,
) -> GradeResult:
    """Grade ``code`` against ``tests`` -> :class:`GradeResult`; the grading entry point.

    Selects the verdict via :func:`select_verdict` and runs every test at the problem's ``time_limit``
    (clamped to ``spec.max_time_limit``), falling back to ``spec.default_timeout``.
    ``spec.max_grading_seconds`` bounds the total sequential grading cost per submission (see
    :func:`run_solution_against_tests`).
    """
    limit = time_limit or spec.default_timeout
    language = resolve_language(spec.language)
    if language is not None and not language.is_compiled:
        # Floor an interpreted language's budget so a C++-tuned limit does not TLE a slower CPython solution.
        limit = max(limit, spec.default_timeout)
    return run_solution_against_tests(
        code,
        tests,
        timeout_per_test=min(limit, spec.max_time_limit),
        max_output_size=spec.max_output_size,
        sandbox=spec.sandbox,
        language=spec.language,
        verdict_fn=select_verdict(checker, spec.comparison, spec.sandbox),
        stop_on_first_failure=spec.stop_on_first_failure,
        max_grading_seconds=spec.max_grading_seconds,
    )
