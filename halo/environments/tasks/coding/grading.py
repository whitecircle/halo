"""Grading a contest program against its tests, and the run's grading contract (:class:`GradingSpec`), shared by the
environment and the offline re-grader so both score alike."""

import logging
import textwrap
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, replace
from math import isfinite
from typing import Any, NamedTuple

from halo.config_validation import require_bool, require_int
from halo.environments.sandbox.base import SANDBOX_DEFAULT_TIMEOUT, SandboxExecutor
from halo.environments.sandbox.factory import build_sandbox
from halo.environments.sandbox.languages import find_language
from halo.environments.sandbox.repl import signal_description, stderr_head, stderr_tail
from halo.environments.sandbox.results import SandboxResult, run_reporting_faults
from halo.environments.tasks.coding.program_source import host_build_error
from halo.environments.tasks.coding.verdicts import (
    VERDICT_DETAIL_FULL,
    VERDICT_DETAIL_OUTCOME,
    CheckerInfraError,
    FloatTolerance,
    VerdictFn,
    as_verdict,
    build_verdict,
    exact_output_match,
    require_comparator,
    require_verdict_detail,
)

logger = logging.getLogger(__name__)

# Stdout characters a test accepts before OUTPUT LIMIT EXCEEDED, raised per test to a multiple of its expected output:
# a fixed cap fails every correct solution whose answer is larger.
DEFAULT_MAX_OUTPUT_SIZE = 1_000_000
OUTPUT_CAP_EXPECTED_MULTIPLE = 4

_STDERR_EXCERPT_CHARS = 200
_OUTPUT_EXCERPT_CHARS = 100
# Failures only, one entry per distinct verdict, capped: an uncapped list turns a broken solution into a page of noise.
_MAX_FAILURE_DETAILS = 5
_MAX_FOLDED_TESTS_NAMED = 6


class GradeResult(NamedTuple):
    """``ran_ok`` counts tests that exited cleanly with output, ``infra_errors`` those lost to the backend, ``graded``
    those judged: below ``total``, the scoring denominator, once a budget stop or ``stop_on_first_failure`` cuts the
    run short. ``graded`` has no default because every value is wrong for some grade."""

    passed: int
    total: int
    details: str
    ran_ok: int
    graded: int
    infra_errors: int = 0
    budget_hit: bool = False


@dataclass
class _Failure:
    body: str
    tests: list[int]

    def render(self) -> str:
        if len(self.tests) == 1:
            return f"Test {self.tests[0]}: {self.body}"
        named = ", ".join(str(t) for t in self.tests[:_MAX_FOLDED_TESTS_NAMED])
        rest = len(self.tests) - _MAX_FOLDED_TESTS_NAMED
        return f"Tests {named}{f' and {rest} more' if rest > 0 else ''}: {self.body}"


@dataclass
class _FailureLog:
    """One grade's non-passing tests folded by verdict and capped; ``full`` is the ``verdict_detail`` that shows more
    than each failure's class."""

    full: bool
    failures: list[_Failure] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    suppressed: int = 0
    infra_texts: Counter[str] = field(default_factory=Counter)
    infra_errors: int = 0

    def add(self, test: int, body: str) -> None:
        for failure in self.failures:
            if failure.body == body:
                failure.tests.append(test)
                return
        if len(self.failures) < _MAX_FAILURE_DETAILS:
            self.failures.append(_Failure(body, [test]))
        else:
            self.suppressed += 1

    def add_infra(self, test: int, error: str) -> None:
        """Under ``outcome`` it shows its class alone: the error's text (a service's message, an exception quoting a
        path) can carry what the program wrote."""
        self.infra_texts[error] += 1
        self.infra_errors += 1
        self.add(
            test, f"ERROR -- {error}" if self.full else "ERROR -- grading infrastructure failure, not your program"
        )

    def details(self) -> list[str]:
        lines = [failure.render() for failure in self.failures] + self.notes
        if self.suppressed:
            lines.append(f"...and {self.suppressed} more non-passing tests (details omitted).")
        return lines


@contextmanager
def _grading_runner(
    sandbox: SandboxExecutor, code: str, *, language: str, timeout: float
) -> Iterator[Callable[[str], SandboxResult]]:
    """A ``run(stdin)`` over one session, so a compiled submission is built once, reset after every run so no test sees
    an earlier one's files. An executor without sessions, or a session that fails to open, grades one-shot."""

    def one_shot(stdin: str) -> SandboxResult:
        return run_reporting_faults(sandbox.run, code, stdin=stdin, timeout=timeout, language=language)

    try:
        session = sandbox.open_session()
    except NotImplementedError:
        yield one_shot
        return
    except Exception:
        logger.warning("Sandbox session failed to open for grading; running each test one-shot", exc_info=True)
        yield one_shot
        return

    def in_session(stdin: str) -> SandboxResult:
        result = run_reporting_faults(session.run, code, stdin=stdin, timeout=timeout, language=language)
        try:
            session.reset_to_prepared()
        except Exception as exc:  # the next test would inherit this one's files: no credit for either
            logger.debug(
                "Sandbox session reset failed during grading; scoring the test as an infra error", exc_info=True
            )
            return SandboxResult(error=f"sandbox reset failure: {type(exc).__name__}: {exc}")
        return result

    try:
        yield in_session
    finally:
        session.close()


def _compile_error_note(result: SandboxResult, shown: bool) -> str:
    """Every test fails the same way, so it is judged once with the whole pool counted. The head of the compiler's
    stderr follows where ``shown``: under ``full``, for a host build error, or on a backend that builds apart from
    every test's stdin (``COMPILES_WITHOUT_TEST_INPUT``)."""
    note = "COMPILATION ERROR (every test fails)"
    if result.stderr and shown:
        note += "\n" + textwrap.indent(stderr_head(result.stderr, _STDERR_EXCERPT_CHARS), "  ")
    return note


def _judge_test(
    result: SandboxResult,
    test: int,
    test_input: str,
    expected: str,
    verdict_fn: VerdictFn,
    log: _FailureLog,
    timeout: float,
    max_output_size: int,
) -> tuple[bool, bool, bool]:
    """``(passed, ran_ok, lost)``; ``lost`` is a test the backend lost, no verdict on the program."""
    if result.timed_out:
        log.add(test, f"TIME LIMIT EXCEEDED ({timeout:g}s)")
        return False, False, False
    if result.error:
        # Backend or transport failure; bucket as ERROR even with partial stdout.
        log.add_infra(test, result.error)
        return False, False, True
    if result.returncode not in (0, None):
        # Non-zero exit is a Runtime Error on every judge, never a pass even if stdout matches.
        body = "RUNTIME ERROR"
        if log.full:
            body += f" ({signal_description(result.returncode) or f'exit {result.returncode}'})"
            if result.stderr:
                body += f"\n  Stderr: {stderr_tail(result.stderr, _STDERR_EXCERPT_CHARS)}"
        log.add(test, body)
        return False, False, False
    cap = max(max_output_size, OUTPUT_CAP_EXPECTED_MULTIPLE * len(expected))
    if len(result.stdout) > cap:
        # Over-cap output is its own verdict: truncate-and-compare would grade a correct-but-long answer wrong.
        log.add(
            test,
            f"OUTPUT LIMIT EXCEEDED ({len(result.stdout)} > {cap} bytes)" if log.full else "OUTPUT LIMIT EXCEEDED",
        )
        return False, False, False
    actual = result.stdout
    try:
        passed = bool(verdict_fn(test_input, expected, actual))
    except CheckerInfraError as e:
        # Verdict lost to infra: no ran_ok/pass credit, keeping an all-infra outage visible.
        log.add_infra(test, str(e))
        return False, False, True
    if not passed:
        body = "FAIL"
        if log.full:
            body += (
                f"\n  Expected: {expected.strip()[:_OUTPUT_EXCERPT_CHARS]}"
                f"\n  Got:      {actual.strip()[:_OUTPUT_EXCERPT_CHARS]}"
            )
            if result.stderr:
                body += f"\n  Stderr: {stderr_tail(result.stderr, _STDERR_EXCERPT_CHARS)}"
        log.add(test, body)
    return passed, passed or bool(actual.strip()) or not expected.strip(), False


def _grade_summary(passed: int, total: int, graded: int, budget_hit: bool, max_grading_seconds: float | None) -> str:
    """The pass count, and after a budget stop how far grading got, for diagnosis only: the full pool stays the scoring
    denominator, so an ungraded test is never passed."""
    summary = f"Passed {passed}/{total} test cases."
    if budget_hit:
        summary += (
            f" (the {max_grading_seconds:g} s total grading time ran out after {graded} of {total} tests, so this "
            "submission is not accepted; a program fast enough to finish every test within it is graded in full)"
        )
    return summary


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
    verdict_detail: str = VERDICT_DETAIL_OUTCOME,
) -> GradeResult:
    """One sandbox run per test. ``max_grading_seconds`` bounds the wall clock, checked between tests (at least one
    runs), since a several-hundred-test problem would stall the rollout round; the full pool stays the denominator, so
    size the budget to the pool."""
    full = require_verdict_detail(verdict_detail) == VERDICT_DETAIL_FULL
    if not test_cases:
        return GradeResult(0, 0, "No test cases provided.", 0, graded=0)
    if sandbox is None:
        sandbox = build_sandbox()
    if verdict_fn is None:
        verdict_fn = as_verdict(exact_output_match)
    log = _FailureLog(full)
    total = len(test_cases)
    passed = ran_ok = graded = 0
    budget_hit = False
    deadline = None if max_grading_seconds is None else time.monotonic() + max_grading_seconds
    # Built here, before any test's stdin reaches the program, so its error line can show under ``outcome``.
    build_error = host_build_error(code, language, sandbox)
    host_build_failure = SandboxResult(stderr=build_error, compile_failed=True) if build_error else None
    with _grading_runner(sandbox, code, language=language, timeout=timeout_per_test) as run_test:
        for test, case in enumerate(test_cases, 1):
            if deadline is not None and graded and time.monotonic() >= deadline:
                budget_hit = True
                break
            # A null input is no input: the program reads end-of-file and a checker an empty input.txt.
            test_input = case.get("input") or ""
            result = host_build_failure or run_test(test_input)
            if result.compile_failed:
                shown = log.full or host_build_failure is not None or sandbox.COMPILES_WITHOUT_TEST_INPUT
                log.notes.append(_compile_error_note(result, shown))
                graded = total
                break
            test_passed, ran, lost = _judge_test(
                result, test, test_input, case.get("output", ""), verdict_fn, log, timeout_per_test, max_output_size
            )
            passed += test_passed
            ran_ok += ran
            graded = test
            # A test lost to the backend is not the program's failure, so it stops nothing.
            if stop_on_first_failure and not (test_passed or lost):
                log.notes.append(f"Stopped after first failing test ({total - test} not run).")
                break
    if log.infra_texts:
        texts = "; ".join(f"{text} (x{count})" for text, count in log.infra_texts.items())
        logger.warning("Grading lost %d test(s) to infra errors: %s", log.infra_errors, texts)
    summary = _grade_summary(passed, total, graded, budget_hit, max_grading_seconds)
    return GradeResult(
        passed, total, "\n".join([summary, *log.details()]), ran_ok, graded, log.infra_errors, budget_hit
    )


@dataclass(frozen=True)
class GradingSpec:
    """The grading contract a run is scored under: everything constant across its problems, so the offline re-grader
    reproduces a run's verdicts from the same object. Per-problem facts stay :func:`grade_solution` arguments."""

    sandbox: SandboxExecutor
    comparison: str = "exact"
    language: str = "python"
    max_output_size: int = DEFAULT_MAX_OUTPUT_SIZE
    stop_on_first_failure: bool = False
    default_timeout: float = SANDBOX_DEFAULT_TIMEOUT
    max_time_limit: float = SANDBOX_DEFAULT_TIMEOUT
    # Multiplies a compiled language's per-test limit; the interpreted floor and the clamp stay unscaled.
    compiled_time_limit_scale: float = 1.0
    max_grading_seconds: float | None = None
    verdict_detail: str = VERDICT_DETAIL_OUTCOME

    # Rebuilt offline from the run's env kwargs, never carried through a JSON dump.
    _META_EXCLUDED = frozenset({"sandbox"})

    def __post_init__(self) -> None:
        # ``replace`` re-runs this, so a restored meta block is held to the same contract.
        require_bool(type(self).__name__, stop_on_first_failure=self.stop_on_first_failure)
        for name, seconds in (
            ("default_timeout (timeout_per_test)", self.default_timeout),
            ("max_time_limit", self.max_time_limit),
        ):
            if not (isfinite(seconds) and seconds > 0):
                raise ValueError(f"{name} must be a finite number of seconds > 0, got {seconds!r}")
        if self.max_time_limit < self.default_timeout:
            # The prompt promises an interpreted solution at least default_timeout per test; a clamp below it
            # would grade under a contract the model was never told.
            raise ValueError(
                f"max_time_limit ({self.max_time_limit}) must be >= timeout_per_test ({self.default_timeout}), "
                f"the per-test floor the task contract states"
            )
        if not (isfinite(self.compiled_time_limit_scale) and self.compiled_time_limit_scale > 0):
            raise ValueError(
                f"compiled_time_limit_scale must be a finite number > 0, got {self.compiled_time_limit_scale}"
            )
        if self.max_grading_seconds is not None and not (
            isfinite(self.max_grading_seconds) and self.max_grading_seconds > 0
        ):
            raise ValueError(
                f"max_grading_seconds must be a finite number > 0 or None, got {self.max_grading_seconds}"
            )
        require_int(type(self).__name__, ge=1, max_output_size=self.max_output_size)
        require_comparator(self.comparison)
        require_verdict_detail(self.verdict_detail)

    def to_meta(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in self._META_EXCLUDED}

    def with_meta(self, meta: dict[str, Any], **overrides: Any) -> "GradingSpec":
        """A key no field declares raises, so a trajectory written under a retired spelling is never re-graded under
        today's defaults."""
        unknown = sorted(set(meta) - set(self.to_meta()))
        if unknown:
            raise ValueError(
                f"grading meta carries {unknown}, which name no GradingSpec field; the trajectory was "
                f"written by an older run and cannot be re-graded under its own contract."
            )
        return replace(self, **{**meta, **overrides})

    def time_limit_for(self, language: str, time_limit: float | None) -> float:
        """An interpreted language floors at ``default_timeout``, a compiled one scales by
        ``compiled_time_limit_scale``, either clamped to ``max_time_limit``."""
        limit = time_limit or self.default_timeout
        resolved = find_language(language)
        if resolved is not None and not resolved.is_compiled:
            # Floor an interpreted language's budget so a C++-tuned limit doesn't TLE a slower CPython solution.
            limit = max(limit, self.default_timeout)
        elif resolved is not None:
            # Statement limits are calibrated for C++ on a dedicated judge; the scale pays for shared grading cores.
            limit *= self.compiled_time_limit_scale
        return min(limit, self.max_time_limit)


def grade_solution(
    code: str,
    tests: list[dict[str, str]],
    spec: GradingSpec,
    *,
    checker: str | None = None,
    time_limit: float | None = None,
    stated_tolerance: FloatTolerance | None = None,
    language: str | None = None,
) -> GradeResult:
    """The single grading entry point; each test runs at :meth:`GradingSpec.time_limit_for`."""
    language = language or spec.language
    return run_solution_against_tests(
        code,
        tests,
        timeout_per_test=spec.time_limit_for(language, time_limit),
        max_output_size=spec.max_output_size,
        sandbox=spec.sandbox,
        language=language,
        verdict_fn=build_verdict(checker, spec.comparison, spec.sandbox, stated_tolerance=stated_tolerance),
        stop_on_first_failure=spec.stop_on_first_failure,
        max_grading_seconds=spec.max_grading_seconds,
        verdict_detail=spec.verdict_detail,
    )
