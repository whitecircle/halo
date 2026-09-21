"""Grading for competitive-programming solutions: code + tests -> pass count.

Shared by CodeContestsEnvironment and the offline eval runner (identical online/offline scoring). Verdict
primitives share the ``(test_input, expected, actual) -> bool`` signature.

``CheckerVerdict`` contract (``open-r1/codeforces`` ``generated_checker``):
``python checker.py input.txt correct_output.txt solution_output.txt`` printing ``1``/``0`` to stdout.
"""

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from math import isfinite
from typing import Any, NamedTuple

from src.environments.sandbox.base import (
    SANDBOX_DEFAULT_TIMEOUT,
    SandboxExecutor,
    SandboxResult,
    SandboxSession,
    resolve_language,
)
from src.environments.sandbox.resolve import resolve_sandbox

logger = logging.getLogger(__name__)

VerdictFn = Callable[[str, str, str], bool]

_FLOAT_TOL = 1e-6

# Bytes of stdout compared per test; generous, since truncating below real output fails a correct solution.
DEFAULT_MAX_OUTPUT_SIZE = 1_000_000

_STDERR_EXCERPT_CHARS = 200
_OUTPUT_EXCERPT_CHARS = 100
# Verdict detail is failures-only, one entry per distinct verdict, capped: per-test PASS lines carry no
# information the summary's pass count doesn't, tests failing the same way say it once, and an
# uncapped failure list turns a broken solution into a page of noise.
_MAX_FAILURE_DETAILS = 5
# Tests named on one folded verdict line before the rest are counted.
_MAX_FOLDED_TESTS_NAMED = 6
# What a non-passing test's detail line shows the policy. ``full`` adds the expected and produced
# output to a wrong answer; ``outcome`` states the verdict alone, the Codeforces contract.
VERDICT_DETAIL_FULL = "full"
VERDICT_DETAIL_OUTCOME = "outcome"
VERDICT_DETAILS = (VERDICT_DETAIL_FULL, VERDICT_DETAIL_OUTCOME)

# The checker's argv contract: the test input, the reference output, the candidate output, in this order.
CHECKER_FILES = ("input.txt", "correct_output.txt", "solution_output.txt")
# run() cannot pass argv, so a runpy shim supplies the Codeforces checker argv contract.
_CHECKER_DRIVER = (
    "import runpy, sys\n"
    f"sys.argv = {['checker.py', *CHECKER_FILES]!r}\n"
    'runpy.run_path("checker.py", run_name="__main__")\n'
)


def _stderr_tail(stderr: str) -> str:
    """The end of a program's stderr: a traceback names the exception on its last line, so a head
    excerpt of a long one shows the frames and drops the error."""
    text = stderr.strip()
    return text if len(text) <= _STDERR_EXCERPT_CHARS else "…" + text[-_STDERR_EXCERPT_CHARS:]


@dataclass
class _Failure:
    """One distinct non-passing verdict and the tests that produced it, in order."""

    body: str
    tests: list[int]

    def render(self) -> str:
        if len(self.tests) == 1:
            return f"Test {self.tests[0]}: {self.body}"
        named = ", ".join(str(t) for t in self.tests[:_MAX_FOLDED_TESTS_NAMED])
        rest = len(self.tests) - _MAX_FOLDED_TESTS_NAMED
        return f"Tests {named}{f' and {rest} more' if rest > 0 else ''}: {self.body}"


def _tokenize(text: str) -> list[str]:
    """Split into whitespace-separated tokens (handles ``\\r\\n``, trailing spaces, blank lines)."""
    return text.split()


def _tokens_equal(expected: str, actual: str) -> bool:
    """Compare one output token, tolerant of float rounding.

    Exact match first; only when the *expected* token looks like a float (``.``/``e``/``E``) do both parse
    as floats within :data:`_FLOAT_TOL` — so integer problems stay byte-exact (``5`` != ``5.0000001``).
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
    """Exact comparison of a program's OUTPUT after stripping leading/trailing whitespace from both
    sides (legacy CodeContests). Distinct from :func:`src.rewards.matching.exact_match`, which
    normalizes a free-text answer."""
    return expected.strip() == actual.strip()


def as_verdict(comparator: Callable[[str, str], bool]) -> VerdictFn:
    """Adapt an output-only ``(expected, actual)`` comparator to the ``(input, expected, actual)`` signature."""

    def _verdict(_test_input: str, expected: str, actual: str) -> bool:
        return comparator(expected, actual)

    return _verdict


def _run_in_sandbox(sandbox: SandboxExecutor | SandboxSession, code: str, **kwargs) -> SandboxResult:
    """Execute ``code`` through ``sandbox`` (an executor or an open session), reporting an executor
    fault as ``SandboxResult(error=...)``.

    The one contract grading depends on: a run lost to the backend is an ``error`` result, never an
    exception. The remote backend already obeys it; a local one raises instead (no interpreter, missing
    ``bwrap``, fork exhaustion, ENOSPC writing the working dir). An escaping exception leaves
    ``submit_solution`` as an ordinary tool error, so the episode grades 0 with the infra-outage
    guard never firing — a host fault averaged into the GRPO baseline as a wrong program.
    """
    try:
        return sandbox.run(code, **kwargs)
    except Exception as exc:  # anything raised on the grading side is infra, not a verdict
        logger.warning("Sandbox run failed during grading; scoring the test as an infra error", exc_info=True)
        return SandboxResult(error=f"sandbox backend failure: {type(exc).__name__}: {exc}")


@contextmanager
def _grading_runner(
    sandbox: SandboxExecutor, code: str, *, language: str, timeout: float
) -> Iterator[Callable[[str], SandboxResult]]:
    """A ``run(stdin)`` for one grade: one session for the whole grade, so a compiled submission is
    built once and its binary reused across the tests, reset to its staged state after every run so no
    test sees files an earlier one produced. An executor without sessions (``NotImplementedError``)
    grades test-by-test through one-shot runs; a session that fails to open is an infra fault worth a
    log line, and the grade falls back the same way, each run then reporting its own fault as an
    error result."""

    def one_shot(stdin: str) -> SandboxResult:
        return _run_in_sandbox(sandbox, code, stdin=stdin, timeout=timeout, language=language)

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
        result = _run_in_sandbox(session, code, stdin=stdin, timeout=timeout, language=language)
        try:
            session.reset_to_staged()
        except Exception as exc:  # the next test would inherit this one's files: no credit for either
            logger.warning(
                "Sandbox session reset failed during grading; scoring the test as an infra error", exc_info=True
            )
            return SandboxResult(error=f"sandbox reset failure: {type(exc).__name__}: {exc}")
        return result

    try:
        yield in_session
    finally:
        session.close()


class CheckerInfraError(RuntimeError):
    """The checker run was lost to a grading-backend failure (transport error, backend down).

    Raised by :class:`CheckerVerdict` and consumed by :func:`run_solution_against_tests` into
    ``infra_errors`` — a plain ``False`` would score the outage as a wrong answer, feeding the whole
    GRPO group a zero grade as fake signal and hiding it from the ``_grading_infra_outage`` guard.
    """


class CheckerVerdict:
    """Special-judge grader: runs a problem's ``generated_checker`` (always Python) through a sandbox per test.

    Verdict is ``True`` iff the checker exits cleanly and prints ``1``; a checker crash, timeout, or
    ``0`` all reject. A sandbox *backend* failure raises :class:`CheckerInfraError` instead — the test
    was lost to infra, not judged. The timeout is an infra bound on trusted problem-setter code, not
    the solution's per-test limit.
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
                **dict(zip(CHECKER_FILES, (test_input, expected, actual), strict=True)),
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

    ``ran_ok`` = tests whose code ran to completion AND produced output (powers the "runnable" reward
    rung, so a no-output stub doesn't tie an honest attempt). ``infra_errors`` = tests lost to a
    grading-backend failure; ``infra_errors == total`` means the grade carries no signal.
    ``graded`` = tests actually judged (``< total`` once a budget stop or ``stop_on_first_failure``
    cuts the run short, while ``total`` stays the scoring denominator), and ``budget_hit`` says which
    of the two it was — partial grading is otherwise indistinguishable from a wrong solution.
    ``graded`` carries no default: every value is wrong for some grade (0 would claim a fully judged
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
    verdict_detail: str = VERDICT_DETAIL_FULL,
) -> GradeResult:
    """Run a solution against test cases through a :class:`SandboxExecutor` -> :class:`GradeResult`.

    Each test feeds ``input`` to stdin and compares stdout to expected ``output`` via ``verdict_fn``
    (default: trimmed exact match) in an independent sandbox run. Details list only non-passing tests,
    one entry per distinct verdict (tests failing the same way are folded into it), capped at
    ``_MAX_FAILURE_DETAILS`` distinct entries; ``verdict_detail`` decides whether a wrong answer shows
    the expected and produced output (``full``) or the verdict alone (``outcome``). A runtime error
    shows the tail of stderr, where a traceback names the exception.

    ``max_grading_seconds`` bounds one grade's total wall clock, since tests run sequentially and a
    several-hundred-test problem would otherwise stall the whole rollout round. It is checked between
    tests (hard bound: the budget plus one per-test timeout) and at least one test always runs. A
    budget stop keeps the FULL pool as the denominator, so a solution too slow to reach its remaining
    tests cannot outscore one that ran them all — size the budget to the pool.
    """
    if verdict_detail not in VERDICT_DETAILS:
        raise ValueError(f"verdict_detail must be one of {VERDICT_DETAILS}, got {verdict_detail!r}")
    if not test_cases:
        return GradeResult(0, 0, "No test cases provided.", 0, graded=0)

    if sandbox is None:
        sandbox = resolve_sandbox()
    if verdict_fn is None:
        verdict_fn = as_verdict(exact_output_match)

    passed = 0
    ran_ok = 0  # clean exit + produced output (PASS or FAIL), NOT crash/TLE/overflow/backend error
    infra_errors = 0  # tests lost to a backend/transport failure, not the program's fault
    total = len(test_cases)
    failures: list[_Failure] = []
    notes: list[str] = []
    suppressed = 0
    deadline = None if max_grading_seconds is None else time.monotonic() + max_grading_seconds
    graded = 0
    budget_hit = False

    def add_detail(i: int, body: str) -> None:
        """Book a non-passing test: onto the failure with the same verdict, else as a new one while
        under the cap, else counted as suppressed."""
        nonlocal suppressed
        for failure in failures:
            if failure.body == body:
                failure.tests.append(i)
                return
        if len(failures) < _MAX_FAILURE_DETAILS:
            failures.append(_Failure(body, [i]))
        else:
            suppressed += 1

    with _grading_runner(sandbox, code, language=language, timeout=timeout_per_test) as run_test:
        for i, tc in enumerate(test_cases, 1):
            if deadline is not None and graded and time.monotonic() >= deadline:
                budget_hit = True
                break
            test_input = tc.get("input", "")
            expected_output = tc.get("output", "")

            result = run_test(test_input)

            if result.compile_failed:
                # The source never built, so every test fails the same way: judged once, the whole
                # pool counted, with the compiler's diagnostics as the verdict.
                # The compiler names the first error first, so the head of its output is the excerpt.
                line = "COMPILATION ERROR (every test fails)"
                if result.stderr:
                    line += f"\n  {result.stderr.strip()[:_STDERR_EXCERPT_CHARS]}"
                notes.append(line)
                graded = total
                break

            test_passed = False
            if result.timed_out:
                add_detail(i, f"TIME LIMIT EXCEEDED ({timeout_per_test:g}s)")
            elif result.error:
                # Backend/transport failure (not the program's stderr); bucket as ERROR even with partial stdout.
                infra_errors += 1
                add_detail(i, f"ERROR -- {result.error}")
            elif result.returncode not in (0, None):
                # Non-zero exit is a Runtime Error on every judge, never a pass even if stdout matches.
                body = f"RUNTIME ERROR (exit {result.returncode})"
                if result.stderr:
                    body += f"\n  Stderr: {_stderr_tail(result.stderr)}"
                add_detail(i, body)
            elif len(result.stdout) > max_output_size:
                # Over-cap output is its own verdict: truncate-and-compare would grade a correct-but-long answer wrong.
                add_detail(i, f"OUTPUT LIMIT EXCEEDED ({len(result.stdout)} > {max_output_size} bytes)")
            else:
                actual_output = result.stdout
                try:
                    test_passed = verdict_fn(test_input, expected_output, actual_output)
                except CheckerInfraError as e:
                    # Verdict lost to infra: no ran_ok/pass credit, keeping an all-infra outage visible.
                    infra_errors += 1
                    add_detail(i, f"ERROR -- {e}")
                else:
                    if test_passed or actual_output.strip() or not expected_output.strip():
                        # Requiring output stops a no-output stub tying an honest attempt on this rung.
                        ran_ok += 1
                    if test_passed:
                        passed += 1
                    else:
                        body = "FAIL"
                        if verdict_detail == VERDICT_DETAIL_FULL:
                            body += (
                                f"\n  Expected: {expected_output.strip()[:_OUTPUT_EXCERPT_CHARS]}"
                                f"\n  Got:      {actual_output.strip()[:_OUTPUT_EXCERPT_CHARS]}"
                            )
                        if result.stderr:
                            body += f"\n  Stderr: {_stderr_tail(result.stderr)}"
                        add_detail(i, body)

            graded = i
            if stop_on_first_failure and not test_passed:
                notes.append(f"Stopped after first failing test ({total - i} not run).")
                break

    details = [failure.render() for failure in failures] + notes
    if suppressed:
        details.append(f"...and {suppressed} more non-passing tests (details omitted).")

    summary = f"Passed {passed}/{total} test cases."
    if budget_hit:
        # The graded prefix is reported for diagnosis, not for credit: the full pool stays the
        # scoring denominator.
        summary += f" (graded the first {graded} of {total}; {max_grading_seconds:g}s grading budget reached)"
    return GradeResult(passed, total, summary + "\n" + "\n".join(details), ran_ok, graded, infra_errors, budget_hit)


def select_verdict(checker: str | None, comparison: str, sandbox: SandboxExecutor) -> VerdictFn:
    """Pick the per-test verdict: a special-judge ``checker`` if given (always wins), else ``comparison``
    (``"tokens"`` = whitespace-token equality, ``"exact"`` = trimmed byte equality).

    The checker runs at the infra default timeout, never the solution's clamped per-test limit: a
    tight C++-tuned limit must TLE the solution, not the trusted judge grading it."""
    if checker:
        return CheckerVerdict(checker, sandbox, timeout=SANDBOX_DEFAULT_TIMEOUT)
    if comparison == "tokens":
        return as_verdict(compare_tokens)
    if comparison == "exact":
        return as_verdict(exact_output_match)
    raise ValueError(f"unknown output comparison {comparison!r} (expected 'tokens' or 'exact')")


@dataclass(frozen=True)
class GradingSpec:
    """The grading contract a run is scored under: everything that is constant across its problems.

    Built once by the environment and handed to every :func:`grade_solution` call, so the offline
    re-grader reproduces a run's verdicts by taking the same object rather than re-threading each
    knob and silently defaulting one it forgot. The two per-problem facts (``checker``,
    ``time_limit``) come from the problem payload and stay arguments.

    :meth:`to_meta` / :meth:`with_meta` carry that same contract across a trajectory dump, derived
    from the field list so a field added here reaches the re-grade without a second edit.
    """

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
    verdict_detail: str = VERDICT_DETAIL_FULL

    # The live executor: rebuilt from the run's env kwargs offline, never carried through a JSON dump.
    _META_EXCLUDED = frozenset({"sandbox"})

    def __post_init__(self) -> None:
        # ``replace`` re-runs this, so a restored meta block is held to the same contract.
        if not (isfinite(self.compiled_time_limit_scale) and self.compiled_time_limit_scale > 0):
            raise ValueError(
                f"compiled_time_limit_scale must be a finite number > 0, got {self.compiled_time_limit_scale}"
            )

    def to_meta(self) -> dict[str, Any]:
        """This contract as a JSON-able block for a trajectory meta line."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in self._META_EXCLUDED}

    def with_meta(self, meta: dict[str, Any], **overrides: Any) -> "GradingSpec":
        """This contract with a dumped :meth:`to_meta` block applied over it, then ``overrides``.

        Keys no field declares raise rather than being ignored, so a trajectory written under a
        retired spelling is refused instead of silently re-graded under today's defaults.
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
    language: str | None = None,
) -> GradeResult:
    """Grade ``code`` against ``tests`` -> :class:`GradeResult`; the single grading entry point.

    Selects the verdict via :func:`select_verdict` and runs every test at the problem's ``time_limit``,
    falling back to ``spec.default_timeout``: an interpreted language is floored at that default, a
    compiled one is multiplied by ``spec.compiled_time_limit_scale``, and either is then clamped to
    ``spec.max_time_limit``. ``spec.max_grading_seconds`` bounds the total sequential grading cost per
    submission (see :func:`run_solution_against_tests`). ``language`` is the submission's own language
    when the run lets the model choose per submission; unset, the contract's ``spec.language`` applies.
    """
    language = language or spec.language
    limit = time_limit or spec.default_timeout
    resolved = resolve_language(language)
    if resolved is not None and not resolved.is_compiled:
        # Floor an interpreted language's budget so a C++-tuned limit doesn't TLE a slower CPython solution.
        limit = max(limit, spec.default_timeout)
    elif resolved is not None:
        # Statement limits are calibrated for C++ on a dedicated judge; the scale pays for shared grading cores.
        limit *= spec.compiled_time_limit_scale
    return run_solution_against_tests(
        code,
        tests,
        timeout_per_test=min(limit, spec.max_time_limit),
        max_output_size=spec.max_output_size,
        sandbox=spec.sandbox,
        language=language,
        verdict_fn=select_verdict(checker, spec.comparison, spec.sandbox),
        stop_on_first_failure=spec.stop_on_first_failure,
        max_grading_seconds=spec.max_grading_seconds,
        verdict_detail=spec.verdict_detail,
    )
