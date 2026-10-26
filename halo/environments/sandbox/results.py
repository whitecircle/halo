"""What one sandbox run comes to: its result, the faults that end a tool call short of a verdict on the program,
and the verdicts a build reaches."""

import logging
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# The shell's exit code for a command it could not exec: on a build step, a missing compiler (a backend
# failure), never a verdict on the source.
COMMAND_NOT_FOUND_RETURNCODE = 127


class SandboxFault(RuntimeError):
    """A tool call ended on its sandbox rather than on a verdict about the program: the protocols catch
    it ahead of any other exception and book it by its class."""


class SandboxInfraError(SandboxFault):
    """A sandbox run was lost to a backend/transport failure (remote HTTP error, missing compiler), not a verdict on
    the program: raised by a backend or the REPL layer so the outage reaches the tool layer by type, where the call is
    booked unpriced and the episode leaves the GRPO group baseline instead of pricing the fault as the policy's.
    """


class SandboxAgentFault(SandboxFault):
    """The program broke its own sandbox beyond safe repair: it put a link or a file in its working directory's place
    (only the ``local`` backend's unconfined program can). The fault is the policy's, never :class:`SandboxInfraError`,
    which would let an episode void itself out of the baseline: grading judges it a runtime error, and the episode ends
    uncompleted, inside the baseline.
    """


@dataclass
class SandboxResult:
    """Outcome of one program execution, uniform across backends."""

    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    timed_out: bool = False
    # The program's source never built: the compiler rejected it (``returncode`` is the compiler's) or
    # the build ran past the compile limit. A missing compiler is a backend failure and sets ``error``.
    compile_failed: bool = False
    # Backend/transport failure, distinct from the program's own non-zero exit or a compile error.
    error: str | None = None
    # What the program did to its own sandbox (:class:`SandboxAgentFault`); set with a non-zero
    # ``returncode`` so grading reads it as a runtime error, never as ``error``.
    agent_fault: str | None = None

    @property
    def ok(self) -> bool:
        """True only when the program built and ran to completion with a zero exit code."""
        return (
            self.returncode == 0
            and not self.timed_out
            and not self.compile_failed
            and self.error is None
            and self.agent_fault is None
        )


def compile_limit_verdict(message: str) -> SandboxResult:
    """A build killed at its compile limit: the source's verdict (an ``#include`` bomb, a template
    blow-up), judged like a compiler rejection and never as a backend failure, which would let a
    program void its own episode."""
    return SandboxResult(stderr=message, compile_failed=True)


def compile_failure_verdict(stderr: str, stdout: str, returncode: int) -> SandboxResult:
    """A build the compiler rejected: the source's verdict, its diagnostics read off stderr (gcc/g++), else
    off stdout for a toolchain that reports there."""
    diagnostics = stderr.strip() or stdout.strip() or "compilation failed"
    return SandboxResult(stderr=diagnostics, returncode=returncode, compile_failed=True)


def run_reporting_faults(run: Callable[..., SandboxResult], code: str, **kwargs) -> SandboxResult:
    """``run(code, **kwargs)`` (an executor's or an open session's), an exception it raises reported as
    ``SandboxResult(error=...)``: grading depends on a lost run being an ``error`` result, since an escaping exception
    would grade the episode 0 as a wrong program with the infra-outage guard never firing.
    """
    try:
        return run(code, **kwargs)
    except Exception as exc:  # anything raised on the grading side is infra, not a verdict
        # Per test at debug: the grade logs one line counting its infra errors by text.
        logger.debug("Sandbox run failed during grading; scoring the test as an infra error", exc_info=True)
        return SandboxResult(error=f"sandbox backend failure: {type(exc).__name__}: {exc}")
