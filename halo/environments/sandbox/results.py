"""What one sandbox run comes to: its result, the faults that end a tool call short of a verdict on the program,
and the verdicts a build reaches."""

import logging
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# A command the shell could not exec: on a build step, a missing compiler (a backend failure), not a verdict.
COMMAND_NOT_FOUND_RETURNCODE = 127


class SandboxFault(RuntimeError):
    """A tool call ended on its sandbox rather than on a verdict about the program; the protocols book it by class."""


class SandboxInfraError(SandboxFault):
    """Lost to a backend or transport failure: the call is booked unpriced and the episode leaves the GRPO group
    baseline instead of pricing the fault as the policy's."""


class SandboxAgentFault(SandboxFault):
    """The program put a link or file in its working directory's place (only ``local``'s unconfined program can). The
    policy's fault: an infra error would let an episode void itself out of the baseline, so grading judges it a runtime
    error and the episode ends uncompleted, inside the baseline."""


@dataclass
class SandboxResult:
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
        return (
            self.returncode == 0
            and not self.timed_out
            and not self.compile_failed
            and self.error is None
            and self.agent_fault is None
        )


def compile_limit_verdict(message: str) -> SandboxResult:
    """The source's verdict (an ``#include`` bomb, a template blow-up), never a backend failure, which would let a
    program void its own episode."""
    return SandboxResult(stderr=message, compile_failed=True)


def compile_failure_verdict(stderr: str, stdout: str, returncode: int) -> SandboxResult:
    """Diagnostics read off stderr (gcc/g++), else stdout for a toolchain that reports there."""
    diagnostics = stderr.strip() or stdout.strip() or "compilation failed"
    return SandboxResult(stderr=diagnostics, returncode=returncode, compile_failed=True)


def run_reporting_faults(run: Callable[..., SandboxResult], code: str, **kwargs) -> SandboxResult:
    """An exception becomes ``SandboxResult(error=...)``: an escaping one would grade the episode 0 as a wrong program
    with the infra-outage guard never firing."""
    try:
        return run(code, **kwargs)
    except Exception as exc:  # anything raised on the grading side is infra, not a verdict
        # Per test at debug: the grade logs one line counting its infra errors by text.
        logger.debug("Sandbox run failed during grading; scoring the test as an infra error", exc_info=True)
        return SandboxResult(error=f"sandbox backend failure: {type(exc).__name__}: {exc}")
