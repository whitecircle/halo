"""The scratchpad reply a model reads for a :class:`SandboxResult`, and a REPL tool handler that runs code through a
backend: a structured result adapted to the string-in/string-out shape a tool returns."""

import signal

from src.environments.sandbox.base import SANDBOX_DEFAULT_TIMEOUT, SandboxExecutor, SandboxSession
from src.environments.sandbox.results import SandboxAgentFault, SandboxInfraError, SandboxResult

# Shared REPL observation for a run that produced no output, so every backend uses the same wording.
REPL_NO_OUTPUT_MESSAGE = "Code executed successfully (no output)"
# What the usual fatal signals mean for a contest program, so a crash reads as its cause, not a number.
_SIGNAL_CAUSES: dict[int, str] = {
    signal.SIGSEGV: "invalid memory access or stack overflow",
    signal.SIGABRT: "aborted: a failed assert, an uncaught exception, or an allocation past the memory limit",
    signal.SIGFPE: "arithmetic error, e.g. integer division by zero",
    signal.SIGBUS: "misaligned or out-of-range memory access",
    signal.SIGKILL: "killed from outside, e.g. by the out-of-memory killer",
}
# Characters of compiler diagnostics / runtime stderr a REPL reply carries: enough for the first
# errors of a compile or the frames of a traceback, bounded so the reply leaves room for stdout.
_REPL_STDERR_EXCERPT_CHARS = 2000


def repl_timeout_message(timeout: float) -> str:
    """The REPL observation for a run killed by its wall-clock ``timeout``."""
    return f"Error: execution exceeded {timeout:g}s timeout"


def signal_description(returncode: int | None) -> str | None:
    """``killed by SIGSEGV (<cause>)`` for a return code that reports a signal death (negative, as
    ``subprocess`` reports it); None for a normal exit."""
    if returncode is None or returncode >= 0:
        return None
    try:
        name = signal.Signals(-returncode).name
    except ValueError:
        name = f"signal {-returncode}"
    cause = _SIGNAL_CAUSES.get(-returncode)
    return f"killed by {name} ({cause})" if cause else f"killed by {name}"


def stderr_head(text: str, limit: int) -> str:
    """The start of a stream, at most ``limit`` characters: a compiler names its first error first."""
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "…"


def stderr_tail(text: str, limit: int) -> str:
    """The end of a stream, at most ``limit`` characters: a traceback names the exception on its last
    line, so a head excerpt of a long one shows the frames and drops the error."""
    text = text.strip()
    return text if len(text) <= limit else "…" + text[-limit:]


def format_sandbox_repl_output(result: SandboxResult, timeout: float) -> str:
    """The scratchpad reply the model reads for ``result``: a verdict on the program (timeout, non-zero exit, compile
    error) leads with its error and then the stdout, so a reply cut from the end keeps the error. A backend failure
    raises :class:`SandboxInfraError` and a sandbox the program broke :class:`SandboxAgentFault`: the tool layer tells
    them apart by type, never by text.
    """
    if result.agent_fault:
        raise SandboxAgentFault(result.agent_fault)
    if result.timed_out:
        return repl_timeout_message(timeout)
    # Raise even if a partial response carried stdout/stderr: partial output is not a verdict.
    if result.error:
        raise SandboxInfraError(f"sandbox backend failure: {result.error}")
    if result.compile_failed:
        diagnostics = stderr_head(result.stderr, _REPL_STDERR_EXCERPT_CHARS)
        return "Error: compilation failed" + (f"\n{diagnostics}" if diagnostics else "")

    stdout = result.stdout.rstrip("\n")
    if result.returncode not in (0, None):
        stderr = stderr_tail(result.stderr, _REPL_STDERR_EXCERPT_CHARS)
        killed = signal_description(result.returncode)
        if killed:
            error = f"{killed}\n{stderr}" if stderr else killed
        else:
            error = stderr or f"process exited with code {result.returncode}"
        return f"Error: {error}\nOutput:\n{stdout}" if stdout else f"Error: {error}"
    return stdout if stdout else REPL_NO_OUTPUT_MESSAGE


def run_code_via_sandbox(
    code: str,
    runner: SandboxExecutor | SandboxSession,
    timeout: float = SANDBOX_DEFAULT_TIMEOUT,
    language: str = "python",
) -> str:
    """REPL handler running ``code`` on ``runner`` (an executor one-shot, or a live session) in a real interpreter or
    binary, so only for a sandbox that isolates it; faults raise as in :func:`format_sandbox_repl_output`. The program
    gets an empty stdin, since the tools here declare none: one reading input sees end-of-file instead of blocking.
    """
    result = runner.run(code, timeout=timeout, language=language)
    return format_sandbox_repl_output(result, timeout)
