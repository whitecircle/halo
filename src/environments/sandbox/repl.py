"""REPL-facing helpers: render a :class:`SandboxResult` as a string and run code through a backend.

Adapts the structured result to the string-in/string-out shape a tool handler returns to the model.
"""

from src.environments.sandbox.base import (
    REPL_NO_OUTPUT_MESSAGE,
    SANDBOX_DEFAULT_TIMEOUT,
    SandboxAgentFault,
    SandboxExecutor,
    SandboxInfraError,
    SandboxResult,
    SandboxSession,
    repl_timeout_message,
    signal_description,
    stderr_head,
    stderr_tail,
)

# Characters of compiler diagnostics / runtime stderr a REPL reply carries: enough for the first
# errors of a compile or the frames of a traceback, bounded so the reply leaves room for stdout.
REPL_STDERR_EXCERPT_CHARS = 2000


def format_sandbox_repl_output(result: SandboxResult, timeout: float) -> str:
    """Render a :class:`SandboxResult` as the scratchpad reply the model reads.

    Program-level outcomes (timeout, non-zero exit, compile error) are verdicts on the submitted code
    and render as strings. A failure leads with its error — a compile error with the compiler's first
    diagnostics, a crash with the signal and the tail of stderr — and the program's stdout follows,
    so an observation cut from the end keeps the error. A backend/transport failure (``result.error``)
    is NOT a verdict: it raises :class:`SandboxInfraError`, and a sandbox the program broke
    (``result.agent_fault``) raises :class:`SandboxAgentFault`; the tool layer tells the two apart by
    type, never by the text.
    """
    if result.agent_fault:
        raise SandboxAgentFault(result.agent_fault)
    if result.timed_out:
        return repl_timeout_message(timeout)
    # Raise even if a partial response carried stdout/stderr: partial output is not a verdict.
    if result.error:
        raise SandboxInfraError(f"sandbox backend failure: {result.error}")
    if result.compile_failed:
        diagnostics = stderr_head(result.stderr, REPL_STDERR_EXCERPT_CHARS)
        return "Error: compilation failed" + (f"\n{diagnostics}" if diagnostics else "")

    stdout = result.stdout.rstrip("\n")
    if result.returncode not in (0, None):
        stderr = stderr_tail(result.stderr, REPL_STDERR_EXCERPT_CHARS)
        killed = signal_description(result.returncode)
        if killed:
            error = f"{killed}\n{stderr}" if stderr else killed
        else:
            error = stderr or f"process exited with code {result.returncode}"
        return f"Error: {error}\nOutput:\n{stdout}" if stdout else f"Error: {error}"
    return stdout if stdout else REPL_NO_OUTPUT_MESSAGE


def run_code_via_sandbox(
    code: str,
    sandbox: SandboxExecutor | None,
    timeout: float = SANDBOX_DEFAULT_TIMEOUT,
    language: str = "python",
    session: SandboxSession | None = None,
    stdin: str = "",
) -> str:
    """REPL handler that executes ``code`` through a :class:`SandboxExecutor` (or a live session).

    Runs in a real interpreter / compiled binary (imports + stdlib), so only appropriate when the
    sandbox provides isolation. Pass a ``session`` for a persistent working dir; omit for one-shot.
    Raises :class:`SandboxInfraError` when the backend itself failed and :class:`SandboxAgentFault`
    when the program broke its own sandbox (see :func:`format_sandbox_repl_output`).

    ``stdin`` is what the program reads; a tool whose schema declares no stdin leaves it empty, so a
    program reading input there sees end-of-file instead of blocking on input nothing can supply.
    """
    runner = session if session is not None else sandbox
    result = runner.run(code, stdin=stdin, timeout=timeout, language=language)
    return format_sandbox_repl_output(result, timeout)
