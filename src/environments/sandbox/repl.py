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
)


def format_sandbox_repl_output(result: SandboxResult, timeout: float) -> str:
    """Render a :class:`SandboxResult` as a REPL-style string matching the in-process sandbox.

    Program-level outcomes (timeout, non-zero exit, compile error) are verdicts on the submitted code
    and render as strings. A backend/transport failure (``result.error``) instead raises
    :class:`SandboxInfraError`, and a sandbox the program broke (``result.agent_fault``) raises
    :class:`SandboxAgentFault`: the tool layer tells the two apart by type, never by the text.
    """
    if result.agent_fault:
        raise SandboxAgentFault(result.agent_fault)
    if result.timed_out:
        return repl_timeout_message(timeout)
    # Raise even if a partial response carried stdout/stderr: partial output is not a verdict.
    if result.error:
        raise SandboxInfraError(f"sandbox backend failure: {result.error}")
    if result.compile_failed:
        detail = result.stderr.strip()
        return f"Error: {detail.splitlines()[-1]}" if detail else "Error: compilation failed"

    stdout = result.stdout.rstrip("\n")
    if result.returncode not in (0, None):
        detail = result.stderr.strip()
        tail = detail.splitlines()[-1] if detail else f"process exited with code {result.returncode}"
        return f"{stdout}\nError: {tail}" if stdout else f"Error: {tail}"
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
