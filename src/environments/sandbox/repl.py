"""REPL-facing helpers: render a :class:`SandboxResult` as a string and run code through a backend.

Adapts the structured result to the string-in/string-out shape a tool handler returns to the model.
"""

from src.environments.sandbox.base import (
    REPL_NO_OUTPUT_MESSAGE,
    SANDBOX_DEFAULT_TIMEOUT,
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
    :class:`SandboxInfraError`, so the tool layer records a failed call rather than a string the
    protocol would score as successful.
    """
    if result.timed_out:
        return repl_timeout_message(timeout)
    # Raise even if a partial response carried stdout/stderr: partial output is not a verdict.
    if result.error:
        raise SandboxInfraError(f"sandbox backend failure: {result.error}")

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
) -> str:
    """REPL handler that executes ``code`` through a :class:`SandboxExecutor` (or a live session).

    Runs in a real interpreter / compiled binary (imports + stdlib), so only appropriate when the
    sandbox provides isolation. Pass a ``session`` for a persistent working dir; omit for one-shot.
    Raises :class:`SandboxInfraError` when the backend itself failed (see
    :func:`format_sandbox_repl_output`).

    No stdin: the REPL tool schemas declare none, so a program reading stdin here would block on an
    input the model has no way to supply. Stdin-fed grading goes through the executor directly.
    """
    runner = session if session is not None else sandbox
    result = runner.run(code, timeout=timeout, language=language)
    return format_sandbox_repl_output(result, timeout)
