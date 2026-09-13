"""Remote SandboxFusion-compatible backend: HTTP client for the stateless ``/run_code`` API.

The endpoint is stateless, so :class:`RemoteSession` carries the working set client-side: written
files are accumulated and resent each :meth:`run`, and a compiled language is rebuilt by the service
on every request. Files *produced* on the service aren't read back.
See https://github.com/bytedance/SandboxFusion.
"""

import math

import requests

from src.environments.sandbox.base import (
    SANDBOX_DEFAULT_TIMEOUT,
    SandboxExecutor,
    SandboxResult,
    SandboxSession,
)

# HTTP budget on top of the program's own timeout: the service still has to queue, provision and
# tear down the run, so the client must not give up before the server's own deadline.
_REQUEST_OVERHEAD_SECONDS = 30.0

# SandboxFusion ``CommandRunStatus`` values that mean the step ran to completion (lower-cased).
_STEP_FINISHED_STATUSES = ("finished", "success", "")
# The shell's exit code for a command it could not exec: a missing compiler, not a source verdict.
_COMMAND_NOT_FOUND = 127


def _command_result(value: object) -> dict[str, object]:
    """A SandboxFusion ``compile_result`` / ``run_result`` block; ``{}`` when absent or malformed."""
    return value if isinstance(value, dict) else {}


def _return_code(value: object) -> int | None:
    """``return_code`` may arrive as an int, a numeric string, or be absent; normalize to int/None."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _is_time_limit(status: str) -> bool:
    return "timelimit" in status or "timeout" in status


class RemoteSandbox(SandboxExecutor):
    """HTTP client for a SandboxFusion-compatible ``/run_code`` service.

    The ``/run_code`` suffix is appended to ``url`` if absent; inject ``session`` to share a
    connection pool across rollout workers (or a fake one in tests).
    """

    def __init__(self, url: str, *, session: requests.Session | None = None):
        base = url.rstrip("/")
        self.endpoint = base if base.endswith("run_code") else base + "/run_code"
        self._session = session or requests.Session()

    def open_session(self) -> "RemoteSession":
        """Open a session that accumulates files client-side and resends them each run."""
        return RemoteSession(self)

    def run(
        self,
        code: str,
        *,
        stdin: str = "",
        timeout: float = SANDBOX_DEFAULT_TIMEOUT,
        language: str = "python",
        files: dict[str, str] | None = None,
    ) -> SandboxResult:
        payload: dict[str, object] = {
            "code": code,
            "language": language,
            "run_timeout": int(math.ceil(timeout)),
        }
        if stdin:
            payload["stdin"] = stdin
        if files:
            payload["files"] = files

        try:
            resp = self._session.post(self.endpoint, json=payload, timeout=timeout + _REQUEST_OVERHEAD_SECONDS)
            resp.raise_for_status()
            data = resp.json()
        except requests.Timeout:
            return SandboxResult(timed_out=True, error="remote sandbox request timed out")
        except (requests.RequestException, ValueError) as exc:
            return SandboxResult(error=f"remote sandbox error: {exc}")
        return self._parse(data)

    @staticmethod
    def _parse(data: dict[str, object]) -> SandboxResult:
        """Map a SandboxFusion response into a :class:`SandboxResult` (tolerant of partial bodies).

        A ``compile_result`` the compiler rejected is the program's verdict (``compile_failed``); a
        compile time limit is a backend/limit failure (``error``) — the local backend's split.
        """
        compile_step = _command_result(data.get("compile_result"))
        compile_status = str(compile_step.get("status", "")).lower()
        if _is_time_limit(compile_status):
            message = "remote compilation exceeded the service's compile time limit"
            return SandboxResult(stderr=message, error=message)
        compile_rc = _return_code(compile_step.get("return_code"))
        if compile_step and (compile_status not in _STEP_FINISHED_STATUSES or compile_rc == _COMMAND_NOT_FOUND):
            # The compiler step did not run to completion (or the compiler is absent): the service's
            # fault, never a verdict on the source.
            diagnostics = str(compile_step.get("stderr") or compile_step.get("stdout") or compile_status)
            return SandboxResult(
                stderr=diagnostics.strip(), error=f"remote compile step failed: {diagnostics.strip()}"
            )
        if compile_step and compile_rc not in (0, None):
            diagnostics = str(compile_step.get("stderr") or compile_step.get("stdout") or "compilation failed")
            return SandboxResult(stderr=diagnostics.strip(), returncode=compile_rc, compile_failed=True)

        run = _command_result(data.get("run_result"))
        timed_out = _is_time_limit(str(run.get("status", "")).lower())

        status = str(data.get("status", "")).lower()
        error: str | None = None
        if status not in ("success", "") and not timed_out:
            error = str(data.get("message") or data.get("status"))
        elif not run and not timed_out:
            # A body with no run block carries no program output: reporting it as an empty clean run
            # would pass a test whose expected output is empty and read as a no-output REPL success.
            error = "remote sandbox returned no run result"

        return SandboxResult(
            stdout=str(run.get("stdout", "") or ""),
            stderr=str(run.get("stderr", "") or ""),
            returncode=_return_code(run.get("return_code")),
            timed_out=timed_out,
            error=error,
        )


class RemoteSession(SandboxSession):
    """Stateful view over a stateless ``/run_code`` service: tracked files are merged into every
    :meth:`run` request (per-call ``files`` win on key collisions).
    """

    def __init__(self, executor: RemoteSandbox):
        self._executor = executor
        self._files: dict[str, str] = {}

    def run(
        self,
        code: str,
        *,
        stdin: str = "",
        timeout: float = SANDBOX_DEFAULT_TIMEOUT,
        language: str = "python",
        files: dict[str, str] | None = None,
    ) -> SandboxResult:
        # `or None` is intentional: an empty merged dict must forward as files=None, not {}
        merged = {**self._files, **(files or {})} or None  # noqa: SIM222
        return self._executor.run(code, stdin=stdin, timeout=timeout, language=language, files=merged)

    def write_file(self, path: str, content: str) -> None:
        self._files[path] = content

    def read_file(self, path: str) -> str | None:
        return self._files.get(path)

    def list_files(self) -> list[str]:
        return sorted(self._files)

    def close(self) -> None:
        self._files.clear()
