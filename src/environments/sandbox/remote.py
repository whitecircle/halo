"""Remote SandboxFusion-compatible backend: HTTP client for the stateless ``/run_code`` API.

The endpoint is stateless, so :class:`RemoteSession` carries the working set client-side: written
files are accumulated and resent each :meth:`run`. Files *produced* on the service aren't read back.
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
        """Map a SandboxFusion response into a :class:`SandboxResult` (tolerant of partial bodies)."""
        run = data.get("run_result") or {}
        if not isinstance(run, dict):
            run = {}
        run_status = str(run.get("status", "")).lower()
        timed_out = "timelimit" in run_status or "timeout" in run_status

        status = str(data.get("status", "")).lower()
        error: str | None = None
        if status not in ("success", "") and not timed_out:
            error = str(data.get("message") or data.get("status"))

        # ``return_code`` may arrive as an int, a numeric string, or be absent; normalize to int/None.
        returncode = run.get("return_code")
        if isinstance(returncode, str):
            try:
                returncode = int(returncode)
            except ValueError:
                returncode = None

        return SandboxResult(
            stdout=str(run.get("stdout", "") or ""),
            stderr=str(run.get("stderr", "") or ""),
            returncode=returncode,
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
