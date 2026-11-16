"""Baseten Sandboxes backend (https://docs.baseten.co/sandboxes/overview): every session runs in its own Baseten
microVM, created on first use and deleted on close. ``BASETEN_API_KEY`` mints the short-lived sandbox access token
every sandbox call carries."""

import contextlib
import math
import os
import shlex
import threading
import time
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from halo.environments.sandbox.base import SANDBOX_DEFAULT_TIMEOUT, SandboxExecutor, SandboxSession
from halo.environments.sandbox.languages import INTERPRETER_PLACEHOLDER, require_language, utf8_encodable
from halo.environments.sandbox.results import (
    COMMAND_NOT_FOUND_RETURNCODE,
    SandboxResult,
    compile_failure_verdict,
    compile_limit_verdict,
)

BASETEN_API_URL = "https://api.baseten.co/v1"
# An organization runs at most 10 sandboxes at once (https://docs.baseten.co/reference/sandbox-api/overview).
BASETEN_SANDBOX_LIMIT = 10
# Tokens expire 2 h after issue and cannot be renewed; mint a fresh one well before that.
_TOKEN_REFRESH_SECONDS = 90 * 60
_COMPILE_TIMEOUT_SECONDS = 30.0
_WORKDIR = "/tmp/halo/work"
_STDIN_PATH = "/tmp/halo/stdin"
# Entries present when the last program launched; reset_to_prepared removes everything else.
_PREPARED_PATH = "/tmp/halo/prepared"


class BasetenSandbox(SandboxExecutor):
    """``max_concurrency`` caps live sandboxes in this process (a session holds one until closed); a session past it
    waits. 429, 5xx and connection errors retry with exponential backoff honoring ``Retry-After``."""

    isolated = True

    def __init__(
        self,
        *,
        api_key_env: str = "BASETEN_API_KEY",
        api_url: str = BASETEN_API_URL,
        image: str = "baseten/base-image:latest",
        cpu: int = 2,
        idle_ttl: str = "30m",
        interpreter: str = "python3",
        request_timeout: float = 60.0,
        request_overhead: float = 30.0,
        create_timeout: float = 600.0,
        max_retries: int = 4,
        backoff_factor: float = 1.0,
        max_concurrency: int = BASETEN_SANDBOX_LIMIT,
        session: requests.Session | None = None,
    ):
        if not os.environ.get(api_key_env):
            raise ValueError(f"the baseten sandbox backend needs the {api_key_env} environment variable")
        if not 1 <= max_concurrency <= BASETEN_SANDBOX_LIMIT:
            raise ValueError(
                f"max_concurrency={max_concurrency} must be within 1..{BASETEN_SANDBOX_LIMIT}: each live session holds "
                f"its own sandbox and an organization runs at most {BASETEN_SANDBOX_LIMIT} at once"
            )
        self.api_key_env, self.api_url = api_key_env, api_url.rstrip("/")
        self.image, self.cpu, self.idle_ttl, self.interpreter = image, cpu, idle_ttl, interpreter
        self.request_timeout, self.request_overhead, self.create_timeout = (
            request_timeout,
            request_overhead,
            create_timeout,
        )
        self.slots = threading.BoundedSemaphore(max_concurrency)
        # A read timeout is a program running long or a service not answering; re-sending would run it again.
        retry = Retry(
            total=max_retries,
            read=False,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=None,
            raise_on_status=False,
        )
        self._http = session or requests.Session()
        for prefix in ("https://", "http://"):
            self._http.mount(prefix, HTTPAdapter(max_retries=retry))
        self._token_lock = threading.Lock()
        self._token: str | None = None
        self._token_refresh_at = 0.0

    def open_session(self) -> "BasetenSession":
        return BasetenSession(self)

    def _access_token(self) -> str:
        with self._token_lock:
            if self._token is None or time.monotonic() >= self._token_refresh_at:
                key = os.environ.get(self.api_key_env, "")  # read per mint, never stored or logged
                resp = self._http.post(
                    f"{self.api_url}/token",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"scopes": ["sandboxes"]},
                    timeout=self.request_timeout,
                )
                resp.raise_for_status()
                self._token = resp.json()["token"]
                self._token_refresh_at = time.monotonic() + _TOKEN_REFRESH_SECONDS
            return self._token

    def request(self, method: str, url: str, *, timeout: float, **kwargs) -> requests.Response:
        resp = self._http.request(
            method, url, headers={"Authorization": f"Bearer {self._access_token()}"}, timeout=timeout, **kwargs
        )
        if resp.status_code != 404:
            resp.raise_for_status()
        return resp


class BasetenSession(SandboxSession):
    """One sandbox, created on first use, whose working directory persists across :meth:`run` calls."""

    def __init__(self, executor: BasetenSandbox):
        self._executor = executor
        self._name: str | None = None
        self._url: str | None = None
        # (language, code, files) of the build in the working directory; a write invalidates it.
        self._prepared: tuple | None = None

    def _sandbox(self) -> str:
        if self._url is None:
            ex = self._executor
            ex.slots.acquire()
            body = {
                "image": ex.image,
                "cpu": ex.cpu,
                "labels": {"app": "halo"},
                # Deletes a sandbox whose session was never closed.
                "lifecycle": {"expiration_policies": [{"type": "TTL_IDLE", "value": ex.idle_ttl, "action": "DELETE"}]},
            }
            try:
                data = ex.request(
                    "POST", f"{ex.api_url}/sandboxes/instances", json=body, timeout=ex.create_timeout
                ).json()
                if data.get("status") != "DEPLOYED":
                    raise RuntimeError(f"baseten sandbox {data.get('name')!r} is {data.get('status')}, not DEPLOYED")
            except BaseException:
                ex.slots.release()
                raise
            self._name, self._url = data["name"], data["url"].rstrip("/")
            self._shell(f"mkdir -p {_WORKDIR}", "/")
        return self._url

    def _process(self, command: str, workdir: str, timeout: float) -> dict:
        ex = self._executor
        payload = {
            "command": command,
            "workingDir": workdir,
            "timeout": max(1, math.ceil(timeout)),
            "waitForCompletion": True,
        }
        return ex.request(
            "POST", f"{self._sandbox()}/process", json=payload, timeout=timeout + ex.request_overhead
        ).json()

    def _shell(self, script: str, workdir: str = _WORKDIR, timeout: float | None = None) -> dict:
        return self._process(f"sh -c {shlex.quote(script)}", workdir, timeout or self._executor.request_timeout)

    def _put(self, path: str, content: str) -> None:
        # An absolute path takes an encoded leading slash (sandbox API filesystem reference).
        url = f"{self._sandbox()}/filesystem/%2F{quote(path.lstrip('/'))}"
        self._executor.request(
            "PUT", url, json={"content": utf8_encodable(content)}, timeout=self._executor.request_timeout
        )

    def run(
        self,
        code: str,
        *,
        stdin: str = "",
        timeout: float = SANDBOX_DEFAULT_TIMEOUT,
        language: str = "python",
        files: dict[str, str] | None = None,
    ) -> SandboxResult:
        try:
            spec = require_language(language)
            for name in files or {}:
                self.reject_unsafe_path(name)
        except ValueError as exc:
            return SandboxResult(error=str(exc))
        key = (spec.name, code, tuple(sorted((files or {}).items())))
        try:
            if self._prepared != key:
                for name, content in {**(files or {}), spec.source_name: code}.items():
                    self.write_file(name, content)
                if spec.is_compiled:
                    built = self._shell(shlex.join(spec.compile_argv), timeout=_COMPILE_TIMEOUT_SECONDS)
                    if built.get("status") == "killed":
                        return compile_limit_verdict(f"compilation exceeded {_COMPILE_TIMEOUT_SECONDS:g}s")
                    if built.get("exitCode") == COMMAND_NOT_FOUND_RETURNCODE:
                        return SandboxResult(error=f"compiler not found: {spec.compile_argv[0]!r}")
                    if built.get("exitCode") != 0:
                        return compile_failure_verdict(
                            built.get("stderr") or "", built.get("stdout") or "", built.get("exitCode")
                        )
                self._prepared = key
            self._put(_STDIN_PATH, stdin)
            argv = [self._executor.interpreter if tok == INTERPRETER_PLACEHOLDER else tok for tok in spec.run_argv]
            out = self._shell(
                f"find . -mindepth 1 > {_PREPARED_PATH} && exec {shlex.join(argv)} < {_STDIN_PATH}", timeout=timeout
            )
        except requests.Timeout:
            # The service kills the program at ``timeout``; a request outliving that is the service not answering.
            return SandboxResult(error="baseten sandbox request timed out")
        except requests.HTTPError as exc:
            return SandboxResult(error=f"baseten sandbox: HTTP {exc.response.status_code}: {exc.response.text[:200]}")
        except (requests.RequestException, RuntimeError, ValueError, KeyError) as exc:
            return SandboxResult(error=f"baseten sandbox error: {exc}")
        killed = out.get("status") == "killed"
        return SandboxResult(
            stdout=out.get("stdout") or "",
            stderr=out.get("stderr") or "",
            returncode=None if killed else out.get("exitCode"),
            timed_out=killed,
        )

    def reset_to_prepared(self) -> None:
        if self._url is not None:
            self._shell(
                f"[ -f {_PREPARED_PATH} ] && find . -mindepth 1 -depth | grep -vxFf {_PREPARED_PATH} "
                "| xargs -r -d '\\n' rm -rf; true"
            )

    def write_file(self, path: str, content: str) -> None:
        self.reject_unsafe_path(path)
        self._prepared = None
        if "/" in path:
            self._shell(f"mkdir -p {shlex.quote(os.path.dirname(path))}")
        self._put(f"{_WORKDIR}/{path}", content)

    def read_file(self, path: str) -> str | None:
        self.reject_unsafe_path(path)
        if self._url is None:
            return None
        ex = self._executor
        url = f"{self._url}/filesystem/%2F{quote(f'{_WORKDIR}/{path}'.lstrip('/'))}"
        resp = ex.request("GET", url, timeout=ex.request_timeout)
        if resp.status_code == 404 or resp.headers.get("Content-Type", "").startswith("application/json"):
            return None  # missing, or a directory listing
        return resp.content.decode("utf-8", errors="replace")

    def list_files(self) -> list[str]:
        if self._url is None:
            return []
        out = self._shell("find . -type f -printf '%P\\n'")
        return sorted(line for line in (out.get("stdout") or "").splitlines() if line)

    def close(self) -> None:
        if self._url is None:
            return
        ex = self._executor
        self._url = None
        try:
            with contextlib.suppress(requests.RequestException):  # the idle TTL deletes it otherwise
                ex.request("DELETE", f"{ex.api_url}/sandboxes/instances/{self._name}", timeout=ex.request_timeout)
        finally:
            ex.slots.release()
