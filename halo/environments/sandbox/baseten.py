"""Baseten Sandboxes backend (https://docs.baseten.co/sandboxes/overview): every session runs in its own Baseten
microVM, created on first use and deleted on close, and every call is one run of ``program_runner`` in it.
``BASETEN_API_KEY`` mints the short-lived sandbox access token every sandbox call carries."""

import contextlib
import json
import math
import os
import threading
import time
from typing import Any
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from halo.environments.sandbox.base import SandboxExecutor
from halo.environments.sandbox.local import (
    SANDBOX_DEFAULT_COMPILE_MEMORY_MB,
    SANDBOX_DEFAULT_COMPILE_TIMEOUT,
    SANDBOX_DEFAULT_MEMORY_MB,
)
from halo.environments.sandbox.results import SandboxInfraError
from halo.environments.sandbox.runner_session import (
    RUNNER_SOURCE,
    SESSION_UIDS,
    ProgramLimits,
    RunnerSession,
    parse_runner_reply,
)

BASETEN_API_URL = "https://api.baseten.co/v1"
# An organization runs at most 10 sandboxes at once (https://docs.baseten.co/reference/sandbox-api/overview).
BASETEN_SANDBOX_LIMIT = 10
# Tokens expire 2 h after issue and cannot be renewed; mint a fresh one well before that.
TOKEN_REFRESH_SECONDS = 90 * 60
# In the sandbox: root's files in /tmp, which a program (another user) can read but not replace.
RUNNER_PATH = "/tmp/halo_program_runner.py"
REQUEST_PATH = "/tmp/halo_runner_request.json"
SESSION_DIR = "/halo/session"


class SandboxGone(SandboxInfraError):
    """The sandbox no longer exists: its idle expiration or a delete removed it."""


class BasetenApi:
    """HTTP to the Baseten management API and to a sandbox's runtime API, as the access token's bearer. 429, 5xx and
    connection errors retry with exponential backoff honoring ``Retry-After``; a read timeout never does, since
    sending the request again would run its program again. Every failure raises :class:`SandboxInfraError`."""

    def __init__(
        self,
        *,
        api_key_env: str,
        api_url: str,
        request_timeout: float,
        max_retries: int,
        backoff_factor: float,
        session: requests.Session | None = None,
    ):
        self.api_key_env = api_key_env
        self.api_url = api_url.rstrip("/")
        self.request_timeout = request_timeout
        retry = Retry(
            total=max_retries,
            read=False,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=None,
            raise_on_status=False,
        )
        self.http = session or requests.Session()
        for scheme in ("https://", "http://"):
            self.http.mount(scheme, HTTPAdapter(max_retries=retry))
        self._token_lock = threading.Lock()
        self._token: str | None = None
        self._token_refresh_at = 0.0

    def send(
        self, method: str, url: str, *, timeout: float | None = None, missing_ok: bool = False, **kwargs
    ) -> requests.Response:
        """The response to an authenticated request. A 404 raises :class:`SandboxGone` unless ``missing_ok``."""
        try:
            headers = {"Authorization": f"Bearer {self._access_token()}"}
            response = self.http.request(
                method, url, headers=headers, timeout=timeout or self.request_timeout, **kwargs
            )
            if response.status_code == 404:
                if missing_ok:
                    return response
                raise SandboxGone(f"baseten sandbox: HTTP 404 from {method} {url}")
            response.raise_for_status()
            return response
        except requests.Timeout as exc:
            raise SandboxInfraError("baseten sandbox request timed out") from exc
        except requests.HTTPError as exc:
            detail = exc.response.text[:200]
            raise SandboxInfraError(f"baseten sandbox: HTTP {exc.response.status_code}: {detail}") from exc
        except requests.RequestException as exc:
            raise SandboxInfraError(f"baseten sandbox error: {exc}") from exc

    def put_file(self, sandbox_url: str, path: str, content: str) -> None:
        # An absolute path takes an encoded leading slash (sandbox API filesystem reference).
        self.send("PUT", f"{sandbox_url}/filesystem/%2F{quote(path.lstrip('/'))}", json={"content": content})

    def _access_token(self) -> str:
        with self._token_lock:
            if self._token is None or time.monotonic() >= self._token_refresh_at:
                key = os.environ.get(self.api_key_env, "")  # read per mint, never stored or logged
                response = self.http.post(
                    f"{self.api_url}/token",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"scopes": ["sandboxes"]},
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                self._token = response.json()["token"]
                self._token_refresh_at = time.monotonic() + TOKEN_REFRESH_SECONDS
            return self._token


class BasetenSandbox(SandboxExecutor):
    """``max_concurrency`` caps live sandboxes in this process (a session holds one until closed); a session past it
    waits. Programs run under ``local``'s limits (``memory_limit_mb``, ``compile_timeout``,
    ``compile_memory_limit_mb``)."""

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
        memory_limit_mb: int = SANDBOX_DEFAULT_MEMORY_MB,
        compile_timeout: float = SANDBOX_DEFAULT_COMPILE_TIMEOUT,
        compile_memory_limit_mb: int = SANDBOX_DEFAULT_COMPILE_MEMORY_MB,
        session: requests.Session | None = None,
    ):
        if not os.environ.get(api_key_env):
            raise ValueError(f"the baseten sandbox backend needs the {api_key_env} environment variable")
        if not 1 <= max_concurrency <= BASETEN_SANDBOX_LIMIT:
            raise ValueError(
                f"max_concurrency={max_concurrency} must be within 1..{BASETEN_SANDBOX_LIMIT}: each live session holds "
                f"its own sandbox and an organization runs at most {BASETEN_SANDBOX_LIMIT} at once"
            )
        self.api = BasetenApi(
            api_key_env=api_key_env,
            api_url=api_url,
            request_timeout=request_timeout,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            session=session,
        )
        self.api_url = self.api.api_url
        self.image = image
        self.cpu = cpu
        self.idle_ttl = idle_ttl
        self.interpreter = interpreter
        self.request_overhead = request_overhead
        self.create_timeout = create_timeout
        self.limits = ProgramLimits(memory_limit_mb, compile_timeout, compile_memory_limit_mb)
        self.slots = threading.BoundedSemaphore(max_concurrency)

    def open_session(self) -> "BasetenSession":
        return BasetenSession(self)

    def create_sandbox(self) -> tuple[str, str]:
        """The name and runtime URL of a new sandbox with the runner in place."""
        body = {
            "image": self.image,
            "cpu": self.cpu,
            "labels": {"app": "halo"},
            # Deletes a sandbox whose session was never closed.
            "lifecycle": {"expiration_policies": [{"type": "TTL_IDLE", "value": self.idle_ttl, "action": "DELETE"}]},
        }
        url = f"{self.api_url}/sandboxes/instances"
        record = self.api.send("POST", url, json=body, timeout=self.create_timeout).json()
        if record.get("status") != "DEPLOYED":
            raise SandboxInfraError(f"baseten sandbox {record.get('name')!r} is {record.get('status')}, not DEPLOYED")
        name, sandbox_url = record["name"], record["url"].rstrip("/")
        try:
            self.api.put_file(sandbox_url, RUNNER_PATH, RUNNER_SOURCE)
        except SandboxInfraError:
            with contextlib.suppress(SandboxInfraError):
                self.delete_sandbox(name)
            raise
        return name, sandbox_url

    def delete_sandbox(self, name: str) -> None:
        self.api.send("DELETE", f"{self.api_url}/sandboxes/instances/{name}", missing_ok=True)


class BasetenSession(RunnerSession):
    """One sandbox, created on first use, whose working directory persists across :meth:`run` calls. A sandbox that
    expired is replaced on the next call and the session continues empty."""

    def __init__(self, executor: BasetenSandbox):
        # One session per sandbox, so every session can take the first user.
        super().__init__(SESSION_DIR, SESSION_UIDS[0], executor.limits, executor.interpreter)
        self.executor = executor
        self._holds_slot = False
        self._name: str | None = None
        self._url: str | None = None

    def exchange(self, request: dict[str, Any], *, deadline: float) -> dict[str, Any]:
        api, url = self.executor.api, self._sandbox_url()
        try:
            api.put_file(url, REQUEST_PATH, json.dumps(request))
            process = api.send(
                "POST",
                f"{url}/process",
                json={
                    "command": f"{self.interpreter} -I {RUNNER_PATH} {REQUEST_PATH}",
                    "workingDir": "/",
                    "timeout": math.ceil(deadline),
                    "waitForCompletion": True,
                },
                timeout=deadline + self.executor.request_overhead,
            ).json()
        except SandboxGone:
            self._name = self._url = None
            raise
        if process.get("status") != "completed" or process.get("exitCode") != 0:
            last_line = (process.get("stderr") or "").strip().splitlines()[-1:] or ["no stderr"]
            raise SandboxInfraError(
                f"baseten sandbox runner {process.get('status')} with exit code {process.get('exitCode')}: "
                f"{last_line[0]}"
            )
        return parse_runner_reply(process.get("stdout") or "")

    def read_file(self, path: str) -> str | None:
        self.reject_unsafe_path(path)
        return None if self._url is None else super().read_file(path)

    def list_files(self) -> list[str]:
        return [] if self._url is None else super().list_files()

    def close(self) -> None:
        """Delete the sandbox and free its slot. Idempotent; never raises."""
        name, self._name, self._url = self._name, None, None
        if name is not None:
            with contextlib.suppress(SandboxInfraError):  # the idle expiration deletes it otherwise
                self.executor.delete_sandbox(name)
        if self._holds_slot:
            self._holds_slot = False
            self.executor.slots.release()

    def _sandbox_url(self) -> str:
        if self._url is None:
            if not self._holds_slot:
                self.executor.slots.acquire()
                self._holds_slot = True
            try:
                self._name, self._url = self.executor.create_sandbox()
            except BaseException:
                self._holds_slot = False
                self.executor.slots.release()
                raise
        return self._url
