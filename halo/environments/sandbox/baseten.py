"""Baseten Sandboxes backend (https://docs.baseten.co/sandboxes/overview). Each session runs in its own Baseten
microVM, created on first use and deleted on close, and each call is one run of the program runner in it.
``BASETEN_API_KEY`` mints the short-lived access token that every sandbox call carries."""

import atexit
import contextlib
import logging
import math
import os
import threading
import time
import uuid
import weakref
from dataclasses import dataclass
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from halo.environments.sandbox.base import SandboxExecutor
from halo.environments.sandbox.limits import (
    SANDBOX_DEFAULT_COMPILE_MEMORY_MB,
    SANDBOX_DEFAULT_COMPILE_TIMEOUT,
    SANDBOX_DEFAULT_MEMORY_MB,
)
from halo.environments.sandbox.results import SandboxInfraError
from halo.environments.sandbox.runner_messages import Reply, Request, encode
from halo.environments.sandbox.runner_session import (
    SESSION_UIDS,
    ProgramLimits,
    RunnerSession,
    decode_reply,
    runner_source,
)

logger = logging.getLogger(__name__)

BASETEN_API_URL = "https://api.baseten.co/v1"
# An organization runs at most 10 sandboxes at once (https://docs.baseten.co/reference/sandbox-api/overview).
BASETEN_SANDBOX_LIMIT = 10
# A token expires 2 hours after issue and cannot be renewed.
TOKEN_REFRESH_SECONDS = 90 * 60
RETRIED_STATUSES = (429, 500, 502, 503, 504)
IDEMPOTENT_METHODS = frozenset({"GET", "PUT", "DELETE"})

# Root's files in the sandbox's /tmp: a program, which runs as another user, can read but not replace them.
RUNNER_PATH = "/tmp/halo_program_runner.py"
REQUEST_PATH = "/tmp/halo_runner_request.json"
SESSION_DIR = "/halo/session"
# At interpreter exit a delete is sent once and waits this long, so an unreachable API does not hold the process.
EXIT_DELETE_TIMEOUT = 5.0


live_executors: "weakref.WeakSet[BasetenSandbox]" = weakref.WeakSet()


@atexit.register
def close_live_executors() -> None:
    for executor in list(live_executors):
        executor.close(at_exit=True)


class SandboxGone(SandboxInfraError):
    """The sandbox no longer exists: its idle expiration or a delete removed it."""


@dataclass(frozen=True)
class Sandbox:
    name: str
    url: str


@dataclass(frozen=True)
class Process:
    status: str
    exit_code: int | None
    stdout: str
    stderr: str

    @property
    def completed(self) -> bool:
        return self.status == "completed" and self.exit_code == 0

    @property
    def last_error_line(self) -> str:
        lines = self.stderr.strip().splitlines()
        return lines[-1] if lines else "no stderr"


def service_message(response: requests.Response) -> str:
    """The sentence Baseten put in an error body, or the start of a body that has none."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    error = body.get("error", body) if isinstance(body, dict) else body
    message = error.get("message") if isinstance(error, dict) else None
    return message or response.text[:200]


def is_registry_reference(image: str) -> bool:
    """``docker.io/library/python:3.12`` names a registry host; ``baseten/base-image:latest`` and
    ``sandbox/my-image:tag`` name images Baseten already holds."""
    host, slash, _ = image.partition("/")
    return bool(slash) and ("." in host or ":" in host or host == "localhost")


class BasetenApi:
    """HTTP to Baseten's management API and to a sandbox's runtime API. Every failure raises
    :class:`SandboxInfraError`.

    Connection errors, 429 and 5xx are retried with exponential backoff, except where a retry could run a program
    twice: a sandbox's ``POST /process`` is sent again only when the connection was never made."""

    def __init__(
        self,
        *,
        api_key_env: str,
        api_url: str,
        request_timeout: float,
        max_retries: int,
        backoff_factor: float,
        session: requests.Session | None,
    ):
        self.api_key_env = api_key_env
        self.api_url = api_url.rstrip("/")
        self.request_timeout = request_timeout
        self.http = session or requests.Session()
        sandbox_calls = HTTPAdapter(max_retries=self._retry(max_retries, backoff_factor, IDEMPOTENT_METHODS))
        for scheme in ("https://", "http://"):
            self.http.mount(scheme, sandbox_calls)
        # Minting a token and creating a named sandbox can be repeated, so every method is retried there.
        management_calls = HTTPAdapter(max_retries=self._retry(max_retries, backoff_factor, None))
        self.http.mount(f"{self.api_url}/", management_calls)
        self.http_once = requests.Session()
        self._token_lock = threading.Lock()
        self._token: str | None = None
        self._token_refresh_at = 0.0

    @staticmethod
    def _retry(max_retries: int, backoff_factor: float, methods: frozenset[str] | None) -> Retry:
        return Retry(
            total=max_retries,
            read=False,
            backoff_factor=backoff_factor,
            status_forcelist=RETRIED_STATUSES,
            allowed_methods=methods,
            raise_on_status=False,
        )

    def create_sandbox(self, *, image: str, cpu: int, idle_ttl: str, timeout: float) -> Sandbox:
        # The name makes a retried create return the sandbox the first attempt made, and lets a failed create be
        # undone without the service's answer.
        name = f"halo-{uuid.uuid4().hex[:16]}"
        body = {
            "name": name,
            "create_if_not_exists": True,
            "image": image,
            "cpu": cpu,
            "labels": {"app": "halo"},
            "lifecycle": {"expiration_policies": [{"type": "TTL_IDLE", "value": idle_ttl, "action": "DELETE"}]},
        }
        try:
            record = self._send("POST", f"{self.api_url}/sandboxes/instances", json=body, timeout=timeout).json()
            if record.get("status") != "DEPLOYED":
                raise SandboxInfraError(f"baseten sandbox {name!r} is {record.get('status')}, not DEPLOYED")
        except SandboxInfraError:
            # The service may hold the sandbox although the create failed here, and it would keep one of the
            # organization's slots until its idle expiration.
            if self._token is not None:
                with contextlib.suppress(SandboxInfraError):
                    self.delete_sandbox(name)
            raise
        return Sandbox(name=record["name"], url=record["url"].rstrip("/"))

    def delete_sandbox(self, name: str, *, at_exit: bool = False) -> None:
        url = f"{self.api_url}/sandboxes/instances/{name}"
        with contextlib.suppress(SandboxGone):
            if at_exit:
                self._send("DELETE", url, timeout=EXIT_DELETE_TIMEOUT, json=None, http=self.http_once)
            else:
                self._send("DELETE", url, timeout=self.request_timeout, json=None)

    def write_file(self, sandbox: Sandbox, path: str, content: str) -> None:
        # The filesystem API takes an absolute path with its leading slash encoded.
        url = f"{sandbox.url}/filesystem/%2F{quote(path.lstrip('/'))}"
        self._send("PUT", url, json={"content": content}, timeout=self.request_timeout)

    def run(self, sandbox: Sandbox, command: str, *, deadline: float, timeout: float) -> Process:
        body = {"command": command, "workingDir": "/", "timeout": math.ceil(deadline), "waitForCompletion": True}
        record = self._send("POST", f"{sandbox.url}/process", json=body, timeout=timeout).json()
        return Process(
            status=record.get("status") or "unknown",
            exit_code=record.get("exitCode"),
            stdout=record.get("stdout") or "",
            stderr=record.get("stderr") or "",
        )

    def _send(
        self, method: str, url: str, *, timeout: float, json: dict | None, http: requests.Session | None = None
    ) -> requests.Response:
        try:
            headers = {"Authorization": f"Bearer {self._access_token()}"}
            response = (http or self.http).request(method, url, headers=headers, timeout=timeout, json=json)
            if response.status_code == 404:
                raise SandboxGone(f"baseten sandbox: HTTP 404 from {method} {url}: {service_message(response)}")
            response.raise_for_status()
            return response
        except requests.Timeout as error:
            raise SandboxInfraError("baseten sandbox request timed out") from error
        except requests.HTTPError as error:
            detail = error.response.text[:200]
            raise SandboxInfraError(f"baseten sandbox: HTTP {error.response.status_code}: {detail}") from error
        except requests.RequestException as error:
            raise SandboxInfraError(f"baseten sandbox error: {error}") from error

    def _access_token(self) -> str:
        with self._token_lock:
            if self._token is None or time.monotonic() >= self._token_refresh_at:
                # The key is read for each token and never stored or logged.
                key = os.environ.get(self.api_key_env, "")
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
    """``max_concurrency`` caps the sandboxes this process holds at once. A session holds one until it is closed, and
    a session past the cap waits."""

    # Baseten's API has no setting that blocks a sandbox's outbound network.
    isolated = False

    def __init__(
        self,
        *,
        api_key_env: str = "BASETEN_API_KEY",
        api_url: str = BASETEN_API_URL,
        image: str = "baseten/base-image:latest",
        cpu: int = 2,
        idle_ttl: str = "10m",
        interpreter: str = "python3",
        request_timeout: float = 60.0,
        request_overhead: float = 30.0,
        create_timeout: float = 600.0,
        max_retries: int = 4,
        backoff_factor: float = 1.0,
        max_concurrency: int = BASETEN_SANDBOX_LIMIT,
        slot_timeout: float = 600.0,
        memory_limit_mb: int = SANDBOX_DEFAULT_MEMORY_MB,
        compile_timeout: float = SANDBOX_DEFAULT_COMPILE_TIMEOUT,
        compile_memory_limit_mb: int = SANDBOX_DEFAULT_COMPILE_MEMORY_MB,
        session: requests.Session | None = None,
    ):
        if not os.environ.get(api_key_env):
            raise ValueError(f"the baseten sandbox backend needs the {api_key_env} environment variable")
        if is_registry_reference(image):
            raise ValueError(
                f"image={image!r} is a registry reference, which Baseten imports again for every sandbox. Import it "
                f"once (POST {api_url.rstrip('/')}/sandboxes/images with name=<name> and image={image!r}) and pass "
                f"image='sandbox/<name>:<tag>'."
            )
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
        self.limits = ProgramLimits(
            memory_limit_mb=memory_limit_mb,
            compile_timeout=compile_timeout,
            compile_memory_limit_mb=compile_memory_limit_mb,
        )
        self.max_concurrency = max_concurrency
        self.slot_timeout = slot_timeout
        self.slots = threading.BoundedSemaphore(max_concurrency)
        self._live_sandboxes: set[Sandbox] = set()
        self._lock = threading.Lock()
        live_executors.add(self)

    def close(self, *, at_exit: bool = False) -> None:
        """Delete every sandbox this executor still holds. Idempotent; never raises."""
        with self._lock:
            sandboxes, self._live_sandboxes = self._live_sandboxes, set()
        for sandbox in sandboxes:
            self.delete_sandbox(sandbox, at_exit=at_exit)
        live_executors.discard(self)

    def take_slot(self) -> None:
        if not self.slots.acquire(timeout=self.slot_timeout):
            raise SandboxInfraError(
                f"no baseten sandbox became free in {self.slot_timeout:g} s: max_concurrency={self.max_concurrency} "
                f"sessions hold one each. Close sessions that are done, or raise max_concurrency within the "
                f"organization's limit of {BASETEN_SANDBOX_LIMIT}."
            )

    def open_session(self) -> "BasetenSession":
        return BasetenSession(self)

    def create_sandbox(self) -> Sandbox:
        """A new sandbox with the runner uploaded."""
        sandbox = self.api.create_sandbox(
            image=self.image, cpu=self.cpu, idle_ttl=self.idle_ttl, timeout=self.create_timeout
        )
        try:
            self.api.write_file(sandbox, RUNNER_PATH, runner_source())
        except SandboxInfraError:
            self.delete_sandbox(sandbox)
            raise
        with self._lock:
            self._live_sandboxes.add(sandbox)
        return sandbox

    def delete_sandbox(self, sandbox: Sandbox, *, at_exit: bool = False) -> None:
        """Never raises: a sandbox that could not be deleted is removed by its idle expiration."""
        with self._lock:
            self._live_sandboxes.discard(sandbox)
        try:
            self.api.delete_sandbox(sandbox.name, at_exit=at_exit)
        except SandboxInfraError as error:
            logger.debug("baseten sandbox %s was not deleted: %s", sandbox.name, error)


class BasetenSession(RunnerSession):
    """One sandbox, created on first use. If Baseten removed it, the next call creates another: a session that only
    ran programs continues with an empty directory, and one that had written files raises from then on."""

    def __init__(self, executor: BasetenSandbox):
        # A sandbox holds one session, so every session can use the same user.
        super().__init__(SESSION_DIR, uid=SESSION_UIDS[0], limits=executor.limits, interpreter=executor.interpreter)
        self.executor = executor
        self._sandbox: Sandbox | None = None
        self._holds_slot = False

    def exchange(self, request: Request, *, deadline: float) -> Reply:
        api = self.executor.api
        sandbox = self._current_sandbox()
        try:
            api.write_file(sandbox, REQUEST_PATH, encode(request))
            process = api.run(
                sandbox,
                f"{self.interpreter} -I {RUNNER_PATH} {REQUEST_PATH}",
                deadline=deadline,
                timeout=deadline + self.executor.request_overhead,
            )
            if not process.completed:
                raise SandboxInfraError(
                    f"baseten sandbox runner {process.status} with exit code {process.exit_code}: "
                    f"{process.last_error_line}"
                )
            return decode_reply(process.stdout)
        except SandboxGone:
            self._sandbox = None
            self.directory_lost()
            raise
        except SandboxInfraError:
            # The sandbox may still hold the program: the next call gets a new one.
            self.executor.delete_sandbox(sandbox)
            self._sandbox = None
            self.directory_lost()
            raise

    def read_file(self, path: str) -> str | None:
        self.reject_unsafe_path(path)
        self.raise_if_files_lost()
        if self._sandbox is None:
            return None
        return super().read_file(path)

    def list_files(self) -> list[str]:
        self.raise_if_files_lost()
        if self._sandbox is None:
            return []
        return super().list_files()

    def close(self) -> None:
        """Delete the sandbox and free its slot. Idempotent; never raises."""
        if self._sandbox is not None:
            self.executor.delete_sandbox(self._sandbox)
            self._sandbox = None
        if self._holds_slot:
            self._holds_slot = False
            self.executor.slots.release()

    def _current_sandbox(self) -> Sandbox:
        if self._sandbox is not None:
            return self._sandbox
        if not self._holds_slot:
            self.executor.take_slot()
            self._holds_slot = True
        try:
            self._sandbox = self.executor.create_sandbox()
        except BaseException:
            self._holds_slot = False
            self.executor.slots.release()
            raise
        return self._sandbox
