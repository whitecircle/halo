"""Modal backend: programs run in one long-lived Modal Sandbox per executor (gVisor, network blocked). Each session is a
directory and a user of its own in it, and every call is one exec of ``program_runner``. ``modal`` is an optional
dependency; Modal's own config (``MODAL_TOKEN_ID`` / ``MODAL_TOKEN_SECRET`` or ``~/.modal.toml``) authenticates."""

import atexit
import contextlib
import itertools
import json
import math
import threading
import uuid
import weakref
from typing import Any

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

WORKSPACE = "/workspace"
# Modal terminates a sandbox no exec has reached for this long, reclaiming one a killed trainer left behind.
IDLE_TIMEOUT_SECONDS = 300
# Modal's longest sandbox lifetime. A sandbox that reaches it takes its runs in flight with it.
MAX_LIFETIME_SECONDS = 86400
# What ContainerProcess.wait() returns once the exec ``timeout`` expires (it maps ExecTimeoutError to -1); an exit is
# 0-255 and a signal 128+N, so no program or runner exit reads as it.
EXEC_DEADLINE_RETURNCODE = -1

live_executors: "weakref.WeakSet[ModalSandbox]" = weakref.WeakSet()


def require_modal():
    try:
        import modal  # noqa: PLC0415 — optional dependency, absent unless the modal backend is used
    except ImportError as exc:
        raise ImportError(
            "the 'modal' sandbox backend needs the Modal SDK: install the 'modal' extra (pip install 'halo[modal]')"
        ) from exc
    return modal


@atexit.register
def close_live_executors() -> None:
    for executor in list(live_executors):
        executor.close()


class ModalSandbox(SandboxExecutor):
    """``cpu`` / ``memory_mb`` reserve the sandbox; at most ``max_concurrency`` (default ``int(cpu)``) runs execute in
    it at once and the rest queue before their clock starts, so a time limit measures a reserved core."""

    # The program runs in the Modal container with its network blocked, never on this host.
    isolated = True

    def __init__(
        self,
        *,
        app_name: str = "halo-sandbox",
        image: Any = None,
        cpu: float = 4.0,
        memory_mb: int = 8192,
        max_concurrency: int | None = None,
        sandbox_timeout: int = MAX_LIFETIME_SECONDS,
        memory_limit_mb: int = SANDBOX_DEFAULT_MEMORY_MB,
        compile_timeout: float = SANDBOX_DEFAULT_COMPILE_TIMEOUT,
        compile_memory_limit_mb: int = SANDBOX_DEFAULT_COMPILE_MEMORY_MB,
    ):
        self.max_concurrency = max_concurrency or max(1, int(cpu))
        if memory_mb < self.max_concurrency * max(memory_limit_mb, compile_memory_limit_mb):
            raise ValueError(
                f"ModalSandbox memory_mb={memory_mb} cannot hold max_concurrency={self.max_concurrency} runs at "
                f"max(memory_limit_mb, compile_memory_limit_mb)={max(memory_limit_mb, compile_memory_limit_mb)} MiB "
                f"each; raise memory_mb or lower max_concurrency."
            )
        self.modal = require_modal()
        if image is None:
            image = self.modal.Image.debian_slim(python_version="3.12").apt_install("gcc", "g++")
        elif isinstance(image, str):
            image = self.modal.Image.from_registry(image)
        self.image = image
        self.cpu = cpu
        self.memory_mb = memory_mb
        self.sandbox_timeout = sandbox_timeout
        self.limits = ProgramLimits(memory_limit_mb, compile_timeout, compile_memory_limit_mb)
        # An authenticated call: missing or invalid credentials fail at construction, not on the first run.
        self.app = self.modal.App.lookup(app_name, create_if_missing=True)
        self.run_slots = threading.BoundedSemaphore(self.max_concurrency)
        self._lock = threading.Lock()
        self._sandbox: Any = None
        self._retired_workdirs: list[str] = []
        self._session_uids = itertools.cycle(SESSION_UIDS)
        live_executors.add(self)

    def open_session(self) -> "ModalSession":
        return ModalSession(self, uid=next(self._session_uids))

    def close(self) -> None:
        """Terminate the sandbox. Idempotent; never raises."""
        with self._lock:
            sandbox, self._sandbox = self._sandbox, None
        if sandbox is not None:
            self._terminate(sandbox)
        live_executors.discard(self)

    def retire_workdir(self, workdir: str) -> None:
        """Queue a closed session's directory for removal by the next exec, so closing makes no remote call."""
        with self._lock:
            self._retired_workdirs.append(workdir)

    def exec_runner(self, request: dict[str, Any], *, deadline: float) -> dict[str, Any]:
        """One runner exec on the sandbox (created on first use), returning its reply. Every failure raises
        :class:`SandboxInfraError` and drops the sandbox, which may still hold the program. Once the runner has
        started nothing is sent again: the program may have run."""
        sandbox, process = self._start_runner(deadline)
        with self._lock:
            retired, self._retired_workdirs = self._retired_workdirs, []
        try:
            process.stdin.write(json.dumps({**request, "remove": retired}))
            process.stdin.write_eof()
            process.stdin.drain()
            stdout = process.stdout.read()
            stderr = process.stderr.read()
            returncode = process.wait()
        except Exception as exc:
            self._drop(sandbox)
            raise SandboxInfraError(f"modal sandbox lost: {type(exc).__name__}: {exc}") from exc
        try:
            if returncode == EXEC_DEADLINE_RETURNCODE:
                raise SandboxInfraError(f"modal sandbox exec passed its {deadline:g} s deadline")
            if returncode != 0:
                last_line = (stderr or "").strip().splitlines()[-1:] or ["no stderr"]
                raise SandboxInfraError(f"modal sandbox runner exited {returncode}: {last_line[0]}")
            return parse_runner_reply(stdout)
        except SandboxInfraError:
            self._drop(sandbox)
            raise

    def _start_runner(self, deadline: float) -> tuple[Any, Any]:
        """The sandbox and a runner process on it. A sandbox that refuses the exec (its idle timeout or lifetime ended
        it) is replaced once: the runner never started, so no program ran."""
        for last_attempt in (False, True):
            sandbox = self._current_sandbox()
            try:
                return sandbox, sandbox.exec("python3", "-I", "-c", RUNNER_SOURCE, timeout=int(math.ceil(deadline)))
            except Exception as exc:
                self._drop(sandbox)
                if last_attempt:
                    raise SandboxInfraError(f"modal sandbox lost: {type(exc).__name__}: {exc}") from exc
        raise AssertionError("unreachable")

    def _current_sandbox(self) -> Any:
        with self._lock:
            if self._sandbox is None:
                try:
                    self._sandbox = self.modal.Sandbox.create(
                        app=self.app,
                        image=self.image,
                        cpu=self.cpu,
                        memory=self.memory_mb,
                        timeout=self.sandbox_timeout,
                        idle_timeout=IDLE_TIMEOUT_SECONDS,
                        block_network=True,
                        workdir=WORKSPACE,
                    )
                except Exception as exc:
                    raise SandboxInfraError(f"modal sandbox provisioning failed: {type(exc).__name__}: {exc}") from exc
                self._retired_workdirs.clear()
            return self._sandbox

    def _drop(self, sandbox: Any) -> None:
        with self._lock:
            if self._sandbox is sandbox:
                self._sandbox = None
        self._terminate(sandbox)

    @staticmethod
    def _terminate(sandbox: Any) -> None:
        with contextlib.suppress(Exception):  # already gone; the idle timeout reclaims it otherwise
            sandbox.terminate()


class ModalSession(RunnerSession):
    """A directory in the executor's sandbox, owned by the session's own user."""

    def __init__(self, executor: ModalSandbox, uid: int):
        super().__init__(f"{WORKSPACE}/{uuid.uuid4().hex}", uid, executor.limits)
        self.executor = executor

    def exchange(self, request: dict[str, Any], *, deadline: float) -> dict[str, Any]:
        if request["op"] != "run":
            return self.executor.exec_runner(request, deadline=deadline)
        with self.executor.run_slots:
            return self.executor.exec_runner(request, deadline=deadline)

    def close(self) -> None:
        self.executor.retire_workdir(self.workdir)
