"""Modal backend. An executor keeps one Modal Sandbox (gVisor, network blocked); each session is a directory and a
user of its own in it, and each call is one exec of the program runner. ``modal`` is an optional dependency, and
Modal's own configuration (``MODAL_TOKEN_ID`` / ``MODAL_TOKEN_SECRET`` or ``~/.modal.toml``) authenticates."""

import atexit
import contextlib
import itertools
import logging
import math
import threading
import uuid
import weakref
from dataclasses import replace
from typing import Any, NamedTuple

from halo.environments.sandbox.base import SandboxExecutor
from halo.environments.sandbox.limits import (
    SANDBOX_DEFAULT_COMPILE_MEMORY_MB,
    SANDBOX_DEFAULT_COMPILE_TIMEOUT,
    SANDBOX_DEFAULT_MEMORY_MB,
)
from halo.environments.sandbox.results import SandboxInfraError
from halo.environments.sandbox.runner_messages import Reply, Request, RunProgram, encode
from halo.environments.sandbox.runner_session import (
    SESSION_UIDS,
    ProgramLimits,
    RunnerSession,
    decode_reply,
    runner_source,
)

logger = logging.getLogger(__name__)

WORKSPACE = "/workspace"
# Modal terminates a sandbox that no exec has reached for this long, so one a killed trainer left is reclaimed.
IDLE_TIMEOUT_SECONDS = 300
# The longest lifetime Modal gives a sandbox.
MAX_LIFETIME_SECONDS = 86400
# What Modal's ContainerProcess.wait() returns when the exec timeout expires. No real exit code is negative.
EXEC_DEADLINE_RETURNCODE = -1


class SessionUids:
    """Hands each open session a user no other open session has."""

    def __init__(self, uids: range):
        self._uids = uids
        self._next = itertools.cycle(uids)
        self._taken: set[int] = set()
        self._lock = threading.Lock()

    def take(self) -> int:
        with self._lock:
            if len(self._taken) == len(self._uids):
                raise SandboxInfraError(f"all {len(self._uids)} session users of the modal sandbox are in use")
            uid = next(uid for uid in self._next if uid not in self._taken)
            self._taken.add(uid)
            return uid

    def give_back(self, uid: int) -> None:
        with self._lock:
            self._taken.discard(uid)


class ClosedSession(NamedTuple):
    workdir: str
    uid: int


class RunnerExec(NamedTuple):
    sandbox: Any
    process: Any


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
    """``cpu`` and ``memory_mb`` are what the sandbox reserves. At most ``max_concurrency`` programs (default
    ``int(cpu)``) run in it at once; the rest wait, and their time limit starts only when they run."""

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
        self.limits = ProgramLimits(
            memory_limit_mb=memory_limit_mb,
            compile_timeout=compile_timeout,
            compile_memory_limit_mb=compile_memory_limit_mb,
        )
        # This call authenticates, so bad credentials fail here and not on the first run.
        self.app = self.modal.App.lookup(app_name, create_if_missing=True)
        self.run_slots = threading.BoundedSemaphore(self.max_concurrency)
        self._lock = threading.Lock()
        self._sandbox: Any = None
        self._closed_sessions: list[ClosedSession] = []
        self.session_uids = SessionUids(SESSION_UIDS)
        live_executors.add(self)

    def open_session(self) -> "ModalSession":
        return ModalSession(self, uid=self.session_uids.take())

    def close(self) -> None:
        """Terminate the sandbox. Idempotent; never raises."""
        with self._lock:
            sandbox, self._sandbox = self._sandbox, None
        if sandbox is not None:
            self._terminate(sandbox)
        live_executors.discard(self)

    def remove_later(self, session: ClosedSession) -> None:
        """Queue a closed session's directory for removal by the next exec, so closing makes no remote call. Its
        user is free again once that exec has finished."""
        with self._lock:
            self._closed_sessions.append(session)

    def exec_runner(self, request: Request, *, deadline: float) -> Reply:
        """Run the runner once on the sandbox, which is created on first use. Every failure raises
        :class:`SandboxInfraError` and drops the sandbox, since it may still hold the program. Nothing is sent a
        second time once the runner has started: the program may have run."""
        sandbox, process = self._start_runner(deadline)
        with self._lock:
            closed, self._closed_sessions = self._closed_sessions, []
        try:
            return self._exchange(
                sandbox, process, replace(request, closed_workdirs=[s.workdir for s in closed]), deadline
            )
        finally:
            # Their directories are removed, or gone with a dropped sandbox.
            for session in closed:
                self.session_uids.give_back(session.uid)

    def _exchange(self, sandbox: Any, process: Any, request: Request, deadline: float) -> Reply:
        try:
            process.stdin.write(encode(request))
            process.stdin.write_eof()
            process.stdin.drain()
            stdout = process.stdout.read()
            stderr = process.stderr.read()
            returncode = process.wait()
        except Exception as error:  # the SDK raises its own types for a lost sandbox or connection
            self._drop(sandbox)
            raise SandboxInfraError(f"modal sandbox lost: {type(error).__name__}: {error}") from error
        try:
            return self._reply(returncode, stdout, stderr, deadline)
        except SandboxInfraError:
            self._drop(sandbox)
            raise

    @staticmethod
    def _reply(returncode: int, stdout: str, stderr: str, deadline: float) -> Reply:
        if returncode == EXEC_DEADLINE_RETURNCODE:
            raise SandboxInfraError(f"modal sandbox exec passed its {deadline:g} s deadline")
        if returncode != 0:
            last_line = (stderr or "").strip().splitlines()[-1:] or ["no stderr"]
            raise SandboxInfraError(f"modal sandbox runner exited {returncode}: {last_line[0]}")
        return decode_reply(stdout)

    def _start_runner(self, deadline: float) -> RunnerExec:
        """Start the runner. A sandbox that refuses the exec was ended by its idle timeout or lifetime; it is replaced
        once, which is safe because no runner started."""
        sandbox = self._current_sandbox()
        try:
            return RunnerExec(sandbox, self._spawn_runner(sandbox, deadline))
        except Exception as error:  # the SDK raises its own types for a sandbox that is gone
            logger.info("modal sandbox refused an exec (%s: %s); starting a new one", type(error).__name__, error)
            self._drop(sandbox)
        sandbox = self._current_sandbox()
        try:
            return RunnerExec(sandbox, self._spawn_runner(sandbox, deadline))
        except Exception as exc:
            self._drop(sandbox)
            raise SandboxInfraError(f"modal sandbox lost: {type(exc).__name__}: {exc}") from exc

    @staticmethod
    def _spawn_runner(sandbox: Any, deadline: float) -> Any:
        return sandbox.exec("python3", "-I", "-c", runner_source(), timeout=math.ceil(deadline))

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
                for session in self._closed_sessions:
                    self.session_uids.give_back(session.uid)
                self._closed_sessions.clear()
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
        super().__init__(f"{WORKSPACE}/{uuid.uuid4().hex}", uid=uid, limits=executor.limits, interpreter="python3")
        self.executor = executor
        self._closed = False

    def exchange(self, request: Request, *, deadline: float) -> Reply:
        if not isinstance(request.operation, RunProgram):
            return self.executor.exec_runner(request, deadline=deadline)
        with self.executor.run_slots:
            return self.executor.exec_runner(request, deadline=deadline)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.executor.remove_later(ClosedSession(self.workdir, self.uid))
