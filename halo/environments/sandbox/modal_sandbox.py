"""Modal backend: programs run in one long-lived Modal Sandbox per executor (gVisor, network blocked), each session a
directory in it, every call one exec of ``modal_runner``. ``modal`` is an optional dependency; Modal's own config
(``MODAL_TOKEN_ID`` / ``MODAL_TOKEN_SECRET`` or ``~/.modal.toml``) authenticates."""

import atexit
import contextlib
import json
import math
import signal
import threading
import uuid
import weakref
from pathlib import Path
from typing import Any

from halo.environments.sandbox.base import SANDBOX_DEFAULT_TIMEOUT, SandboxExecutor, SandboxSession
from halo.environments.sandbox.languages import INTERPRETER_PLACEHOLDER, LanguageSpec, require_language, utf8_encodable
from halo.environments.sandbox.local import (
    LOCAL_FSIZE_LIMIT,
    LOCAL_NPROC_LIMIT,
    RLIMIT_CPU_SLACK_SECONDS,
    SANDBOX_DEFAULT_COMPILE_MEMORY_MB,
    SANDBOX_DEFAULT_COMPILE_TIMEOUT,
    SANDBOX_DEFAULT_MEMORY_MB,
    build_program_env,
)
from halo.environments.sandbox.results import (
    COMMAND_NOT_FOUND_RETURNCODE,
    SandboxInfraError,
    SandboxResult,
    compile_failure_verdict,
    compile_limit_verdict,
)

_RUNNER_SOURCE = Path(__file__).with_name("modal_runner.py").read_text()
_WORKSPACE = "/workspace"
# Modal terminates a sandbox no exec has reached for this long, reclaiming one a killed trainer left behind.
_IDLE_TIMEOUT_SECONDS = 300
# Exec deadline slack over the runner's own limits, for transport and the runner's start.
_EXEC_OVERHEAD_SECONDS = 30.0
# What ContainerProcess.wait() returns once the exec ``timeout`` expires (it maps ExecTimeoutError to -1); an exit is
# 0-255 and a signal 128+N, so no program or runner exit reads as it.
_EXEC_DEADLINE_RETURNCODE = -1
_MIB = 1024 * 1024

_live_executors: "weakref.WeakSet[ModalSandbox]" = weakref.WeakSet()


def _require_modal():
    try:
        import modal  # noqa: PLC0415 — optional dependency, absent unless the modal backend is used
    except ImportError as exc:
        raise ImportError(
            "the 'modal' sandbox backend needs the Modal SDK: install the 'modal' extra (pip install 'halo[modal]')"
        ) from exc
    return modal


@atexit.register
def _close_live_executors() -> None:
    for executor in list(_live_executors):
        executor.close()


def _limits(*, timeout: float, memory_mb: int) -> dict[str, int]:
    limits = {
        "RLIMIT_CPU": int(math.ceil(timeout + RLIMIT_CPU_SLACK_SECONDS)),
        "RLIMIT_FSIZE": LOCAL_FSIZE_LIMIT,
        "RLIMIT_CORE": 0,
    }
    if memory_mb:
        limits["RLIMIT_AS"] = memory_mb * _MIB
    return limits


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
        sandbox_timeout: int = 3600,
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
        self._modal = _require_modal()
        if image is None:
            image = self._modal.Image.debian_slim(python_version="3.12").apt_install("gcc", "g++")
        elif isinstance(image, str):
            image = self._modal.Image.from_registry(image)
        self.image = image
        self.cpu = cpu
        self.memory_mb = memory_mb
        self.sandbox_timeout = sandbox_timeout
        self.memory_limit_mb = memory_limit_mb
        self.compile_timeout = compile_timeout
        self.compile_memory_limit_mb = compile_memory_limit_mb
        # An authenticated call: missing or invalid credentials fail at construction, not on the first run.
        self.app = self._modal.App.lookup(app_name, create_if_missing=True)
        self._slots = threading.BoundedSemaphore(self.max_concurrency)
        self._lock = threading.Lock()
        self._sandbox: Any = None
        self._stale_workdirs: list[str] = []
        _live_executors.add(self)

    def open_session(self) -> "ModalSession":
        return ModalSession(self)

    def close(self) -> None:
        """Terminate the sandbox. Idempotent; never raises."""
        with self._lock:
            sandbox, self._sandbox = self._sandbox, None
        if sandbox is not None:
            with contextlib.suppress(Exception):  # already gone; the idle timeout reclaims it otherwise
                sandbox.terminate()
        _live_executors.discard(self)

    def retire_workdir(self, workdir: str) -> None:
        """Queue a closed session's directory for removal by the next exec, so closing makes no remote call."""
        with self._lock:
            self._stale_workdirs.append(workdir)

    def execute(self, request: dict[str, Any], *, deadline: float) -> dict[str, Any]:
        """One runner exec on the sandbox (created on first use), returning its reply. A lost sandbox or transport is
        dropped and raised as :class:`SandboxInfraError`, never retried: the program may have run."""
        sandbox = self._current()
        with self._lock:
            stale, self._stale_workdirs = self._stale_workdirs, []
        try:
            proc = sandbox.exec("python3", "-I", "-c", _RUNNER_SOURCE, timeout=int(math.ceil(deadline)))
            proc.stdin.write(json.dumps({**request, "remove": stale}))
            proc.stdin.write_eof()
            proc.stdin.drain()
            raw = proc.stdout.read()
            stderr = proc.stderr.read()
            returncode = proc.wait()
        except Exception as exc:
            self._drop(sandbox)
            raise SandboxInfraError(f"modal sandbox lost: {type(exc).__name__}: {exc}") from exc
        if returncode == _EXEC_DEADLINE_RETURNCODE:
            raise SandboxInfraError(f"modal sandbox exec passed its {deadline:g} s deadline")
        if returncode != 0:
            tail = (stderr or "").strip().splitlines()
            raise SandboxInfraError(f"modal sandbox runner exited {returncode}: {tail[-1] if tail else 'no stderr'}")
        try:
            return json.loads(raw)
        except ValueError as exc:
            raise SandboxInfraError("modal sandbox runner returned malformed output") from exc

    def _current(self) -> Any:
        with self._lock:
            if self._sandbox is None:
                self._sandbox = self._modal.Sandbox.create(
                    app=self.app,
                    image=self.image,
                    cpu=self.cpu,
                    memory=self.memory_mb,
                    timeout=self.sandbox_timeout,
                    idle_timeout=_IDLE_TIMEOUT_SECONDS,
                    block_network=True,
                    workdir=_WORKSPACE,
                )
                self._stale_workdirs.clear()
            return self._sandbox

    def _drop(self, sandbox: Any) -> None:
        with self._lock:
            if self._sandbox is sandbox:
                self._sandbox = None
        with contextlib.suppress(Exception):
            sandbox.terminate()

    def run_request(self, spec: LanguageSpec, *, timeout: float, build: bool) -> dict[str, Any]:
        """The commands and limits of one run, from the local backend's constants."""
        limits = {**_limits(timeout=timeout, memory_mb=self.memory_limit_mb), "RLIMIT_NPROC": LOCAL_NPROC_LIMIT}
        if spec.is_compiled:
            limits["RLIMIT_STACK"] = self.memory_limit_mb * _MIB
        return {
            "op": "run",
            "source_name": spec.source_name,
            "run_argv": ["python3" if tok == INTERPRETER_PLACEHOLDER else tok for tok in spec.run_argv],
            "timeout": timeout,
            "limits": limits,
            "compile_argv": list(spec.compile_argv) if spec.is_compiled else None,
            "build": build,
            "compile_timeout": self.compile_timeout,
            "compile_limits": _limits(timeout=self.compile_timeout, memory_mb=self.compile_memory_limit_mb),
        }


class ModalSession(SandboxSession):
    """A directory in the executor's sandbox. A compiled program rebuilds only when language, source or ``files``
    change, or when a replaced sandbox lost the directory, which the session then continues empty."""

    def __init__(self, executor: ModalSandbox):
        self._executor = executor
        self.workdir = f"{_WORKSPACE}/{uuid.uuid4().hex}"
        self._build: tuple[tuple, SandboxResult | None] | None = None
        self._prepared: list[str] | None = None
        self._reset_pending = False

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
        except ValueError as exc:
            return SandboxResult(error=str(exc))
        files = files or {}
        for name in files:
            if not self.is_safe_member_name(name):
                return SandboxResult(error=f"unsafe auxiliary file path: {name!r}")
        key = (spec.name, code, tuple(sorted(files.items())))
        built = spec.is_compiled and self._build is not None and self._build[0] == key
        if built and self._build[1] is not None:
            return self._build[1]
        request = {
            **self._executor.run_request(spec, timeout=timeout, build=not built),
            "workdir": self.workdir,
            "code": utf8_encodable(code),
            "files": {name: utf8_encodable(content) for name, content in files.items()},
            "stdin": utf8_encodable(stdin),
            "env": build_program_env(self.workdir),
            "reset_to": self._prepared if self._reset_pending else None,
        }
        deadline = _EXEC_OVERHEAD_SECONDS + timeout + (self._executor.compile_timeout if spec.is_compiled else 0)
        try:
            with self._executor._slots:
                reply = self._executor.execute(request, deadline=deadline)
        except SandboxInfraError as exc:
            return SandboxResult(error=str(exc))
        self._reset_pending = False
        if "compile" in reply:
            verdict = _compile_verdict(reply["compile"], spec, self._executor.compile_timeout)
            self._build = (key, verdict)
            if verdict is not None:
                return verdict
        self._prepared = reply["entries"]
        run = reply["run"]
        return SandboxResult(
            stdout=run["stdout"],
            stderr=run["stderr"],
            returncode=None if run["timed_out"] else run["returncode"],
            timed_out=run["timed_out"] or run["returncode"] == -signal.SIGXCPU,
        )

    def reset_to_prepared(self) -> None:
        """Applied by the next run, before it writes its sources: one exec per graded test."""
        self._reset_pending = self._prepared is not None

    def write_file(self, path: str, content: str) -> None:
        self.reject_unsafe_path(path)
        self._build, self._prepared = None, None
        self._file_op({"op": "write", "files": {path: content}})

    def read_file(self, path: str) -> str | None:
        self.reject_unsafe_path(path)
        return self._file_op({"op": "read", "path": path}).get("content")

    def list_files(self) -> list[str]:
        return list(self._file_op({"op": "list"})["files"])

    def close(self) -> None:
        self._executor.retire_workdir(self.workdir)

    def _file_op(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._executor.execute(
            {**request, "workdir": self.workdir, "env": build_program_env(self.workdir)},
            deadline=_EXEC_OVERHEAD_SECONDS,
        )


def _compile_verdict(step: dict[str, Any], spec: LanguageSpec, compile_timeout: float) -> SandboxResult | None:
    """As the local backend judges a build: past the limit or rejected is the source's verdict, a missing compiler
    sets ``error``."""
    if step["timed_out"]:
        return compile_limit_verdict(f"compilation timed out after {compile_timeout:g} s")
    if step["returncode"] == COMMAND_NOT_FOUND_RETURNCODE:
        return SandboxResult(error=f"compiler not found: {spec.compile_argv[0]!r}", stderr=step["stderr"].strip())
    if step["returncode"] != 0:
        return compile_failure_verdict(step["stderr"], step["stdout"], step["returncode"])
    return None
