"""Local subprocess sandbox: a child process under POSIX resource limits (CPU, address space, file size).
No network or filesystem namespace isolation; the bubblewrap and remote backends provide that."""

import contextlib
import math
import os
import platform
import resource
import select
import signal
import stat
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from halo.env import env_int
from halo.environments.sandbox.base import SANDBOX_DEFAULT_TIMEOUT, SandboxExecutor, SandboxSession
from halo.environments.sandbox.languages import INTERPRETER_PLACEHOLDER, LanguageSpec, require_language, utf8_encodable
from halo.environments.sandbox.limits import (
    LOCAL_FSIZE_LIMIT,
    LOCAL_NPROC_LIMIT,
    RLIMIT_CPU_SLACK_SECONDS,
    SANDBOX_DEFAULT_COMPILE_MEMORY_MB,
    SANDBOX_DEFAULT_COMPILE_TIMEOUT,
    SANDBOX_DEFAULT_MEMORY_MB,
    TAMPERED_WORKDIR_RETURNCODE,
    build_program_env,
)
from halo.environments.sandbox.results import (
    COMMAND_NOT_FOUND_RETURNCODE,
    SandboxAgentFault,
    SandboxInfraError,
    SandboxResult,
    compile_failure_verdict,
    compile_limit_verdict,
)
from halo.environments.sandbox.workspace import Entry, Owner, SessionPathError, Workspace
from halo.hardware import available_cpu_count

_PYTHON_INTERPRETER = sys.executable or "python"
# As tool descriptions name it (``CPython 3.12``).
_PYTHON_RUNTIME = f"{platform.python_implementation()} {sys.version_info.major}.{sys.version_info.minor}"

# Seconds a launch (limit shell, bubblewrap's namespaces and mounts) may take to reach the program before the run
# is lost to the host. The program's time limit starts where the launch ends.
LAUNCH_TIMEOUT_SECONDS = 30.0

# (language, source text, auxiliary file contents)
_BuildKey = tuple[str, str, tuple[tuple[str, str], ...]]


def _resolve_execution_slots() -> int:
    """CPUs this process may use, or ``HALO_SANDBOX_MAX_CONCURRENCY``."""
    override = env_int("HALO_SANDBOX_MAX_CONCURRENCY", None)
    if override is not None:
        return max(1, override)
    return available_cpu_count()


class ExecutionGate:
    """Fixed pool of execution slots, so a per-test wall-clock limit measures near-dedicated-core time.
    A run acquires its slot before its timeout starts."""

    def __init__(self, slots: int):
        self._semaphore = threading.BoundedSemaphore(max(1, slots))

    @contextmanager
    def slot(self) -> Iterator[None]:
        self._semaphore.acquire()
        try:
            yield
        finally:
            self._semaphore.release()


sandbox_execution_gate = ExecutionGate(_resolve_execution_slots())


def _captured_text(capture) -> str:
    """Decoded with replacement, so non-UTF-8 output (C++ UB, a binary dump) is judged, not lost to a decode error."""
    capture.seek(0)
    return capture.read(LOCAL_FSIZE_LIMIT).decode("utf-8", errors="replace")


def _readable_within(fds: tuple[int, ...], timeout: float) -> bool:
    # poll, not select: a busy host hands out descriptors past select's FD_SETSIZE.
    poller = select.poll()
    for fd in fds:
        poller.register(fd, select.POLLIN)
    return bool(poller.poll(math.ceil(timeout * 1000)))


def _exited_within(pid: int, timeout: float, marker_fd: int | None = None) -> bool:
    """Whether ``pid`` exits within ``timeout`` seconds, without reaping it, so its zombie keeps the pid and group
    id. With ``marker_fd`` (:func:`_start_marked`), ``timeout`` runs from the program's start; a launch past
    :data:`LAUNCH_TIMEOUT_SECONDS` raises :class:`SandboxInfraError`."""
    pidfd = os.pidfd_open(pid)
    try:
        if marker_fd is not None and not _readable_within((marker_fd, pidfd), LAUNCH_TIMEOUT_SECONDS):
            raise SandboxInfraError(
                f"the sandbox launch did not reach the program within {LAUNCH_TIMEOUT_SECONDS:g} s"
            )
        return _readable_within((pidfd,), timeout)
    finally:
        os.close(pidfd)


def _start_marked(argv: list[str], fd: int) -> list[str]:
    """``argv`` behind a shell that writes one byte to ``fd``, closes it and ``exec``s ``argv``, marking the start
    its time limit runs from. The program never holds ``fd`` and never runs if the byte cannot be written, so a lost
    marker cannot leave it timed by the launch bound alone."""
    return ["/bin/bash", "-c", f'printf . >&{fd} && exec {fd}>&- && exec "$@"', "halo-start", *argv]


@contextmanager
def _start_marker() -> Iterator[tuple[int, int]]:
    read_fd, write_fd = os.pipe()
    try:
        yield read_fd, write_fd
    finally:
        os.close(read_fd)
        os.close(write_fd)


def _stack_limit_kib(stack_mb: int) -> int:
    hard = resource.getrlimit(resource.RLIMIT_STACK)[1]
    kib = stack_mb * 1024
    return kib if hard == resource.RLIM_INFINITY else min(kib, hard // 1024)


class LocalSubprocessSandbox(SandboxExecutor):
    """Run code in a child process under POSIX resource limits, in a managed working directory.

    Stateless aside from config, so one instance is safe across threads and Ray actors. Limits go through a
    ``ulimit`` shell wrapper, so launches stay ``vfork``/``exec`` fast under a large parent. :meth:`_wrap_command` and
    :meth:`_reap_adopted` let a subclass interpose an isolation wrapper and reap what it leaves behind.
    """

    # An unconfined program that read a test's input can write it to a host file, remove its workdir to force a
    # rebuild, and have the rebuilt source include that file into the compiler's message.
    COMPILES_WITHOUT_TEST_INPUT = False

    # The user the program runs as, who owns its working directory; None: this process's own.
    program_owner: Owner | None = None

    def __init__(
        self,
        memory_limit_mb: int = SANDBOX_DEFAULT_MEMORY_MB,
        compile_timeout: float = SANDBOX_DEFAULT_COMPILE_TIMEOUT,
        compile_memory_limit_mb: int = SANDBOX_DEFAULT_COMPILE_MEMORY_MB,
    ):
        self.memory_limit_mb = memory_limit_mb
        self.compile_timeout = compile_timeout
        self.compile_memory_limit_mb = compile_memory_limit_mb
        # Every run waits through a pidfd; where refused, each run would fail as a backend error and void its episode.
        try:
            os.close(os.pidfd_open(os.getpid()))
        except OSError as exc:
            raise RuntimeError(
                f"the local sandbox waits on its programs through pidfd_open, which this environment refuses "
                f"({exc}); it needs Linux 5.3+ and a seccomp profile that allows pidfd_open. Otherwise use the "
                "'remote' sandbox backend."
            ) from exc

    def open_session(self) -> "_LocalSession":
        workspace = Workspace(tempfile.mkdtemp(prefix="halo_sandbox_"), self.program_owner)
        workspace.hand_over()
        return _LocalSession(workspace, self)

    def toolchain(self, language: str) -> str | None:
        """A tool description's phrase for ``language``: compile flags, Python runtime, or ``None`` if run off PATH."""
        spec = require_language(language)
        if spec.is_compiled:
            compiler, *args = spec.compile_argv
            flags = [arg for arg in args if arg.startswith("-") and arg != "-o"]
            return f"is compiled with {' '.join([compiler, *flags])}"
        if INTERPRETER_PLACEHOLDER in spec.run_argv:
            return f"runs on {_PYTHON_RUNTIME}"
        return None

    def _wrap_command(self, argv: list[str], workdir: str) -> list[str]:
        """Isolation-launcher hook; bubblewrap overrides it."""
        return argv

    def _reap_adopted(self, leader: int) -> None:
        """Reap what the run led by ``leader`` left to this process, once the group is killed and the leader reaped.
        The local backend adopts nothing; bubblewrap overrides this."""

    @staticmethod
    def _limit_wrap(
        argv: list[str], cpu_seconds: int, memory_mb: int, nproc: int | None = None, stack_mb: int | None = None
    ) -> list[str]:
        """``argv`` behind a shell that applies the run's RLIMITs through ``ulimit``, then ``exec``s it.
        A ``preexec_fn`` would force CPython's ``fork`` path, copying the parent's page tables per run; the kernel
        carries the limits into a bwrap jail."""
        # bash ulimit units (outside POSIX mode): -t CPU seconds; -f, -v, -s KiB; -u processes. -c 0, or the host's
        # crash handler stores a core at its full size.
        limits = [f"ulimit -t {cpu_seconds}", f"ulimit -f {LOCAL_FSIZE_LIMIT // 1024}", "ulimit -c 0"]
        if memory_mb:
            limits.append(f"ulimit -v {memory_mb * 1024}")
        if nproc:
            limits.append(f"ulimit -u {nproc}")
        if stack_mb:
            limits.append(f"ulimit -s {_stack_limit_kib(stack_mb)}")
        script = "; ".join(limits) + '; exec "$@"'
        return ["/bin/bash", "-c", script, "halo-sandbox", *argv]

    def _run_in_new_session(
        self,
        argv: list[str],
        *,
        stdin: str,
        timeout: float,
        cwd: str,
        env: dict[str, str],
        started: tuple[int, int] | None = None,
    ) -> tuple[str, str, int | None, bool]:
        """Run ``argv`` in its own session; returns ``(stdout, stderr, returncode, timed_out)``.

        The group is SIGKILLed whenever the run ends, after the leader exits too (a child left in it would outlive
        the run) and before the leader is reaped, while its zombie holds the group's id. Stdio are temp files: nothing
        left running holds the run open, and ``RLIMIT_FSIZE`` ends an output flood as the program's own failure. With
        ``started`` (:func:`_start_marked`), ``timeout`` runs from the program's start.
        """
        # poll() reads a negative timeout as none: the host would wait as long as the program runs.
        if not (math.isfinite(timeout) and timeout > 0):
            raise ValueError(f"a run's timeout must be a finite number of seconds > 0, got {timeout!r}")
        with tempfile.TemporaryFile() as feed, tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            # A model-written stdin can hold a lone surrogate.
            feed.write(utf8_encodable(stdin).encode("utf-8"))
            feed.seek(0)
            proc = subprocess.Popen(
                argv,
                stdin=feed,
                stdout=out,
                stderr=err,
                cwd=cwd,
                env=env,
                start_new_session=True,
                pass_fds=(started[1],) if started else (),
            )
            try:
                timed_out = not _exited_within(proc.pid, timeout, started[0] if started else None)
            finally:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                self._reap_adopted(proc.pid)
            return _captured_text(out), _captured_text(err), proc.returncode, timed_out

    @staticmethod
    def _child_env(workdir: str) -> dict[str, str]:
        """Minimal environment, with no inherited proxy or secrets."""
        return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), **build_program_env(workdir)}

    def _write_sources(
        self, workspace: Workspace, spec: LanguageSpec, code: str, files: dict[str, str] | None
    ) -> SandboxResult | None:
        """Write ``files`` and the source. Returns None, an error for an unsafe path (the caller's fault), or a
        runtime-error verdict when the program replaced or locked a prepared entry (its own fault, never an infra
        error that could void its episode). A write refused after the restore means an escaped child locked it."""
        files = files or {}
        refused = SandboxSession.refused_auxiliary_file(files)
        if refused is not None:
            return refused
        try:
            for name, content in files.items():
                workspace.write(name, utf8_encodable(content))
            workspace.write(spec.source_name, utf8_encodable(code))
        except (SessionPathError, PermissionError) as exc:
            return SandboxResult(stderr=f"working directory tampered: {exc}", returncode=TAMPERED_WORKDIR_RETURNCODE)
        return None

    def _compile(self, workdir: str, spec: LanguageSpec) -> SandboxResult | None:
        """Build the written source; None on success. A rejection or a build past ``compile_timeout`` is the source's
        verdict (``compile_failed``); a missing compiler sets ``error``."""
        compile_argv = self._wrap_command(list(spec.compile_argv), workdir)
        compile_argv = self._limit_wrap(
            compile_argv,
            int(math.ceil(self.compile_timeout + RLIMIT_CPU_SLACK_SECONDS)),
            self.compile_memory_limit_mb,
        )
        with sandbox_execution_gate.slot():
            stdout, stderr, returncode, timed_out = self._run_in_new_session(
                compile_argv, stdin="", timeout=self.compile_timeout, cwd=workdir, env=self._child_env(workdir)
            )
        if timed_out:
            # The compile holds an execution slot, so an overrun is the source's doing, not a starved host.
            return compile_limit_verdict(f"compilation timed out after {self.compile_timeout:g} s")
        if returncode == COMMAND_NOT_FOUND_RETURNCODE:
            return SandboxResult(error=f"compiler not found: {spec.compile_argv[0]!r}", stderr=stderr.strip())
        if returncode != 0:
            return compile_failure_verdict(stderr, stdout, returncode)
        return None

    def _run_program(self, workdir: str, spec: LanguageSpec, *, stdin: str, timeout: float) -> SandboxResult:
        """Run the built program, ``timeout`` measured from its start. The launch before it (on bubblewrap, jail
        setup queueing on a host-wide kernel lock) is the host's cost."""
        run_argv = [_PYTHON_INTERPRETER if tok == INTERPRETER_PLACEHOLDER else tok for tok in spec.run_argv]
        # The marker's pipe opens inside the slot, so waiting runs hold no descriptors.
        with sandbox_execution_gate.slot(), _start_marker() as started:
            run_argv = self._wrap_command(_start_marked(run_argv, started[1]), workdir)
            # A compiled program's stack gets the whole memory limit, as on a contest judge. An interpreter's does
            # not: glibc sizes every thread's stack by that limit and each worker thread would take it.
            run_argv = self._limit_wrap(
                run_argv,
                int(math.ceil(timeout + RLIMIT_CPU_SLACK_SECONDS)),
                self.memory_limit_mb,
                nproc=LOCAL_NPROC_LIMIT,
                stack_mb=self.memory_limit_mb if spec.is_compiled else None,
            )
            stdout, stderr, returncode, timed_out = self._run_in_new_session(
                run_argv, stdin=stdin, timeout=timeout, cwd=workdir, env=self._child_env(workdir), started=started
            )
        # SIGXCPU (the backstop) is reported as a timeout so callers bucket it as TLE.
        return SandboxResult(
            stdout=stdout or "",
            stderr=stderr or "",
            returncode=None if timed_out else returncode,
            timed_out=timed_out or returncode == -signal.SIGXCPU,
        )


class _LocalSession(SandboxSession):
    """Persistent working directory: source and build survive between turns, and a compiled program reruns from its
    binary while language, source and ``files`` stay the same. A removed ``workdir`` is recreated empty; a link or
    file in its place breaks the session (:class:`SandboxAgentFault`), since any path under it could reach the host."""

    def __init__(self, workspace: Workspace, executor: LocalSubprocessSandbox):
        self.workspace = workspace
        self._executor = executor
        # Build key and compile verdict (None = built, runnable).
        self._build: tuple[_BuildKey, SandboxResult | None] | None = None
        # What reset_to_prepared keeps; typed so an entry the program swapped for a link is dropped.
        self._prepared_entries: set[Entry] | None = None

    @property
    def workdir(self) -> str:
        return self.workspace.path

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
        broken = self._ensure_workspace()
        if broken is not None:
            return broken
        failure = self._prepare(spec, code, files)
        if failure is not None:
            return failure
        result = self._executor._run_program(self.workdir, spec, stdin=stdin, timeout=timeout)
        return self._ensure_workspace() or result

    def _ensure_workspace(self) -> SandboxResult | None:
        """None once the working directory is usable (recreated empty if the program removed it), or the agent fault
        when the program put a link or file at its path."""
        try:
            mode = os.lstat(self.workdir).st_mode
        except FileNotFoundError:
            os.makedirs(self.workdir, mode=0o700, exist_ok=True)
            self.workspace.hand_over()
            self._build = None
            self._prepared_entries = None
            return None
        if stat.S_ISDIR(mode):
            return None
        message = "the program replaced its working directory"
        return SandboxResult(stderr=message, returncode=TAMPERED_WORKDIR_RETURNCODE, agent_fault=message)

    def _require_intact(self) -> None:
        broken = self._ensure_workspace()
        if broken is not None:
            raise SandboxAgentFault(broken.agent_fault)

    def _prepare(self, spec: LanguageSpec, code: str, files: dict[str, str] | None) -> SandboxResult | None:
        """Write ``code`` + ``files`` and build when ``spec`` compiles; None once the program is runnable.
        Every registered compile writes ``./main``, so a session keeps one build and its verdict until language, source
        or ``files`` change; any other write through the session drops it, since it may have changed a header."""
        key = (spec.name, code, tuple(sorted((files or {}).items())))
        if spec.is_compiled and self._build is not None and self._build[0] == key:
            return self._build[1]
        self._build = None
        self.workspace.restore_owner_access()
        refused = self._executor._write_sources(self.workspace, spec, code, files)
        if refused is not None:
            return refused
        if spec.is_compiled:
            failure = self._executor._compile(self.workdir, spec)
            self._build = (key, failure)
        else:
            failure = None
        self._prepared_entries = self.workspace.entries()
        return failure

    def reset_to_prepared(self) -> None:
        # A replaced working directory lists whatever it now points at; the next run reports the fault.
        if self._ensure_workspace() is not None or self._prepared_entries is None:
            return
        self.workspace.keep_only(self._prepared_entries)

    def write_file(self, path: str, content: str) -> None:
        self.reject_unsafe_path(path)
        self._require_intact()
        self._build = None
        self._prepared_entries = None
        self.workspace.write(path, utf8_encodable(content))

    def read_file(self, path: str) -> str | None:
        """The file's text, or ``None`` when ``path`` is not a regular file, so a planted link never carries a host
        file into the trajectory. Decoded as UTF-8 with replacement, the encoding the write side uses."""
        self.reject_unsafe_path(path)
        self._require_intact()
        return self.workspace.read(path)

    def list_files(self) -> list[str]:
        self._require_intact()
        return self.workspace.files()

    def close(self) -> None:
        """Remove the workdir, or the link or file in its place. Never raises; what the walk cannot remove stays."""
        try:
            mode = os.lstat(self.workdir).st_mode
        except OSError:
            return
        if stat.S_ISDIR(mode):
            self.workspace.remove()
        else:
            with contextlib.suppress(OSError):
                os.unlink(self.workdir)
