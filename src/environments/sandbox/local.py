"""Local subprocess sandbox: runs code in a child process under POSIX resource limits.

Bounds CPU time, address space, and file size, runs Python with the parent environment stripped, and
compiles C/C++ before running. It does **not** namespace-isolate the network or wider filesystem; for
that use :class:`~src.environments.sandbox.bubblewrap.BubblewrapSandbox` or the remote backend.
"""

import contextlib
import errno
import math
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile

from src.environments.sandbox.base import (
    INTERPRETER_PLACEHOLDER,
    LOCAL_FSIZE_LIMIT,
    LOCAL_NPROC_LIMIT,
    SANDBOX_DEFAULT_COMPILE_MEMORY_MB,
    SANDBOX_DEFAULT_COMPILE_TIMEOUT,
    SANDBOX_DEFAULT_MEMORY_MB,
    SANDBOX_DEFAULT_TIMEOUT,
    SANDBOX_EXECUTION_GATE,
    LanguageSpec,
    SandboxAgentFault,
    SandboxExecutor,
    SandboxResult,
    SandboxSession,
    compile_limit_verdict,
    require_language,
)

# The interpreter :data:`INTERPRETER_PLACEHOLDER` resolves to: this process's own, so a sandboxed
# Python program runs on the same interpreter as the toolkit.
PYTHON_INTERPRETER = sys.executable or "python"

# Post-kill wait bound, for a child the group kill cannot reap at once.
KILL_DRAIN_TIMEOUT = 10.0
# RLIMIT_CPU headroom over the wall-clock timeout, so SIGXCPU only fires as the backstop.
RLIMIT_CPU_SLACK_SECONDS = 1.0

# The exit code a run is booked with when the program tampered with its working directory (a staged
# entry or the directory itself replaced): its own runtime error, never an infra fault that would void
# the episode it belongs to.
TAMPERED_WORKDIR_RETURNCODE = 1

# What a session's staged build was made from: language, source text, auxiliary file contents.
_BuildKey = tuple[str, str, tuple[tuple[str, str], ...]]
# Directory entries by name and file type (``lstat``, so a link is a link, not its target).
_EntryKinds = set[tuple[str, int]]


class SessionPathError(ValueError):
    """A session path the host must not touch: an entry the program replaced with a link or a
    non-regular file, or one whose resolution leaves the working directory."""


def _safe_member_name(name: str) -> bool:
    """Reject auxiliary-file names that would escape the sandbox working directory."""
    return name not in ("", ".", "..") and not name.startswith(("/", "\\")) and ".." not in name.split("/")


def _open_member(workdir: str, name: str, flags: int) -> int:
    """Open ``<workdir>/<name>`` for the host process without following a link at any component.

    The program owns the working directory between runs and can replace any entry with a symlink
    (``main.py -> /root/.aws/credentials``); the host — staging the next run's source, reading a file
    for the model — would otherwise write or read through it. Refused (:class:`SessionPathError`) when
    the path does not resolve to itself under the real working directory; ``O_NOFOLLOW`` closes the
    window on the final component between that check and the open. A FIFO or device in its place is
    refused too: ``O_NONBLOCK`` keeps the open from blocking the host on it, and only a regular file
    passes the ``fstat``.
    """
    dest = os.path.normpath(os.path.join(os.path.realpath(workdir), name))
    if os.path.realpath(dest) != dest:
        raise SessionPathError(f"session path {name!r} is a link or resolves outside the working directory")
    if flags & os.O_CREAT:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        fd = os.open(dest, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o644)
    except OSError as exc:
        # ELOOP: a link raced in at the final component; EISDIR: a write met a directory in its place;
        # ENXIO: a write met a FIFO with no reader.
        if exc.errno in (errno.ELOOP, errno.EISDIR, errno.ENXIO):
            raise SessionPathError(f"session path {name!r} is not a regular file") from exc
        raise
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise SessionPathError(f"session path {name!r} is not a regular file")
    return fd


def _captured_text(capture) -> str:
    """A run's captured output stream as text. Decoded with replacement: a program that emits bytes
    that are not UTF-8 (C++ undefined behavior, a binary dump) is judged on the replaced text, never
    lost to a decode error."""
    capture.seek(0)
    return capture.read(LOCAL_FSIZE_LIMIT).decode("utf-8", errors="replace")


def _entry_kinds(workdir: str) -> _EntryKinds:
    """The working directory's top-level entries with their file types."""
    kinds: _EntryKinds = set()
    for name in os.listdir(workdir):
        with contextlib.suppress(FileNotFoundError):
            kinds.add((name, stat.S_IFMT(os.lstat(os.path.join(workdir, name)).st_mode)))
    return kinds


def _build_key(spec: LanguageSpec, code: str, files: dict[str, str] | None) -> _BuildKey:
    """Identity of a compiled program's inputs; runs with equal keys share one build."""
    return (spec.name, code, tuple(sorted((files or {}).items())))


class LocalSubprocessSandbox(SandboxExecutor):
    """Run code in a child process under POSIX resource limits, in a managed working directory.

    Python runs with ``-s -E``; C/C++ compile with ``g++``/``gcc``. An execution is three steps —
    :meth:`_stage_sources`, :meth:`_compile` (compiled languages), :meth:`_run_program` — which
    :class:`LocalSession` sequences so a session can reuse a build. The run step is bounded by CPU
    time, address space, and file size; the compile step gets a larger time/memory budget.

    Stateless aside from config, with a per-execution working dir and per-call limits, so one instance
    is safe across threads / Ray actors. Limits use a ``ulimit`` shell wrapper (:meth:`_limit_wrap`)
    not a ``preexec_fn``, so launches use ``vfork``/``exec`` and stay fast under a large parent. The
    :meth:`_wrap_command` hook lets subclasses interpose an isolation wrapper.
    """

    def __init__(
        self,
        memory_limit_mb: int = SANDBOX_DEFAULT_MEMORY_MB,
        compile_timeout: float = SANDBOX_DEFAULT_COMPILE_TIMEOUT,
        compile_memory_limit_mb: int = SANDBOX_DEFAULT_COMPILE_MEMORY_MB,
    ):
        self.memory_limit_mb = memory_limit_mb
        self.compile_timeout = compile_timeout
        self.compile_memory_limit_mb = compile_memory_limit_mb

    def open_session(self) -> "LocalSession":
        """Open a persistent session backed by a fresh temp working directory."""
        return LocalSession(self._new_workdir(), self)

    @staticmethod
    def _new_workdir() -> str:
        """Create a fresh throwaway working directory for a session/execution."""
        return tempfile.mkdtemp(prefix="halo_sandbox_")

    def _wrap_command(self, argv: list[str], workdir: str, *, allow_network: bool) -> list[str]:
        """Wrap a child command with an isolation launcher. Identity here; overridden by bubblewrap."""
        return argv

    @staticmethod
    def _limit_wrap(argv: list[str], cpu_seconds: int, memory_mb: int, nproc: int | None = None) -> list[str]:
        """Wrap ``argv`` in a shell that applies per-run RLIMITs via ``ulimit`` then ``exec``s it.

        Used instead of a ``preexec_fn``, which forces CPython down the ``fork`` path and copies the
        parent's page tables on every execution. ``vfork`` + ``exec`` keeps launch cost flat as
        resident memory grows; the kernel carries the RLIMITs across ``exec`` and into a bwrap jail.
        Bounds are per-call, so concurrent executions do not share them. ``nproc`` (run step only)
        caps process/thread count so a fork bomb cannot outrun the timeout's process-group kill; the
        compile step omits it, since the compiler's fork tree is trusted.
        """
        # bash ulimit units (outside POSIX mode): -t seconds (CPU), -f and -v KiB, -u processes.
        limits = [f"ulimit -t {cpu_seconds}", f"ulimit -f {LOCAL_FSIZE_LIMIT // 1024}"]
        if memory_mb:
            limits.append(f"ulimit -v {memory_mb * 1024}")
        if nproc:
            limits.append(f"ulimit -u {nproc}")
        script = "; ".join(limits) + '; exec "$@"'
        return ["/bin/bash", "-c", script, "halo-sandbox", *argv]

    @staticmethod
    def _run_in_new_session(
        argv: list[str], *, stdin: str, timeout: float, cwd: str, env: dict[str, str]
    ) -> tuple[str, str, int | None, bool]:
        """Run ``argv`` in its own session; returns ``(stdout, stderr, returncode, timed_out)``.

        ``start_new_session`` puts the child in a fresh process group, so a timeout SIGKILLs the whole
        group; killing only the child would leave forked grandchildren running. Output is captured in
        temp files rather than pipes, so the child's ``RLIMIT_FSIZE`` bounds it: an output flood ends
        as the program's own failure at the file-size limit, never as host memory the grader runs out
        of.
        """
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            with subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=out,
                stderr=err,
                cwd=cwd,
                env=env,
                start_new_session=True,
            ) as proc:
                try:
                    proc.communicate(input=stdin.encode("utf-8"), timeout=timeout)
                    timed_out = False
                except subprocess.TimeoutExpired:
                    # The group may already be gone (leader exited between timeout and kill).
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGKILL)
                    try:
                        proc.wait(timeout=KILL_DRAIN_TIMEOUT)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    timed_out = True
            return _captured_text(out), _captured_text(err), proc.returncode, timed_out

    @staticmethod
    def _child_env(workdir: str) -> dict[str, str]:
        """Minimal environment: PATH, a HOME inside the sandbox, no inherited proxy/secrets."""
        return {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": workdir,
            "TMPDIR": workdir,
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        }

    def _stage_sources(
        self, workdir: str, spec: LanguageSpec, code: str, files: dict[str, str] | None
    ) -> SandboxResult | None:
        """Write ``files`` and the source into ``workdir``. Returns None, an error result for an unsafe
        path (the caller's fault), or a runtime-error verdict when the program replaced a staged entry
        with a link (its own fault, so never an infra error it could void its episode with)."""
        files = files or {}
        for name in files:
            if not _safe_member_name(name):
                return SandboxResult(error=f"unsafe auxiliary file path: {name!r}")
        try:
            for name, content in files.items():
                self._write_member(workdir, name, content)
            self._write_member(workdir, spec.source_name, code)
        except SessionPathError as exc:
            return SandboxResult(stderr=f"working directory tampered: {exc}", returncode=TAMPERED_WORKDIR_RETURNCODE)
        return None

    def _compile(self, workdir: str, spec: LanguageSpec, *, allow_network: bool) -> SandboxResult | None:
        """Build a compiled language's staged source. Returns None on success, a failure result otherwise.

        A non-zero compiler exit or a build past ``compile_timeout`` is the source's fault
        (``compile_failed``, ``returncode``/``stderr``, ``error`` unset); a missing compiler is a
        backend failure (``error`` set).
        """
        compile_argv = self._wrap_command(list(spec.compile_argv), workdir, allow_network=allow_network)
        compile_argv = self._limit_wrap(
            compile_argv,
            int(math.ceil(self.compile_timeout + RLIMIT_CPU_SLACK_SECONDS)),
            self.compile_memory_limit_mb,
        )
        with SANDBOX_EXECUTION_GATE.slot():
            stdout, stderr, returncode, timed_out = self._run_in_new_session(
                compile_argv, stdin="", timeout=self.compile_timeout, cwd=workdir, env=self._child_env(workdir)
            )
        if timed_out:
            return compile_limit_verdict(f"compilation exceeded {self.compile_timeout:g}s")
        # 127 = the wrapper shell could not exec the compiler: a backend failure, not a bad-source verdict.
        if returncode == 127:
            return SandboxResult(error=f"compiler not found: {spec.compile_argv[0]!r}", stderr=stderr.strip())
        if returncode != 0:
            # gcc/g++ emit diagnostics on stderr; fall back to stdout if a toolchain uses it.
            diagnostics = stderr.strip() or stdout.strip() or "compilation failed"
            return SandboxResult(stderr=diagnostics, returncode=returncode, compile_failed=True)
        return None

    def _run_program(
        self, workdir: str, spec: LanguageSpec, *, stdin: str, timeout: float, allow_network: bool
    ) -> SandboxResult:
        """Run the staged (and built) program in ``workdir`` under the run-step limits."""
        run_argv = [PYTHON_INTERPRETER if tok == INTERPRETER_PLACEHOLDER else tok for tok in spec.run_argv]
        run_argv = self._wrap_command(run_argv, workdir, allow_network=allow_network)
        # RLIMIT_CPU backstop: SIGXCPU still kills a busy loop if timeout delivery lags.
        run_argv = self._limit_wrap(
            run_argv,
            int(math.ceil(timeout + RLIMIT_CPU_SLACK_SECONDS)),
            self.memory_limit_mb,
            nproc=LOCAL_NPROC_LIMIT,
        )
        # The slot is held for the whole run, so ``timeout`` measures near-dedicated-core time.
        with SANDBOX_EXECUTION_GATE.slot():
            stdout, stderr, returncode, timed_out = self._run_in_new_session(
                run_argv, stdin=stdin, timeout=timeout, cwd=workdir, env=self._child_env(workdir)
            )
        # SIGXCPU (the backstop) is reported as a timeout so callers bucket it as TLE.
        return SandboxResult(
            stdout=stdout or "",
            stderr=stderr or "",
            returncode=None if timed_out else returncode,
            timed_out=timed_out or returncode == -signal.SIGXCPU,
        )

    @staticmethod
    def _write_member(workdir: str, name: str, content: str) -> None:
        with os.fdopen(_open_member(workdir, name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC), "w") as fh:
            fh.write(content)


class LocalSession(SandboxSession):
    """Persistent working directory for a local backend, reused across :meth:`run` calls.

    Source and compiled artifacts written into ``workdir`` survive between turns. A compiled program
    is built once and rerun from its binary while language, source and ``files`` stay the same.

    A removed ``workdir`` is recreated empty on the next use. One the program replaced with a link or
    a file has broken the session (:class:`SandboxAgentFault`): every later run reports it instead of
    staging and the file operations raise it, since any path under it could resolve to a host file.
    """

    def __init__(self, workdir: str, executor: LocalSubprocessSandbox, *, allow_network: bool = False):
        self.workdir = workdir
        self._executor = executor
        self._allow_network = allow_network
        # The compiled program staged in ``workdir`` and its compile verdict (None = built, runnable).
        self._build: tuple[_BuildKey, SandboxResult | None] | None = None
        # Directory entries (name and file type) present once the program was staged and built: what
        # :meth:`reset_to_staged` keeps. Typed so an entry the program swapped for a link is dropped.
        self._staged_entries: _EntryKinds | None = None

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
        result = self._executor._run_program(
            self.workdir, spec, stdin=stdin, timeout=timeout, allow_network=self._allow_network
        )
        return self._ensure_workspace() or result

    def _ensure_workspace(self) -> SandboxResult | None:
        """None once the working directory is usable — recreated empty, with nothing staged, when the
        program removed it — or the agent fault when the program put a link or file at its path."""
        try:
            mode = os.lstat(self.workdir).st_mode
        except FileNotFoundError:
            os.makedirs(self.workdir, mode=0o700, exist_ok=True)
            self._build = None
            self._staged_entries = None
            return None
        if stat.S_ISDIR(mode):
            return None
        message = "the program replaced its working directory"
        return SandboxResult(stderr=message, returncode=TAMPERED_WORKDIR_RETURNCODE, agent_fault=message)

    def _require_intact(self) -> None:
        """Raise :class:`SandboxAgentFault` when the program replaced the working directory."""
        broken = self._ensure_workspace()
        if broken is not None:
            raise SandboxAgentFault(broken.agent_fault)

    def _prepare(self, spec: LanguageSpec, code: str, files: dict[str, str] | None) -> SandboxResult | None:
        """Stage ``code`` + ``files`` and build them when ``spec`` compiles; None once the program is runnable.

        One build slot per session (every registered compile writes ``./main``): a compiled program is
        rebuilt only when language, source or ``files`` differ from the staged build, and its compile
        verdict — built, or the failure result — is reused until then. Any other write through the
        session (an interpreted run, :meth:`write_file`) drops the slot, since it may have changed an
        included header.
        """
        key = _build_key(spec, code, files)
        if spec.is_compiled and self._build is not None and self._build[0] == key:
            return self._build[1]
        self._build = None
        staged = self._executor._stage_sources(self.workdir, spec, code, files)
        if staged is not None:
            return staged
        if spec.is_compiled:
            failure = self._executor._compile(self.workdir, spec, allow_network=self._allow_network)
            self._build = (key, failure)
        else:
            failure = None
        self._staged_entries = _entry_kinds(self.workdir)
        return failure

    def reset_to_staged(self) -> None:
        # A replaced working directory lists whatever it now points at; the next run reports the fault.
        if self._ensure_workspace() is not None or self._staged_entries is None:
            return
        for name, kind in _entry_kinds(self.workdir) - self._staged_entries:
            path = os.path.join(self.workdir, name)
            if kind == stat.S_IFDIR:
                shutil.rmtree(path, ignore_errors=True)
            else:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(path)

    def write_file(self, path: str, content: str) -> None:
        if not _safe_member_name(path):
            raise ValueError(f"unsafe session file path: {path!r}")
        self._require_intact()
        self._build = None
        self._staged_entries = None
        LocalSubprocessSandbox._write_member(self.workdir, path, content)

    def read_file(self, path: str) -> str | None:
        """The file's text, or ``None`` when there is no regular file at ``path`` — a link the program
        planted included, so a host file never reaches the trajectory through it."""
        if not _safe_member_name(path):
            raise ValueError(f"unsafe session file path: {path!r}")
        self._require_intact()
        try:
            fd = _open_member(self.workdir, path, os.O_RDONLY)
        except (SessionPathError, FileNotFoundError, NotADirectoryError):
            return None
        with os.fdopen(fd) as fh:
            return fh.read()

    def list_files(self) -> list[str]:
        self._require_intact()
        out: list[str] = []
        for root, _dirs, names in os.walk(self.workdir):
            for n in names:
                out.append(os.path.relpath(os.path.join(root, n), self.workdir))
        return sorted(out)

    def close(self) -> None:
        try:
            mode = os.lstat(self.workdir).st_mode
        except FileNotFoundError:
            return
        if stat.S_ISDIR(mode):
            shutil.rmtree(self.workdir, ignore_errors=True)
        else:
            # A link or file the program put in the directory's place: rmtree refuses a link.
            os.unlink(self.workdir)
