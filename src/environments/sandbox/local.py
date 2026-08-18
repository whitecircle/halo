"""Local subprocess sandbox: runs code in a child process under POSIX resource limits.

Bounds CPU time, address space, and file size, runs Python with the parent environment stripped, and
compiles C/C++ before running. It does **not** namespace-isolate the network or wider filesystem; for
that use :class:`~src.environments.sandbox.bubblewrap.BubblewrapSandbox` or the remote backend.
"""

import contextlib
import math
import os
import shutil
import signal
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
    SandboxExecutor,
    SandboxResult,
    SandboxSession,
    require_language,
)

# The interpreter :data:`INTERPRETER_PLACEHOLDER` resolves to: this process's own, so a sandboxed
# Python program runs on the same interpreter as the toolkit.
PYTHON_INTERPRETER = sys.executable or "python"

# Post-kill drain bound: a setsid-escaped child holds the pipe open, hanging an unbounded communicate().
KILL_DRAIN_TIMEOUT = 10.0
# RLIMIT_CPU headroom over the wall-clock timeout, so SIGXCPU only fires as the backstop.
RLIMIT_CPU_SLACK_SECONDS = 1.0


def _safe_member_name(name: str) -> bool:
    """Reject auxiliary-file names that would escape the sandbox working directory."""
    return name not in ("", ".", "..") and not name.startswith(("/", "\\")) and ".." not in name.split("/")


class LocalSubprocessSandbox(SandboxExecutor):
    """Run code in a child process under POSIX resource limits, in a managed working directory.

    Python runs with ``-s -E``; C/C++ compile with ``g++``/``gcc``. The run step is bounded by CPU
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
        # ulimit units: -t seconds (CPU), -f 512-byte blocks (file size), -v KiB (address space), -u processes.
        limits = [f"ulimit -t {cpu_seconds}", f"ulimit -f {LOCAL_FSIZE_LIMIT // 512}"]
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
        group; killing only the child would leave forked grandchildren running.
        """
        with subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env=env,
            start_new_session=True,
        ) as proc:
            try:
                stdout, stderr = proc.communicate(input=stdin, timeout=timeout)
                return stdout, stderr, proc.returncode, False
            except subprocess.TimeoutExpired:
                # The group may already be gone (leader exited between timeout and kill).
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                try:
                    stdout, stderr = proc.communicate(timeout=KILL_DRAIN_TIMEOUT)
                except subprocess.TimeoutExpired as e:
                    proc.kill()
                    stdout, stderr = e.stdout or "", e.stderr or ""
                return stdout, stderr, proc.returncode, True

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

    def _exec_in_workdir(
        self,
        workdir: str,
        code: str,
        *,
        stdin: str,
        timeout: float,
        language: str,
        files: dict[str, str] | None,
        allow_network: bool,
    ) -> SandboxResult:
        """Compile (if needed) and run ``code`` inside an already-prepared ``workdir``."""
        try:
            spec = require_language(language)
        except ValueError as exc:
            return SandboxResult(error=str(exc))

        for name, content in (files or {}).items():
            if not _safe_member_name(name):
                return SandboxResult(error=f"unsafe auxiliary file path: {name!r}")
            self._write_member(workdir, name, content)

        self._write_member(workdir, spec.source_name, code)

        # One slot for compile+run, so the timeout below measures near-dedicated-core time.
        with SANDBOX_EXECUTION_GATE.slot():
            if spec.is_compiled:
                compile_failure = self._compile(workdir, spec, allow_network=allow_network)
                if compile_failure is not None:
                    return compile_failure

            run_argv = [PYTHON_INTERPRETER if tok == INTERPRETER_PLACEHOLDER else tok for tok in spec.run_argv]
            run_argv = self._wrap_command(run_argv, workdir, allow_network=allow_network)

            # RLIMIT_CPU backstop: SIGXCPU still kills a busy loop if timeout delivery lags.
            run_argv = self._limit_wrap(
                run_argv,
                int(math.ceil(timeout + RLIMIT_CPU_SLACK_SECONDS)),
                self.memory_limit_mb,
                nproc=LOCAL_NPROC_LIMIT,
            )
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

    def _compile(self, workdir: str, spec: LanguageSpec, *, allow_network: bool) -> SandboxResult | None:
        """Build a compiled language's source. Returns None on success, a failure result otherwise.

        A non-zero compiler exit is the source's fault (``returncode``/``stderr``, ``error`` unset);
        a missing compiler or compile timeout is a backend/limit failure (``error`` set).
        """
        compile_argv = self._wrap_command(list(spec.compile_argv), workdir, allow_network=allow_network)
        compile_argv = self._limit_wrap(
            compile_argv,
            int(math.ceil(self.compile_timeout + RLIMIT_CPU_SLACK_SECONDS)),
            self.compile_memory_limit_mb,
        )
        stdout, stderr, returncode, timed_out = self._run_in_new_session(
            compile_argv, stdin="", timeout=self.compile_timeout, cwd=workdir, env=self._child_env(workdir)
        )
        if timed_out:
            return SandboxResult(
                stderr=f"compilation exceeded {self.compile_timeout:g}s",
                error=f"compilation exceeded {self.compile_timeout:g}s",
            )
        # 127 = the wrapper shell could not exec the compiler: a backend failure, not a bad-source verdict.
        if returncode == 127:
            return SandboxResult(error=f"compiler not found: {spec.compile_argv[0]!r}", stderr=stderr.strip())
        if returncode != 0:
            # gcc/g++ emit diagnostics on stderr; fall back to stdout if a toolchain uses it.
            diagnostics = stderr.strip() or stdout.strip() or "compilation failed"
            return SandboxResult(stderr=diagnostics, returncode=returncode)
        return None

    @staticmethod
    def _write_member(workdir: str, name: str, content: str) -> None:
        dest = os.path.join(workdir, name)
        os.makedirs(os.path.dirname(dest) or workdir, exist_ok=True)
        with open(dest, "w") as fh:
            fh.write(content)


class LocalSession(SandboxSession):
    """Persistent working directory for a local backend, reused across :meth:`run` calls.

    Source and compiled artifacts written into ``workdir`` survive between turns.
    """

    def __init__(self, workdir: str, executor: LocalSubprocessSandbox, *, allow_network: bool = False):
        self.workdir = workdir
        self._executor = executor
        self._allow_network = allow_network

    def run(
        self,
        code: str,
        *,
        stdin: str = "",
        timeout: float = SANDBOX_DEFAULT_TIMEOUT,
        language: str = "python",
        files: dict[str, str] | None = None,
    ) -> SandboxResult:
        return self._executor._exec_in_workdir(
            self.workdir,
            code,
            stdin=stdin,
            timeout=timeout,
            language=language,
            files=files,
            allow_network=self._allow_network,
        )

    def write_file(self, path: str, content: str) -> None:
        if not _safe_member_name(path):
            raise ValueError(f"unsafe session file path: {path!r}")
        LocalSubprocessSandbox._write_member(self.workdir, path, content)

    def read_file(self, path: str) -> str | None:
        if not _safe_member_name(path):
            raise ValueError(f"unsafe session file path: {path!r}")
        full = os.path.join(self.workdir, path)
        if not os.path.isfile(full):
            return None
        with open(full) as fh:
            return fh.read()

    def list_files(self) -> list[str]:
        out: list[str] = []
        for root, _dirs, names in os.walk(self.workdir):
            for n in names:
                out.append(os.path.relpath(os.path.join(root, n), self.workdir))
        return sorted(out)

    def close(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)
