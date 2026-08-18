"""Core sandbox abstractions: results, executor/session interfaces, and the language registry.

Separates *what* runs (:class:`LanguageSpec`) from *where* it runs (:class:`SandboxExecutor`).
One-shot :meth:`SandboxExecutor.run` uses a throwaway working dir; a :class:`SandboxSession` persists
its working dir across calls (one per episode isolates concurrent rollouts).
"""

import os
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from src.env import env_int

# Per-execution wall-clock cap (seconds); large enough that a slower but correct CPython solution passes.
SANDBOX_DEFAULT_TIMEOUT = 15.0
SANDBOX_DEFAULT_MEMORY_MB = 1024
# Build-step wall-clock cap (seconds); bounds an #include bomb in untrusted source.
SANDBOX_DEFAULT_COMPILE_TIMEOUT = 30.0
SANDBOX_DEFAULT_COMPILE_MEMORY_MB = 2048
# Largest file a local-backend child may write (bytes); bounds FS disk-fill and binary size.
LOCAL_FSIZE_LIMIT = 64 * 1024 * 1024
# RLIMIT_NPROC for the run step. The kernel ignores it for root (how the training containers run), so
# it binds only unprivileged runs such as a bubblewrap jail; the timeout's process-group kill remains
# the primary fork-bomb defense. It counts the user's entire task set, so it must clear the
# trainer + Ray baseline.
LOCAL_NPROC_LIMIT = 4096

INTERPRETER_PLACEHOLDER = "$INTERPRETER"

# Shared REPL observation for a run that produced no output, so every backend uses the same wording.
REPL_NO_OUTPUT_MESSAGE = "Code executed successfully (no output)"


def repl_timeout_message(timeout: float) -> str:
    """The REPL observation for a run killed by its wall-clock ``timeout``."""
    return f"Error: execution exceeded {timeout:g}s timeout"


def _resolve_execution_slots() -> int:
    """Max concurrent sandboxed executions; host CPU count by default, ``HALO_SANDBOX_MAX_CONCURRENCY``
    overrides (parsed via :func:`env_int` — a malformed value warns and falls back to the default)."""
    override = env_int("HALO_SANDBOX_MAX_CONCURRENCY", None)
    if override is not None:
        return max(1, override)
    return max(1, os.cpu_count() or 1)


class ExecutionGate:
    """Fixed pool of execution slots; caps host oversubscription so a per-test wall-clock limit
    measures near-dedicated-core time. Thread-safe; a run acquires its slot before its timeout starts.
    """

    def __init__(self, slots: int):
        self._semaphore = threading.BoundedSemaphore(max(1, slots))

    @contextmanager
    def slot(self) -> Iterator[None]:
        """Hold one execution slot for the ``with`` block, queueing if saturated."""
        self._semaphore.acquire()
        try:
            yield
        finally:
            self._semaphore.release()


# Process-global so total concurrency stays at the host core budget across all env instances.
SANDBOX_EXECUTION_GATE = ExecutionGate(_resolve_execution_slots())


class SandboxInfraError(RuntimeError):
    """A sandbox run was lost to a backend/transport failure (remote HTTP error, missing compiler).

    Distinct from the program's own non-zero exit, compile error, or timeout; those are verdicts on
    the submitted code and stay ordinary string results. The REPL layer raises this so an
    infrastructure outage surfaces as a failed tool call rather than a string the protocol would
    score as successful.
    """


@dataclass
class SandboxResult:
    """Outcome of one program execution, uniform across backends."""

    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    timed_out: bool = False
    # Backend/transport failure, distinct from the program's own non-zero exit or a compile error.
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True only when the program ran to completion with a zero exit code."""
        return self.returncode == 0 and not self.timed_out and self.error is None


@dataclass(frozen=True)
class LanguageSpec:
    """How a language's source is named, compiled, and launched inside a sandbox working dir.

    ``compile_argv`` is ``None`` for interpreted languages. In ``run_argv``, the
    :data:`INTERPRETER_PLACEHOLDER` token (if present) is replaced with the backend's interpreter.
    """

    name: str
    source_name: str
    run_argv: tuple[str, ...]
    compile_argv: tuple[str, ...] | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_compiled(self) -> bool:
        return self.compile_argv is not None


LANGUAGES: dict[str, LanguageSpec] = {
    "python": LanguageSpec(
        name="python",
        source_name="main.py",
        # -s -E (not -I) drops user site-packages / PYTHON* env but keeps the script's dir on sys.path
        # so a session can import workspace modules written on earlier turns.
        run_argv=(INTERPRETER_PLACEHOLDER, "-s", "-E", "main.py"),
        aliases=("py", "python3"),
    ),
    "cpp": LanguageSpec(
        name="cpp",
        source_name="main.cpp",
        compile_argv=("g++", "-O2", "-pipe", "-std=c++17", "-o", "main", "main.cpp"),
        run_argv=("./main",),
        aliases=("c++", "cxx", "cc"),
    ),
    "c": LanguageSpec(
        name="c",
        source_name="main.c",
        compile_argv=("gcc", "-O2", "-pipe", "-std=c11", "-o", "main", "main.c"),
        run_argv=("./main",),
        aliases=(),
    ),
}


def resolve_language(language: str) -> LanguageSpec | None:
    """Look up a :class:`LanguageSpec` by name or alias (case-insensitive); None if unknown."""
    key = (language or "").strip().lower()
    for spec in LANGUAGES.values():
        if key == spec.name or key in spec.aliases:
            return spec
    return None


def supported_languages() -> list[str]:
    """Canonical names of all registered languages."""
    return list(LANGUAGES.keys())


def require_language(language: str) -> LanguageSpec:
    """:func:`resolve_language`, raising ``ValueError`` instead of returning None for an unknown name."""
    spec = resolve_language(language)
    if spec is None:
        raise ValueError(f"unsupported language {language!r}; supported: {', '.join(supported_languages())}")
    return spec


class SandboxSession(ABC):
    """Persistent execution context: a working directory that survives across :meth:`run` calls.

    Files written on one turn are visible on the next (multi-turn rollouts). One session per episode
    keeps concurrent rollouts isolated. Use as a context manager for cleanup.
    """

    @abstractmethod
    def run(
        self,
        code: str,
        *,
        stdin: str = "",
        timeout: float = SANDBOX_DEFAULT_TIMEOUT,
        language: str = "python",
        files: dict[str, str] | None = None,
    ) -> SandboxResult:
        """Compile (if needed) and execute ``code`` in this session's persistent working directory."""
        raise NotImplementedError

    @abstractmethod
    def write_file(self, path: str, content: str) -> None:
        """Create or overwrite a file in the session (path relative to the working directory)."""
        raise NotImplementedError

    @abstractmethod
    def read_file(self, path: str) -> str | None:
        """Read a session file's text, or return None if it does not exist."""
        raise NotImplementedError

    @abstractmethod
    def list_files(self) -> list[str]:
        """List relative paths of files currently in the session."""
        raise NotImplementedError

    def close(self) -> None:  # noqa: B027 — optional hook; a base session holds no resources to free
        """Release session resources (e.g. delete the working directory). Idempotent."""

    def __enter__(self) -> "SandboxSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class SandboxExecutor(ABC):
    """Runs a complete program against stdin and returns a uniform :class:`SandboxResult`.

    Backends differ only in isolation/locality; they share this interface and the
    :class:`LanguageSpec` registry, so multi-language and session semantics are identical.
    """

    @abstractmethod
    def open_session(self) -> SandboxSession:
        """Open a persistent :class:`SandboxSession` (one per episode for multi-turn rollouts)."""
        raise NotImplementedError

    def run(
        self,
        code: str,
        *,
        stdin: str = "",
        timeout: float = SANDBOX_DEFAULT_TIMEOUT,
        language: str = "python",
        files: dict[str, str] | None = None,
    ) -> SandboxResult:
        """One-shot execution: run ``code`` in a throwaway session and discard it (stateless grading)."""
        with self.open_session() as session:
            return session.run(code, stdin=stdin, timeout=timeout, language=language, files=files)
