"""Where a sandboxed program runs: the executor every backend implements, and the session that keeps one working
directory across runs (one per episode keeps concurrent rollouts apart)."""

from abc import ABC, abstractmethod

from halo.environments.sandbox.results import SandboxResult

# Per-execution wall-clock cap (seconds); large enough that a slower but correct CPython solution passes.
SANDBOX_DEFAULT_TIMEOUT = 15.0


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

    @abstractmethod
    def write_file(self, path: str, content: str) -> None:
        """Create or overwrite a file in the session (path relative to the working directory)."""

    @abstractmethod
    def read_file(self, path: str) -> str | None:
        """Read a session file's text, or return None if it does not exist."""

    @abstractmethod
    def list_files(self) -> list[str]:
        """List relative paths of files currently in the session."""

    def reset_to_prepared(self) -> None:  # noqa: B027 — optional hook; a stateless backend keeps nothing to drop
        """Drop whatever the last :meth:`run` produced in the working directory, keeping what was prepared
        for it (its source and build), so consecutive runs of one program start from the same state (a grader running
        hidden tests through one session). A backend whose service starts every run fresh has nothing
        to drop."""

    def close(self) -> None:  # noqa: B027 — optional hook; a base session holds no resources to free
        """Release session resources (e.g. delete the working directory). Idempotent."""

    @staticmethod
    def is_safe_member_name(name: str) -> bool:
        """Whether ``name`` stays inside a working directory: relative, naming an entry, never ``..``."""
        return name not in ("", ".", "..") and not name.startswith(("/", "\\")) and ".." not in name.split("/")

    @staticmethod
    def reject_unsafe_path(path: str) -> None:
        """Refuse a session file path no backend may take with ``ValueError``, a priced tool error: one
        UTF-8 cannot encode (a lone surrogate from model-written JSON) or one that leaves the working
        directory. A backend that failed on it instead would read as the backend's fault. Refused rather
        than renamed, so the program never finds its file under a name it did not write."""
        try:
            path.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError(f"session file path {path!r} is not valid UTF-8") from None
        if not SandboxSession.is_safe_member_name(path):
            raise ValueError(f"unsafe session file path: {path!r}")

    def __enter__(self) -> "SandboxSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class SandboxExecutor(ABC):
    """Runs a complete program against stdin and returns a uniform :class:`SandboxResult`.

    Backends differ only in isolation/locality; they share this interface and the language registry
    (:mod:`~halo.environments.sandbox.languages`), so multi-language and session semantics are identical.
    """

    # Whether a program is kept from writing the host's filesystem and from its network; it may still
    # read what the backend exposes (bubblewrap's read-only system paths and ``extra_ro_binds``). Only
    # a backend that confines it so declares True: a coding environment warns on every other one.
    isolated: bool = False
    # Whether a compiled submission is built before, and apart from, every test's stdin, and no program
    # can make a later build read what an earlier test left behind, so a compiler's message can quote
    # only the submission. Only a backend that builds so declares True: grading shows that message
    # under ``verdict_detail: outcome`` there alone.
    COMPILES_WITHOUT_TEST_INPUT: bool = False

    @abstractmethod
    def open_session(self) -> SandboxSession:
        """Open a persistent :class:`SandboxSession` (one per episode for multi-turn rollouts)."""

    def toolchain(self, language: str) -> str | None:
        """How this backend builds or runs a ``language`` program, as the predicate a tool description
        states (``is compiled with g++ -O2 -pipe -std=c++17``, ``runs on CPython 3.12``); ``None`` where the
        backend does not say, as a service answering for its own toolchain."""
        return None

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
