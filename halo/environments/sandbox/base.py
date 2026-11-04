"""Where a sandboxed program runs: the executor every backend implements, and the session keeping one working directory
across runs."""

from abc import ABC, abstractmethod

from halo.environments.sandbox.results import SandboxResult

# Per-execution wall-clock cap (seconds); large enough that a slower but correct CPython solution passes.
SANDBOX_DEFAULT_TIMEOUT = 15.0


class SandboxSession(ABC):
    """A working directory surviving across :meth:`run` calls. One per episode keeps concurrent rollouts isolated."""

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
        """Compile (if needed) and execute ``code`` in the persistent working directory."""

    @abstractmethod
    def write_file(self, path: str, content: str) -> None:
        """Create or overwrite a file; ``path`` is relative to the working directory."""

    @abstractmethod
    def read_file(self, path: str) -> str | None:
        """``None`` if the file does not exist."""

    @abstractmethod
    def list_files(self) -> list[str]:
        """Relative paths of the files in the session."""

    def reset_to_prepared(self) -> None:  # noqa: B027 — optional hook; a stateless backend keeps nothing to drop
        """Drop what the last :meth:`run` produced, keeping its source and build, so consecutive runs of one program
        start from the same state. A backend whose service starts every run fresh has nothing to drop."""

    def close(self) -> None:  # noqa: B027 — optional hook; a base session holds no resources to free
        """Release session resources. Idempotent."""

    @staticmethod
    def is_safe_member_name(name: str) -> bool:
        """Relative, naming an entry, never ``..``."""
        return name not in ("", ".", "..") and not name.startswith(("/", "\\")) and ".." not in name.split("/")

    @staticmethod
    def reject_unsafe_path(path: str) -> None:
        """Refuse with ``ValueError``, a priced tool error: a path UTF-8 cannot encode (a lone surrogate in model JSON)
        or one leaving the working directory. A backend failing on it would read as its own fault."""
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
    """Runs a complete program against stdin. Backends differ only in isolation and locality; they share this interface
    and the language registry."""

    # Whether a program is kept from writing the host's filesystem and from its network; it may still read what the
    # backend exposes. A coding environment warns on every backend that does not declare True.
    isolated: bool = False
    # Whether a compiled submission is built apart from every test's stdin, with no later build reading what an earlier
    # test left, so a compiler's message can quote only the submission and grading shows it under ``outcome``.
    COMPILES_WITHOUT_TEST_INPUT: bool = False

    @abstractmethod
    def open_session(self) -> SandboxSession:
        """One per episode for multi-turn rollouts."""

    def toolchain(self, language: str) -> str | None:
        """The predicate a tool description states (``runs on CPython 3.12``, ``is compiled with g++ -O2 ...``);
        ``None`` where the backend does not say."""
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
        """Runs in a throwaway session (stateless grading)."""
        with self.open_session() as session:
            return session.run(code, stdin=stdin, timeout=timeout, language=language, files=files)
