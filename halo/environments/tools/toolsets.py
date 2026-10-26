"""The tool sets an environment registers (math, code, search, files). The plain builders are stateless (Python runs
in the in-process sandbox unless a ``sandbox`` or another ``language`` asks for a real interpreter); the
``build_session_*`` ones bind the active episode's :class:`SandboxSession`, resolved per call, whose working dir
persists across turns."""

import functools
from collections.abc import Callable
from contextvars import ContextVar

from halo.environments.sandbox.base import SANDBOX_DEFAULT_TIMEOUT, SandboxExecutor, SandboxSession
from halo.environments.sandbox.factory import build_sandbox
from halo.environments.sandbox.inprocess import calculate_sandboxed, run_python_sandboxed
from halo.environments.sandbox.languages import require_language
from halo.environments.sandbox.repl import run_code_via_sandbox
from halo.environments.sandbox.results import SandboxInfraError
from halo.environments.tools.definitions import (
    NativeTool,
    NativeToolRegistry,
    ToolParameter,
)
from halo.environments.tools.web_search import (
    DEFAULT_MAX_RESULTS,
    async_web_search,
    validate_search_backend,
    web_search,
)

# What the file tools tell the model over each store: a simulated absolute-path filesystem, a persistent workspace.
_SIMULATED_FILE_TEXTS = {
    "read": "Read the contents of a file.",
    "write": "Write content to a file.",
    "list": "List files in a directory.",
    "read_path": "Path to the file to read",
    "write_path": "Path to the file to write",
    "list_dir": "Directory path to list files from",
}
_WORKSPACE_FILE_TEXTS = {
    "read": "Read the contents of a workspace file.",
    "write": "Write content to a file in the workspace (persists across turns).",
    "list": "List files in the workspace (optionally filtered by a path prefix).",
    "read_path": "Relative path of the file to read",
    "write_path": "Relative path of the file to write",
    "list_dir": "Path prefix to filter by",
}


def build_math_tools(timeout: float = SANDBOX_DEFAULT_TIMEOUT) -> NativeToolRegistry:
    """A safe-calculator tool, ``timeout`` bounding one evaluation: an expression the AST guard admits can still run
    indefinitely, and sync tool execution occupies a Ray actor's loop.
    """
    registry = NativeToolRegistry()
    registry.register(
        NativeTool(
            name="calculate",
            description="Evaluate a mathematical expression. Supports basic arithmetic (+, -, *, /, **, %), parentheses, and functions like sqrt, sin, cos, log, exp, floor, ceil, factorial, pi, e.",
            parameters=[
                ToolParameter(
                    name="expression",
                    type="string",
                    description="The mathematical expression to evaluate, e.g., '2 + 3 * 4', 'sqrt(16)', 'log(100, 10)'",
                ),
            ],
            handler=functools.partial(calculate_sandboxed, timeout=timeout),
        )
    )
    return registry


def build_code_tools(
    language: str = "python",
    timeout: float = SANDBOX_DEFAULT_TIMEOUT,
    sandbox: SandboxExecutor | None = None,
    tool_name: str | None = None,
) -> NativeToolRegistry:
    """A single-language code-execution tool named ``tool_name`` (default: the language's canonical name) for any
    ``language`` the sandbox registry holds, ``timeout`` seconds per run. ``sandbox`` runs it as a subprocess with
    imports and the stdlib (confined on ``bubblewrap`` / ``remote``); ``None`` keeps the in-process Python REPL, and a
    non-Python language takes the default backend.
    """
    spec = require_language(language)

    if spec.name == "python" and sandbox is None:
        handler = functools.partial(run_python_sandboxed, timeout=timeout)
        # The math names are the in-process REPL's injected builtins; in a real interpreter they are
        # not bare names, so this wording belongs to that handler, not to the language.
        description = (
            "Execute Python code. Can perform calculations, define variables, use loops, and "
            "print results. Common math functions are available: sqrt, sin, cos, log, exp, "
            "floor, ceil, factorial, pi, e."
        )
    else:
        handler = functools.partial(
            run_code_via_sandbox, runner=sandbox or build_sandbox(), timeout=timeout, language=spec.name
        )
        if spec.is_compiled:
            description = (
                f"Execute a complete {spec.name} program (compiled, then run). It is given no stdin and "
                "writes to stdout. Include a main() and any needed includes."
            )
        else:
            description = f"Execute a complete {spec.name} script. It is given no stdin and writes to stdout."

    registry = NativeToolRegistry()
    registry.register(
        NativeTool(
            name=tool_name or spec.name,
            description=description,
            parameters=[
                ToolParameter(
                    name="code",
                    type="string",
                    description="Source code to execute. Writes results to stdout/print; it is given no stdin.",
                ),
            ],
            handler=handler,
        )
    )
    return registry


def build_python_tools(
    timeout: float = SANDBOX_DEFAULT_TIMEOUT, sandbox: SandboxExecutor | None = None
) -> NativeToolRegistry:
    """A sandboxed Python REPL tool (named ``python``). See :func:`build_code_tools`."""
    return build_code_tools(language="python", timeout=timeout, sandbox=sandbox, tool_name="python")


def build_search_tools(backend: str | None = None) -> NativeToolRegistry:
    """A web_search tool; ``backend=None`` auto-selects by available API key (Serper, Brave, Tavily, then keyless
    DuckDuckGo), and ``mock`` also needs ``HALO_ALLOW_MOCK_SEARCH=1`` (its fabricated snippets score as a success). The
    name is validated here, so an environment never runs every episode on a backend it cannot select.
    """
    validate_search_backend(backend)
    registry = NativeToolRegistry()
    registry.register(
        NativeTool(
            name="web_search",
            description="Search the web for information on any topic. Returns titles, snippets, and URLs.",
            parameters=[
                ToolParameter(name="query", type="string", description="The search query"),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description=f"Maximum number of results (default {DEFAULT_MAX_RESULTS})",
                    required=False,
                ),
            ],
            handler=lambda query, max_results=DEFAULT_MAX_RESULTS: web_search(
                query, max_results=int(max_results), backend=backend
            ),
            async_handler=lambda query, max_results=DEFAULT_MAX_RESULTS: async_web_search(
                query, max_results=int(max_results), backend=backend
            ),
        )
    )
    return registry


def _file_tools(
    read_file: Callable[..., str],
    write_file: Callable[..., str],
    list_files: Callable[..., str],
    texts: dict[str, str],
) -> NativeToolRegistry:
    """The read/write/list file trio over any store, given its three handlers and its wording (``texts``): the
    model calls the same three tools with the same parameters whichever store backs them."""
    registry = NativeToolRegistry()
    registry.register(
        NativeTool(
            name="read_file",
            description=texts["read"],
            parameters=[ToolParameter("path", "string", texts["read_path"])],
            handler=read_file,
        )
    )
    registry.register(
        NativeTool(
            name="write_file",
            description=texts["write"],
            parameters=[
                ToolParameter("path", "string", texts["write_path"]),
                ToolParameter("content", "string", "Content to write to the file"),
            ],
            handler=write_file,
        )
    )
    registry.register(
        NativeTool(
            name="list_files",
            description=texts["list"],
            parameters=[ToolParameter("directory", "string", texts["list_dir"], required=False)],
            handler=list_files,
        )
    )
    return registry


def build_simulated_file_tools() -> NativeToolRegistry:
    """Simulated in-memory filesystem tools (tests, closed-world demos) over a per-context store, so episodes share no
    process-wide state; writes carry across turns while the driver keeps one context per episode.
    """
    # None default so each context lazily seeds its own dict (a shared mutable default breaks isolation).
    files_var: ContextVar[dict[str, str] | None] = ContextVar("native_file_tools_store", default=None)

    def _files() -> dict[str, str]:
        store = files_var.get()
        if store is None:
            store = {"/home/user/notes.txt": "My notes...", "/home/user/data.csv": "name,value\na,1\nb,2"}
            files_var.set(store)
        return store

    def list_files(directory: str = "/home/user") -> str:
        matching = [f for f in _files() if f.startswith(directory)]
        return "\n".join(matching) if matching else "No files found"

    def read_file(path: str) -> str:
        return _files().get(path, f"Error: File not found: {path}")

    def write_file(path: str, content: str) -> str:
        _files()[path] = content
        return f"Successfully wrote {len(content)} bytes to {path}"

    return _file_tools(read_file, write_file, list_files, _SIMULATED_FILE_TEXTS)


def build_combined_tools(
    timeout: float = SANDBOX_DEFAULT_TIMEOUT,
    sandbox: SandboxExecutor | None = None,
    search_backend: str | None = None,
) -> NativeToolRegistry:
    """Registry with all native tools: math, python, search (on ``search_backend``, see
    :func:`build_search_tools`), and (simulated) file."""
    return NativeToolRegistry.combine(
        build_math_tools(timeout=timeout),
        build_python_tools(timeout=timeout, sandbox=sandbox),
        build_search_tools(backend=search_backend),
        build_simulated_file_tools(),
    )


def _require_session(session_getter: Callable[[], SandboxSession]) -> SandboxSession:
    """Resolve the active episode's session, raising when there is none.

    A missing session is an infrastructure fault, not a tool outcome: returned as a string, the protocol
    would score the call as a successful observation; raised untyped, it would price it as the policy's.
    """
    session = session_getter()
    if session is None:
        raise SandboxInfraError("no active sandbox session for this episode")
    return session


def build_session_code_tools(
    session_getter: Callable[[], SandboxSession],
    language: str = "python",
    timeout: float = SANDBOX_DEFAULT_TIMEOUT,
    tool_name: str | None = None,
) -> NativeToolRegistry:
    """A code-execution tool on the active episode's :class:`SandboxSession`, which ``session_getter`` resolves per
    call, so earlier turns' files are in scope and concurrent episodes keep their own working dirs; ``tool_name``
    defaults to ``run_<language>``.
    """
    spec = require_language(language)

    def _run(code: str) -> str:
        return run_code_via_sandbox(code, _require_session(session_getter), timeout=timeout, language=spec.name)

    registry = NativeToolRegistry()
    registry.register(
        NativeTool(
            name=tool_name or f"run_{spec.name}",
            description=(
                f"Execute {spec.name} code in a persistent workspace that keeps files and state "
                "across turns. Use print()/stdout for output."
            ),
            parameters=[ToolParameter("code", "string", "Source code to execute in the workspace")],
            handler=_run,
        )
    )
    return registry


def build_session_bash_tools(
    session_getter: Callable[[], SandboxSession],
    timeout: float = SANDBOX_DEFAULT_TIMEOUT,
) -> NativeToolRegistry:
    """A shell tool (``run_bash_command``) on the active episode's session, the backend, working directory and limits
    of :func:`build_session_code_tools`; a non-zero exit is an ordinary observation, a sandbox fault raises by class.
    """

    def _run(command: str) -> str:
        return run_code_via_sandbox(command, _require_session(session_getter), timeout=timeout, language="bash")

    registry = NativeToolRegistry()
    registry.register(
        NativeTool(
            name="run_bash_command",
            description=(
                "Run a shell command with bash in the persistent workspace directory, where the file "
                "tools write. It is given no stdin. The observation is the command's stdout; a non-zero "
                "exit leads it with the signal that killed the command or the tail of its stderr, so "
                "redirect with 2>&1 when you need the error output of a command that succeeds."
            ),
            parameters=[ToolParameter("command", "string", "Shell command to run in the workspace")],
            handler=_run,
        )
    )
    return registry


def build_session_file_tools(
    session_getter: Callable[[], SandboxSession],
) -> NativeToolRegistry:
    """Real read/write/list file tools on the active episode's session, sharing its working dir with the code it runs.
    A path that leaves the workspace or that UTF-8 cannot encode raises ``ValueError`` (a failed call); a missing
    session or a replaced working directory raises its sandbox fault; an absent file is a valid string result.
    """

    def write_file(path: str, content: str) -> str:
        _require_session(session_getter).write_file(path, content)
        return f"Successfully wrote {len(content)} bytes to {path}"

    def read_file(path: str) -> str:
        content = _require_session(session_getter).read_file(path)
        return content if content is not None else f"Error: File not found: {path}"

    def list_files(directory: str = "") -> str:
        names = [f for f in _require_session(session_getter).list_files() if f.startswith(directory)]
        return "\n".join(names) if names else "No files found"

    return _file_tools(read_file, write_file, list_files, _WORKSPACE_FILE_TEXTS)
