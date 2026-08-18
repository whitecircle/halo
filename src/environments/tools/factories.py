"""Pre-built tool-set factories (math, code, search, file).

``create_native_*`` are stateless (Python defaults to the in-process sandbox; pass a ``sandbox`` or
non-Python ``language`` for a real interpreter). ``create_session_*`` bind a :class:`SandboxSession`
whose working dir persists across turns, resolved per call so concurrent episodes stay isolated.
"""

import functools
from collections.abc import Callable
from contextvars import ContextVar

from src.environments.sandbox.base import SANDBOX_DEFAULT_TIMEOUT, SandboxExecutor, SandboxSession, require_language
from src.environments.sandbox.inprocess import run_python_sandboxed, safe_calculate
from src.environments.sandbox.repl import run_code_via_sandbox
from src.environments.sandbox.resolve import resolve_sandbox
from src.environments.tools.definitions import (
    NativeTool,
    NativeToolRegistry,
    ToolParameter,
)
from src.environments.tools.web_search import async_web_search, validate_search_backend, web_search


def _code_repl_handler(
    language: str,
    timeout: float,
    allow_imports: bool,
    sandbox: SandboxExecutor | None,
) -> Callable[..., str]:
    """Pick the code-execution handler. Python with no ``sandbox`` uses the in-process REPL; otherwise
    (or any non-Python language) runs via a :class:`SandboxExecutor`, resolving a default when none given."""
    spec = require_language(language)

    if spec.name == "python" and sandbox is None:
        return functools.partial(run_python_sandboxed, timeout=timeout, allow_imports=allow_imports)

    executor = sandbox or resolve_sandbox()
    return functools.partial(run_code_via_sandbox, sandbox=executor, timeout=timeout, language=spec.name)


def _code_tool(
    name: str,
    description: str,
    language: str,
    timeout: float,
    allow_imports: bool,
    sandbox: SandboxExecutor | None,
) -> NativeTool:
    """Build a code-execution :class:`NativeTool`."""
    return NativeTool(
        name=name,
        description=description,
        parameters=[
            ToolParameter(
                name="code",
                type="string",
                description="Source code to execute. Writes results to stdout/print; it is given no stdin.",
            ),
        ],
        handler=_code_repl_handler(language, timeout, allow_imports, sandbox),
    )


def create_native_math_tools(timeout: float = SANDBOX_DEFAULT_TIMEOUT) -> NativeToolRegistry:
    """Create a safe-calculator tool (arithmetic and common math functions).

    ``timeout`` bounds one evaluation, the same wall-clock cap the code tools take: an expression the
    AST guard admits can still compute forever, and sync tool execution runs on a Ray actor's loop.
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
            handler=functools.partial(safe_calculate, timeout=timeout),
        )
    )
    return registry


def create_native_code_tools(
    language: str = "python",
    timeout: float = SANDBOX_DEFAULT_TIMEOUT,
    allow_imports: bool = False,
    sandbox: SandboxExecutor | None = None,
    tool_name: str | None = None,
) -> NativeToolRegistry:
    """Create a single-language code-execution tool.

    Args:
        language: ``"python"``, ``"cpp"``/``"c++"``, or ``"c"`` (see the sandbox language registry).
        timeout: wall-clock cap (seconds) per execution.
        allow_imports: permit ``import`` in the in-process Python sandbox only (no effect with a real
            ``sandbox`` or compiled language). Only safe in an externally isolated context.
        sandbox: run on this :class:`SandboxExecutor` (OS-isolated; imports/stdlib available). ``None``
            keeps the in-process Python REPL; a non-Python language resolves a default backend.
        tool_name: name exposed to the model (defaults to canonical language name).
    """
    spec = require_language(language)

    if spec.is_compiled:
        description = (
            f"Execute a complete {spec.name} program (compiled, then run). It is given no stdin and "
            "writes to stdout. Include a main() and any needed includes."
        )
    else:
        description = (
            "Execute Python code. Can perform calculations, define variables, use loops, and "
            "print results. Common math functions are available: sqrt, sin, cos, log, exp, "
            "floor, ceil, factorial, pi, e."
        )

    registry = NativeToolRegistry()
    registry.register(
        _code_tool(
            name=tool_name or spec.name,
            description=description,
            language=spec.name,
            timeout=timeout,
            allow_imports=allow_imports,
            sandbox=sandbox,
        )
    )
    return registry


def create_native_python_tools(
    timeout: float = SANDBOX_DEFAULT_TIMEOUT,
    allow_imports: bool = False,
    sandbox: SandboxExecutor | None = None,
) -> NativeToolRegistry:
    """Create a sandboxed Python REPL tool (named ``python``). See :func:`create_native_code_tools`."""
    return create_native_code_tools(
        language="python", timeout=timeout, allow_imports=allow_imports, sandbox=sandbox, tool_name="python"
    )


def create_native_search_tools(backend: str | None = None) -> NativeToolRegistry:
    """Create a web_search tool with pluggable backends.

    ``backend=None`` auto-selects by available API-key env var, in order: Serper (SERPER_API_KEY),
    Brave (BRAVE_API_KEY), Tavily (TAVILY_API_KEY), then keyless DuckDuckGo. Pass a name
    ("serper"/"brave"/"tavily"/"duckduckgo") to force one; "mock" additionally needs
    ``HALO_ALLOW_MOCK_SEARCH=1`` (its fabricated snippets would score as a successful search).

    The name is validated HERE, not at the first search: an env built with an unselectable backend
    would otherwise run every episode to completion on tool errors.
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
                    description="Maximum number of results (default 5)",
                    required=False,
                ),
            ],
            handler=lambda query, max_results=5: web_search(query, max_results=int(max_results), backend=backend),
            async_handler=lambda query, max_results=5: async_web_search(
                query, max_results=int(max_results), backend=backend
            ),
        )
    )
    return registry


def _seed_simulated_files() -> dict[str, str]:
    """Fresh copy of the demo filesystem's initial contents (one per episode)."""
    return {
        "/home/user/notes.txt": "My notes...",
        "/home/user/data.csv": "name,value\na,1\nb,2",
    }


def _file_tools(
    read_file: Callable[..., str],
    write_file: Callable[..., str],
    list_files: Callable[..., str],
    *,
    read_desc: str,
    write_desc: str,
    list_desc: str,
    read_path_desc: str,
    write_path_desc: str,
    list_dir_desc: str,
) -> NativeToolRegistry:
    """Build the read/write/list file trio over any store, given its three handlers.

    The schema is fixed — the model calls the same three tools with the same parameter names whichever
    store backs them — while every description differs per store (a simulated absolute-path filesystem
    versus a persistent relative-path workspace), so the wording is supplied rather than shared.
    """
    registry = NativeToolRegistry()
    registry.register(
        NativeTool(
            name="read_file",
            description=read_desc,
            parameters=[ToolParameter("path", "string", read_path_desc)],
            handler=read_file,
        )
    )
    registry.register(
        NativeTool(
            name="write_file",
            description=write_desc,
            parameters=[
                ToolParameter("path", "string", write_path_desc),
                ToolParameter("content", "string", "Content to write to the file"),
            ],
            handler=write_file,
        )
    )
    registry.register(
        NativeTool(
            name="list_files",
            description=list_desc,
            parameters=[ToolParameter("directory", "string", list_dir_desc, required=False)],
            handler=list_files,
        )
    )
    return registry


def create_native_file_tools() -> NativeToolRegistry:
    """Create simulated in-memory filesystem tools (tests / closed-world demos).

    The store is a per-call :class:`ContextVar`, so episodes never share process-wide state. Writes carry
    across turns whenever the driver keeps one context per episode (:class:`EpisodeDispatcher`). For a
    real workspace shared with executed code, use :func:`create_session_file_tools`.
    """
    # None default so each context lazily seeds its own dict (a shared mutable default breaks isolation).
    files_var: ContextVar[dict[str, str] | None] = ContextVar("native_file_tools_store", default=None)

    def _files() -> dict[str, str]:
        store = files_var.get()
        if store is None:
            store = _seed_simulated_files()
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

    return _file_tools(
        read_file,
        write_file,
        list_files,
        read_desc="Read the contents of a file.",
        write_desc="Write content to a file.",
        list_desc="List files in a directory.",
        read_path_desc="Path to the file to read",
        write_path_desc="Path to the file to write",
        list_dir_desc="Directory path to list files from",
    )


def create_all_native_tools(
    timeout: float = SANDBOX_DEFAULT_TIMEOUT,
    allow_imports: bool = False,
    sandbox: SandboxExecutor | None = None,
) -> NativeToolRegistry:
    """Registry with all native tools: math, python, search, and (simulated) file."""
    return NativeToolRegistry.combine(
        create_native_math_tools(timeout=timeout),
        create_native_python_tools(timeout=timeout, allow_imports=allow_imports, sandbox=sandbox),
        create_native_search_tools(),
        create_native_file_tools(),
    )


def _require_session(session_getter: Callable[[], SandboxSession]) -> SandboxSession:
    """Resolve the active episode's session, raising when there is none.

    A missing session is an infrastructure fault, not a tool outcome: returning it as a string would
    make the protocol score the call as a successful observation (and pay ``tool_success_reward``).
    """
    session = session_getter()
    if session is None:
        raise RuntimeError("no active sandbox session for this episode")
    return session


def create_session_code_tools(
    session_getter: Callable[[], SandboxSession],
    language: str = "python",
    timeout: float = SANDBOX_DEFAULT_TIMEOUT,
    tool_name: str | None = None,
) -> NativeToolRegistry:
    """Create a code-execution tool bound to the current episode's persistent session.

    ``session_getter`` is invoked every call and returns the active episode's :class:`SandboxSession`,
    so earlier turns' files are in scope and concurrent episodes never share a working dir. ``tool_name``
    defaults to ``run_<language>``.
    """
    spec = require_language(language)

    def _run(code: str) -> str:
        session = _require_session(session_getter)
        return run_code_via_sandbox(code, sandbox=None, timeout=timeout, language=spec.name, session=session)

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


def create_session_file_tools(
    session_getter: Callable[[], SandboxSession],
) -> NativeToolRegistry:
    """Create real read/write/list file tools backed by the current episode's persistent session.

    Files live in the session working dir, so code run via :func:`create_session_code_tools` shares them
    across turns. ``session_getter`` resolves the active episode's session per call.

    A missing session and a rejected (workspace-escaping) path both raise, so the protocol marks the
    call failed; a genuinely absent file is a real answer and stays a string.
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

    return _file_tools(
        read_file,
        write_file,
        list_files,
        read_desc="Read the contents of a workspace file.",
        write_desc="Write content to a file in the workspace (persists across turns).",
        list_desc="List files in the workspace (optionally filtered by a path prefix).",
        read_path_desc="Relative path of the file to read",
        write_path_desc="Relative path of the file to write",
        list_dir_desc="Path prefix to filter by",
    )
