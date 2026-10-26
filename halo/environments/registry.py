"""Registry mapping ``env_type`` strings to environment factories. Built-ins register at import
time; custom types via register_environment(). build_environment() instantiates by name."""

import logging
from collections.abc import Callable
from typing import Any

from halo.environments.environment_base import EnvironmentBase
from halo.environments.protocols.mcp import build_mcp_environment
from halo.environments.protocols.native_tool_use import NativeToolUseEnvironment
from halo.environments.protocols.react import build_react_math_environment, build_react_search_environment
from halo.environments.tasks.coding.code_contests import CodeContestsEnvironment
from halo.environments.tasks.coding.swe import SweEnvironment
from halo.environments.tasks.qa import ExamQAEnvironment, build_qa_search_environment
from halo.environments.tools.definitions import NativeToolRegistry
from halo.environments.tools.toolsets import (
    build_code_tools,
    build_combined_tools,
    build_math_tools,
    build_python_tools,
)

logger = logging.getLogger(__name__)

_environment_registry: dict[str, Callable[[dict[str, Any]], EnvironmentBase]] = {}


def register_environment(
    name: str,
    factory: Callable[[dict[str, Any]], EnvironmentBase],
    override: bool = False,
) -> None:
    """Register an environment ``factory`` (env_config dict → EnvironmentBase) under ``name``.
    Raises unless ``override`` when the name is already registered."""
    name = name.lower()
    if name in _environment_registry and not override:
        raise ValueError(f"Environment type '{name}' is already registered. Use override=True to replace it.")
    _environment_registry[name] = factory
    logger.debug(f"Registered environment type: {name}")


def build_environment(env_type: str | tuple[type, dict[str, Any]], env_config: dict[str, Any]) -> EnvironmentBase:
    """Build an environment from a registered type name (case-insensitive) or an explicit ``(cls, kwargs)``
    pair, whose ``kwargs`` override ``env_config``."""
    if isinstance(env_type, tuple):
        cls, kwargs = env_type
        return cls(**{**env_config, **kwargs})
    if not isinstance(env_type, str):
        raise TypeError(f"env_type must be str or (cls, kwargs) tuple, got {type(env_type)}")
    name = env_type.lower()
    if name not in _environment_registry:
        available = sorted(_environment_registry.keys())
        raise ValueError(
            f"Unknown environment type: '{name}'. "
            f"Available types: {available}. "
            f"Use register_environment() to add custom types, "
            f"or pass environment_cls directly to the trainer."
        )
    return _environment_registry[name](env_config)


def get_registered_environments() -> list[str]:
    """Return sorted list of registered environment type names."""
    return sorted(_environment_registry.keys())


def _build_native_combined(search_backend: str | None = None, **kwargs) -> NativeToolUseEnvironment:
    """The native protocol over every built-in tool, web search on ``search_backend``."""
    return NativeToolUseEnvironment(tool_registry=build_combined_tools(search_backend=search_backend), **kwargs)


def _register_builtins():
    """Register all built-in environment types; every factory forwards the full env_config, so YAML
    ``environment_kwargs`` reach the environment."""
    register_environment("react_math", lambda c: build_react_math_environment(**c))
    register_environment("react_search", lambda c: build_react_search_environment(**c))

    register_environment(
        "native_math",
        lambda c: NativeToolUseEnvironment(
            tool_registry=NativeToolRegistry.combine(build_math_tools(), build_python_tools()), **c
        ),
    )
    register_environment(
        "native_coding",
        lambda c: NativeToolUseEnvironment(
            tool_registry=build_code_tools(language="python", tool_name="python_repl"), **c
        ),
    )
    register_environment("native_combined", lambda c: _build_native_combined(**c))

    register_environment("mcp", lambda c: build_mcp_environment(**c))
    register_environment("swe", lambda c: SweEnvironment(**c))

    register_environment("qa_search", lambda c: build_qa_search_environment(**c))
    # code_contests / codeforces are one environment; only the default output_comparison differs.
    # code_contests takes the class default ("exact"), so only codeforces carries a preset.
    register_environment("code_contests", lambda c: CodeContestsEnvironment(**c))
    register_environment("codeforces", lambda c: CodeContestsEnvironment(**{"output_comparison": "tokens", **c}))
    register_environment("exam_qa", lambda c: ExamQAEnvironment(**c))


_register_builtins()
