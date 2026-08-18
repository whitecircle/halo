"""Registry mapping ``env_type`` strings to environment factories. Built-ins register at import
time; custom types via register_environment(). resolve_environment() instantiates by name."""

import logging
from collections.abc import Callable
from typing import Any

from src.environments.envs.protocols.mcp import create_native_mcp_environment
from src.environments.envs.protocols.native import NativeToolUseEnvironment
from src.environments.envs.protocols.react import create_react_math_environment, create_react_search_environment
from src.environments.envs.tasks.coding.code_contests import CodeContestsEnvironment
from src.environments.envs.tasks.coding.swe import SweEnvironment
from src.environments.envs.tasks.qa import ExamQAEnvironment, create_qa_search_environment
from src.environments.tools.definitions import NativeToolRegistry
from src.environments.tools.factories import (
    create_all_native_tools,
    create_native_code_tools,
    create_native_math_tools,
    create_native_python_tools,
)

logger = logging.getLogger(__name__)

_ENVIRONMENT_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_environment(
    name: str,
    factory: Callable[[dict[str, Any]], Any],
    override: bool = False,
) -> None:
    """Register an environment ``factory`` (env_config dict → BaseEnvironment) under ``name``.
    Raises unless ``override`` when the name is already registered."""
    name = name.lower()
    if name in _ENVIRONMENT_REGISTRY and not override:
        raise ValueError(f"Environment type '{name}' is already registered. Use override=True to replace it.")
    _ENVIRONMENT_REGISTRY[name] = factory
    logger.debug(f"Registered environment type: {name}")


def resolve_environment(name: str, env_config: dict[str, Any]):
    """Resolve a registered env-type name to a BaseEnvironment instance via its factory."""
    name = name.lower()
    if name not in _ENVIRONMENT_REGISTRY:
        available = sorted(_ENVIRONMENT_REGISTRY.keys())
        raise ValueError(
            f"Unknown environment type: '{name}'. "
            f"Available types: {available}. "
            f"Use register_environment() to add custom types, "
            f"or pass environment_cls directly to the trainer."
        )
    return _ENVIRONMENT_REGISTRY[name](env_config)


def create_environment(env_type: str | tuple[type, dict[str, Any]], env_config: dict[str, Any]):
    """Build an environment from a registered type name or an explicit ``(cls, kwargs)`` pair. For the
    pair form, ``kwargs`` override ``env_config``."""
    if isinstance(env_type, tuple):
        cls, kwargs = env_type
        return cls(**{**env_config, **kwargs})
    if isinstance(env_type, str):
        return resolve_environment(env_type, env_config)
    raise TypeError(f"env_type must be str or (cls, kwargs) tuple, got {type(env_type)}")


def get_registered_environments() -> list[str]:
    """Return sorted list of registered environment type names."""
    return sorted(_ENVIRONMENT_REGISTRY.keys())


def _without(env_config: dict, *exclude: str) -> dict:
    """The env_config minus keys a factory binds itself, where a duplicate keyword would raise.

    Only for that case: every other factory forwards ``**env_config`` whole, and nothing is defaulted
    here — each knob's default belongs to the environment class that declares it, so a
    registry-injected value would silently override the per-class one.
    """
    return {k: v for k, v in env_config.items() if k not in exclude}


def _native_env(tool_registry, env_config: dict):
    """Create a NativeToolUseEnvironment with the full env config forwarded."""
    return NativeToolUseEnvironment(tool_registry=tool_registry, **env_config)


def _register_builtins():
    """Register all built-in environment types.

    Every factory forwards the FULL env_config, so YAML ``environment_kwargs`` reach the environment;
    :func:`_without` covers the few that bind a key themselves.
    """
    # ReAct: system_prompt is hardcoded per template, so exclude it from the forward.
    register_environment("react_math", lambda c: create_react_math_environment(**_without(c, "system_prompt")))
    register_environment("react_search", lambda c: create_react_search_environment(**_without(c, "system_prompt")))

    register_environment(
        "native_math",
        lambda c: _native_env(
            NativeToolRegistry.combine(create_native_math_tools(), create_native_python_tools()),
            c,
        ),
    )
    register_environment(
        "native_coding",
        lambda c: _native_env(create_native_code_tools(language="python", tool_name="python_repl"), c),
    )
    register_environment("native_combined", lambda c: _native_env(create_all_native_tools(), c))

    # No default for mcp_server here: the factory owns it, because an sse config names no preset.
    register_environment(
        "mcp",
        lambda c: create_native_mcp_environment(server_name=c.get("mcp_server"), **_without(c, "mcp_server")),
    )
    register_environment("swe", lambda c: SweEnvironment(**c))

    register_environment("qa_search", lambda c: create_qa_search_environment(**c))
    # code_contests / codeforces are one environment; only the default output_comparison differs —
    # code_contests takes the class default ("exact"), so only codeforces carries a preset.
    register_environment("code_contests", lambda c: CodeContestsEnvironment(**c))
    register_environment("codeforces", lambda c: CodeContestsEnvironment(**{"output_comparison": "tokens", **c}))
    register_environment("exam_qa", lambda c: ExamQAEnvironment(**c))


_register_builtins()
