#!/usr/bin/env python
"""Every shipped environmental-GRPO example must resolve its environment from its own YAML.

``test_examples_parse.py`` pins the parse layer, where ``environment_kwargs`` is an opaque dict: a
misspelled env kwarg, one meant for another ``environment_type``, or an invalid
``reasoning_effort_profiles`` entry parses cleanly and only raises inside a Ray actor — after the
cluster and the rollout servers are up. Here each example's environment fields go through the chain
the trainer uses (YAML → ``EnvironmentConfig`` → ``to_env_config`` → ``resolve_environment``), so
those raise on the CPU tier instead.

Run: pytest tests/cpu/config/test_examples_env_kwargs_resolve.py
"""

from pathlib import Path

import pytest
import yaml

from src.configs.environment_config import EnvironmentConfig
from src.environments.registry import resolve_environment

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_EXAMPLES_ROOT = PROJECT_ROOT / "examples" / "grpo" / "environmental"

_EXAMPLES = sorted(ENV_EXAMPLES_ROOT.rglob("*.yaml"))
_ID = {"ids": lambda config: config.relative_to(ENV_EXAMPLES_ROOT).as_posix()}

# The negative controls below need an example whose environment binds these kwargs.
_CODING_ENV_TYPES = ("code_contests", "codeforces")


def environment_config(config: Path) -> EnvironmentConfig:
    """The example's ``EnvironmentConfig``, built from the raw YAML's environment fields alone."""
    raw = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    fields = set(EnvironmentConfig.__dataclass_fields__)
    return EnvironmentConfig(**{key: value for key, value in raw.items() if key in fields})


def coding_example() -> Path:
    """One shipped code-contests example, for the controls that inject a bad kwarg into a real config."""
    for config in _EXAMPLES:
        if environment_config(config).environment_type in _CODING_ENV_TYPES:
            return config
    raise AssertionError("no shipped environmental example runs code_contests / codeforces")


@pytest.fixture(autouse=True)
def _in_process_sandbox(monkeypatch):
    """The coding envs build their sandbox at construction; the host's own choice must not reach here."""
    monkeypatch.delenv("HALO_SANDBOX_BACKEND", raising=False)
    monkeypatch.delenv("HALO_SANDBOX_URL", raising=False)


def test_environmental_examples_tree_is_not_empty():
    """A glob that stops matching would make every parametrized case vanish silently."""
    assert len(_EXAMPLES) > 10, f"expected the shipped environmental examples, found {len(_EXAMPLES)}"


@pytest.mark.parametrize("config", _EXAMPLES, **_ID)
def test_example_environment_resolves(config):
    """The environment the example names accepts every kwarg and profile the example gives it."""
    cfg = environment_config(config)
    env = resolve_environment(cfg.environment_type, cfg.to_env_config())
    try:
        assert env.max_turns >= 1
    finally:
        env.close()


def test_an_unknown_environment_kwarg_is_still_rejected():
    """Negative control: the sweep above asserts nothing if a stray kwarg is silently absorbed."""
    cfg = environment_config(coding_example())
    env_config = cfg.to_env_config()
    env_config["timout_per_test"] = 5
    with pytest.raises(TypeError, match="timout_per_test"):
        resolve_environment(cfg.environment_type, env_config)


def test_an_invalid_effort_profile_is_still_rejected():
    """Negative control for the profile validation the sweep relies on."""
    cfg = environment_config(coding_example())
    env_config = cfg.to_env_config()
    env_config["reasoning_effort_profiles"] = {"high": {"thinking_budget": 16384}}
    with pytest.raises(ValueError, match="thinking_budget"):
        resolve_environment(cfg.environment_type, env_config)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
