#!/usr/bin/env python
"""Every shipped environmental-GRPO example must resolve its environment from its own YAML.

``test_examples_parse.py`` pins the parse layer, where ``environment_kwargs`` is an opaque dict: a
misspelled env kwarg, one meant for another ``environment_type``, or an invalid
``reasoning_effort_profiles`` entry parses cleanly and only raises inside a Ray actor — after the
cluster and the rollout servers are up. Here each example's environment fields go through the chain
the trainer uses (YAML → ``EnvironmentConfig`` → ``to_env_config`` → ``resolve_environment``), so
those raise on the CPU tier instead. The resolved environment also pins what each example's judge is told: a
tool it names must be one that environment offers.

Run: pytest tests/cpu/config/test_examples_env_kwargs_resolve.py
"""

import functools
import re
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


@functools.cache
def offered_tools(config: Path) -> frozenset[str]:
    """The tool names the example's environment offers the policy, as its transcripts spell them."""
    cfg = environment_config(config)
    env = resolve_environment(cfg.environment_type, cfg.to_env_config())
    try:
        return frozenset(schema["function"]["name"] for schema in env.get_tools_schema() or [])
    finally:
        env.close()


def judge_text(config: Path) -> str:
    """What the example's judge terms tell the judge in their own words: each context and each check's or
    requirement's description."""
    raw = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    texts = []
    for term in raw.get("rewards") or []:
        if term.get("source") == "judge":
            items = (*(term.get("checks") or ()), *(term.get("requirements") or ()))
            texts += [term.get("context") or "", *(item.get("description", "") for item in items)]
    return "\n".join(text for text in texts if text)


def foreign_tools(text: str, offered: frozenset[str]) -> set[str]:
    """The tools some shipped example offers that ``text`` names but ``offered`` does not hold."""
    shipped = frozenset().union(*map(offered_tools, _EXAMPLES))
    return {name for name in shipped - offered if re.search(rf"\b{re.escape(name)}\b", text)}


_JUDGED_EXAMPLES = [config for config in _EXAMPLES if judge_text(config)]


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


def test_some_example_runs_a_judge():
    """A judge-term scan that stops matching would make the tool-name sweep below vanish silently."""
    assert _JUDGED_EXAMPLES, "no shipped environmental example runs a judge term"


@pytest.mark.parametrize("config", _JUDGED_EXAMPLES, **_ID)
def test_a_judge_names_only_the_tools_its_environment_offers(config):
    """The judge reads each call under the name the environment gave its tool: a context or check naming a tool
    this environment does not offer (``run_code`` where a Python-only run offers ``python_repl``) describes calls
    its transcripts never hold."""
    assert not foreign_tools(judge_text(config), offered_tools(config))


def test_a_judge_naming_another_runs_scratchpad_is_caught():
    """Negative control: the sweep above asserts nothing unless the shipped tool names include the scratchpad a
    Python-only run does not offer."""
    config = next(config for config in _JUDGED_EXAMPLES if "python_repl" in offered_tools(config))
    assert foreign_tools("run_code is the policy's scratchpad", offered_tools(config)) == {"run_code"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
