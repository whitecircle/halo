#!/usr/bin/env python
"""Every shipped environmental-GRPO example resolves its environment from its own YAML, through the chain the
trainer uses. A misspelled env kwarg parses cleanly and otherwise raises only inside a Ray actor, after the
cluster and servers are up."""

import functools
import re
from pathlib import Path

import pytest
import yaml

from halo.configs.grpo_environmental import EnvironmentConfig
from halo.environments.registry import build_environment
from halo.rewards.terms import JudgeTerm
from tests.common.repo import REPO_ROOT

ENV_EXAMPLES_ROOT = REPO_ROOT / "examples" / "grpo" / "environmental"

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
    env = build_environment(cfg.environment_type, cfg.to_env_config())
    try:
        return frozenset(schema["function"]["name"] for schema in env.get_tools_schema() or [])
    finally:
        env.close()


def runs_judge(config: Path) -> bool:
    raw = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    return any(term.get("source") == "judge" for term in raw.get("rewards") or [])


def judge_text(config: Path) -> str:
    """What the example's judge terms tell the judge, as its environment resolves them (a rubric the environment
    declares included): each context and each check's or requirement's description."""
    cfg = environment_config(config)
    env = build_environment(cfg.environment_type, cfg.to_env_config())
    try:
        judges = [term for term in env.reward_terms if isinstance(term, JudgeTerm)]
    finally:
        env.close()
    texts = [
        text
        for judge in judges
        for text in (judge.context or "", *(item.description for item in (*judge.requirements, *judge.checks)))
    ]
    return "\n".join(text for text in texts if text)


def foreign_tools(text: str, offered: frozenset[str]) -> set[str]:
    """The tools some shipped example offers that ``text`` names but ``offered`` does not hold."""
    shipped = frozenset().union(*map(offered_tools, _EXAMPLES))
    return {name for name in shipped - offered if re.search(rf"\b{re.escape(name)}\b", text)}


_JUDGED_EXAMPLES = [config for config in _EXAMPLES if runs_judge(config)]


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
    cfg = environment_config(config)
    env = build_environment(cfg.environment_type, cfg.to_env_config())
    try:
        assert env.max_turns >= 1
    finally:
        env.close()


def test_an_unknown_environment_kwarg_is_still_rejected():
    cfg = environment_config(coding_example())
    env_config = cfg.to_env_config()
    env_config["timout_per_test"] = 5
    with pytest.raises(TypeError, match="timout_per_test"):
        build_environment(cfg.environment_type, env_config)


def test_an_invalid_effort_profile_is_still_rejected():
    cfg = environment_config(coding_example())
    env_config = cfg.to_env_config()
    env_config["reasoning_effort_profiles"] = {"high": {"thinking_budget": 16384}}
    with pytest.raises(ValueError, match="thinking_budget"):
        build_environment(cfg.environment_type, env_config)


def test_some_example_runs_a_judge():
    assert _JUDGED_EXAMPLES, "no shipped environmental example runs a judge term"


@pytest.mark.parametrize("config", _JUDGED_EXAMPLES, **_ID)
def test_a_judge_names_only_the_tools_its_environment_offers(config):
    """The judge reads each call under the environment's tool name, so naming a tool it does not offer describes
    calls its transcripts never hold."""
    assert not foreign_tools(judge_text(config), offered_tools(config))


def test_a_judge_naming_another_runs_scratchpad_is_caught():
    """The sweep above asserts nothing unless the shipped tool names include the scratchpad a Python-only run does
    not offer."""
    config = next(config for config in _JUDGED_EXAMPLES if "python_repl" in offered_tools(config))
    assert foreign_tools("run_code is the policy's scratchpad", offered_tools(config)) == {"run_code"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
