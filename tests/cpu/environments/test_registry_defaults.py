#!/usr/bin/env python
"""The environment registry does not override per-class defaults it did not configure: class defaults win unless the
config sets ``max_turns``. Since the shipped templates rely on ``max_turns: null``, the overview's registry table
is held to the classes."""

import pytest

from halo.configs.grpo_environmental import EnvironmentConfig
from halo.environments.environment_base import EnvironmentBase
from halo.environments.episode_keys import EPISODE_INVALID_KEY, EPISODE_INVALID_REASON_KEY
from halo.environments.protocols.react import build_react_math_environment
from halo.environments.registry import build_environment, get_registered_environments
from halo.rewards.terms import EnvironmentTerm
from scripts.docs.markdown import parse_tables
from tests.common.repo import REPO_ROOT

_ENVIRONMENTS_OVERVIEW = REPO_ROOT / "agent-docs/training-methods/async-grpo/environments/README.md"
_MAX_TURNS_COLUMN = "Default `max_turns`"

# Minimal valid construction kwargs per registered env. ``sandbox_backend`` is declared only by the
# sandboxed coding envs; passing it to the others raises instead of being absorbed and ignored,
# so each row carries its own.
_ENV_KWARGS = {
    "code_contests": {"sandbox_backend": "local"},
    "codeforces": {"sandbox_backend": "local"},
    "swe": {"sandbox_backend": "local"},
    "exam_qa": {},
    "native_math": {},
    "native_coding": {},
    "native_combined": {},
    "react_math": {},
    "react_search": {},
    "qa_search": {},
    "mcp": {},
}

# The turn budget each env class declares; absent means the base 10. Written out, since deriving it from the classes
# the overview must describe would make the check agree with itself.
_CLASS_DEFAULT_MAX_TURNS = {"code_contests": 15, "codeforces": 15, "swe": 20, "exam_qa": 8}

# The environments whose reward grades only against ``context["answer"]``. Written out for the same reason.
_REQUIRES_ANSWER = {"code_contests", "codeforces", "exam_qa", "qa_search", "react_math", "react_search", "swe"}

# The environments that register ``web_search``, with the options that make them register it.
_SEARCH_ENV_KWARGS = {"exam_qa": {"open_book": True}, "native_combined": {}, "qa_search": {}, "react_search": {}}


def _base_default_max_turns() -> int:
    return EnvironmentBase.DEFAULT_MAX_TURNS


def _declared_max_turns(env_type: str) -> int:
    return type(build_environment(env_type, _ENV_KWARGS[env_type])).DEFAULT_MAX_TURNS


def _documented_max_turns() -> dict[str, int]:
    """``environment_type`` → the default ``max_turns`` the environments overview's registry table states."""
    table = next(
        table
        for table in parse_tables(_ENVIRONMENTS_OVERVIEW.read_text(encoding="utf-8"))
        if table.headers[0] == "`environment_type`"
    )
    column = table.headers.index(_MAX_TURNS_COLUMN)
    return {row[0].strip("`"): int(row[column]) for row in table.rows}


@pytest.mark.parametrize("env_type", sorted(_ENV_KWARGS))
def test_class_default_max_turns_wins_when_unconfigured(env_type):
    expected = _CLASS_DEFAULT_MAX_TURNS.get(env_type, _base_default_max_turns())
    env = build_environment(env_type, _ENV_KWARGS[env_type])
    assert env.max_turns == expected


def test_class_default_table_names_every_environment_that_declares_one():
    """A class declaring its own ``DEFAULT_MAX_TURNS`` must appear in the table above, or the docs test passes by
    not knowing about it."""
    declared = {
        env_type: _declared_max_turns(env_type)
        for env_type in _ENV_KWARGS
        if _declared_max_turns(env_type) != _base_default_max_turns()
    }
    assert declared == _CLASS_DEFAULT_MAX_TURNS


def test_every_registered_environment_is_covered():
    """A new env_type must arrive with its own turn-budget assertion."""
    assert sorted(_ENV_KWARGS) == get_registered_environments()


def test_the_environments_overview_states_every_environments_turn_budget():
    """A missing or wrong row leaves the templates' ``max_turns: null`` documenting the wrong budget."""
    expected = {
        env_type: _CLASS_DEFAULT_MAX_TURNS.get(env_type, _base_default_max_turns()) for env_type in _ENV_KWARGS
    }
    assert _documented_max_turns() == expected, (
        f"{_ENVIRONMENTS_OVERVIEW.relative_to(REPO_ROOT)}'s registry table must state each environment's "
        f"default max_turns in its '{_MAX_TURNS_COLUMN}' column"
    )


@pytest.mark.parametrize("env_type", sorted(_ENV_KWARGS))
def test_each_environment_declares_whether_its_reward_needs_an_answer(env_type):
    """The trainer refuses an answer-less dataset by this flag: a class forgetting it trains on a constant reward,
    one declaring it spuriously refuses a dataset."""
    env = build_environment(env_type, _ENV_KWARGS[env_type])
    assert env.requires_answer is (env_type in _REQUIRES_ANSWER)


def test_the_answer_requirement_table_names_only_registered_environments():
    assert set(get_registered_environments()) >= _REQUIRES_ANSWER


def test_a_react_answer_validator_grades_in_place_of_the_answer_column():
    """The validator is the grader once passed, so it clears the requirement; without one an answer-less episode
    collects the objective for any Final Answer."""
    assert build_react_math_environment().requires_answer

    seen = []

    def validator(answer, expected):
        seen.append((answer, expected))
        return answer == "7"

    env = build_react_math_environment(answer_validator=validator, thought_reward=0.0, no_thought_penalty=0.0)
    assert not env.requires_answer

    ids, _ = env.reset(["2 + 5?"])
    reward = env.step(ids, ["Thought: add\nFinal Answer: 8"], [{}])[0].trajectory.total_reward
    assert seen == [("8", None)]
    assert reward == 0.0


def test_a_raising_react_validator_voids_the_episode_instead_of_paying_it():
    """With no expected answer the default check would pay any Final Answer, so a raising grader voids the episode."""

    def validator(answer, expected):
        raise RuntimeError("grader down")

    env = build_react_math_environment(answer_validator=validator, thought_reward=0.0, no_thought_penalty=0.0)
    ids, _ = env.reset(["2 + 5?"])
    trajectory = env.step(ids, ["Thought: add\nFinal Answer: 7"], [{}])[0].trajectory
    assert trajectory.total_reward == 0.0
    assert trajectory.info[EPISODE_INVALID_KEY] is True
    assert trajectory.info[EPISODE_INVALID_REASON_KEY] == "answer_validator raised RuntimeError: grader down"


def test_an_explicit_requires_answer_overrides_the_class_declaration():
    """The escape hatch for a run that grades some other way: an environment_kwargs entry, like every
    other constructor knob."""
    assert not build_environment("exam_qa", {"requires_answer": False}).requires_answer
    assert build_environment("native_math", {"requires_answer": True}).requires_answer


def test_explicit_max_turns_still_overrides():
    env = build_environment("swe", {"sandbox_backend": "local", "max_turns": 7})
    assert env.max_turns == 7


@pytest.mark.parametrize("env_type", ["code_contests", "react_math"])
def test_unknown_environment_kwarg_is_rejected(env_type):
    """A misspelled key or one meant for another ``env_type`` would otherwise parse cleanly and change nothing."""
    with pytest.raises(TypeError, match="timout_per_test"):
        build_environment(env_type, {"timout_per_test": 10})


def test_valid_environment_kwarg_still_reaches_the_environment():
    env = build_environment("code_contests", {"timeout_per_test": 3, "sandbox_backend": "local"})
    assert env.grading_spec.default_timeout == 3


@pytest.mark.parametrize("env_type", sorted(_SEARCH_ENV_KWARGS))
def test_every_search_environment_takes_its_backend_from_the_config(env_type, monkeypatch):
    """A name no backend answers to is refused at build; refused as an unknown option instead, a preset can only
    auto-select, searching on whichever API key its actor nodes hold."""
    monkeypatch.setenv("HALO_ALLOW_MOCK_SEARCH", "1")
    config = {**_ENV_KWARGS[env_type], **_SEARCH_ENV_KWARGS[env_type]}
    env = build_environment(env_type, {**config, "search_backend": "mock"})
    assert "Wikipedia: pytest" in env.registry.get("web_search").execute(query="pytest")
    with pytest.raises(ValueError, match="Unknown search backend: nonexistent"):
        build_environment(env_type, {**config, "search_backend": "nonexistent"})


def test_the_search_table_names_every_environment_that_registers_web_search():
    """A preset gaining ``web_search`` must arrive in the table above."""
    registering = set()
    for env_type, kwargs in _ENV_KWARGS.items():
        env = build_environment(env_type, {**kwargs, **_SEARCH_ENV_KWARGS.get(env_type, {})})
        if "web_search" in env.registry.names():
            registering.add(env_type)
    assert registering == set(_SEARCH_ENV_KWARGS)


@pytest.mark.parametrize("max_turns", [0, -1])
def test_non_positive_max_turns_is_rejected(max_turns):
    """``max_turns: 0`` makes the rollout loop a no-op: a silently all-zero batch that reads as a failing policy."""
    with pytest.raises(ValueError, match="max_turns"):
        build_environment("react_math", {"max_turns": max_turns})


def test_the_reward_defaults_to_the_environment_term_alone():
    """An env config naming no ``reward_terms`` prices the environment's grade at weight 1, exponent 1, and a
    per-env reward knob is refused as an unknown option."""
    env = build_environment("exam_qa", {})
    assert env.reward_terms == (EnvironmentTerm(),)
    for knob in ("success_reward", "failure_reward", "partial_reward"):
        with pytest.raises(TypeError, match=knob):
            build_environment("exam_qa", {knob: 0.5})


def test_environment_config_yaml_path_defers_to_class_default():
    """An unset max_turns lets the class default (code_contests 15) win."""
    cfg = EnvironmentConfig(environment_type="code_contests", environment_kwargs={"sandbox_backend": "local"})
    assert "max_turns" not in cfg.to_env_config()
    env = build_environment(cfg.environment_type, cfg.to_env_config())
    assert env.max_turns == 15


def test_environment_config_explicit_max_turns_overrides():
    cfg = EnvironmentConfig(
        environment_type="code_contests", max_turns=12, environment_kwargs={"sandbox_backend": "local"}
    )
    env = build_environment(cfg.environment_type, cfg.to_env_config())
    assert env.max_turns == 12


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
