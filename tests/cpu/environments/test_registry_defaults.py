#!/usr/bin/env python
"""CPU tests: the environment registry must not override per-class defaults it didn't configure.

``_common_kwargs`` must not inject ``max_turns=10`` into every factory call: that silently clobbers
the per-class defaults. Class defaults win unless the user set ``max_turns`` in the env config —
and because ``max_turns: null`` is what the shipped templates rely on, the ``EnvironmentConfig``
help that tells users which default they are getting is held to the classes themselves here.

Run: python tests/cpu/environments/test_registry_defaults.py  (or pytest)
"""

import sys

import pytest

from src.configs.environment_config import EnvironmentConfig
from src.environments.base import BaseEnvironment
from src.environments.registry import get_registered_environments, resolve_environment

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

# The turn budget each env class declares for itself; anything absent here defaults to the base 10.
# Written out, not derived: this is the table the user-facing help below is held to, and deriving it
# from the same classes the help must describe would make that check agree with itself.
_CLASS_DEFAULT_MAX_TURNS = {"code_contests": 15, "codeforces": 15, "swe": 20, "exam_qa": 8}


def _base_default_max_turns() -> int:
    """``BaseEnvironment``'s own default — the value every env inherits unless it declares one."""
    return BaseEnvironment.DEFAULT_MAX_TURNS


def _declared_max_turns(env_type: str) -> int:
    """The ``DEFAULT_MAX_TURNS`` the class registered under ``env_type`` declares."""
    return type(resolve_environment(env_type, _ENV_KWARGS[env_type])).DEFAULT_MAX_TURNS


@pytest.mark.parametrize("env_type", sorted(_ENV_KWARGS))
def test_class_default_max_turns_wins_when_unconfigured(env_type):
    expected = _CLASS_DEFAULT_MAX_TURNS.get(env_type, _base_default_max_turns())
    env = resolve_environment(env_type, _ENV_KWARGS[env_type])
    assert env.max_turns == expected


def test_class_default_table_names_every_environment_that_declares_one():
    """Anti-rot: a class that declares its own ``DEFAULT_MAX_TURNS`` must appear in the table above,
    which is what holds the user-facing help to account. Without this, a new override would ship
    with no help entry and the help test would pass by not knowing about it."""
    declared = {
        env_type: _declared_max_turns(env_type)
        for env_type in _ENV_KWARGS
        if _declared_max_turns(env_type) != _base_default_max_turns()
    }
    assert declared == _CLASS_DEFAULT_MAX_TURNS


def test_every_registered_environment_is_covered():
    """A new env_type must arrive with its own turn-budget assertion, else the table below (and the
    user-facing help it holds to account) silently stops covering the registry."""
    assert sorted(_ENV_KWARGS) == get_registered_environments()


def test_max_turns_help_names_every_environment_that_overrides_the_base_default():
    """An env that declares its own budget while the help enumerates only the others leaves the
    shipped template's ``max_turns: null`` documenting no budget at all. Derived from the
    classes, not restated — a new env with its own budget fails here until the help says so."""
    help_text = EnvironmentConfig.__dataclass_fields__["max_turns"].metadata["help"]
    for env_type, default in _CLASS_DEFAULT_MAX_TURNS.items():
        assert f"{env_type} {default}" in help_text, (
            f"EnvironmentConfig.max_turns help does not state that '{env_type}' defaults to "
            f"{default} turns; a null max_turns then silently picks a budget the user cannot read."
        )
    # The catch-all the help states for everything it does not name.
    assert f"every other environment {_base_default_max_turns()}" in help_text


def test_explicit_max_turns_still_overrides():
    env = resolve_environment("swe", {"sandbox_backend": "local", "max_turns": 7})
    assert env.max_turns == 7


@pytest.mark.parametrize("env_type", ["code_contests", "react_math"])
def test_unknown_environment_kwarg_is_rejected(env_type):
    """An ``environment_kwargs`` key no constructor binds must raise, not be silently absorbed.

    Parking leftover ``**kwargs`` in ``BaseEnvironment.__init__``'s ``self.config``, which nothing
    ever reads, lets a misspelled key (``timout_per_test``) or one meant for a different ``env_type``
    (``sandbox_backend`` on a non-sandboxed env) parse cleanly and change nothing about the run.
    """
    with pytest.raises(TypeError, match="timout_per_test"):
        resolve_environment(env_type, {"timout_per_test": 10})


def test_valid_environment_kwarg_still_reaches_the_environment():
    """Guards the rejection above from being satisfied by rejecting everything."""
    env = resolve_environment("code_contests", {"timeout_per_test": 3, "sandbox_backend": "local"})
    assert env.grading_spec.default_timeout == 3


@pytest.mark.parametrize("max_turns", [0, -1])
def test_non_positive_max_turns_is_rejected(max_turns):
    """``max_turns: 0`` makes the rollout loop a no-op, so every episode returns reward 0 with no
    error — a silently all-zero batch that reads as a legitimately failing policy."""
    with pytest.raises(ValueError, match="max_turns"):
        resolve_environment("react_math", {"max_turns": max_turns})


def test_reward_defaults_still_ensured():
    """No ``partial_reward``: ``compute_answer_reward`` grades all-or-nothing, so a partial tier
    would name a reward the grader never pays."""
    env = resolve_environment("exam_qa", {})
    assert (env.success_reward, env.failure_reward) == (1.0, 0.0)
    assert not hasattr(env, "partial_reward")


def test_environment_config_yaml_path_defers_to_class_default():
    """The trainer YAML path (EnvironmentConfig.to_env_config) must not inject its own dataclass
    default — an unset max_turns lets the class default (code_contests 15) win."""
    cfg = EnvironmentConfig(environment_type="code_contests", environment_kwargs={"sandbox_backend": "local"})
    assert "max_turns" not in cfg.to_env_config()
    env = resolve_environment(cfg.environment_type, cfg.to_env_config())
    assert env.max_turns == 15


def test_environment_config_explicit_max_turns_overrides():
    cfg = EnvironmentConfig(
        environment_type="code_contests", max_turns=12, environment_kwargs={"sandbox_backend": "local"}
    )
    env = resolve_environment(cfg.environment_type, cfg.to_env_config())
    assert env.max_turns == 12


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
