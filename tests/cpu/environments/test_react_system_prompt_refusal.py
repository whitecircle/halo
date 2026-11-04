#!/usr/bin/env python
"""The ReAct presets refuse ``environment_kwargs.system_prompt``: they hardcode the prompt stating the format their
parser reads. An explicit ``null`` is no request and builds the preset."""

import pytest

from halo.configs.grpo_environmental import EnvironmentConfig
from halo.environments.registry import build_environment

_REACT_TYPES = ["react_math", "react_search"]


def _resolve(environment_type: str, environment_kwargs: dict):
    config = EnvironmentConfig(environment_type=environment_type, environment_kwargs=environment_kwargs)
    return build_environment(config.environment_type, config.to_env_config())


@pytest.mark.parametrize("environment_type", _REACT_TYPES)
def test_react_preset_refuses_a_configured_system_prompt(environment_type):
    with pytest.raises(ValueError, match=rf"{environment_type}.*environment_kwargs\.system_prompt"):
        _resolve(environment_type, {"system_prompt": "Answer tersely."})


@pytest.mark.parametrize("environment_type", _REACT_TYPES)
@pytest.mark.parametrize("environment_kwargs", [{}, {"system_prompt": None}])
def test_react_preset_keeps_its_own_prompt_without_a_request(environment_type, environment_kwargs):
    env = _resolve(environment_type, environment_kwargs)
    assert "Final Answer:" in env.system_prompt


def test_an_environment_that_takes_a_system_prompt_still_gets_it():
    """The refusal is the ReAct presets' alone; the named alternative honors the key."""
    env = _resolve("native_math", {"system_prompt": "Answer tersely."})
    assert env.system_prompt == "Answer tersely."


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
