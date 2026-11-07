#!/usr/bin/env python
"""CPU tests: the template variables a rollout's requests carry and the trainer's renders reproduce.

``generation_control_fields`` (the request side) and ``rollout_template_kwargs`` (the trainer's render
side) must agree on the per-episode variable — the level, as the top-level ``reasoning_effort`` field —
or the trainer scores log-probs against a prompt the engine never rendered. The engine's reasoning cap
rides its own request field and never reaches the template: the model is told its level, no token count.

Run: python tests/cpu/grpo/test_rollout_template_kwargs.py  (or pytest)
"""

from dataclasses import replace

import pytest

from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.environments.engine_wire import generation_control_fields
from halo.environments.rollout_config import RolloutConfig
from halo.trainers.grpo.environmental.trajectory_tokenize import rollout_template_kwargs


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_request_and_render_agree_on_the_episode_variables(backend):
    """vLLM merges the top-level level into the template's variables and SGLang reads the nested copy; either
    way the template sees the run's kwargs plus the level, which is what the trainer renders under."""
    run_kwargs = {"preserve_thinking": True}
    config = RolloutConfig(backend=backend, max_tokens=20000, chat_template_kwargs=run_kwargs)
    fields = generation_control_fields(config, "medium")
    assert fields["reasoning_effort"] == "medium"
    render_kwargs = rollout_template_kwargs(run_kwargs, "medium")
    assert render_kwargs == {**fields["chat_template_kwargs"], "reasoning_effort": fields["reasoning_effort"]}
    assert render_kwargs == {"preserve_thinking": True, "reasoning_effort": "medium"}
    assert run_kwargs == {"preserve_thinking": True}, "the run's kwargs are never mutated"


def test_the_engine_enforces_the_turns_cap_and_the_template_is_told_no_count():
    """Under an output budget the engine enforces the narrowed cap of this turn; the template states the level
    alone, on the request and in every trainer-side render, so no turn's cap leaks into the prompt."""
    run = AsyncRolloutConfig(
        rollout_max_thinking_tokens=4000, rollout_chat_template_kwargs={"preserve_thinking": True}
    )
    turn = replace(run.get_rollout_config(), max_thinking_tokens=2500)
    fields = generation_control_fields(turn, "high")
    assert fields["thinking_token_budget"] == 2500
    assert fields["chat_template_kwargs"] == {"preserve_thinking": True}
    assert rollout_template_kwargs(run.rollout_chat_template_kwargs, "high") == {
        "preserve_thinking": True,
        "reasoning_effort": "high",
    }


def test_the_control_fields_never_ask_for_the_sampled_ids():
    """The id capture is the payload builder's, from ``capture_token_ids``, so an eval samples without one."""
    for config in (
        RolloutConfig(max_tokens=20000, max_thinking_tokens=4000, reasoning_end_token_id=5),
        RolloutConfig(backend="sglang", max_tokens=20000),
    ):
        assert "return_token_ids" not in generation_control_fields(config, "high")


def test_sglang_also_nests_the_level_for_its_template():
    config = RolloutConfig(backend="sglang", max_tokens=20000)
    fields = generation_control_fields(config, "high")
    assert fields["reasoning_effort"] == "high"
    assert fields["chat_template_kwargs"] == {"reasoning_effort": "high"}
    assert "chat_template_kwargs" not in generation_control_fields(RolloutConfig(max_tokens=20000), "high")


def test_episodes_without_a_level_carry_no_variable():
    fields = generation_control_fields(RolloutConfig(max_tokens=20000), None)
    assert "reasoning_effort" not in fields and "chat_template_kwargs" not in fields
    assert rollout_template_kwargs({}, None) == {}
    assert rollout_template_kwargs({}, "low") == {"reasoning_effort": "low"}


def test_run_wide_template_kwargs_refuse_the_per_episode_level():
    with pytest.raises(ValueError, match="must not carry 'reasoning_effort'"):
        AsyncRolloutConfig(rollout_chat_template_kwargs={"reasoning_effort": "low"})


def test_run_wide_template_kwargs_must_be_a_mapping():
    """A JSON string satisfies ``in`` and ``set()`` character-wise, so the per-episode key check alone passes it, and
    the run loads the model and brings up the servers before the trainer rejects the value."""
    for wrong in ('{"preserve_thinking": true}', ["preserve_thinking"], 3):
        with pytest.raises(ValueError, match="must be a mapping"):
            AsyncRolloutConfig(rollout_chat_template_kwargs=wrong)
    kwargs = {"preserve_thinking": True}
    assert AsyncRolloutConfig(rollout_chat_template_kwargs=kwargs).rollout_chat_template_kwargs == kwargs


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
