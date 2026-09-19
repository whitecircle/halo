#!/usr/bin/env python
"""CPU tests: the template variables a rollout's requests carry and the trainer's renders reproduce.

``generation_control_fields`` (the request side) and ``rollout_template_kwargs`` (the trainer's render
side) must agree on the per-episode variables — the level as the top-level ``reasoning_effort`` field and
its thinking budget under ``reasoning_budget`` in the nested form — or the trainer scores log-probs
against a prompt the engine never rendered.

Run: python tests/cpu/grpo/test_rollout_template_kwargs.py  (or pytest)
"""

import pytest

from src.configs.async_training_config import AsyncTrainingConfig
from src.configs.rollout_config import REASONING_BUDGET_TEMPLATE_VAR, RolloutConfig
from src.environments.engine_wire import generation_control_fields
from src.trainers.grpo.rollout.trajectory_tokenize import rollout_template_kwargs


def test_request_and_render_agree_on_the_episode_variables():
    run_kwargs = {"preserve_thinking": True}
    config = RolloutConfig(max_tokens=20000, max_thinking_tokens=12288, chat_template_kwargs=run_kwargs)
    fields = generation_control_fields(config, "medium")
    assert fields["reasoning_effort"] == "medium"
    assert fields["chat_template_kwargs"] == {"preserve_thinking": True, REASONING_BUDGET_TEMPLATE_VAR: 12288}
    render_kwargs = rollout_template_kwargs(run_kwargs, "medium", 12288)
    assert render_kwargs == {**fields["chat_template_kwargs"], "reasoning_effort": "medium"}
    assert run_kwargs == {"preserve_thinking": True}, "the run's kwargs are never mutated"


def test_sglang_also_nests_the_level_for_its_template():
    config = RolloutConfig(backend="sglang", max_tokens=20000)
    fields = generation_control_fields(config, "high")
    assert fields["reasoning_effort"] == "high"
    assert fields["chat_template_kwargs"] == {"reasoning_effort": "high"}
    assert "chat_template_kwargs" not in generation_control_fields(RolloutConfig(max_tokens=20000), "high")


def test_episodes_without_a_level_or_budget_carry_neither_variable():
    fields = generation_control_fields(RolloutConfig(max_tokens=20000), None)
    assert "reasoning_effort" not in fields and "chat_template_kwargs" not in fields
    assert rollout_template_kwargs({}, None, None) == {}
    assert rollout_template_kwargs({}, "low", None) == {"reasoning_effort": "low"}


def test_run_wide_template_kwargs_refuse_the_per_episode_variables():
    with pytest.raises(ValueError, match="reasoning_budget"):
        AsyncTrainingConfig(rollout_chat_template_kwargs={REASONING_BUDGET_TEMPLATE_VAR: 8192})
    with pytest.raises(ValueError, match="reasoning_effort"):
        AsyncTrainingConfig(rollout_chat_template_kwargs={"reasoning_effort": "low"})


def test_run_wide_template_kwargs_must_be_a_mapping():
    """A JSON string satisfies ``in`` and ``set()`` character-wise, so the per-episode key check alone
    passes it through: the run then loads the model and brings up the servers before the trainer
    rejects the value."""
    for wrong in ('{"preserve_thinking": true}', ["preserve_thinking"], 3):
        with pytest.raises(ValueError, match="must be a mapping"):
            AsyncTrainingConfig(rollout_chat_template_kwargs=wrong)
    kwargs = {"preserve_thinking": True}
    assert AsyncTrainingConfig(rollout_chat_template_kwargs=kwargs).rollout_chat_template_kwargs == kwargs


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
