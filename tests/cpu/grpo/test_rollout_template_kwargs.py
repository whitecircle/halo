#!/usr/bin/env python
"""CPU tests: the template variables a rollout's requests carry and the trainer's renders reproduce.

``generation_control_fields`` (the request side) and ``rollout_template_kwargs`` (the trainer's render
side) must agree on the per-episode variables — the level as the top-level ``reasoning_effort`` field and
its thinking budget under ``reasoning_budget`` in the nested form — or the trainer scores log-probs
against a prompt the engine never rendered.

Run: python tests/cpu/grpo/test_rollout_template_kwargs.py  (or pytest)
"""

from dataclasses import replace

import pytest

from src.configs.async_training_config import AsyncTrainingConfig
from src.configs.rollout_config import REASONING_BUDGET_TEMPLATE_VAR, RolloutConfig
from src.environments.engine_wire import generation_control_fields
from src.trainers.grpo.rollout.trajectory_tokenize import rollout_template_kwargs


def test_request_and_render_agree_on_the_episode_variables():
    run_kwargs = {"preserve_thinking": True}
    config = RolloutConfig(max_tokens=20000, max_thinking_tokens=12288, chat_template_kwargs=run_kwargs)
    fields = generation_control_fields(config, "medium", 12288)
    assert fields["reasoning_effort"] == "medium"
    assert fields["chat_template_kwargs"] == {"preserve_thinking": True, REASONING_BUDGET_TEMPLATE_VAR: 12288}
    render_kwargs = rollout_template_kwargs(run_kwargs, "medium", 12288)
    assert render_kwargs == {**fields["chat_template_kwargs"], "reasoning_effort": "medium"}
    assert run_kwargs == {"preserve_thinking": True}, "the run's kwargs are never mutated"


def test_the_template_states_the_levels_cap_while_the_engine_enforces_the_turns():
    """Under an output budget the two diverge: the engine enforces the narrowed cap of this turn, the
    template states the level's whole per-turn cap — the one value the trajectory carries and every
    trainer-side render reproduces. Omitted, the template is told no budget: the drivers always pass the level's."""
    run = AsyncTrainingConfig(
        rollout_max_thinking_tokens=4000, rollout_chat_template_kwargs={"preserve_thinking": True}
    )
    # What the drivers send on a turn the output budget narrowed to 2500 of the level's 4000.
    turn = replace(run.get_rollout_config(), max_thinking_tokens=2500)
    fields = generation_control_fields(turn, "high", 4000)
    assert fields["thinking_token_budget"] == 2500
    assert fields["chat_template_kwargs"] == {"preserve_thinking": True, REASONING_BUDGET_TEMPLATE_VAR: 4000}
    render_kwargs = rollout_template_kwargs(run.rollout_chat_template_kwargs, "high", 4000)
    assert render_kwargs == {**fields["chat_template_kwargs"], "reasoning_effort": "high"}
    omitted = generation_control_fields(RolloutConfig(max_tokens=20000, max_thinking_tokens=12288), "high")
    assert omitted["thinking_token_budget"] == 12288
    assert REASONING_BUDGET_TEMPLATE_VAR not in omitted.get("chat_template_kwargs", {}), (
        "nothing stands in for the level's budget"
    )


def test_the_control_fields_never_ask_for_the_sampled_ids():
    """The id capture is the payload builder's, from ``capture_token_ids``; the control fields the eval
    sends as its extra body carry no capture flag on either engine, so an eval samples without one."""
    for config in (
        RolloutConfig(max_tokens=20000, max_thinking_tokens=4000, reasoning_end_token_id=5),
        RolloutConfig(backend="sglang", max_tokens=20000),
    ):
        assert "return_token_ids" not in generation_control_fields(config, "high", 4000)


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
