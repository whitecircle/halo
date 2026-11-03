#!/usr/bin/env python
"""Parity between ``AsyncRolloutConfig`` and the ``RolloutConfig`` it builds for the Ray actors: every field is
reachable and each copy happens. ``capture_token_ids``' default does not agree on purpose: off in a hand-built
config, on through the YAML path."""

from dataclasses import fields

import pytest

from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.environments.rollout_config import RolloutConfig

# RolloutConfig field -> the AsyncRolloutConfig field supplying it, read from the builder's own mapping.
_FORWARDED = AsyncRolloutConfig.rollout_field_sources()

# The fields get_rollout_config computes instead of copying, each named with what it computes from.
_DERIVED = {
    "capture_token_ids": "train_on_sampled_tokens",
    "capture_routed_experts": "routing_replay",
    "stop_token_ids": "rollout_stop_tokens, resolved to ids by the trainer",
    "reasoning_end_token_id": "rollout_reasoning_end_token, resolved to an id by the caller that owns the tokenizer",
}

# A non-default value per mirrored knob, applied one at a time (an sglang backend and a thinking budget are mutually
# exclusive), so a dropped copy cannot pass on the RolloutConfig default.
_NON_DEFAULT_VALUES = {
    "rollout_backend": "sglang",
    "rollout_temperature": 0.3,
    "rollout_top_p": 0.5,
    "rollout_top_k": 20,
    "rollout_min_p": 0.05,
    "rollout_repetition_penalty": 1.1,
    "rollout_max_tokens": 4096,
    "rollout_max_episode_tokens": 65536,
    "rollout_max_thinking_tokens": 1024,
    "rollout_max_answer_tokens": 2048,
    "model_name": "org/rollout-model",
    "request_timeout": 45.0,
    "episode_timeout": 600.0,
    "max_retries": 1,
    "retry_base_wait": 2.5,
    "rollout_chat_template_kwargs": {"preserve_thinking": True},
}


def test_every_rollout_field_is_either_mirrored_or_derived():
    """A field the YAML surface does not spell falls out of the map silently; unless derived, no config can reach
    it."""
    unreachable = {f.name for f in fields(RolloutConfig)} - set(_FORWARDED) - set(_DERIVED)
    assert unreachable == set(), (
        f"{sorted(unreachable)} are RolloutConfig fields with no AsyncRolloutConfig knob: declare "
        f"one (same name, or rollout_-prefixed), or derive it in get_rollout_config and name it here."
    )


@pytest.mark.parametrize(("rollout_field", "async_field"), sorted(_FORWARDED.items()))
def test_every_mirrored_knob_reaches_the_built_rollout_config(rollout_field, async_field):
    """Matching defaults only prove the two declarations agree. The actors read nothing but the built RolloutConfig."""
    requested = _NON_DEFAULT_VALUES[async_field]

    built = AsyncRolloutConfig(**{async_field: requested}).get_rollout_config()

    assert getattr(built, rollout_field) == requested, (
        f"AsyncRolloutConfig.{async_field}={requested!r} never reached RolloutConfig.{rollout_field} "
        f"(got {getattr(built, rollout_field)!r}): the knob parses and no rollout sees it."
    )


def test_the_non_default_table_covers_every_mirrored_knob():
    """Guards the table above: a knob added to the mirror without an entry would go untested."""
    assert set(_NON_DEFAULT_VALUES) == set(_FORWARDED.values())


@pytest.mark.parametrize("sampled_tokens", [True, False])
def test_capture_token_ids_follows_train_on_sampled_tokens(sampled_tokens):
    """A broken derivation trains env-GRPO on a re-tokenized re-render of every turn, with nothing in the logs."""
    built = AsyncRolloutConfig(train_on_sampled_tokens=sampled_tokens).get_rollout_config()
    assert built.capture_token_ids is sampled_tokens


def test_reasoning_end_token_id_is_the_id_the_caller_resolved():
    """A string knob on the YAML surface and an id on the wire: the caller owning the tokenizer resolves it and the
    builder passes it through."""
    config = AsyncRolloutConfig(rollout_max_thinking_tokens=8192)
    assert config.get_rollout_config(reasoning_end_token_id=7).reasoning_end_token_id == 7
    assert config.get_rollout_config().reasoning_end_token_id is None
    assert AsyncRolloutConfig().get_rollout_config().reasoning_end_token_id is None


def test_the_yaml_path_captures_token_ids_where_a_hand_built_config_does_not():
    """Linking the two defaults would change what every hand-built RolloutConfig sends to the server."""
    assert AsyncRolloutConfig().get_rollout_config().capture_token_ids is True
    assert RolloutConfig().capture_token_ids is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
