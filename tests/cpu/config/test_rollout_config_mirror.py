#!/usr/bin/env python
"""Parity between ``AsyncTrainingConfig`` (the env-GRPO YAML surface) and the ``RolloutConfig`` it
builds for the Ray rollout actors.

``AsyncTrainingConfig.get_rollout_config`` copies every same-named (or ``rollout_``-prefixed) field
under a mirrored name; this pins that every RolloutConfig field is reachable and that each copy happens.
The other fields are DERIVED from different knobs, and ``capture_token_ids``' default does not agree:
off in a hand-built config, on through the YAML path. Those derivations are pinned below.

Run: pytest tests/cpu/config/test_rollout_config_mirror.py
"""

from dataclasses import fields

import pytest

from src.configs.grpo_environmental import AsyncTrainingConfig
from src.environments.rollout_config import RolloutConfig

# RolloutConfig field -> the AsyncTrainingConfig field that supplies it, read from the builder's own
# mapping rather than restated: the map has to exist exactly once to be worth testing.
_FORWARDED = AsyncTrainingConfig.rollout_field_sources()

# The fields get_rollout_config computes instead of copying, each named with what it computes from.
_DERIVED = {
    "capture_token_ids": "train_on_sampled_tokens",
    "capture_routed_experts": "routing_replay",
    "stop_token_ids": "rollout_stop_tokens, resolved to ids by the trainer",
    "reasoning_end_token_id": "rollout_reasoning_end_token, resolved to an id by the caller that owns the tokenizer",
}

# A non-default value per mirrored knob. Applied one at a time (an sglang backend and a thinking
# budget are mutually exclusive), so a knob that stops being copied cannot pass by leaving the
# RolloutConfig default in place — which is the value the YAML was overriding.
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
    """``rollout_field_sources`` pairs the two declarations by name, so a RolloutConfig field the
    YAML surface does not spell falls out of the map silently. Unless it is one of the derived
    three, that is a request field no config can reach and no rollout ever sets."""
    unreachable = {f.name for f in fields(RolloutConfig)} - set(_FORWARDED) - set(_DERIVED)
    assert unreachable == set(), (
        f"{sorted(unreachable)} are RolloutConfig fields with no AsyncTrainingConfig knob: declare "
        f"one (same name, or rollout_-prefixed), or derive it in get_rollout_config and name it here."
    )


@pytest.mark.parametrize(("rollout_field", "async_field"), sorted(_FORWARDED.items()))
def test_every_mirrored_knob_reaches_the_built_rollout_config(rollout_field, async_field):
    """Matching defaults only prove the two declarations agree; this proves the copy happens. The
    actors read nothing but the built RolloutConfig, so a knob that stops being copied is one the
    YAML sets and every rollout ignores."""
    requested = _NON_DEFAULT_VALUES[async_field]

    built = AsyncTrainingConfig(**{async_field: requested}).get_rollout_config()

    assert getattr(built, rollout_field) == requested, (
        f"AsyncTrainingConfig.{async_field}={requested!r} never reached RolloutConfig.{rollout_field} "
        f"(got {getattr(built, rollout_field)!r}): the knob parses and no rollout sees it."
    )


def test_the_non_default_table_covers_every_mirrored_knob():
    """Guards the table above: a knob added to the mirror without an entry would go untested."""
    assert set(_NON_DEFAULT_VALUES) == set(_FORWARDED.values())


@pytest.mark.parametrize("sampled_tokens", [True, False])
def test_capture_token_ids_follows_train_on_sampled_tokens(sampled_tokens):
    """The knob is the only thing that turns id capture on, and the actors read only the derived
    field: a broken derivation trains env-GRPO on a re-tokenized re-render of every turn — the exact
    mismatch ``train_on_sampled_tokens`` exists to eliminate — with nothing in the logs saying so."""
    built = AsyncTrainingConfig(train_on_sampled_tokens=sampled_tokens).get_rollout_config()
    assert built.capture_token_ids is sampled_tokens


def test_reasoning_end_token_id_is_the_id_the_caller_resolved():
    """The marker is a string knob on the YAML surface and an id on the wire, so no name pairing can
    carry it: the caller that owns the tokenizer resolves it and the builder passes it through
    untouched (the per-turn reasoning count ``episode/thinking_cap_turns`` reads stops at it)."""
    config = AsyncTrainingConfig(rollout_max_thinking_tokens=8192)
    assert config.get_rollout_config(reasoning_end_token_id=7).reasoning_end_token_id == 7
    assert config.get_rollout_config().reasoning_end_token_id is None
    assert AsyncTrainingConfig().get_rollout_config().reasoning_end_token_id is None


def test_the_yaml_path_captures_token_ids_where_a_hand_built_config_does_not():
    """The one place the two sides legitimately disagree, pinned so it stays a decision. Linking the
    two defaults would change what every hand-built RolloutConfig sends to the server; if that is
    wanted, flip the default here and correct the note on the field."""
    assert AsyncTrainingConfig().get_rollout_config().capture_token_ids is True
    assert RolloutConfig().capture_token_ids is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
