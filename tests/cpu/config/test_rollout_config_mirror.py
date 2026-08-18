#!/usr/bin/env python
"""Parity between ``AsyncTrainingConfig`` (the env-GRPO YAML surface) and the ``RolloutConfig`` it
builds for the Ray rollout actors.

``AsyncTrainingConfig.get_rollout_config`` copies ten fields under a mirrored name. Every pair reads
one module constant; this pins that they still do. Unlinked declarations agree only by coincidence —
and for ``max_tokens`` they do not: a directly built ``RolloutConfig()`` generating 1024 tokens per
turn where the YAML path allows 32768 is a 32x truncation that reads as the model refusing to finish
rather than as a config bug.

The other three fields are DERIVED from different knobs, so their defaults are not required to
agree, and ``capture_token_ids``' does not: off in a hand-built config, on through the YAML path.
Those derivations are pinned below instead of exempted.

Run: pytest tests/cpu/config/test_rollout_config_mirror.py
"""

from dataclasses import fields

import pytest

from src.configs.async_training_config import AsyncTrainingConfig, rollout_field_sources
from src.configs.rollout_config import RolloutConfig

# RolloutConfig field -> the AsyncTrainingConfig field that supplies it, read from the builder's own
# mapping rather than restated: the map has to exist exactly once to be worth testing.
_FORWARDED = rollout_field_sources(AsyncTrainingConfig)

# The fields get_rollout_config computes instead of copying, each named with what it computes from.
_DERIVED = {
    "capture_token_ids": "train_on_sampled_tokens",
    "capture_routed_experts": "routing_replay",
    "stop_token_ids": "rollout_stop_tokens, resolved to ids by the trainer",
}

# A non-default value per mirrored knob. Applied one at a time (an sglang backend and a thinking
# budget are mutually exclusive), so a knob that stops being copied cannot pass by leaving the
# RolloutConfig default in place — which is the value the YAML was overriding.
_NON_DEFAULT_VALUES = {
    "rollout_backend": "sglang",
    "rollout_temperature": 0.3,
    "rollout_top_p": 0.5,
    "rollout_max_tokens": 4096,
    "rollout_max_thinking_tokens": 1024,
    "model_name": "org/rollout-model",
    "request_timeout": 45.0,
    "episode_timeout": 600.0,
    "max_retries": 1,
    "retry_base_wait": 2.5,
}


def _defaults(cls) -> dict:
    return {f.name: f.default for f in fields(cls)}


@pytest.mark.parametrize(("rollout_field", "async_field"), sorted(_FORWARDED.items()))
def test_forwarded_default_matches_on_both_sides(rollout_field, async_field):
    async_default = _defaults(AsyncTrainingConfig)[async_field]
    rollout_default = _defaults(RolloutConfig)[rollout_field]
    assert async_default == rollout_default, (
        f"AsyncTrainingConfig.{async_field}={async_default!r} but RolloutConfig.{rollout_field}="
        f"{rollout_default!r}: a directly built RolloutConfig silently runs a different budget from "
        f"the identical YAML run. Point both at one module constant in src/configs/rollout_config.py."
    )


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


def test_default_async_config_builds_an_identical_rollout_config():
    """The end-to-end statement of the same fact: on stock settings the built config equals the
    directly constructed one, so neither entry point is the odd one out."""
    built = AsyncTrainingConfig().get_rollout_config()
    for rollout_field in _FORWARDED:
        assert getattr(built, rollout_field) == getattr(RolloutConfig(), rollout_field), rollout_field


@pytest.mark.parametrize("sampled_tokens", [True, False])
def test_capture_token_ids_follows_train_on_sampled_tokens(sampled_tokens):
    """The knob is the only thing that turns id capture on, and the actors read only the derived
    field: a broken derivation trains env-GRPO on a re-tokenized re-render of every turn — the exact
    mismatch ``train_on_sampled_tokens`` exists to eliminate — with nothing in the logs saying so."""
    built = AsyncTrainingConfig(train_on_sampled_tokens=sampled_tokens).get_rollout_config()
    assert built.capture_token_ids is sampled_tokens


def test_the_yaml_path_captures_token_ids_where_a_hand_built_config_does_not():
    """The one place the two sides legitimately disagree, pinned so it stays a decision. Linking the
    two defaults would change what every hand-built RolloutConfig sends to the server; if that is
    wanted, flip the default here and correct the note on the field."""
    assert AsyncTrainingConfig().get_rollout_config().capture_token_ids is True
    assert RolloutConfig().capture_token_ids is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
