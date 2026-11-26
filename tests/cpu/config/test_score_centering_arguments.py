#!/usr/bin/env python
"""The ``score_centering_*`` knobs: one bundle on the environmental GRPO config, validated at parse. Online GRPO rolls
out through vLLM only, whose candidates cost most of its throughput, and offline GRPO has no sampler: neither has the
knob, so their parsers refuse it as an unknown key."""

import dataclasses

import pytest

from halo.configs.grpo import ScoreCenteringArguments, ScoreCenteringConfig
from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.configs.grpo_offline import OfflineGRPOConfig
from halo.configs.grpo_online import OnlineGRPOScriptArguments


def test_the_bundle_is_on_the_environmental_config_and_off_by_default():
    assert issubclass(AsyncRolloutConfig, ScoreCenteringArguments)
    assert AsyncRolloutConfig().build_score_centering_config() is None


@pytest.mark.parametrize("config", [OnlineGRPOScriptArguments, OfflineGRPOConfig])
def test_online_and_offline_grpo_have_no_knob(config):
    assert not issubclass(config, ScoreCenteringArguments)
    assert "score_centering_top_k" not in {field.name for field in dataclasses.fields(config)}


def test_the_config_carries_the_knobs():
    config = AsyncRolloutConfig(
        score_centering_top_k=16,
        score_centering_weights="mis",
        score_centering_mis_low=0.25,
        score_centering_mis_high=4,
    ).build_score_centering_config()
    assert config == ScoreCenteringConfig(top_k=16, weights="mis", mis_low=0.25, mis_high=4)
    assert AsyncRolloutConfig(score_centering_top_k=32).build_score_centering_config() == ScoreCenteringConfig(
        top_k=32
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"score_centering_top_k": 0}, "score_centering_top_k must be an integer >= 1"),
        ({"score_centering_top_k": 1.5}, "score_centering_top_k must be an integer"),
        ({"score_centering_top_k": 8, "score_centering_weights": "huber"}, "score_centering_weights must be one of"),
        (
            {"score_centering_top_k": 8, "score_centering_weights": "mis", "score_centering_mis_low": 6.0},
            "must not exceed",
        ),
        (
            {"score_centering_top_k": 8, "score_centering_weights": "mis", "score_centering_mis_low": 0.0},
            "score_centering_mis_low must be > 0",
        ),
        ({"score_centering_weights": "none"}, "score_centering_top_k is unset"),
        ({"score_centering_mis_high": 9.0}, "score_centering_top_k is unset"),
        ({"score_centering_top_k": 8, "score_centering_mis_high": 9.0}, "bound the 'mis' weight"),
    ],
)
def test_a_malformed_or_inert_knob_is_refused_at_parse(kwargs, match):
    with pytest.raises(ValueError, match=match):
        AsyncRolloutConfig(**kwargs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
