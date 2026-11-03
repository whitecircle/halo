#!/usr/bin/env python
"""The IS trust-region breaker's verdict holds for the whole generation round. A skip that consumed the flag on first
use would leave the round's later steps running Adam on momentum alone, the movement the skip exists to prevent."""

import types
from collections import defaultdict

import pytest
import torch

from halo.trainers.grpo.environmental.importance_sampling import _BreakerOptimizerSkipCallback
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer


class _TrainerStandIn(types.SimpleNamespace):
    """Weak-referenceable, as the trainer the breaker's hook holds is."""


def _skip_hook(host):
    callback = _BreakerOptimizerSkipCallback(host)
    return lambda: callback.on_pre_optimizer_step(None, None, None)


def _breaker_host(threshold: float, tripped: bool = False):
    host = types.SimpleNamespace(
        async_config=types.SimpleNamespace(skip_update_masked_frac=threshold),
        _breaker_tripped_this_step=tripped,
        accelerator=types.SimpleNamespace(gather=lambda x: x),
        _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
    )
    return DistributedAsyncEnvironmentalGRPOTrainer._update_breaker_tripped.__get__(host), host


def _batch(masked_trajs: list[bool]):
    n = len(masked_trajs)
    ratio = torch.ones(n, 4)
    for i, masked in enumerate(masked_trajs):
        if masked:
            ratio[i] = 0.0
    return ratio, torch.ones(n, 4, dtype=torch.bool), torch.arange(n)


def test_the_skip_fires_on_every_optimizer_step_while_the_round_is_tripped():
    calls = []
    host = _TrainerStandIn(
        _breaker_tripped_this_step=True,
        optimizer=types.SimpleNamespace(zero_grad=lambda set_to_none: calls.append(set_to_none)),
    )
    skip = _skip_hook(host)
    for _ in range(3):
        skip()
    assert calls == [True, True, True], "a round of three optimizer steps must skip all three"
    assert host._breaker_tripped_this_step is True, "the skip must not consume the round's verdict"


def test_the_next_round_verdict_replaces_a_tripped_one():
    tripped, host = _breaker_host(0.5, tripped=True)
    assert tripped(*_batch([True, False, False, False]), 4, "train") is False
    assert host._breaker_tripped_this_step is False
    assert tripped(*_batch([True, True, True, False]), 4, "train") is True
    assert host._breaker_tripped_this_step is True


def test_an_eval_round_leaves_the_training_verdict_standing():
    """Evaluation can run between two optimizer steps of one generation round."""
    tripped, host = _breaker_host(0.5, tripped=True)
    assert tripped(*_batch([False, False, False, False]), 4, "eval") is False
    assert host._breaker_tripped_this_step is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
