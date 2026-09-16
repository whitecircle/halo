#!/usr/bin/env python
"""The IS trust-region breaker's verdict holds for the WHOLE generation round.

A round spans ``steps_per_generation * num_iterations`` micro-batches — ``num_iterations`` optimizer
steps at the default ``steps_per_generation`` — and ``_build_training_tensors`` zeroes the advantages
of every one of them. The optimizer skip must therefore fire on every one of those steps too: a skip
that consumed the flag on first use left the later steps running Adam on momentum alone, exactly the
weight movement the skip exists to prevent. ``_update_breaker_tripped`` owns the flag: it sets this
round's verdict and, on the next round, replaces it.

    python tests/cpu/grpo/test_env_trainer_breaker_round.py
"""

import types
from collections import defaultdict

import pytest
import torch
from accelerate import PartialState

from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer

PartialState()  # the breaker warns through accelerate's logger, which refuses to log without it


def _breaker_host(threshold: float, tripped: bool = False):
    host = types.SimpleNamespace(
        _skip_update_masked_frac=threshold,
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
    host = types.SimpleNamespace(
        _breaker_tripped_this_step=True,
        optimizer=types.SimpleNamespace(zero_grad=lambda set_to_none: calls.append(set_to_none)),
    )
    skip = DistributedAsyncEnvironmentalGRPOTrainer._skip_optimizer_step_if_breaker_tripped.__get__(host)
    assert [skip(), skip(), skip()] == [True, True, True], "a round of three optimizer steps must skip all three"
    assert calls == [True, True, True]
    assert host._breaker_tripped_this_step is True, "the skip must not consume the round's verdict"


def test_the_next_round_verdict_replaces_a_tripped_one():
    """The flag comes down only where it went up: a healthy round after a tripped one re-enables the
    optimizer step, and a tripped round after a healthy one arms it."""
    tripped, host = _breaker_host(0.5, tripped=True)
    assert tripped(*_batch([True, False, False, False]), 4, "train") is False
    assert host._breaker_tripped_this_step is False
    assert tripped(*_batch([True, True, True, False]), 4, "train") is True
    assert host._breaker_tripped_this_step is True


def test_an_eval_round_leaves_the_training_verdict_standing():
    """Evaluation can run between two optimizer steps of one generation round; its batch build must
    not clear the verdict those steps still act on."""
    tripped, host = _breaker_host(0.5, tripped=True)
    assert tripped(*_batch([False, False, False, False]), 4, "eval") is False
    assert host._breaker_tripped_this_step is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
