#!/usr/bin/env python
"""CPU tests for the trainer-side per-effort token cost.

The env stamps ``episode_token_cost`` (reward units per 1k generated tokens) from the effort
profile; the trainer charges it against the episode's TOTAL generated tokens. A cost, never a
target: it can only be avoided by economy, so unlike length floors it is not farmable.

Run: python tests/cpu/grpo/test_effort_token_costs.py  (or pytest)
"""

import types
from collections import defaultdict

import pytest
import torch

from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer
from tests.common.grpo_metrics import attach_world_metrics, flushed_metrics


def _rollout(price: float | None, tokens: int):
    info = {} if price is None else {"episode_token_cost": price}
    return types.SimpleNamespace(trajectory=types.SimpleNamespace(info=info), generation_tokens=tokens)


def _apply(rollouts):
    trainer = attach_world_metrics(object.__new__(DistributedAsyncEnvironmentalGRPOTrainer))
    trainer._metrics = {"train": defaultdict(list)}
    rewards = torch.zeros(len(rollouts))
    trainer._apply_effort_token_costs(rewards, rollouts)
    return rewards, flushed_metrics(trainer)


def test_cost_charges_per_generated_token():
    rewards, metrics = _apply([_rollout(0.05, 2000), _rollout(None, 2000), _rollout(0.0, 5000)])
    assert rewards[0].item() == pytest.approx(-0.1)  # 0.05/1k x 2000
    assert rewards[1].item() == 0.0  # unstamped episode is free
    assert rewards[2].item() == 0.0  # explicit zero price is free
    assert metrics["reward/token_cost"] == [pytest.approx(-0.1 / 3)]


def test_all_free_episodes_log_a_zero_cost_over_the_whole_batch():
    """The key is recorded on every rank whatever it charged: ``WorldMetrics`` folds the union of what
    the ranks recorded, so a rank that skipped the record would drop out of the world denominator and
    the logged mean would be an average over the ranks that charged something, not over the batch."""
    rewards, metrics = _apply([_rollout(None, 3000), _rollout(0.0, 3000)])
    assert rewards.sum().item() == 0.0
    assert metrics["reward/token_cost"] == [pytest.approx(0.0)]


def test_missing_trajectory_is_free():
    rollout = types.SimpleNamespace(trajectory=None, generation_tokens=9999)
    rewards, _ = _apply([rollout])
    assert rewards[0].item() == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
