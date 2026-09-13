#!/usr/bin/env python
"""The capped, effort-conditioned reasoning-length penalty: the pure term and its trainer application.

``-min(c_max, k(effort) * reasoning_tokens / l_norm)`` with ``k(effort) = k0 * exp(-(effort - effort_min) / tau)``:
capped exactly at ``c_max``, one ``e`` cheaper per ``tau`` effort units, priced on reasoning tokens summed over
the turns, and applied per episode from the trajectory's effort level — off until ``k0`` is set.

    python tests/cpu/grpo/test_effort_length_penalty.py
"""

import math
import types
from collections import defaultdict

import pytest
import torch

from src.configs.async_training_config import AsyncTrainingConfig
from src.environments.episode import effort_length_penalty
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer

K0, TAU, C_MAX, L_NORM = 0.4, 25.0, 0.5, 8192.0


def test_penalty_prices_summed_reasoning_tokens_and_caps_at_c_max():
    assert effort_length_penalty([], 25, 25, K0, TAU, C_MAX, L_NORM) == 0.0
    assert effort_length_penalty([0, 0], 25, 25, K0, TAU, C_MAX, L_NORM) == 0.0
    one_turn = effort_length_penalty([4096], 25, 25, K0, TAU, C_MAX, L_NORM)
    assert one_turn == pytest.approx(-K0 * 4096 / L_NORM)
    assert effort_length_penalty([2048, 2048], 25, 25, K0, TAU, C_MAX, L_NORM) == pytest.approx(one_turn), "turns sum"
    assert effort_length_penalty([10**6], 25, 25, K0, TAU, C_MAX, L_NORM) == -C_MAX, "capped, not linear"
    assert effort_length_penalty([int(C_MAX * L_NORM / K0)], 25, 25, K0, TAU, C_MAX, L_NORM) == pytest.approx(-C_MAX)


def test_coefficient_falls_by_e_per_tau_effort_units():
    low = effort_length_penalty([4096], 25, 25, K0, TAU, C_MAX, L_NORM)
    one_tau_higher = effort_length_penalty([4096], 50, 25, K0, TAU, C_MAX, L_NORM)
    assert one_tau_higher / low == pytest.approx(math.exp(-1))
    assert abs(one_tau_higher) < abs(low)


def _rollout(level, reasoning):
    traj = types.SimpleNamespace(reasoning_effort=level, reasoning=reasoning)
    return types.SimpleNamespace(trajectory=traj)


def _trainer(k0=K0, levels=None):
    host = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    host.async_config = AsyncTrainingConfig(
        effort_length_penalty_k0=k0,
        effort_length_penalty_tau=TAU,
        effort_length_penalty_c_max=C_MAX,
        effort_length_penalty_l_norm=L_NORM,
        **({"effort_length_penalty_levels": levels} if levels is not None else {}),
    )
    host._metrics = {"train": defaultdict(list)}
    host._assistant_turn_reasoning_tokens = lambda traj: traj.reasoning
    return host


def test_same_trace_costs_more_at_low_effort_and_is_logged_per_level():
    host = _trainer()
    rollouts = [_rollout("low", [4096, 4096]), _rollout("high", [4096, 4096]), _rollout(None, [4096])]
    rewards = torch.ones(3)
    host._apply_effort_length_penalty(rewards, rollouts, "train")
    assert rewards[0] < rewards[1] < 1.0, "low pays more than high; both pay"
    assert rewards[2] == 1.0, "an episode without an effort level is not priced"
    metrics = host._metrics["train"]
    assert metrics["reward/effort_length_penalty"] == [pytest.approx(((rewards[0] - 1) + (rewards[1] - 1)).item() / 3)]
    assert metrics["effort/low/length_penalty"][0] < metrics["effort/high/length_penalty"][0]


def test_penalty_is_off_until_k0_is_set():
    host = _trainer(k0=None)
    rewards = torch.ones(1)
    host._apply_effort_length_penalty(rewards, [_rollout("low", [10**6])], "train")
    assert rewards[0] == 1.0 and not host._metrics["train"]


def test_level_table_must_name_exactly_the_effort_levels():
    _trainer()._validate_effort_length_penalty_levels()
    _trainer(k0=None, levels={"low": 1.0})._validate_effort_length_penalty_levels()
    with pytest.raises(ValueError, match="missing \\['high'\\]"):
        _trainer(levels={"low": 25.0, "medium": 50.0})._validate_effort_length_penalty_levels()
    with pytest.raises(ValueError, match="unknown \\['max'\\]"):
        _trainer(
            levels={"low": 25.0, "medium": 50.0, "high": 100.0, "max": 200.0}
        )._validate_effort_length_penalty_levels()


def test_config_refuses_non_positive_penalty_parameters():
    with pytest.raises(ValueError, match="effort_length_penalty_tau"):
        AsyncTrainingConfig(effort_length_penalty_k0=0.4, effort_length_penalty_tau=0.0)
    with pytest.raises(ValueError, match="effort_length_penalty_c_max"):
        AsyncTrainingConfig(effort_length_penalty_k0=0.4, effort_length_penalty_c_max=-1.0)
    AsyncTrainingConfig(effort_length_penalty_tau=0.0)  # inert while the penalty is off


def test_config_refuses_level_values_that_are_not_finite_numbers():
    """A string or NaN effort would reach the penalty's subtraction at the first rollout."""
    for bad in ("1", float("nan"), True, None):
        with pytest.raises(ValueError, match="effort_length_penalty_levels\\['medium'\\] must be a finite number"):
            AsyncTrainingConfig(
                effort_length_penalty_k0=0.4, effort_length_penalty_levels={"low": 0.0, "medium": bad, "high": 2.0}
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
