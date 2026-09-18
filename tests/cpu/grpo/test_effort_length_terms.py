#!/usr/bin/env python
"""CPU tests for the trainer-side effort length terms: the capped, effort-conditioned reasoning-length
price and the under-use floor.

The price is ``-min(c_max, k(effort) * reasoning_tokens / l_norm)`` with ``k`` falling by ``e`` per
``tau`` effort units, so one trace costs most at the lowest level and the cap keeps a long trace from
outweighing the task reward. The floor is the only term that pays for MORE reasoning: it prices an
episode's shortfall against a multiple of the per-turn thinking budget it ran under. Both read the
episode's reasoning summed over its turns — a short repair turn is not under-use.

Run: python tests/cpu/grpo/test_effort_length_terms.py  (or pytest)
"""

import math
import types
from collections import defaultdict

import pytest
import torch

from src.configs.async_training_config import AsyncTrainingConfig
from src.environments.base import Message, Trajectory
from src.environments.episode import RolloutResult, effort_length_floor, effort_length_penalty
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer
from tests.common.grpo_metrics import attach_world_metrics, flushed_metrics

PRICE = {"effort_min": 25.0, "k0": 0.05, "tau": 25.0, "c_max": 0.1, "l_norm": 8192.0}


def test_price_is_linear_in_reasoning_tokens_summed_over_turns():
    assert effort_length_penalty([4096], 25.0, **PRICE) == pytest.approx(-0.025)
    assert effort_length_penalty([1024, 3072], 25.0, **PRICE) == pytest.approx(-0.025), "turns are summed"
    assert effort_length_penalty([], 25.0, **PRICE) == 0.0
    assert effort_length_penalty([0, 0], 25.0, **PRICE) == 0.0


def test_price_is_capped_exactly_at_c_max():
    # 16384 tokens at the lowest level reaches the cap; ten times that trace pays no more.
    assert effort_length_penalty([16384], 25.0, **PRICE) == pytest.approx(-0.1)
    assert effort_length_penalty([163840], 25.0, **PRICE) == pytest.approx(-0.1)
    assert effort_length_penalty([16383], 25.0, **PRICE) > -0.1


def test_price_coefficient_falls_by_e_per_tau():
    low = effort_length_penalty([4096], 25.0, **PRICE)
    one_tau_up = effort_length_penalty([4096], 50.0, **PRICE)
    three_tau_up = effort_length_penalty([4096], 100.0, **PRICE)
    assert one_tau_up == pytest.approx(low / math.e)
    assert three_tau_up == pytest.approx(low / math.e**3)
    assert low < one_tau_up < three_tau_up < 0, "the same trace costs most at the lowest effort"


def test_floor_prices_the_episode_shortfall_and_nothing_above_the_minimum():
    assert effort_length_floor([8192], 8192, 0.05) == 0.0
    assert effort_length_floor([20000], 8192, 0.05) == 0.0
    assert effort_length_floor([4096], 8192, 0.05) == pytest.approx(-0.025)
    assert effort_length_floor([2048], 8192, 0.05) == pytest.approx(-0.0375)


def test_floor_sums_the_episode_so_a_short_repair_turn_is_not_under_use():
    # One deep turn then two terse repairs: a per-turn mean would price the repairs as under-use.
    assert effort_length_floor([9000, 100, 100], 8192, 0.05) == 0.0
    # An extra thinking-free tool turn cannot lower the score either.
    assert effort_length_floor([4096], 8192, 0.05) == effort_length_floor([4096, 0, 0], 8192, 0.05)


def test_floor_charges_silent_turns_in_full_and_a_lost_episode_nothing():
    # Turns that carry no reasoning at all are the maximal under-use...
    assert effort_length_floor([0, 0, 0], 8192, 0.05) == pytest.approx(-0.05)
    # ...an episode with no assistant turn is one the driver lost, not a policy choice.
    assert effort_length_floor([], 8192, 0.05) == 0.0
    # Off switches: no stamped minimum, or no weight.
    assert effort_length_floor([10], 0, 0.05) == 0.0
    assert effort_length_floor([10], 8192, 0.0) == 0.0


def _rollout(level, tokens, budget=None):
    trajectory = types.SimpleNamespace(reasoning_effort=level, reasoning_budget=budget, tokens=tokens)
    return types.SimpleNamespace(trajectory=trajectory)


def _apply(rollouts, **config):
    trainer = attach_world_metrics(object.__new__(DistributedAsyncEnvironmentalGRPOTrainer))
    trainer.async_config = AsyncTrainingConfig(**config)
    trainer._metrics = {"train": defaultdict(list)}
    trainer._assistant_turn_reasoning_tokens = lambda traj: traj.tokens
    rewards = torch.zeros(len(rollouts))
    trainer._apply_effort_length_terms(rewards, rollouts)
    return rewards, flushed_metrics(trainer)


def test_trainer_charges_each_episode_its_levels_price_and_the_floor_of_its_own_budget():
    rollouts = [
        _rollout("low", [8192], budget=8192),  # price only: at its floor
        _rollout("high", [8192], budget=16384),  # nearly free price, a third short of its 12288 floor
        _rollout(None, [8192]),  # no level, no budget: free of both
    ]
    rewards, metrics = _apply(rollouts, effort_length_penalty_k0=0.05, effort_length_floor_weight=0.05)
    assert rewards[0].item() == pytest.approx(-0.05)
    assert rewards[1].item() == pytest.approx(-0.05 * math.e**-3 - 0.05 / 3)
    assert rewards[2].item() == 0.0
    assert metrics["reward/effort_length_penalty"] == [pytest.approx((-0.05 - 0.05 * math.e**-3) / 3)]
    assert metrics["reward/effort_length_floor"] == [pytest.approx(-0.05 / 3 / 3)]


def test_the_floor_is_a_multiple_of_the_per_turn_budget():
    # The default sits below one budget, so a single assistant turn can clear it under the engine's cap.
    rollouts = [_rollout("low", [2048], budget=8192)]
    rewards, _ = _apply(rollouts, effort_length_floor_weight=0.05)
    assert rewards[0].item() == pytest.approx(-0.05 * 4096 / 6144), "the default floor is 0.75 x the budget"
    rewards, _ = _apply(rollouts, effort_length_floor_weight=0.05, effort_length_floor_budgets=1.0)
    assert rewards[0].item() == pytest.approx(-0.05 * 6144 / 8192)
    rewards, _ = _apply(rollouts, effort_length_floor_weight=0.05, effort_length_floor_budgets=0.25)
    assert rewards[0].item() == 0.0, "2048 tokens meets a quarter of an 8192 budget"
    rewards, _ = _apply(rollouts, effort_length_floor_weight=0.05, effort_length_floor_budgets=0.5)
    assert rewards[0].item() == pytest.approx(-0.05 * 2048 / 4096)


def test_each_term_is_off_on_its_own_and_records_only_its_own_metric():
    rollouts = [_rollout("low", [4096], budget=8192)]
    rewards, metrics = _apply(rollouts, effort_length_penalty_k0=0.05)
    assert rewards[0].item() == pytest.approx(-0.025) and "reward/effort_length_floor" not in metrics
    rewards, metrics = _apply(rollouts, effort_length_floor_weight=0.05)
    assert rewards[0].item() == pytest.approx(-0.05 / 3) and "reward/effort_length_penalty" not in metrics


class _CharTokenizer:
    """One token per character: deterministic reasoning lengths without a real tokenizer."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [0] * len(text)}


def _episode(thinkings, budget=1000, level="low"):
    turns = (Message.assistant("step", thinking=thinking) for thinking in thinkings)
    trajectory = Trajectory(messages=[Message.user("task"), *turns])
    trajectory.reasoning_budget, trajectory.reasoning_effort = budget, level
    return RolloutResult(prompt="task", trajectory=trajectory)


def test_the_reward_build_applies_both_terms_over_every_assistant_turn():
    """The real path: the trainer's own reward build and its per-turn token counter. Every assistant
    turn is one entry, a thinking-free one as 0 — a counter that skipped them would hand the floor an
    empty list, which it reads as a lost episode, so dropping the reasoning entirely would be free
    while brief reasoning paid nearly the whole weight."""
    trainer = attach_world_metrics(object.__new__(DistributedAsyncEnvironmentalGRPOTrainer))
    trainer.async_config = AsyncTrainingConfig(effort_length_penalty_k0=0.05, effort_length_floor_weight=0.05)
    trainer._tokenizer = _CharTokenizer()
    trainer._metrics = {"train": defaultdict(list)}
    trainer._carry_reasoning = False
    trainer._warned_once = set()
    episodes = [
        _episode([None, "", None]),  # three silent turns: the whole floor, nothing to price
        _episode([]),  # no assistant turn: a lost episode, free
        _episode(["x" * 750]),  # exactly 0.75 x its 1000 budget: no floor, 750 tokens priced
    ]
    rewards = trainer._build_rollout_rewards(episodes, torch.device("cpu"))
    assert rewards[0].item() == pytest.approx(-0.05)
    assert rewards[1].item() == 0.0
    assert rewards[2].item() == pytest.approx(-0.05 * 750 / 8192)


def test_a_missing_trajectory_is_free():
    rewards, _ = _apply(
        [types.SimpleNamespace(trajectory=None)], effort_length_penalty_k0=0.05, effort_length_floor_weight=0.05
    )
    assert rewards[0].item() == 0.0


def test_config_refuses_values_that_would_invert_or_poison_the_terms():
    assert AsyncTrainingConfig().effort_length_penalty_k0 is None, "the price is off by default"
    assert AsyncTrainingConfig().effort_length_floor_weight == 0.0, "the floor is off by default"
    for name in ("effort_length_penalty_tau", "effort_length_penalty_c_max", "effort_length_penalty_l_norm"):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError, match=name):
                AsyncTrainingConfig(effort_length_penalty_k0=0.05, **{name: bad})
        AsyncTrainingConfig(**{name: -1.0})  # unchecked while the price is off
    for bad in (0.0, -0.05, float("nan")):
        with pytest.raises(ValueError, match="effort_length_penalty_k0"):
            AsyncTrainingConfig(effort_length_penalty_k0=bad)
    with pytest.raises(ValueError, match="effort_length_penalty_levels"):
        AsyncTrainingConfig(effort_length_penalty_k0=0.05, effort_length_penalty_levels={"low": float("nan")})
    for bad in (-0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="effort_length_floor_weight"):
            AsyncTrainingConfig(effort_length_floor_weight=bad)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="effort_length_floor_budgets"):
            AsyncTrainingConfig(effort_length_floor_weight=0.05, effort_length_floor_budgets=bad)
    AsyncTrainingConfig(effort_length_floor_budgets=-1.0)  # unchecked while the floor is off


def _constructing(budgets, **config):
    trainer = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    trainer.async_config = AsyncTrainingConfig(**config)
    trainer._rollout_env = types.SimpleNamespace(thinking_budget_for_effort=budgets.get)
    return trainer


def test_construction_refuses_a_price_table_that_misses_or_invents_a_level():
    with pytest.raises(ValueError, match="missing \\['high'\\]"):
        _constructing(
            {}, effort_length_penalty_k0=0.05, effort_length_penalty_levels={"low": 25.0, "medium": 50.0}
        )._validate_effort_length_terms()
    with pytest.raises(ValueError, match="unknown \\['extreme'\\]"):
        _constructing(
            {},
            effort_length_penalty_k0=0.05,
            effort_length_penalty_levels={"low": 25.0, "medium": 50.0, "high": 100.0, "extreme": 200.0},
        )._validate_effort_length_terms()
    _constructing({}, effort_length_penalty_k0=0.05)._validate_effort_length_terms()


def test_construction_refuses_a_floor_no_episode_can_ever_be_priced_by():
    with pytest.raises(ValueError, match="no episode can carry a thinking budget"):
        _constructing({}, effort_length_floor_weight=0.05)._validate_effort_length_terms()
    # One budgeted level is enough, and so is the run-wide cap.
    _constructing({"high": 16384}, effort_length_floor_weight=0.05)._validate_effort_length_terms()
    _constructing(
        {}, effort_length_floor_weight=0.05, rollout_max_thinking_tokens=8000
    )._validate_effort_length_terms()
    _constructing({})._validate_effort_length_terms()  # off: nothing to check


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
