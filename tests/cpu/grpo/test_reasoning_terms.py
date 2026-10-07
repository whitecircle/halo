#!/usr/bin/env python
"""CPU tests for the trainer-side reasoning terms: the per-level, capped reasoning price and the
under-use floor.

The price is ``-min(cap, price_per_1k * reasoning_tokens / 1000)`` at the level's own price, so one
trace costs most where little reasoning was asked and the cap keeps a long trace from outweighing the
task reward. The floor is the only term that pays for MORE reasoning: it prices an episode's shortfall
against three quarters of the per-turn thinking budget it ran under. Both read the episode's reasoning
summed over its turns — a short repair turn is not under-use.

Run: python tests/cpu/grpo/test_reasoning_terms.py  (or pytest)
"""

import types
from collections import defaultdict

import pytest
import torch

from src.configs.async_training_config import AsyncTrainingConfig
from src.environments.base import Message, Trajectory
from src.environments.episode import RolloutResult
from src.trainers.grpo.environmental import BatchBuildFence, DistributedAsyncEnvironmentalGRPOTrainer
from src.trainers.grpo.reasoning_terms import reasoning_floor_term, reasoning_price_term
from tests.common.grpo_metrics import attach_world_metrics, flushed_metrics

PRICE = {"low": 0.01, "medium": 0.005, "high": 0.001}
CAP = 0.1


def test_price_is_linear_in_reasoning_tokens_per_thousand_summed_over_turns():
    assert reasoning_price_term([4096], 0.01, CAP) == pytest.approx(-0.04096)
    assert reasoning_price_term([1024, 3072], 0.01, CAP) == pytest.approx(-0.04096), "turns are summed"
    assert reasoning_price_term([500], 0.01, CAP) == pytest.approx(-0.005), "priced per 1,000 tokens"
    assert reasoning_price_term([], 0.01, CAP) == 0.0
    assert reasoning_price_term([0, 0], 0.01, CAP) == 0.0
    assert reasoning_price_term([4096], 0.0, CAP) == 0.0, "a level priced at zero pays nothing"


def test_price_is_capped_exactly_at_the_cap():
    # 10000 tokens at 0.01 per 1k reach the cap; ten times that trace pays no more.
    assert reasoning_price_term([10000], 0.01, CAP) == pytest.approx(-CAP)
    assert reasoning_price_term([100000], 0.01, CAP) == pytest.approx(-CAP)
    assert reasoning_price_term([9999], 0.01, CAP) > -CAP
    assert reasoning_price_term([100000], 0.01, 0.3) == pytest.approx(-0.3)


def test_floor_prices_the_episode_shortfall_and_nothing_above_the_target():
    # The target under an 8192-token cap is 6144 tokens; the shortfall is priced against it.
    assert reasoning_floor_term([8192], 8192, 0.05) == 0.0
    assert reasoning_floor_term([6144], 8192, 0.05) == 0.0
    assert reasoning_floor_term([4096], 8192, 0.05) == pytest.approx(-0.05 * 2048 / 6144)
    assert reasoning_floor_term([2048], 8192, 0.05) == pytest.approx(-0.05 * 4096 / 6144)


def test_floor_sums_the_episode_so_a_short_repair_turn_is_not_under_use():
    # One deep turn then two terse repairs: a per-turn mean would price the repairs as under-use.
    assert reasoning_floor_term([9000, 100, 100], 8192, 0.05) == 0.0
    # An extra thinking-free tool turn cannot lower the score either.
    assert reasoning_floor_term([4096], 8192, 0.05) == reasoning_floor_term([4096, 0, 0], 8192, 0.05)


def test_floor_charges_silent_turns_in_full_and_a_lost_episode_nothing():
    # Turns that carry no reasoning at all are the maximal under-use...
    assert reasoning_floor_term([0, 0, 0], 8192, 0.05) == pytest.approx(-0.05)
    # ...an episode with no assistant turn is one the driver lost, not a policy choice.
    assert reasoning_floor_term([], 8192, 0.05) == 0.0
    # Off switches: no cap, or no weight.
    assert reasoning_floor_term([10], 0, 0.05) == 0.0
    assert reasoning_floor_term([10], 8192, 0.0) == 0.0


def _rollout(level, tokens, budget=None):
    trajectory = types.SimpleNamespace(reasoning_effort=level, reasoning_budget=budget, tokens=tokens)
    return types.SimpleNamespace(trajectory=trajectory)


def _apply(rollouts, **config):
    trainer = attach_world_metrics(object.__new__(DistributedAsyncEnvironmentalGRPOTrainer))
    trainer.async_config = AsyncTrainingConfig(**config)
    trainer._metrics = {"train": defaultdict(list)}
    rewards = torch.zeros(len(rollouts))
    trainer._apply_reasoning_terms(rewards, rollouts, [r.trajectory.tokens if r.trajectory else [] for r in rollouts])
    return rewards, flushed_metrics(trainer)


def test_trainer_charges_each_episode_its_levels_price_and_the_floor_of_its_own_budget():
    rollouts = [
        _rollout("low", [8192], budget=8192),  # price only: at its floor
        _rollout("high", [8192], budget=16384),  # a tenth of low's price, a third short of its 12288 floor
        _rollout(None, [8192]),  # no level, no budget: free of both
    ]
    rewards, metrics = _apply(rollouts, reasoning_price=PRICE, reasoning_floor=0.05)
    assert rewards[0].item() == pytest.approx(-0.08192)
    assert rewards[1].item() == pytest.approx(-0.008192 - 0.05 / 3)
    assert rewards[2].item() == 0.0
    assert metrics["reward/reasoning_price"] == [pytest.approx((-0.08192 - 0.008192) / 3)]
    assert metrics["reward/reasoning_floor"] == [pytest.approx(-0.05 / 3 / 3)]


def test_the_same_trace_costs_most_at_the_lowest_level_and_is_capped_per_episode():
    rollouts = [_rollout(level, [4096]) for level in ("low", "medium", "high")]
    rewards, _ = _apply(rollouts, reasoning_price=PRICE)
    low, medium, high = rewards.tolist()
    assert low < medium < high < 0
    assert (low, medium, high) == pytest.approx((-0.04096, -0.02048, -0.004096))
    capped, _ = _apply([_rollout("low", [50000])], reasoning_price=PRICE, reasoning_price_cap=0.2)
    assert capped[0].item() == pytest.approx(-0.2)


def test_the_floor_is_three_quarters_of_the_per_turn_budget():
    # Below one budget, so a single assistant turn can clear it under the cap the engine enforces per turn.
    rewards, _ = _apply([_rollout("low", [2048], budget=8192)], reasoning_floor=0.05)
    assert rewards[0].item() == pytest.approx(-0.05 * 4096 / 6144)
    rewards, _ = _apply([_rollout("low", [750], budget=1000)], reasoning_floor=0.05)
    assert rewards[0].item() == 0.0, "750 tokens clear three quarters of a 1000-token budget"
    rewards, _ = _apply([_rollout("low", [749], budget=1000)], reasoning_floor=0.05)
    assert rewards[0].item() == pytest.approx(-0.05 / 750)


def test_each_term_is_off_on_its_own_and_records_only_its_own_metric():
    rollouts = [_rollout("low", [4096], budget=8192)]
    rewards, metrics = _apply(rollouts, reasoning_price=PRICE)
    assert rewards[0].item() == pytest.approx(-0.04096) and "reward/reasoning_floor" not in metrics
    rewards, metrics = _apply(rollouts, reasoning_floor=0.05)
    assert rewards[0].item() == pytest.approx(-0.05 / 3) and "reward/reasoning_price" not in metrics


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
    trainer.async_config = AsyncTrainingConfig(reasoning_price=PRICE, reasoning_floor=0.05)
    trainer._tokenizer = _CharTokenizer()
    trainer._metrics = {"train": defaultdict(list)}
    trainer._carry_reasoning = False
    trainer._warned_once = set()
    episodes = [
        _episode([None, "", None]),  # three silent turns: the whole floor, nothing to price
        _episode([]),  # no assistant turn: a lost episode, free
        _episode(["x" * 750]),  # exactly 0.75 x its 1000 budget: no floor, 750 tokens priced
    ]
    rewards = trainer._build_rollout_rewards(
        episodes, trainer._episode_reasoning_tokens(episodes), torch.device("cpu")
    )
    assert rewards[0].item() == pytest.approx(-0.05)
    assert rewards[1].item() == 0.0
    assert rewards[2].item() == pytest.approx(-0.01 * 750 / 1000)


def test_a_missing_trajectory_is_free():
    rewards, _ = _apply([types.SimpleNamespace(trajectory=None)], reasoning_price=PRICE, reasoning_floor=0.05)
    assert rewards[0].item() == 0.0


def test_config_refuses_values_that_would_invert_or_poison_the_terms():
    defaults = AsyncTrainingConfig()
    assert defaults.reasoning_price is None, "the price is off by default"
    assert defaults.reasoning_floor == 0.0, "the floor is off by default"
    assert defaults.reasoning_price_cap == 0.1
    for empty in ({}, [0.01], 0.01, "low: 0.01"):
        with pytest.raises(ValueError, match="reasoning_price must map each effort level to a price"):
            AsyncTrainingConfig(reasoning_price=empty)
    for bad in (-0.01, float("nan"), float("inf"), True, "0.01", None):
        with pytest.raises(ValueError, match=r"reasoning_price\['low'\] must be a finite number >= 0"):
            AsyncTrainingConfig(reasoning_price={**PRICE, "low": bad})
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match=r"reasoning_price_cap must be (a finite number|> 0)"):
            AsyncTrainingConfig(reasoning_price=PRICE, reasoning_price_cap=bad)
    for bad in (-0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="reasoning_floor must be a finite number >= 0"):
            AsyncTrainingConfig(reasoning_floor=bad)
    # A price of zero on one level is a priced level that pays nothing, not a mistake.
    assert AsyncTrainingConfig(reasoning_price={**PRICE, "high": 0}).reasoning_price["high"] == 0


def test_a_price_cap_with_the_price_off_is_refused_as_inert():
    """The cap is read only beside a price: set alone it parses and changes nothing. Its default value
    passes with the price off, and any value is free while the price is on."""
    with pytest.raises(ValueError, match="reasoning_price_cap set with reasoning_price unset"):
        AsyncTrainingConfig(reasoning_price_cap=0.2)
    assert AsyncTrainingConfig(reasoning_price_cap=0.1).reasoning_price is None
    assert AsyncTrainingConfig(reasoning_price=PRICE, reasoning_price_cap=0.2).reasoning_price_cap == 0.2


def _constructing(budgets, effort="random", **config):
    trainer = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    trainer.async_config = AsyncTrainingConfig(**config)
    trainer._rollout_env = types.SimpleNamespace(reasoning_effort=effort, thinking_budget_for_effort=budgets.get)
    return trainer


def test_construction_refuses_a_price_table_that_misses_or_invents_a_level():
    with pytest.raises(ValueError, match=r"missing \['high'\]"):
        _constructing({}, reasoning_price={"low": 0.01, "medium": 0.005})._validate_reasoning_terms()
    with pytest.raises(ValueError, match=r"unknown \['extreme'\]"):
        _constructing({}, reasoning_price={**PRICE, "extreme": 0.0})._validate_reasoning_terms()
    _constructing({}, reasoning_price=PRICE)._validate_reasoning_terms()
    _constructing({})._validate_reasoning_terms()  # the price off: no table to check


def test_construction_refuses_a_price_with_no_level_to_price_at():
    """The price reads the episode's level; an environment drawing none would charge nothing, silently."""
    with pytest.raises(ValueError, match="reasoning_effort is unset"):
        _constructing({}, effort=None, reasoning_price=PRICE)._validate_reasoning_terms()
    _constructing({}, effort="low", reasoning_price=PRICE)._validate_reasoning_terms()


def test_a_budget_on_a_level_the_environment_never_draws_budgets_no_episode():
    """The floor's gate reads the levels the effort setting can draw: a fixed setting draws one, an unset one
    none, so a budget elsewhere would leave the floor never pricing an episode."""
    for effort in ("low", None):
        with pytest.raises(ValueError, match="no episode of this run carries one"):
            _constructing({"high": 16384}, effort=effort, reasoning_floor=0.05)._validate_reasoning_terms()
    _constructing({"high": 16384}, effort="high", reasoning_floor=0.05)._validate_reasoning_terms()
    _constructing(
        {}, effort=None, reasoning_floor=0.05, rollout_max_tokens=30000, rollout_max_thinking_tokens=8000
    )._validate_reasoning_terms()


def test_construction_refuses_a_floor_no_episode_can_ever_be_priced_by():
    with pytest.raises(ValueError, match="no episode of this run carries one"):
        _constructing({}, reasoning_floor=0.05)._validate_reasoning_terms()
    # One budgeted level is enough, and so is the run-wide cap.
    _constructing({"high": 16384}, reasoning_floor=0.05)._validate_reasoning_terms()
    _constructing({}, reasoning_floor=0.05, rollout_max_thinking_tokens=8000)._validate_reasoning_terms()
    _constructing({})._validate_reasoning_terms()  # off: nothing to check


class _CountingTokenizer(_CharTokenizer):
    def __init__(self):
        self.calls = 0

    def __call__(self, text, add_special_tokens=False):
        self.calls += 1
        return super().__call__(text, add_special_tokens)


class _Round:
    """The real round entry over two collected episodes, recording what each consumer was handed.
    ``leader`` stands in for the TP/ETP group leader's rollouts, which the broadcast hands every rank."""

    _generate_and_score_completions_base = (
        DistributedAsyncEnvironmentalGRPOTrainer._generate_and_score_completions_base
    )
    _extract_prompts_and_contexts = DistributedAsyncEnvironmentalGRPOTrainer._extract_prompts_and_contexts
    _episode_reasoning_tokens = DistributedAsyncEnvironmentalGRPOTrainer._episode_reasoning_tokens
    eval_split_rows = DistributedAsyncEnvironmentalGRPOTrainer.eval_split_rows

    def __init__(self, leader: list[RolloutResult] | None = None):
        self.model = types.SimpleNamespace(training=True)
        self.accelerator = types.SimpleNamespace(device=torch.device("cpu"), is_main_process=False)
        self.state, self.args = types.SimpleNamespace(global_step=0), types.SimpleNamespace(report_to=[])
        self._leader = leader
        self._group_random_effort, self._batch_errors, self._prefetch_enabled = False, BatchBuildFence(), False
        self._tokenizer = _CountingTokenizer()
        episodes = [_episode(["xx", None, "xxxx"]), _episode(["xxx"])]
        self._loop = types.SimpleNamespace(run_until_complete=lambda batch: batch)
        self._rollout_manager = types.SimpleNamespace(collect_rollouts=lambda prompts, contexts: episodes)
        self.handed: dict[str, list] = {}

    def _broadcast_rollouts_for_tp(self, rollout_results):
        return rollout_results if self._leader is None else self._leader

    def _build_training_tensors(self, rollout_results, device, mode, num_padding, reasoning_tokens):
        self.handed["rewards"] = reasoning_tokens
        return {}

    def _log_rollout_metrics(self, results, mode, reasoning_tokens):
        self.handed["metrics"] = reasoning_tokens


def test_a_rounds_reasoning_is_counted_once_for_both_consumers():
    """The price and the rollout metrics read one count: tokenizing every trace twice per step costs a
    second pass over the longest text the step holds."""
    host = _Round()
    host._generate_and_score_completions_base([{"prompt": "task"}, {"prompt": "task"}])
    assert host.handed["rewards"] == [[2, 0, 4], [3]]
    assert host.handed["metrics"] is host.handed["rewards"]
    assert host._tokenizer.calls == 3, "each reasoning trace is tokenized once per step"


def test_the_counts_are_the_tp_leaders_episodes():
    """Under TP/ETP every rank trains the group leader's rollouts, so both consumers count the leader's
    reasoning, not the episodes this rank collected."""
    host = _Round(leader=[_episode(["xxxxx"]), _episode(["xxxxxx", None])])
    host._generate_and_score_completions_base([{"prompt": "task"}, {"prompt": "task"}])
    assert host.handed["rewards"] == host.handed["metrics"] == [[5], [6, 0]]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
