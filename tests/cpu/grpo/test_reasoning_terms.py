#!/usr/bin/env python
"""Trainer-side reasoning terms: the per-level capped price ``-min(cap, price_per_1k * reasoning_tokens / 1000)``, on
every token or past a soft budget, and the under-use floor, both reading reasoning summed over an episode's turns."""

import types
from collections import defaultdict

import pytest
import torch

from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.environments.episode import RolloutResult
from halo.environments.episode_keys import EPISODE_INVALID_KEY, REWARD_COMPONENTS_KEY, SOLVE_RATE_KEY
from halo.environments.messages import Message, Trajectory
from halo.environments.reasoning_effort import has_thinking_budget
from halo.rewards.terms import OBJECTIVE_REWARD_KEY, EnvironmentTerm
from halo.trainers.grpo.environmental.environmental_rules import reject_inert_reasoning_terms
from halo.trainers.grpo.environmental.reasoning import (
    failed_group_mask,
    reasoning_floor_term,
    reasoning_price_term,
    reasoning_token_counts,
)
from halo.trainers.grpo.environmental.step_batch import BatchBuildFence
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.phase_timing import PhaseTimedMixin
from tests.common.world_metrics import attach_world_metrics, flushed_metrics

PRICE = {"low": 0.01, "medium": 0.005, "high": 0.001}
# The example prices the async-GRPO templates show.
TEMPLATE_PRICE = {"low": 0.0037, "medium": 0.0013, "high": 0.00018}
CAP = 0.1
# The default reward's environment term: a solve priced at 1.
OBJECTIVE_TERM = EnvironmentTerm()


def test_price_is_linear_in_reasoning_tokens_per_thousand_summed_over_turns():
    assert reasoning_price_term([4096], 0.01, CAP) == pytest.approx(-0.04096)
    assert reasoning_price_term([1024, 3072], 0.01, CAP) == pytest.approx(-0.04096), "turns are summed"
    assert reasoning_price_term([500], 0.01, CAP) == pytest.approx(-0.005), "priced per 1,000 tokens"
    assert reasoning_price_term([], 0.01, CAP) == 0.0
    assert reasoning_price_term([0, 0], 0.01, CAP) == 0.0
    assert reasoning_price_term([4096], 0.0, CAP) == 0.0, "a level priced at zero pays nothing"
    # The template's prices at the low and high levels' whole per-turn budgets.
    assert reasoning_price_term([8192], TEMPLATE_PRICE["low"], CAP) == pytest.approx(-0.0303104)
    assert reasoning_price_term([16384], TEMPLATE_PRICE["high"], CAP) == pytest.approx(-0.00294912)


def test_price_is_capped_exactly_at_the_cap():
    # 10000 tokens at 0.01 per 1k reach the cap.
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
    assert reasoning_floor_term([4096], 8192, 0.05) == reasoning_floor_term([4096, 0, 0], 8192, 0.05)


def test_floor_charges_silent_turns_in_full_and_a_lost_episode_nothing():
    # Turns with no reasoning at all are the maximal under-use.
    assert reasoning_floor_term([0, 0, 0], 8192, 0.05) == pytest.approx(-0.05)
    # An episode with no assistant turn is one the driver lost, never a policy choice.
    assert reasoning_floor_term([], 8192, 0.05) == 0.0
    # Off switches: no cap, or no weight.
    assert reasoning_floor_term([10], 0, 0.05) == 0.0
    assert reasoning_floor_term([10], 8192, 0.0) == 0.0


def test_the_floor_target_rounds_a_cap_the_share_does_not_divide():
    """0.75 x 1001 rounds to 751: 751 tokens clear the floor, 750 fall one short."""
    assert reasoning_floor_term([751], 1001, 1.0) == 0.0
    assert reasoning_floor_term([750], 1001, 1.0) == pytest.approx(-1 / 751)


class _CharTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [0] * len(text)}


def _reward_host(num_generations=1, **config) -> DistributedAsyncEnvironmentalGRPOTrainer:
    trainer = attach_world_metrics(object.__new__(DistributedAsyncEnvironmentalGRPOTrainer))
    trainer.async_config = AsyncRolloutConfig(**config)
    trainer.num_generations = num_generations
    trainer._tokenizer = _CharTokenizer()
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer._carry_reasoning = False
    trainer._warned_once = set()
    return trainer


def _rollout(level, tokens, budget=None, soft=False, objective=1.0):
    """Zero environment reward; each turn carries ``tokens`` reasoning tokens; settled objective ``objective``."""
    trajectory = types.SimpleNamespace(
        reasoning_effort=level,
        reasoning_budget=budget,
        reasoning_budget_soft=soft,
        tokens=tokens,
        messages=[],
        info={REWARD_COMPONENTS_KEY: {OBJECTIVE_REWARD_KEY: objective}},
    )
    return types.SimpleNamespace(trajectory=trajectory, total_reward=0.0, counts_toward_baseline=True)


def _apply(rollouts, mode="train", num_generations=1, **config):
    trainer = _reward_host(num_generations, **config)
    tokens = [r.trajectory.tokens if r.trajectory else [] for r in rollouts]
    rewards = trainer._build_rollout_rewards(rollouts, tokens, torch.device("cpu"), mode)
    return rewards, flushed_metrics(trainer, mode)


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


def _episode(thinkings, budget=1000, level="low", objective=1.0):
    turns = (Message.assistant("step", thinking=thinking) for thinking in thinkings)
    trajectory = Trajectory(messages=[Message.user("task"), *turns])
    trajectory.reasoning_budget, trajectory.reasoning_effort = budget, level
    trajectory.info[REWARD_COMPONENTS_KEY] = {OBJECTIVE_REWARD_KEY: objective}
    return RolloutResult(prompt="task", trajectory=trajectory)


def _built_rewards(trainer, episodes) -> torch.Tensor:
    counts = [reasoning_token_counts(trainer._tokenizer, e.trajectory) for e in episodes]
    return trainer._build_rollout_rewards(episodes, counts, torch.device("cpu"), "train")


def test_a_training_group_that_earned_no_objective_is_charged_neither_term():
    """Where every attempt failed, the length terms are most of what tells attempts apart, so the advantage would teach
    reasoning less wherever the task is out of reach."""
    rollouts = [
        _rollout("low", [4096], budget=8192),  # solved
        _rollout("low", [4096], budget=8192, objective=0.0),  # failed beside a solve: charged
        _rollout("low", [4096], budget=8192, objective=0.0),
        _rollout("low", [4096], budget=8192, objective=0.0),  # a failed group: waived
    ]
    rewards, metrics = _apply(rollouts, num_generations=2, reasoning_price=PRICE, reasoning_floor=0.05)
    charged = -0.04096 - 0.05 / 3
    assert rewards.tolist() == pytest.approx([charged, charged, 0.0, 0.0])
    assert metrics["outcome/reasoning_terms_waived_frac"] == [0.5]
    assert metrics["reward/reasoning_price"] == [pytest.approx(-0.04096 / 2)], "the logged price is what was charged"
    assert metrics["reward/reasoning_floor"] == [pytest.approx(-0.05 / 3 / 2)]


def test_a_vetoed_solve_and_an_episode_outside_the_baseline_earn_their_group_nothing():
    """The group reads the settled objective component, which a veto zeroes even where the solve verdict stands."""
    vetoed = _episode(["x" * 500], objective=0.0)
    vetoed.metrics[SOLVE_RATE_KEY] = 1.0
    invalid = _episode(["x" * 500])
    invalid.trajectory.info[EPISODE_INVALID_KEY] = True
    episodes = [vetoed, _episode(["x" * 500], objective=0.0), invalid, _episode(["x" * 500], objective=0.0)]
    assert failed_group_mask(episodes, 2) == [True] * 4
    trainer = _reward_host(num_generations=2, reasoning_price=PRICE)
    rewards = _built_rewards(trainer, episodes)
    assert rewards.tolist() == [0.0] * 4
    assert failed_group_mask([invalid, _episode(["x"])], 2) == [False, False], "a baseline solve keeps the terms"


def test_eval_episodes_and_the_knob_charge_failed_groups():
    """An eval reward trains nothing, so it reports what the reasoning costs."""
    rollouts = [_rollout("low", [4096], objective=0.0) for _ in range(2)]
    for mode, config in (("eval", {}), ("train", {"reasoning_terms_in_failed_groups": True})):
        rewards, metrics = _apply(rollouts, mode=mode, num_generations=2, reasoning_price=PRICE, **config)
        assert rewards.tolist() == pytest.approx([-0.04096] * 2)
        assert "outcome/reasoning_terms_waived_frac" not in metrics


def test_the_reward_build_applies_both_terms_over_every_assistant_turn():
    """Every assistant turn is one entry, a thinking-free one as 0. A counter that skipped them would hand the floor an
    empty list, which it reads as a lost episode, so dropping the reasoning entirely would be free."""
    trainer = _reward_host(reasoning_price=PRICE, reasoning_floor=0.05)
    episodes = [
        _episode([None, "", None]),  # three silent turns: the whole floor, nothing to price
        _episode([]),  # no assistant turn: a lost episode, free
        _episode(["x" * 750]),  # exactly 0.75 x its 1000 budget: no floor, 750 tokens priced
        _episode(["x" * 300, "x" * 75]),  # 375 summed over two turns: half the target short, 375 priced
    ]
    rewards = _built_rewards(trainer, episodes)
    assert rewards.tolist() == pytest.approx([-0.05, 0.0, -0.0075, -0.025 - 0.00375])
    metrics = flushed_metrics(trainer)
    assert metrics["reward/reasoning_floor"] == [pytest.approx(-0.075 / 4)]
    assert metrics["reward/reasoning_price"] == [pytest.approx(-0.01125 / 4)]


def test_with_both_terms_off_nothing_is_charged_or_logged():
    trainer = _reward_host()
    episodes = [_episode(["x" * 10])]
    rewards = _built_rewards(trainer, episodes)
    assert rewards.tolist() == [0.0]
    metrics = flushed_metrics(trainer)
    assert "reward/reasoning_floor" not in metrics and "reward/reasoning_price" not in metrics


def test_a_missing_trajectory_is_free():
    missing = types.SimpleNamespace(trajectory=None, total_reward=0.0, counts_toward_baseline=True)
    rewards, _ = _apply([missing], reasoning_price=PRICE, reasoning_floor=0.05)
    assert rewards[0].item() == 0.0


def test_config_refuses_values_that_would_invert_or_poison_the_terms():
    defaults = AsyncRolloutConfig()
    assert defaults.reasoning_price is None, "the price is off by default"
    assert defaults.reasoning_floor == 0.0, "the floor is off by default"
    assert defaults.reasoning_price_cap == 0.1
    for empty in ({}, [0.01], 0.01, "low: 0.01"):
        with pytest.raises(ValueError, match="reasoning_price must map each effort level to a price"):
            AsyncRolloutConfig(reasoning_price=empty)
    for bad in (-0.01, float("nan"), float("inf"), True, "0.01", None):
        with pytest.raises(ValueError, match=r"reasoning_price\['low'\] must be (a finite number|>= 0)"):
            AsyncRolloutConfig(reasoning_price={**PRICE, "low": bad})
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match=r"reasoning_price_cap must be (a finite number|> 0)"):
            AsyncRolloutConfig(reasoning_price=PRICE, reasoning_price_cap=bad)
    for bad in (-0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match=r"reasoning_floor must be (a finite number|>= 0 \(0 = off\))"):
            AsyncRolloutConfig(reasoning_floor=bad)
    # A price of zero on one level is a priced level that pays nothing, not a mistake.
    assert AsyncRolloutConfig(reasoning_price={**PRICE, "high": 0}).reasoning_price["high"] == 0
    assert AsyncRolloutConfig(reasoning_price=TEMPLATE_PRICE, reasoning_floor=0.05).reasoning_price == TEMPLATE_PRICE


def test_a_price_cap_with_the_price_off_is_refused_as_inert():
    """The cap is read only beside a price, so set alone it parses and changes nothing."""
    with pytest.raises(ValueError, match="reasoning_price_cap set with reasoning_price unset"):
        AsyncRolloutConfig(reasoning_price_cap=0.2)
    assert AsyncRolloutConfig(reasoning_price_cap=0.1).reasoning_price is None
    assert AsyncRolloutConfig(reasoning_price=PRICE, reasoning_price_cap=0.2).reasoning_price_cap == 0.2


def test_the_failed_group_knob_with_both_terms_off_is_refused_as_inert():
    with pytest.raises(ValueError, match="reasoning_terms_in_failed_groups set with reasoning_price and"):
        AsyncRolloutConfig(reasoning_terms_in_failed_groups=True)
    assert AsyncRolloutConfig(reasoning_floor=0.05, reasoning_terms_in_failed_groups=True).reasoning_floor == 0.05


def _constructing(budgets, effort="random", soft=(), objective_term=OBJECTIVE_TERM, **config):
    trainer = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    trainer.async_config = AsyncRolloutConfig(**config)
    trainer._rollout_env = types.SimpleNamespace(
        reasoning_effort=effort,
        thinking_budget_for_effort=budgets.get,
        thinking_budget_enforced=lambda level: level not in soft,
        objective_term=objective_term,
    )
    return trainer


def _check_terms(host) -> None:
    """The reasoning-terms refusals as the trainer's init runs them, the drawable levels' own contract first."""
    reject_inert_reasoning_terms(
        host.async_config,
        host._rollout_env,
        budgeted=has_thinking_budget(host._efforts_by_level()),
        every_budget_soft=host._every_budget_soft,
    )


def test_construction_refuses_a_price_table_that_misses_or_invents_a_level():
    """The config parses any non-empty map; the trainer, beside the environment that draws the levels, refuses it."""
    with pytest.raises(ValueError, match=r"missing \['high'\]"):
        _check_terms(_constructing({}, reasoning_price={"low": 0.01, "medium": 0.005}))
    with pytest.raises(ValueError, match=r"unknown \['extreme'\]"):
        _check_terms(_constructing({}, reasoning_price={**PRICE, "extreme": 0.0}))
    _check_terms(_constructing({}, reasoning_price=PRICE))
    _check_terms(_constructing({}, reasoning_price=TEMPLATE_PRICE))
    _check_terms(_constructing({}))  # the price off: no table to check


def test_construction_refuses_waiving_the_terms_under_a_reward_that_prices_no_success_above_zero():
    """No episode of a judge-only or zero-weight environment reward earns the objective; all groups read as failed."""
    for objective_term in (None, EnvironmentTerm(weight=0.0), EnvironmentTerm(weight=-1.0)):
        with pytest.raises(ValueError, match="prices no success above zero"):
            _check_terms(_constructing({}, objective_term=objective_term, reasoning_price=PRICE))
        _check_terms(
            _constructing(
                {}, objective_term=objective_term, reasoning_price=PRICE, reasoning_terms_in_failed_groups=True
            )
        )


def test_construction_refuses_a_price_with_no_level_to_price_at():
    """The price reads the episode's level; an environment drawing none would charge nothing, silently."""
    with pytest.raises(ValueError, match="reasoning_effort is unset"):
        _check_terms(_constructing({}, effort=None, reasoning_price=PRICE))
    _check_terms(_constructing({}, effort="low", reasoning_price=PRICE))


def test_construction_refuses_an_answer_room_bound_on_a_level_with_no_reasoning_cap():
    """The answer room is what a turn generates past its reasoning cap, so a drawable level without one bounds nothing.
    Refused up front, or every episode raises in its actor."""
    with pytest.raises(ValueError, match="rollout_max_answer_tokens"):
        _check_terms(_constructing({"low": 8192, "medium": 12288}, rollout_max_answer_tokens=8192))
    _check_terms(_constructing({"low": 8192, "medium": 12288, "high": 16384}, rollout_max_answer_tokens=8192))


def test_a_budget_on_a_level_the_environment_never_draws_budgets_no_episode():
    """The floor's gate reads the levels the effort setting can draw (a fixed setting one, an unset one none), so a
    budget elsewhere would leave the floor never pricing an episode."""
    for effort in ("low", None):
        with pytest.raises(ValueError, match="no episode of this run carries one"):
            _check_terms(_constructing({"high": 16384}, effort=effort, reasoning_floor=0.05))
    _check_terms(_constructing({"high": 16384}, effort="high", reasoning_floor=0.05))
    _check_terms(
        _constructing(
            {}, effort=None, reasoning_floor=0.05, rollout_max_tokens=30000, rollout_max_thinking_tokens=8000
        )
    )


def test_construction_refuses_a_floor_no_episode_can_ever_be_priced_by():
    with pytest.raises(ValueError, match="reasoning_floor prices an episode against its per-turn thinking cap"):
        _check_terms(_constructing({}, reasoning_floor=0.05))
    # One budgeted level is enough, and so is the run-wide cap.
    _check_terms(_constructing({"high": 16384}, reasoning_floor=0.05))
    _check_terms(_constructing({}, reasoning_floor=0.05, rollout_max_thinking_tokens=8000))
    # The floor reads the stated budget, so a soft one no engine cap backs still carries it.
    _check_terms(_constructing({"high": 16384}, soft=("high",), reasoning_floor=0.05))
    _check_terms(_constructing({}))


def test_past_a_budget_the_price_counts_only_what_each_turn_spent_past_it():
    """Per turn, so a long turn is not excused by short ones beside it; nothing within the budget; capped."""
    assert reasoning_price_term([9000, 7000, 10000], 0.01, 0.1, budget=8192) == pytest.approx(-0.01 * 2616 / 1000)
    assert reasoning_price_term([8192, 4000], 0.01, 0.1, budget=8192) == 0.0
    assert reasoning_price_term([21808, 21808], 0.01, 0.1, budget=8192) == pytest.approx(-0.1)
    assert reasoning_price_term([9000, 7000], 0.01, 1.0) == pytest.approx(-0.16), "without a budget, every token"


def test_the_trainer_prices_past_a_budget_only_where_it_was_soft():
    """An enforced budget cannot be overrun, so only a soft one is read here. The floor reads both, and a turn past its
    budget leaves the episode above the floor's reference, so the two never meet."""
    rollouts = [
        _rollout("low", [10192, 2000], budget=8192, soft=True),  # 2000 past the budget
        _rollout("low", [10192], budget=8192, soft=False),  # enforced: not priced past it
        _rollout("low", [1024], budget=8192, soft=True),  # short: the floor, not the price
    ]
    price = {"low": 0.01, "medium": 0.005, "high": 0.001}
    rewards, metrics = _apply(rollouts, reasoning_price=price, reasoning_price_past_budget=True, reasoning_floor=0.05)
    assert rewards[0].item() == pytest.approx(-0.02)
    assert rewards[1].item() == 0.0
    assert rewards[2].item() == pytest.approx(-0.05 * (6144 - 1024) / 6144)
    assert metrics["reward/reasoning_price"] == [pytest.approx(-0.02 / 3)]


def test_pricing_past_a_budget_needs_a_price():
    assert AsyncRolloutConfig().reasoning_price_past_budget is False
    with pytest.raises(ValueError, match="reasoning_price_past_budget set with reasoning_price unset"):
        AsyncRolloutConfig(reasoning_price_past_budget=True)
    assert AsyncRolloutConfig(reasoning_price=PRICE, reasoning_price_past_budget=True).reasoning_price_past_budget


def test_construction_refuses_pricing_past_a_budget_unless_every_drawable_budget_is_soft():
    """An enforced or unset budget cannot be overrun, so its turns go unpriced; a fixed effort checks one level."""
    budgets = {"low": 8192, "medium": 12288, "high": 16384}
    config = {"reasoning_price": PRICE, "reasoning_price_past_budget": True}
    for soft in ((), ("high",)):
        with pytest.raises(ValueError, match="budget is enforced or unset"):
            _check_terms(_constructing(budgets, soft=soft, **config))
    with pytest.raises(ValueError, match="budget is enforced or unset"):
        _check_terms(_constructing({"low": 8192}, effort="high", soft=("low", "high"), **config))
    _check_terms(_constructing(budgets, effort="high", soft=("high",), **config))
    _check_terms(
        _constructing(
            budgets, soft=("low", "medium", "high"), rollout_max_tokens=30000, rollout_max_answer_tokens=8192, **config
        )
    )


class _CountingTokenizer(_CharTokenizer):
    def __init__(self):
        self.calls = 0

    def __call__(self, text, add_special_tokens=False):
        self.calls += 1
        return super().__call__(text, add_special_tokens)


class _Round:
    """Records what each consumer was handed. ``leader`` stands in for the TP/ETP group leader's broadcast rollouts."""

    _collect_and_score_rollouts = DistributedAsyncEnvironmentalGRPOTrainer._collect_and_score_rollouts
    _phase_timers = PhaseTimedMixin._phase_timers
    _tool_call_watch = None
    _extract_prompts_and_contexts = DistributedAsyncEnvironmentalGRPOTrainer._extract_prompts_and_contexts
    eval_split_rows = DistributedAsyncEnvironmentalGRPOTrainer.eval_split_rows

    def _is_tp_follower(self) -> bool:
        return False

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

    def _broadcast_object_from_tp_leader(self, rollout_results):
        return rollout_results if self._leader is None else self._leader

    def _build_training_tensors(self, rollout_results, device, mode, num_padding, reasoning_tokens):
        self.handed["rewards"] = reasoning_tokens
        return {}

    def _log_rollout_metrics(self, results, mode, reasoning_tokens):
        self.handed["metrics"] = reasoning_tokens


def test_a_rounds_reasoning_is_counted_once_for_both_consumers():
    """Tokenizing every trace twice per step costs a second pass over the longest text the step holds."""
    host = _Round()
    host._collect_and_score_rollouts([{"prompt": "task"}, {"prompt": "task"}])
    assert host.handed["rewards"] == [[2, 0, 4], [3]]
    assert host.handed["metrics"] is host.handed["rewards"]
    assert host._tokenizer.calls == 3, "each reasoning trace is tokenized once per step"


def test_the_counts_are_the_tp_leaders_episodes():
    """Under TP/ETP every rank trains the group leader's rollouts."""
    host = _Round(leader=[_episode(["xxxxx"]), _episode(["xxxxxx", None])])
    host._collect_and_score_rollouts([{"prompt": "task"}, {"prompt": "task"}])
    assert host.handed["rewards"] == host.handed["metrics"] == [[5], [6, 0]]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
