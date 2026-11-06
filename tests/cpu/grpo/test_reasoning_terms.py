#!/usr/bin/env python
"""CPU tests for the trainer-side reasoning terms, both read off the episode's effort level: its profile's
``reasoning_budget`` (the reasoning tokens an episode spends free, summed over its turns) and
``reasoning_price`` (reward units per 1,000 reasoning tokens past it).

The price is ``-min(reasoning_price_cap, price * (reasoning - budget) / 1000)`` once an episode reasons past
its level's budget, so one trace costs most where the level prices highest and the cap keeps a long trace
from outweighing the task reward. The floor is the only term that pays for MORE reasoning: a baseline
episode that earned no objective and reasoned under its level's budget pays ``-reasoning_floor * shortfall /
budget``; a solve never pays it, and a failed episode pays at most one of the two. A training group no
baseline episode of which earned the objective is charged neither term (``reasoning_terms_in_failed_groups``
restores the charge). The trainer refuses at construction every setting under which a term would charge
nothing, or every episode.

Run: python tests/cpu/grpo/test_reasoning_terms.py  (or pytest)
"""

import math
import types
from collections import defaultdict

import pytest
import torch

from halo.configs.grpo_environmental import AsyncRolloutConfig, ReasoningCostArguments
from halo.environments.episode import RolloutResult
from halo.environments.episode_keys import EPISODE_INVALID_KEY, REWARD_COMPONENTS_KEY, SOLVE_RATE_KEY
from halo.environments.messages import Message, Trajectory
from halo.environments.registry import build_environment
from halo.rewards.terms import OBJECTIVE_REWARD_KEY, EnvironmentTerm, JudgeTerm, Requirement
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

# The code-contests recipes' per-level profiles, price cap and floor weight.
PROFILES = {
    "low": {"reasoning_budget": 40000, "reasoning_price": 0.008},
    "medium": {"reasoning_budget": 50000, "reasoning_price": 0.006},
    "high": {"reasoning_budget": 60000, "reasoning_price": 0.004},
}
BUDGETS_ONLY = {level: {"reasoning_budget": profile["reasoning_budget"]} for level, profile in PROFILES.items()}
CAP, FLOOR = 0.28, 0.25
# Rewards that price no success above zero: a judge alone prices no environment term.
UNPRICED_SUCCESS = (
    [JudgeTerm(name="quality", requirements=(Requirement(name="r", description="d"),))],
    [EnvironmentTerm(weight=0.0)],
    [EnvironmentTerm(weight=-1.0)],
)


def _env(effort="random", profiles=PROFILES, reward_terms=None):
    """A registered environment built from the kwargs a YAML's ``environment_kwargs`` carries."""
    return build_environment(
        "native_math",
        {"reasoning_effort": effort, "reasoning_effort_profiles": profiles, "reward_terms": reward_terms},
    )


class _CharTokenizer:
    """One token per character: deterministic reasoning lengths without a real tokenizer."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [0] * len(text)}


def _host(env=None, num_generations=1, **config) -> DistributedAsyncEnvironmentalGRPOTrainer:
    trainer = attach_world_metrics(object.__new__(DistributedAsyncEnvironmentalGRPOTrainer))
    trainer.async_config = AsyncRolloutConfig(**config)
    trainer._rollout_env = _env() if env is None else env
    trainer.num_generations = num_generations
    trainer._tokenizer = _CharTokenizer()
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer._carry_reasoning = False
    trainer._warned_once = set()
    return trainer


def _validate(host) -> None:
    reject_inert_reasoning_terms(host.async_config, host._rollout_env, host._level_reasoning_terms)


def _rollout(level, tokens, objective=1.0):
    """Zero environment reward; each turn carries ``tokens`` reasoning tokens; settled objective ``objective``."""
    trajectory = types.SimpleNamespace(
        reasoning_effort=level,
        tokens=tokens,
        messages=[],
        info={REWARD_COMPONENTS_KEY: {OBJECTIVE_REWARD_KEY: objective}},
    )
    return types.SimpleNamespace(trajectory=trajectory, total_reward=0.0, counts_toward_baseline=True)


def _apply(rollouts, mode="train", num_generations=1, env=None, **config):
    trainer = _host(env, num_generations, **config)
    tokens = [r.trajectory.tokens if r.trajectory else [] for r in rollouts]
    rewards = trainer._build_rollout_rewards(rollouts, tokens, torch.device("cpu"), mode)
    return rewards, flushed_metrics(trainer, mode)


def _episode(thinkings, level="low", objective=1.0):
    turns = (Message.assistant("step", thinking=thinking) for thinking in thinkings)
    trajectory = Trajectory(messages=[Message.user("task"), *turns], reasoning_effort=level)
    trajectory.info[REWARD_COMPONENTS_KEY] = {OBJECTIVE_REWARD_KEY: objective}
    return RolloutResult(prompt="task", trajectory=trajectory)


def _reward_build(trainer, episodes, mode="train"):
    """The trainer's own reward build over its own per-turn token counter."""
    counts = [reasoning_token_counts(trainer._tokenizer, e.trajectory) for e in episodes]
    return trainer._build_rollout_rewards(episodes, counts, torch.device("cpu"), mode)


def test_the_price_charges_what_the_turns_reason_together_past_the_budget():
    assert reasoning_price_term([30000], 0.008, CAP, 40000) == 0.0, "within the budget reasoning is free"
    assert reasoning_price_term([40000], 0.008, CAP, 40000) == 0.0
    assert reasoning_price_term([30000, 30000], 0.008, CAP, 40000) == pytest.approx(-0.16), "turns are summed"
    assert reasoning_price_term([20000, 21000], 0.008, CAP, 40000) == pytest.approx(-0.008)
    assert reasoning_price_term([], 0.008, CAP, 40000) == 0.0


def test_a_zero_budget_prices_every_reasoning_token():
    assert reasoning_price_term([4096], 0.01, 0.1, 0) == pytest.approx(-0.04096)
    assert reasoning_price_term([1024, 3072], 0.01, 0.1, 0) == pytest.approx(-0.04096)
    assert reasoning_price_term([500], 0.01, 0.1, 0) == pytest.approx(-0.005), "priced per 1,000 tokens"
    assert reasoning_price_term([0, 0], 0.01, 0.1, 0) == 0.0
    assert reasoning_price_term([4096], 0.0, 0.1, 0) == 0.0, "a level priced at zero pays nothing"


def test_the_price_is_capped_exactly_at_the_cap():
    assert reasoning_price_term([75000], 0.008, CAP, 40000) == pytest.approx(-CAP), "35000 past reach the cap"
    assert reasoning_price_term([74999], 0.008, CAP, 40000) > -CAP
    assert reasoning_price_term([400000], 0.008, CAP, 40000) == pytest.approx(-CAP)
    assert reasoning_price_term([400000], 0.008, 0.1, 40000) == pytest.approx(-0.1)


def test_the_floor_prices_the_shortfall_under_the_budget_and_nothing_at_or_past_it():
    assert reasoning_floor_term([10000], 40000, FLOOR) == pytest.approx(-0.1875)
    assert reasoning_floor_term([30000], 40000, FLOOR) == pytest.approx(-0.0625)
    assert reasoning_floor_term([40000], 40000, FLOOR) == 0.0
    assert reasoning_floor_term([50000], 40000, FLOOR) == 0.0


def test_the_floor_sums_the_episode_so_a_short_repair_turn_is_not_under_use():
    assert reasoning_floor_term([39000, 500, 500], 40000, FLOOR) == 0.0
    assert reasoning_floor_term([20000, 20000], 40000, FLOOR) == 0.0
    assert reasoning_floor_term([10000], 40000, FLOOR) == reasoning_floor_term([10000, 0, 0], 40000, FLOOR)


def test_the_floor_charges_silent_turns_in_full_and_a_lost_episode_nothing():
    assert reasoning_floor_term([0, 0, 0], 40000, FLOOR) == pytest.approx(-FLOOR)
    assert reasoning_floor_term([], 40000, FLOOR) == 0.0, "no assistant turn: an episode the driver lost"
    assert reasoning_floor_term([10], 0, FLOOR) == 0.0, "a zero budget leaves nothing to fall short of"
    assert reasoning_floor_term([10], 40000, 0.0) == 0.0


def test_each_episode_pays_its_own_levels_price_past_its_own_budget():
    rollouts = [_rollout(level, [70000]) for level in ("low", "medium", "high")] + [_rollout(None, [70000])]
    rewards, metrics = _apply(rollouts, reasoning_price_cap=CAP)
    low, medium, high, unleveled = rewards.tolist()
    assert (low, medium, high) == pytest.approx((-0.24, -0.12, -0.04)), "30000, 20000 and 10000 tokens past"
    assert low < medium < high < 0
    assert unleveled == 0.0, "an episode with no level reads no profile"
    assert metrics["reward/reasoning_price"] == [pytest.approx(-0.4 / 4)]
    assert "reward/reasoning_floor" not in metrics


def test_the_price_is_capped_per_episode():
    for cap in (CAP, ReasoningCostArguments.reasoning_price_cap):
        rewards, _ = _apply([_rollout("low", [200000])], reasoning_price_cap=cap)
        assert rewards[0].item() == pytest.approx(-cap)


def test_the_floor_charges_only_a_baseline_episode_that_earned_no_objective():
    rollouts = [
        _rollout("low", [10000]),  # a short solve: free
        _rollout("low", [10000], objective=0.0),  # a short failure: its shortfall
        _rollout("low", [40000], objective=0.0),  # at the budget: free of both
        _rollout("low", [60000], objective=0.0),  # past it: the price, never the floor
        _rollout("low", [60000]),  # a long solve pays the price too
        _rollout("low", [30000, 30000]),  # the turns together, not each against the budget
        _rollout(None, [10000], objective=0.0),  # no level: free of both
    ]
    rewards, metrics = _apply(
        rollouts, reasoning_price_cap=CAP, reasoning_floor=FLOOR, reasoning_terms_in_failed_groups=True
    )
    assert rewards.tolist() == pytest.approx([0.0, -0.1875, 0.0, -0.16, -0.16, -0.16, 0.0])
    assert metrics["reward/reasoning_floor"] == [pytest.approx(-0.1875 / 7)]
    assert metrics["reward/reasoning_price"] == [pytest.approx(-0.48 / 7)]


def test_a_failed_episode_pays_one_term_at_most_and_never_more_than_the_larger_bound():
    """Under the budget the floor alone, past it the price alone: a failed episode's charge reaches
    ``max(reasoning_price_cap, reasoning_floor)`` and no further, whatever it reasoned."""
    totals = [0, 20000, 39999, 40000, 40001, 60000, 75000, 200000]
    rewards, _ = _apply(
        [_rollout("low", [total], objective=0.0) for total in totals],
        reasoning_price_cap=CAP,
        reasoning_floor=FLOOR,
        reasoning_terms_in_failed_groups=True,
    )
    expected = [-0.25, -0.125, -0.25 / 40000, 0.0, -0.008 / 1000, -0.16, -0.28, -0.28]
    assert rewards.tolist() == pytest.approx(expected, abs=1e-7)
    assert min(rewards.tolist()) == pytest.approx(-max(CAP, FLOOR))


def test_each_term_is_off_on_its_own_and_records_only_its_own_metric():
    rollouts = [_rollout("low", [10000], objective=0.0), _rollout("low", [60000], objective=0.0)]
    rewards, metrics = _apply(rollouts, reasoning_price_cap=CAP, reasoning_terms_in_failed_groups=True)
    assert rewards.tolist() == pytest.approx([0.0, -0.16]) and "reward/reasoning_floor" not in metrics
    rewards, metrics = _apply(
        rollouts, env=_env(profiles=BUDGETS_ONLY), reasoning_floor=FLOOR, reasoning_terms_in_failed_groups=True
    )
    assert rewards.tolist() == pytest.approx([-0.1875, 0.0]) and "reward/reasoning_price" not in metrics


def test_a_training_group_that_earned_no_objective_is_charged_neither_term():
    """Where every attempt failed, the length terms are most of what tells the attempts apart, so the advantage
    would train reasoning length wherever the task is out of reach. A group with a success is charged in full,
    its failure included."""
    rollouts = [
        _rollout("low", [60000]),  # solved: the price
        _rollout("low", [10000], objective=0.0),  # failed beside a solve: the floor
        _rollout("low", [10000], objective=0.0),
        _rollout("low", [60000], objective=0.0),  # a failed group: waived
    ]
    rewards, metrics = _apply(rollouts, num_generations=2, reasoning_price_cap=CAP, reasoning_floor=FLOOR)
    assert rewards.tolist() == pytest.approx([-0.16, -0.1875, 0.0, 0.0])
    assert metrics["outcome/reasoning_terms_waived_frac"] == [0.5]
    assert metrics["reward/reasoning_price"] == [pytest.approx(-0.16 / 4)], "the logged price is what was charged"
    assert metrics["reward/reasoning_floor"] == [pytest.approx(-0.1875 / 4)]


def test_a_vetoed_solve_and_an_episode_outside_the_baseline_earn_their_group_nothing():
    """The group is read off the settled objective component, which a veto zeroes even where the environment's
    solve verdict stands, and off the episodes that train."""
    vetoed = _episode(["x" * 500], objective=0.0)
    vetoed.metrics[SOLVE_RATE_KEY] = 1.0
    invalid = _episode(["x" * 500])
    invalid.trajectory.info[EPISODE_INVALID_KEY] = True
    episodes = [vetoed, _episode(["x" * 500], objective=0.0), invalid, _episode(["x" * 500], objective=0.0)]
    assert failed_group_mask(episodes, 2) == [True] * 4
    priced = _env(profiles={level: {"reasoning_budget": 100, "reasoning_price": 0.01} for level in PROFILES})
    assert _reward_build(_host(priced, num_generations=2), episodes).tolist() == [0.0] * 4
    charged = _reward_build(_host(priced, num_generations=2, reasoning_terms_in_failed_groups=True), episodes)
    assert charged.tolist() == pytest.approx([-0.004] * 4), "the waiver, not the episodes, zeroed the charge"
    assert failed_group_mask([invalid, _episode(["x"])], 2) == [False, False], "a baseline solve keeps the terms"


def test_eval_episodes_and_the_knob_charge_failed_groups():
    """An eval reward trains nothing, so it reports what the reasoning costs; the knob restores the charge in
    training."""
    rollouts = [_rollout("low", [60000], objective=0.0), _rollout("low", [10000], objective=0.0)]
    for mode, config in (("eval", {}), ("train", {"reasoning_terms_in_failed_groups": True})):
        rewards, metrics = _apply(
            rollouts, mode=mode, num_generations=2, reasoning_price_cap=CAP, reasoning_floor=FLOOR, **config
        )
        assert rewards.tolist() == pytest.approx([-0.16, -0.1875])
        assert "outcome/reasoning_terms_waived_frac" not in metrics


def test_an_episode_outside_the_baseline_pays_no_floor():
    """A void episode trains nothing, and a floor charge would only inflate its record and the logged mean."""
    void = _rollout("high", [30000], objective=0.0)
    void.counts_toward_baseline = False
    rewards, metrics = _apply([void, _rollout("high", [30000])], num_generations=2, reasoning_floor=FLOOR)
    assert rewards.tolist() == [0.0, 0.0]
    assert metrics["reward/reasoning_floor"] == [0.0]


def test_a_missing_trajectory_is_free():
    missing = types.SimpleNamespace(trajectory=None, total_reward=0.0, counts_toward_baseline=True)
    rewards, _ = _apply([missing], reasoning_price_cap=CAP, reasoning_floor=FLOOR)
    assert rewards[0].item() == 0.0


def test_the_reward_build_applies_both_terms_over_every_assistant_turn():
    """The real path: the trainer's own reward build and its per-turn token counter. Every assistant turn is one
    entry, a thinking-free one as 0 — a counter that skipped them would hand the floor an empty list, which it
    reads as a lost episode, so dropping the reasoning entirely would be free while brief reasoning paid."""
    env = _env(profiles={level: {"reasoning_budget": 1000, "reasoning_price": 0.01} for level in PROFILES})
    trainer = _host(env, reasoning_floor=0.05, reasoning_terms_in_failed_groups=True)
    episodes = [
        _episode([None, "", None], objective=0.0),  # three silent turns: the whole floor
        _episode([], objective=0.0),  # no assistant turn: a lost episode, free
        _episode(["x" * 1000], objective=0.0),  # exactly its budget: free of both
        _episode(["x" * 300, "x" * 200], objective=0.0),  # 500 summed over two turns: half the budget short
        _episode(["x" * 1500, "x" * 500], objective=0.0),  # 2000 summed: 1000 past the budget priced
    ]
    assert _reward_build(trainer, episodes).tolist() == pytest.approx([-0.05, 0.0, 0.0, -0.025, -0.01])
    metrics = flushed_metrics(trainer)
    assert metrics["reward/reasoning_floor"] == [pytest.approx(-0.075 / 5)]
    assert metrics["reward/reasoning_price"] == [pytest.approx(-0.01 / 5)]


def test_a_zero_budget_level_prices_every_reasoning_token_in_the_reward_build():
    """``reasoning_budget: 0`` is the price-everything setting, not a missing budget."""
    env = _env(profiles={level: {"reasoning_budget": 0, "reasoning_price": 0.01} for level in PROFILES})
    trainer = _host(env, reasoning_terms_in_failed_groups=True)
    assert _reward_build(trainer, [_episode(["x" * 300, "x" * 200])]).tolist() == pytest.approx([-0.005])


def test_with_both_terms_off_nothing_is_charged_or_logged():
    trainer = _host(_env(profiles=None))
    assert _reward_build(trainer, [_episode(["x" * 10], objective=0.0)]).tolist() == [0.0]
    metrics = flushed_metrics(trainer)
    assert "reward/reasoning_floor" not in metrics and "reward/reasoning_price" not in metrics


def test_config_refuses_a_cap_or_floor_that_would_invert_or_poison_the_terms():
    defaults = AsyncRolloutConfig()
    assert defaults.reasoning_floor == 0.0, "the floor is off by default"
    assert defaults.reasoning_price_cap == ReasoningCostArguments.reasoning_price_cap == 0.1
    for bad in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="reasoning_price_cap must be (a finite number|> 0)"):
            AsyncRolloutConfig(reasoning_price_cap=bad)
    for bad in (-0.01, math.nan, math.inf):
        with pytest.raises(ValueError, match=r"reasoning_floor must be (a finite number|>= 0 \(0 = off\))"):
            AsyncRolloutConfig(reasoning_floor=bad)


def test_the_terms_read_only_the_levels_the_environment_draws():
    """``random`` draws every level, a fixed setting its own, and no setting none; a priced level the
    environment never draws turns no price on."""
    every = _host()
    assert every._level_reasoning_terms == {
        level: (profile["reasoning_budget"], profile["reasoning_price"]) for level, profile in PROFILES.items()
    }
    assert every._reasoning_price_on
    fixed = _host(_env(effort="high", profiles={"low": PROFILES["low"]}))
    assert fixed._level_reasoning_terms == {"high": (None, 0.0)} and not fixed._reasoning_price_on
    unleveled = _host(_env(effort=None, profiles=None))
    assert unleveled._level_reasoning_terms == {None: (None, 0.0)} and not unleveled._reasoning_price_on


def test_construction_passes_the_recipes_economy():
    """Guards the refusals below from being satisfied by refusing everything."""
    _validate(_host(reasoning_price_cap=CAP, reasoning_floor=FLOOR))
    _validate(_host(reasoning_price_cap=CAP, reasoning_floor=FLOOR, reasoning_terms_in_failed_groups=True))
    _validate(_host(_env(effort=None, profiles=None)))


def test_construction_refuses_profile_terms_with_no_level_to_read_them_at():
    """The terms read the episode's level; an environment drawing none would charge nothing, silently."""
    for profiles, keys in (
        ({"high": {"reasoning_budget": 60000}}, r"\['reasoning_budget'\]"),
        ({"low": PROFILES["low"]}, r"\['reasoning_budget', 'reasoning_price'\]"),
    ):
        with pytest.raises(ValueError, match=keys + " but the environment's reasoning_effort is unset"):
            _validate(_host(_env(effort=None, profiles=profiles)))
    task_only = build_environment(
        "code_contests",
        {
            "sandbox_backend": "local",
            "reasoning_effort": None,
            "reasoning_effort_profiles": {"high": {"max_submissions": 2}},
        },
    )
    _validate(_host(task_only))


def test_construction_refuses_a_price_cap_no_drawn_level_reads():
    """The cap is read only beside a price; set alone, or beside a price only a level never drawn carries, it
    parses and changes nothing."""
    for env in (_env(profiles=BUDGETS_ONLY), _env(effort="high", profiles={"low": PROFILES["low"]})):
        with pytest.raises(ValueError, match="reasoning_price_cap set while no level the environment draws"):
            _validate(_host(env, reasoning_price_cap=CAP))
    _validate(_host(_env(profiles=BUDGETS_ONLY), reasoning_floor=FLOOR))
    _validate(_host(_env(effort="high", profiles={"low": PROFILES["low"]})))


def test_construction_refuses_the_failed_group_knob_with_no_term_on():
    with pytest.raises(ValueError, match="reasoning_terms_in_failed_groups set while no drawn level prices"):
        _validate(_host(_env(profiles=BUDGETS_ONLY), reasoning_terms_in_failed_groups=True))
    _validate(_host(_env(profiles=BUDGETS_ONLY), reasoning_floor=FLOOR, reasoning_terms_in_failed_groups=True))
    _validate(_host(reasoning_terms_in_failed_groups=True))


def test_construction_refuses_a_floor_at_a_drawn_level_with_no_budget():
    """The floor's reference is the level's budget: a drawn level without one would leave its failures free."""
    partial = {"low": {"reasoning_budget": 40000}}
    with pytest.raises(ValueError, match=r"the drawn level\(s\) \['high', 'medium'\] set none above 0"):
        _validate(_host(_env(profiles=partial), reasoning_floor=FLOOR))
    _validate(_host(_env(effort="low", profiles=partial), reasoning_floor=FLOOR))


def test_construction_refuses_a_floor_its_levels_leave_inert():
    """A floor charges a shortfall under a budget above 0: a drawn level at 0, or no level at all, leaves its
    failures free while the config says the floor is on."""
    zero_low = {**BUDGETS_ONLY, "low": {"reasoning_budget": 0}}
    with pytest.raises(ValueError, match=r"the drawn level\(s\) \['low'\] set none above 0"):
        _validate(_host(_env(profiles=zero_low), reasoning_floor=FLOOR))
    with pytest.raises(ValueError, match="reasoning_effort is unset, so no episode carries a level"):
        _validate(_host(_env(effort=None, profiles=None), reasoning_floor=FLOOR))
    _validate(_host(_env(effort="medium", profiles=zero_low), reasoning_floor=FLOOR))


def test_construction_refuses_a_budget_no_term_reads():
    """A drawn level's budget is read by its price or by the floor; with neither it parses and changes nothing."""
    unpriced_high = {**PROFILES, "high": {"reasoning_budget": 60000}}
    with pytest.raises(ValueError, match=r"the drawn level\(s\) \['high'\] set reasoning_budget with no"):
        _validate(_host(_env(profiles=unpriced_high)))
    _validate(_host(_env(profiles=unpriced_high), reasoning_floor=FLOOR))
    _validate(_host(_env(effort="low", profiles=unpriced_high)))


def test_construction_refuses_a_profile_key_among_the_template_variables():
    """A template variable is text the template may state, and the model must never read a token count."""
    for key in ("reasoning_budget", "reasoning_price"):
        with pytest.raises(ValueError, match=rf"rollout_chat_template_kwargs carries \['{key}'\]"):
            _validate(_host(rollout_chat_template_kwargs={key: 60000}))
    _validate(_host(rollout_chat_template_kwargs={"preserve_thinking": True}))


def test_construction_refuses_a_floor_under_a_reward_that_prices_no_success_above_zero():
    """No episode of such a reward earns the objective, so the floor would charge every one, waiver or not."""
    for reward_terms in UNPRICED_SUCCESS:
        env = _env(profiles=BUDGETS_ONLY, reward_terms=reward_terms)
        for in_failed_groups in (False, True):
            with pytest.raises(ValueError, match="so it would charge every episode"):
                _validate(_host(env, reasoning_floor=FLOOR, reasoning_terms_in_failed_groups=in_failed_groups))


def test_construction_refuses_waiving_the_price_under_a_reward_that_prices_no_success_above_zero():
    """Every group of such a reward reads as failed, so the waived price would charge no training episode;
    charging failed groups too lets it run."""
    for reward_terms in UNPRICED_SUCCESS:
        env = _env(reward_terms=reward_terms)
        with pytest.raises(ValueError, match="the terms would charge no training episode"):
            _validate(_host(env))
        _validate(_host(env, reasoning_terms_in_failed_groups=True))


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
