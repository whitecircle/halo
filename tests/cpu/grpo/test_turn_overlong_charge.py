#!/usr/bin/env python
"""CPU tests for the overlong charge: a turn pays the larger of two ramps, its reasoning against the thinking
cap it ran under (its level's, or a retry's reserve) and every token it sampled against the turn cap, nothing
until the count enters the last quarter under the cap and the whole penalty at it; the episode pays its
most-charged turn once; the trainer's reward build logs the term and the share of turns charged, per level too.

Run: python tests/cpu/grpo/test_turn_overlong_charge.py  (or pytest)
"""

import types
from collections import defaultdict

import pytest
import torch

from src.configs.async_training_config import AsyncTrainingConfig
from src.environments.base import Message, Trajectory
from src.environments.episode import RolloutResult
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer
from src.trainers.grpo.reasoning_terms import reasoning_floor_term, turn_overlong_term
from tests.common.grpo_metrics import attach_world_metrics, flushed_metrics

CAP, PENALTY = 18000, 0.05
RAMP = 4500  # a quarter of the cap
START = CAP - RAMP


def _trajectory(*turns: tuple[int | None, int | None], level: str | None = "low") -> Trajectory:
    """An episode of assistant turns, each ``(reasoning tokens, thinking cap)`` as the rollout recorded them."""
    messages = [Message.user("task")]
    for reasoning, cap in turns:
        messages += [Message.assistant("step", reasoning_tokens=reasoning, thinking_cap=cap), Message.user("tool")]
    trajectory = Trajectory(messages=messages)
    trajectory.reasoning_effort = level
    return trajectory


def _charge(reasoning: int, cap: int) -> float:
    """What one turn of ``reasoning`` tokens under ``cap`` pays."""
    return turn_overlong_term(_trajectory((reasoning, cap)), penalty=PENALTY)[0]


@pytest.mark.parametrize(
    ("reasoning", "charge"),
    [
        (0, 0.0),
        (START - 1000, 0.0),  # below the ramp
        (START, 0.0),  # the ramp's first token
        (START + RAMP // 2, -PENALTY / 2),  # halfway in
        (CAP, -PENALTY),  # at the cap: a forced close counts the close, so a capped turn spends exactly it
        (CAP + 5000, -PENALTY),  # a count past the cap still costs the penalty once
    ],
)
def test_the_charge_ramps_from_the_target_to_the_cap(reasoning, charge):
    assert _charge(reasoning, CAP) == pytest.approx(charge)


def test_the_ramp_scales_with_the_cap_and_is_never_shorter_than_one_token():
    """A level's own cap sets its ramp: the last quarter of 8192 is 2048 tokens. A cap too small to hold a
    target below it still ramps over its last token, so a turn at any cap pays the whole penalty and one
    below it less."""
    assert _charge(6144, 8192) == 0.0
    assert _charge(7168, 8192) == pytest.approx(-PENALTY / 2)
    assert _charge(8192, 8192) == pytest.approx(-PENALTY)
    for cap in (1, 2, 3):
        assert _charge(cap, cap) == pytest.approx(-PENALTY)
        assert _charge(cap - 1, cap) == 0.0


def test_the_floor_target_and_the_ramp_start_coincide_at_a_cap_the_share_does_not_divide():
    """Both terms round the same target off the cap, so no reasoning count is at once short of the
    floor and charged overlong: 0.75 x 1001 rounds to 751."""
    cap = 1001
    assert reasoning_floor_term([751], cap, 1.0) == 0.0 and reasoning_floor_term([750], cap, 1.0) < 0.0
    assert _charge(751, cap) == 0.0 and _charge(752, cap) < 0.0


def test_each_turn_is_charged_against_the_cap_its_level_set():
    """The rollout records the level's cap on every turn, so a turn run to a request cap the output budget
    narrowed to 3000 under a 4000-token level cap only reaches the ramp's start and pays nothing, while
    a turn at a lower level's own 2048 cap pays in full."""
    assert turn_overlong_term(_trajectory((3000, 4000)), penalty=PENALTY) == (0.0, 0, 1)
    assert turn_overlong_term(_trajectory((3500, 4000)), penalty=PENALTY) == (pytest.approx(-PENALTY / 2), 1, 1)
    assert turn_overlong_term(_trajectory((2048, 2048)), penalty=PENALTY) == (pytest.approx(-PENALTY), 1, 1)


def test_an_episode_pays_its_most_charged_turn_once():
    """Two turns at the cap cost the penalty once, not twice: a sum would grow with the turns the run
    allows. A turn under the ramp is not charged, nor is one whose reasoning was not counted."""
    episode = _trajectory((CAP, CAP), (CAP, CAP), (13000, CAP), (None, CAP))
    assert turn_overlong_term(episode, penalty=PENALTY) == (pytest.approx(-PENALTY), 2, 4)
    halfway = _trajectory((START + RAMP // 2, CAP), (START + RAMP // 4, CAP))
    assert turn_overlong_term(halfway, penalty=PENALTY) == (pytest.approx(-PENALTY / 2), 2, 2)


TURN_CAP = 30000


def _turn(reasoning: int | None, cap: int | None, sampled: int | None) -> Trajectory:
    """One assistant turn of ``reasoning`` tokens under ``cap`` that sampled ``sampled`` ids in all."""
    ids = None if sampled is None else [0] * sampled
    turn = Message.assistant("step", reasoning_tokens=reasoning, thinking_cap=cap, token_ids=ids)
    return Trajectory(messages=[Message.user("task"), turn, Message.user("tool")])


@pytest.mark.parametrize(
    ("sampled", "charge"),
    [(22500, 0.0), (26250, -PENALTY / 2), (TURN_CAP, -PENALTY), (TURN_CAP + 100, -PENALTY)],
)
def test_the_turn_ramp_charges_every_sampled_token_against_the_turn_cap(sampled, charge):
    """A turn whose reasoning stays under its target still pays when the whole turn runs into the turn
    cap: reasoning carried past the close into the call is sampled output, and the cut it ends in is
    the wall this ramp prices."""
    assert turn_overlong_term(_turn(1000, CAP, sampled), penalty=PENALTY, turn_cap=TURN_CAP)[0] == pytest.approx(
        charge
    )


def test_the_turn_pays_the_larger_of_its_two_ramps_and_none_it_cannot_read():
    at_cap = _turn(CAP, CAP, 23000)
    assert turn_overlong_term(at_cap, penalty=PENALTY, turn_cap=TURN_CAP) == (pytest.approx(-PENALTY), 1, 1)
    long_turn = _turn(START + RAMP // 4, CAP, TURN_CAP)
    assert turn_overlong_term(long_turn, penalty=PENALTY, turn_cap=TURN_CAP) == (pytest.approx(-PENALTY), 1, 1)
    # No turn cap, or no sampled ids: the reasoning ramp alone.
    assert turn_overlong_term(_turn(1000, CAP, TURN_CAP), penalty=PENALTY) == (0.0, 0, 1)
    assert turn_overlong_term(_turn(1000, CAP, None), penalty=PENALTY, turn_cap=TURN_CAP) == (0.0, 0, 1)
    # An uncapped level still pays the turn ramp.
    assert turn_overlong_term(_turn(None, None, TURN_CAP), penalty=PENALTY, turn_cap=TURN_CAP) == (
        pytest.approx(-PENALTY),
        1,
        1,
    )


def test_the_reward_build_reads_the_turn_cap_off_the_run():
    """The trainer passes the run's ``rollout_max_tokens``, not a constant: under a 40000-token turn a turn
    that sampled 30000 is under the ramp and one that sampled 40000 pays in full, through the real reward
    build."""
    trainer = _host(rollout_max_tokens=40000)
    episodes = [
        RolloutResult(prompt="a", trajectory=_turn(1000, CAP, 40000)),
        RolloutResult(prompt="b", trajectory=_turn(1000, CAP, 30000)),
    ]
    rewards = trainer._build_rollout_rewards(
        episodes, trainer._episode_reasoning_tokens(episodes), torch.device("cpu")
    )
    assert rewards.tolist() == pytest.approx([-PENALTY, 0.0])


def test_an_episode_with_no_cap_or_no_trajectory_pays_nothing():
    assert turn_overlong_term(_trajectory((CAP, None), (50000, None)), penalty=PENALTY) == (0.0, 0, 2)
    assert turn_overlong_term(None, penalty=PENALTY) == (0.0, 0, 0)
    assert turn_overlong_term(_trajectory(), penalty=PENALTY) == (0.0, 0, 0)


class _CharTokenizer:
    """One token per character: deterministic reasoning lengths for the price."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [0] * len(text)}


def _host(rollout_max_tokens: int = 30000, **config) -> DistributedAsyncEnvironmentalGRPOTrainer:
    trainer = attach_world_metrics(object.__new__(DistributedAsyncEnvironmentalGRPOTrainer))
    trainer.async_config = AsyncTrainingConfig(
        rollout_max_tokens=rollout_max_tokens,
        rollout_max_thinking_tokens=CAP,
        turn_overlong_penalty=PENALTY,
        **config,
    )
    trainer._tokenizer = _CharTokenizer()
    trainer._metrics = {"train": defaultdict(list)}
    trainer._carry_reasoning = False
    trainer._warned_once = set()
    return trainer


def test_the_reward_build_charges_each_episode_and_logs_the_term_and_its_share():
    """The real path: the trainer's reward build, the batch metric and the per-level ones."""
    episodes = [
        RolloutResult(prompt="a", trajectory=_trajectory((CAP, CAP))),
        RolloutResult(prompt="b", trajectory=_trajectory((START + RAMP // 2, CAP), (1000, CAP), level="high")),
        RolloutResult(prompt="c", trajectory=_trajectory((4096, 8192), (13000, CAP))),
    ]
    trainer = _host()
    rewards = trainer._build_rollout_rewards(
        episodes, trainer._episode_reasoning_tokens(episodes), torch.device("cpu")
    )
    assert rewards.tolist() == pytest.approx([-PENALTY, -PENALTY / 2, 0.0])
    metrics = flushed_metrics(trainer)
    assert metrics["reward/turn_overlong"] == [pytest.approx(-1.5 * PENALTY / 3)]
    assert metrics["reward/turn_overlong_turn_frac"] == [pytest.approx(2 / 5)]
    assert metrics["effort/low/turn_overlong"] == [pytest.approx(-PENALTY / 2)]
    assert metrics["effort/low/turn_overlong_turn_frac"] == [pytest.approx(1 / 3)]
    assert metrics["effort/high/turn_overlong"] == [pytest.approx(-PENALTY / 2)]
    assert metrics["effort/high/turn_overlong_turn_frac"] == [pytest.approx(1 / 2)]


def test_the_charge_adds_to_the_reasoning_price():
    turn = Message.assistant("step", thinking="x" * 8192, reasoning_tokens=CAP, thinking_cap=CAP)
    trajectory = Trajectory(messages=[Message.user("task"), turn])
    trajectory.reasoning_effort = "low"
    episodes = [RolloutResult(prompt="a", trajectory=trajectory)]
    trainer = _host(reasoning_price={"low": 0.01, "medium": 0.005, "high": 0.001})
    rewards = trainer._build_rollout_rewards(
        episodes, trainer._episode_reasoning_tokens(episodes), torch.device("cpu")
    )
    assert rewards.tolist() == pytest.approx([-0.08192 - PENALTY]), "8192 tokens at 0.01 per 1k, plus the cap"


def test_with_the_charge_off_nothing_is_charged_or_logged():
    trainer = _host()
    trainer.async_config = AsyncTrainingConfig()
    episodes = [RolloutResult(prompt="a", trajectory=_trajectory((CAP, CAP)))]
    rewards = trainer._build_rollout_rewards(
        episodes, trainer._episode_reasoning_tokens(episodes), torch.device("cpu")
    )
    assert rewards.tolist() == [0.0]
    assert "reward/turn_overlong" not in flushed_metrics(trainer)


def test_config_refuses_a_charge_it_could_not_apply():
    assert AsyncTrainingConfig().turn_overlong_penalty == 0.0, "the charge is off by default"
    for bad in (-0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="turn_overlong_penalty must be"):
            AsyncTrainingConfig(turn_overlong_penalty=bad)
    with pytest.raises(ValueError, match="requires train_on_sampled_tokens"):
        AsyncTrainingConfig(turn_overlong_penalty=PENALTY, train_on_sampled_tokens=False)
    with pytest.raises(ValueError, match="turn_overlong_penalty requires rollout_reasoning_end_token"):
        AsyncTrainingConfig(turn_overlong_penalty=PENALTY, rollout_reasoning_end_token="")
    with pytest.raises(ValueError, match="not supported with rollout_backend='sglang'"):
        AsyncTrainingConfig(turn_overlong_penalty=PENALTY, rollout_backend="sglang")
    # Anti-vacuity: the same shapes construct with the charge off, and the charge constructs once met.
    AsyncTrainingConfig(train_on_sampled_tokens=False, rollout_reasoning_end_token="")
    assert _host().async_config.turn_overlong_penalty == PENALTY


_END = 151668
GPT_OSS_FINAL = "<|start|>assistant<|channel|>final<|message|>"


class _MarkerTokenizer:
    """What the marker resolver reads: ``</think>`` as one added token, gpt-oss's final-channel opener as five."""

    added_tokens_decoder = dict.fromkeys((_END, 70, 72, 74))

    def encode(self, text, add_special_tokens=False):
        return {"</think>": [_END], GPT_OSS_FINAL: [70, 71, 72, 73, 74]}[text]


def _constructing(budgets: dict, effort: str | None = "random", **config) -> DistributedAsyncEnvironmentalGRPOTrainer:
    trainer = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    trainer.async_config = AsyncTrainingConfig(**config)
    trainer._rollout_env = types.SimpleNamespace(reasoning_effort=effort, thinking_budget_for_effort=budgets.get)
    trainer._tokenizer = _MarkerTokenizer()
    return trainer


def test_construction_refuses_a_charge_no_turn_can_reach():
    with pytest.raises(ValueError, match="no episode of this run carries one"):
        _constructing({}, turn_overlong_penalty=PENALTY)._validate_reasoning_terms()
    _constructing({"high": 8192}, turn_overlong_penalty=PENALTY)._validate_reasoning_terms()
    _constructing({}, turn_overlong_penalty=PENALTY, rollout_max_thinking_tokens=8192)._validate_reasoning_terms()


def test_the_charge_gate_reads_the_levels_the_environment_can_draw():
    """A budget on a level the effort setting never draws caps no turn, so the charge would never fire."""
    with pytest.raises(ValueError, match="no episode of this run carries one"):
        _constructing({"high": 8192}, effort="low", turn_overlong_penalty=PENALTY)._validate_reasoning_terms()
    _constructing(
        {"low": 2048, "high": 8192}, effort="high", turn_overlong_penalty=PENALTY
    )._validate_reasoning_terms()


def test_the_reasoning_marker_is_resolved_for_the_charge_alone():
    """Nothing else reads the marker's single id; without it the rollout counts no turn's reasoning and
    the charge never fires."""
    assert _constructing({}, turn_overlong_penalty=PENALTY)._resolve_reasoning_end_token_id() == _END
    assert _constructing({})._resolve_reasoning_end_token_id() is None


def test_a_marker_of_several_tokens_is_refused_at_construction_with_the_charges_remedy():
    """gpt-oss closes its reasoning with five tokens, and the count reads up to one. Refused at construction,
    before Ray starts, and told to turn the charge off."""
    trainer = _constructing({"high": 8192}, turn_overlong_penalty=PENALTY, rollout_reasoning_end_token=GPT_OSS_FINAL)
    with pytest.raises(ValueError, match="encodes to 5 tokens") as refused:
        trainer._validate_reasoning_terms()
    assert "turn_overlong_penalty needs a single-token" in str(refused.value)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
