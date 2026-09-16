#!/usr/bin/env python
"""CPU tests for the trainer-side reasoning-calibration term: turn counting and the under-use knob.

Every assistant turn must count toward the calibration band, a thinking-free turn as 0 tokens.
Skipping empty turns scores "no thinking at all" as a zero penalty while brief thinking pays the
under-band penalty — a preference for dropping CoT entirely, the opposite of the term's intent.

``reasoning_compliance_under_use_weight`` scales the below-band side alone: at 0 a short turn is
free while an over-band turn still pays its penalty, and the default is the term's 0.3 under-use weight.

Run: python tests/cpu/grpo/test_reasoning_calibration_counting.py  (or pytest)
"""

import pytest
import torch

from src.configs.async_training_config import AsyncTrainingConfig
from src.environments.base import Message, Trajectory
from src.environments.episode import reasoning_calibration_penalty
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer
from tests.common.grpo_metrics import attach_world_metrics

BUDGET = 1000  # band [300, 900]
WEIGHT = 0.15
DEFAULT_UNDER_USE_WEIGHT = 0.3


class _StubTokenizer:
    """One token per character — deterministic CoT lengths without a real tokenizer."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [0] * len(text)}


class _Rollout:
    def __init__(self, trajectory):
        self.trajectory = trajectory


def _traj(thinkings, budget=BUDGET):
    messages = [Message(role="user", content="task")]
    for t in thinkings:
        messages.append(Message(role="assistant", content="step", thinking=t))
    traj = Trajectory(messages=messages)
    traj.reasoning_budget = budget
    return traj


def _apply(thinkings, budget=BUDGET, weight=WEIGHT, under_use_weight=DEFAULT_UNDER_USE_WEIGHT):
    trainer = attach_world_metrics(object.__new__(DistributedAsyncEnvironmentalGRPOTrainer))
    trainer._tokenizer = _StubTokenizer()
    rewards = torch.zeros(1)
    trainer._apply_reasoning_calibration(rewards, [_Rollout(_traj(thinkings, budget))], weight, under_use_weight)
    return rewards[0].item()


def test_thinking_free_turns_count_as_zero():
    # One in-band turn (500 of 1000: inside [300, 900]) plus two empty turns: the empty turns must
    # drag the mean under-band penalty negative — skipping them scores this episode a perfect 0.
    assert _apply(["x" * 500, "", ""]) < 0.0
    assert _apply(["x" * 500]) == 0.0  # the in-band-only episode is genuinely compliant


def test_all_empty_thinking_pays_the_maximum_under_band_penalty():
    # Zero thinking anywhere: every turn is a full-shortfall under-band turn -> -under_use_weight
    # (0.3) x weight; short-circuiting the empty list to 0.0 would score the episode compliant.
    assert _apply(["", "", ""]) == pytest.approx(-WEIGHT * DEFAULT_UNDER_USE_WEIGHT)


def test_no_assistant_turns_still_scores_zero():
    assert _apply([]) == 0.0


def test_zero_under_use_weight_frees_short_turns_but_still_prices_over_use():
    # 100 of 1000 sits under the 300 floor: priced at the default, free once the under-use side is off.
    assert _apply(["x" * 100]) < 0.0
    assert _apply(["x" * 100], under_use_weight=0.0) == 0.0
    # 1000 of 1000 is full over-use (r >= B): the whole -weight, whatever the under-use side.
    assert _apply(["x" * 1000], under_use_weight=0.0) == pytest.approx(-WEIGHT)
    # Mixed episode: only the over-band turn reaches the per-turn mean.
    assert _apply(["x" * 100, "x" * 1000], under_use_weight=0.0) == pytest.approx(-WEIGHT / 2)


def test_default_under_use_weight_reproduces_the_fixed_term():
    # 100 of 1000 is 200/300 short of the band floor: -0.3 x 2/3 before the run weight scales it.
    expected = -DEFAULT_UNDER_USE_WEIGHT * (300 - 100) / 300
    assert reasoning_calibration_penalty([100], BUDGET) == pytest.approx(expected)
    assert reasoning_calibration_penalty([100], BUDGET, under_use_weight=DEFAULT_UNDER_USE_WEIGHT) == pytest.approx(
        expected
    )
    assert _apply(["x" * 100]) == pytest.approx(WEIGHT * expected)
    assert AsyncTrainingConfig().reasoning_compliance_under_use_weight == DEFAULT_UNDER_USE_WEIGHT


def test_under_use_weight_scales_the_below_band_side_only():
    below_at_default = reasoning_calibration_penalty([100], BUDGET, under_use_weight=0.3)
    assert reasoning_calibration_penalty([100], BUDGET, under_use_weight=0.6) == pytest.approx(2 * below_at_default)
    # In-band and over-band turns never see the knob.
    assert reasoning_calibration_penalty([500], BUDGET, under_use_weight=0.6) == 0.0
    over = reasoning_calibration_penalty([950], BUDGET, under_use_weight=0.0)
    assert over == pytest.approx(-0.5)
    assert reasoning_calibration_penalty([950], BUDGET, under_use_weight=0.6) == over


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
def test_config_refuses_a_negative_or_non_finite_under_use_weight(bad):
    """A negative weight rewards skipping the CoT; NaN passes every ordered comparison."""
    with pytest.raises(ValueError, match="reasoning_compliance_under_use_weight"):
        AsyncTrainingConfig(reasoning_compliance_under_use_weight=bad)
    AsyncTrainingConfig(reasoning_compliance_under_use_weight=0.0)  # no raise: the below-band side off


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
