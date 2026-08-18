#!/usr/bin/env python
"""CPU tests for the trainer-side reasoning-calibration turn counting.

Every assistant turn must count toward the calibration band, a thinking-free turn as 0 tokens.
Skipping empty turns scores "no thinking at all" as a zero penalty while brief thinking pays the
under-band penalty — a preference for dropping CoT entirely, the opposite of the term's intent.

Run: python tests/cpu/grpo/test_reasoning_calibration_counting.py  (or pytest)
"""

from collections import defaultdict

import pytest
import torch

from src.environments.base import Message, Trajectory
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer


class _StubTokenizer:
    """One token per character — deterministic CoT lengths without a real tokenizer."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [0] * len(text)}


class _Rollout:
    def __init__(self, trajectory):
        self.trajectory = trajectory


def _traj(thinkings, budget=1000):
    messages = [Message(role="user", content="task")]
    for t in thinkings:
        messages.append(Message(role="assistant", content="step", thinking=t))
    traj = Trajectory(messages=messages)
    traj.reasoning_budget = budget
    return traj


def _apply(thinkings, budget=1000, weight=0.15):
    trainer = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    trainer._tokenizer = _StubTokenizer()
    trainer._metrics = {"train": defaultdict(list)}
    rewards = torch.zeros(1)
    trainer._apply_reasoning_calibration(rewards, [_Rollout(_traj(thinkings, budget))], weight, "train")
    return rewards[0].item()


def test_thinking_free_turns_count_as_zero():
    # One in-band turn (500 of 1000: inside [300, 900]) plus two empty turns: the empty turns must
    # drag the mean under-band penalty negative — skipping them scores this episode a perfect 0.
    assert _apply(["x" * 500, "", ""]) < 0.0
    assert _apply(["x" * 500]) == 0.0  # the in-band-only episode is genuinely compliant


def test_all_empty_thinking_pays_the_maximum_under_band_penalty():
    # Zero thinking anywhere: every turn is a full-shortfall under-band turn -> -under_use_weight
    # (0.3) x weight; short-circuiting the empty list to 0.0 would score the episode compliant.
    assert _apply(["", "", ""]) == pytest.approx(-0.15 * 0.3)


def test_no_assistant_turns_still_scores_zero():
    assert _apply([]) == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
