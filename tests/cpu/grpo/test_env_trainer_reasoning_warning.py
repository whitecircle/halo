#!/usr/bin/env python
"""A reasoning-consuming knob with no captured reasoning warns once per run.

Reasoning reaches an assistant turn only through the rollout server's reasoning parser. Without one,
the reasoning floor scores every episode as maximal under-use, the reasoning price charges nothing,
and ``carry_reasoning`` sends nothing back — all silently. The trainer warns once, at the point the
length terms are applied, when a step's assistant turns carry no reasoning while either knob is on; a
step with reasoning, a step with no assistant turn, or a run with both knobs off warns nothing.

    python tests/cpu/grpo/test_env_trainer_reasoning_warning.py
"""

import logging
from collections import defaultdict

import pytest
import torch

from src.configs.async_training_config import AsyncTrainingConfig
from src.environments.base import Message, Trajectory
from src.environments.episode import RolloutResult
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer
from src.trainers.grpo.reasoning_terms import reasoning_token_counts
from tests.common.grpo_metrics import attach_world_metrics, flushed_metrics

# The warning goes through the tokenize mixin's stdlib logger (the repo's warn_once convention).
LOGGER = "src.trainers.grpo.rollout.trajectory_tokenize"


class _StubTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [0] * len(text)}


def _trainer(floor_weight: float, carry_reasoning: bool):
    trainer = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    trainer.async_config = AsyncTrainingConfig(reasoning_floor=floor_weight)
    trainer._tokenizer = _StubTokenizer()
    attach_world_metrics(trainer)
    trainer._metrics = {"train": defaultdict(list)}
    trainer._carry_reasoning = carry_reasoning
    trainer._warned_once = set()
    return trainer


def _rollouts(thinkings: list[str | None]) -> list[RolloutResult]:
    traj = Trajectory(messages=[Message.user("task"), *(Message.assistant("step", thinking=t) for t in thinkings)])
    traj.reasoning_budget = 1000
    return [RolloutResult(prompt="task", trajectory=traj)]


def _with_counts(rollouts: list[RolloutResult]) -> tuple:
    """``_build_rollout_rewards``' arguments: the rollouts, their per-turn reasoning counts, the device."""
    counts = [reasoning_token_counts(_StubTokenizer(), r.trajectory) for r in rollouts]
    return rollouts, counts, torch.device("cpu")


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and "reasoning" in r.getMessage()]


def test_length_terms_with_no_captured_reasoning_warn_once_per_run(caplog):
    trainer = _trainer(0.15, carry_reasoning=False)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        for _ in range(2):
            trainer._build_rollout_rewards(*_with_counts(_rollouts([None, None])))
            flushed_metrics(trainer)  # the step boundary: a second record before it is refused
    assert len(_warnings(caplog)) == 1
    assert "the reasoning terms" in _warnings(caplog)[0]
    assert "carry_reasoning" not in _warnings(caplog)[0]


def test_carried_reasoning_with_no_captured_reasoning_warns_by_its_own_name(caplog):
    trainer = _trainer(0.0, carry_reasoning=True)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        trainer._build_rollout_rewards(*_with_counts(_rollouts([None])))
    assert len(_warnings(caplog)) == 1 and "carry_reasoning" in _warnings(caplog)[0]


def test_a_step_with_captured_reasoning_warns_nothing(caplog):
    trainer = _trainer(0.15, carry_reasoning=True)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        trainer._build_rollout_rewards(*_with_counts(_rollouts([None, "let me think"])))
    assert _warnings(caplog) == []


def test_no_consumer_means_no_warning_and_no_turn_means_no_evidence(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _trainer(0.0, carry_reasoning=False)._build_rollout_rewards(*_with_counts(_rollouts([None])))
        _trainer(0.15, carry_reasoning=True)._build_rollout_rewards(*_with_counts(_rollouts([])))
    assert _warnings(caplog) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
