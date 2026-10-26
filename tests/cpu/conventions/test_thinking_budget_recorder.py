"""The thinking-budget e2e check grades the turns the checkpoint's own policy sampled.

The e2e takes one large optimizer step, after which a gpt-oss policy writes turns that never open an
analysis channel: the engine's budget, counted per analysis message, has nothing to cap there, while the
reasoning parser still reports the whole turn as reasoning. The recorder keeps only the turns sampled
before the first optimizer step, so the check measures the engine's budget rather than the step.
"""

import types

import pytest

from src.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from tests.common.thinking_budget import ReasoningTurnRecorder, record_thinking_budget_checks

BUDGET = 64


@pytest.fixture
def recorder(monkeypatch):
    monkeypatch.setattr(
        DistributedAsyncEnvironmentalGRPOTrainer, "_log_rollout_metrics", lambda self, results, mode, tokens: None
    )
    trainer = ReasoningTurnRecorder.__new__(ReasoningTurnRecorder)
    trainer.sampled_reasoning_tokens = []
    trainer.state = types.SimpleNamespace(global_step=0)
    return trainer


def test_turns_sampled_after_the_first_update_are_not_graded(recorder):
    recorder._log_rollout_metrics([], "train", [[30, 12], [0]])
    recorder.state.global_step = 1
    recorder._log_rollout_metrics([], "train", [[124, 128]])
    checks: dict[str, bool] = {}

    record_thinking_budget_checks(recorder, checks, BUDGET)

    assert recorder.sampled_reasoning_tokens == [30, 12, 0]
    assert checks["thinking_budget_capped_every_turn"]


def test_an_over_budget_turn_of_the_checkpoints_policy_fails_the_check(recorder):
    recorder._log_rollout_metrics([], "train", [[30, BUDGET + 35]])
    checks: dict[str, bool] = {}

    record_thinking_budget_checks(recorder, checks, BUDGET)

    assert not checks["thinking_budget_capped_every_turn"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
