"""The thinking-budget e2e check grades the turns the checkpoint's own policy sampled.

The e2e takes one large optimizer step, after which a gpt-oss policy writes turns that never open an
analysis channel: the engine's budget, counted per analysis message, has nothing to cap there, while the
reasoning parser still reports the whole turn as reasoning. The recorder keeps only the turns sampled
before the first optimizer step, so the check measures the engine's budget rather than the step, and it
probes the engine's enforcement only after the trainer's first push: until then a server shared by the rows
of a tier serves the previous row's trained policy.
"""

import types

import pytest

from src.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from tests.common import thinking_budget
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


def test_the_enforcement_probe_reads_the_pushed_checkpoint_once(recorder, monkeypatch):
    events: list[str] = []

    def push(self):
        if self._rollout_generation_started:
            return
        self._rollout_generation_started = True
        events.append("push")

    monkeypatch.setattr(DistributedAsyncEnvironmentalGRPOTrainer, "_start_rollout_generation", push)
    recorder._rollout_generation_started = False
    recorder.budget_probe = lambda: events.append("probe")

    recorder._start_rollout_generation()
    recorder._start_rollout_generation()

    assert events == ["push", "probe"]


def test_a_rank_without_a_probe_only_pushes(recorder, monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(
        DistributedAsyncEnvironmentalGRPOTrainer, "_start_rollout_generation", lambda self: events.append("push")
    )
    recorder._rollout_generation_started = False
    recorder.budget_probe = None

    recorder._start_rollout_generation()

    assert events == ["push"]


def test_an_accepted_field_is_not_enough_until_the_probe_runs(monkeypatch):
    """A run whose probe never fires cannot pass on the acceptance check alone."""
    monkeypatch.setattr(thinking_budget, "_chat", lambda *a, **k: types.SimpleNamespace(status_code=200, text=""))
    checks: dict[str, bool] = {}

    assert thinking_budget.record_budget_acceptance("http://server", "model", BUDGET, checks)
    assert checks == {"server_accepts_the_thinking_budget": True, "thinking_budget_probe_ran": False}


def test_a_refused_field_fails_enforcement(monkeypatch):
    monkeypatch.setattr(
        thinking_budget, "_chat", lambda *a, **k: types.SimpleNamespace(status_code=400, text="no parser")
    )
    checks: dict[str, bool] = {}

    assert not thinking_budget.record_budget_acceptance("http://server", "model", BUDGET, checks)
    assert checks == {"server_accepts_the_thinking_budget": False, "server_enforced_the_thinking_budget": False}


def test_a_probe_that_elicits_no_reasoning_does_not_blame_the_engine(monkeypatch):
    monkeypatch.setattr(thinking_budget, "_chat", lambda *a, **k: types.SimpleNamespace(status_code=200, text=""))
    monkeypatch.setattr(thinking_budget, "_cot", lambda *a, **k: "")
    checks: dict[str, bool] = {}

    thinking_budget.record_budget_acceptance("http://server", "model", BUDGET, checks)
    thinking_budget.record_budget_enforcement("http://server", "model", BUDGET, None, checks)

    assert not checks["thinking_budget_probe_elicited_reasoning"]
    assert "server_enforced_the_thinking_budget" not in checks


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
