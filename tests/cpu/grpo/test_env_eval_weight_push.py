#!/usr/bin/env python
"""An eval generates on what the engines hold. A push ran only at the start of a training step, so the eval right after
an update, with the checkpoint it selects, rolled out on weights up to ``sync_weights_every_n_steps`` steps older."""

import contextlib
from types import SimpleNamespace

import pytest

from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer


def _trainer(monkeypatch, *, every: int = 1, step: int = 0):
    monkeypatch.setattr(DistributedTrainerMixin, "training_step", lambda self, model, inputs, n=None: "loss")
    monkeypatch.setattr(DistributedTrainerMixin, "evaluate", lambda self, *args, **kwargs: "eval")
    trainer = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    trainer.state = SimpleNamespace(global_step=step)
    trainer.async_config = SimpleNamespace(sync_weights_every_n_steps=every)
    trainer._rollout_generation_started = True
    trainer._last_sync_attempt_step = step
    trainer._last_pushed_step = step
    trainer._routing_injector = None
    trainer.pushed: list[int] = []

    def fenced(force=False):
        if trainer._sync_weights_to_engine_would_push(force):
            trainer.pushed.append(trainer.state.global_step)
            trainer._last_pushed_step = trainer.state.global_step

    trainer._sync_weights_to_engine_would_push = lambda force: force or trainer.state.global_step % every == 0
    trainer._sync_weights_to_engine_fenced = fenced
    return trainer


def _run_steps(trainer, eval_at: set[int], steps: int) -> None:
    for step in range(trainer.state.global_step + 1, trainer.state.global_step + 1 + steps):
        trainer.state.global_step = step
        if step in eval_at:
            assert trainer.evaluate() == "eval"
        trainer.training_step(None, {})


def test_an_eval_between_cadence_pushes_pushes_first(monkeypatch):
    trainer = _trainer(monkeypatch, every=4)
    _run_steps(trainer, {10}, 12)
    assert trainer.pushed == [4, 8, 10, 12]


def test_at_every_step_cadence_an_eval_takes_the_next_steps_push(monkeypatch):
    """The update it evaluates was not pushed yet, and the next step would have pushed the same weights."""
    trainer = _trainer(monkeypatch, every=1)
    _run_steps(trainer, {2}, 3)
    assert trainer.pushed == [1, 2, 3], "one push per update, eval or not"


def test_an_eval_on_weights_the_engines_hold_pushes_nothing(monkeypatch):
    """``eval_on_start``, where train-begin's forced push stamped this step, and a second eval at one step."""
    trainer = _trainer(monkeypatch, step=5)
    assert trainer.evaluate() == "eval"
    assert trainer.evaluate() == "eval"
    assert trainer.pushed == []


def test_an_eval_outside_training_pushes_nothing(monkeypatch):
    """No round is running to generate for, and no group is formed."""
    trainer = _trainer(monkeypatch, step=5)
    trainer._rollout_generation_started = False
    trainer.state.global_step = 7
    assert trainer.evaluate() == "eval"
    assert trainer.pushed == []


def test_the_real_cadence_gate_records_what_the_engines_hold(monkeypatch):
    trainer = _trainer(monkeypatch, every=4, step=3)
    del trainer._sync_weights_to_engine_fenced
    module = "halo.trainers.grpo.environmental.async_rollouts"
    monkeypatch.setattr(f"{module}.sync_trainer_weights", lambda trainer, client, **_: None)
    trainer.first_push_verified = True
    monkeypatch.setattr(f"{module}.profiling_context", lambda trainer, name: contextlib.nullcontext())
    trainer._weight_sync_group_formed = True
    trainer._weight_sync_client = object()
    assert trainer._sync_weights_to_engine() is False
    assert trainer._last_pushed_step == 3, "a declined push moved nothing"
    assert trainer._sync_weights_to_engine(force=True) is True
    assert trainer._last_pushed_step == 3
    trainer.state.global_step = 8
    assert trainer._sync_weights_to_engine() is True
    assert trainer._last_pushed_step == 8


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
