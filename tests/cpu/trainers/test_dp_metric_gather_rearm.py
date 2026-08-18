#!/usr/bin/env python
"""The DP-scoped eval metric gather must survive more than one evaluation loop.

``_install_dp_metric_gather`` runs once, at construction, and points ``gather_function`` at the
DP-scoped gather — TP/CP/ETP/PP siblings return the same rows, so a world gather repeats every
replica once per sibling and accelerate's remainder trim then keeps a prefix of the duplicates.
``Trainer.evaluation_loop`` resets ``gather_function`` back to ``accelerator.gather_for_metrics`` on
its way out, so without a per-loop re-arm only the FIRST evaluate() of a run is scoped and every
later one silently reports duplicated metrics.

    python tests/cpu/trainers/test_dp_metric_gather_rearm.py
"""

import inspect
import re
import types

import pytest
from transformers import Trainer

from src.trainers.mixins.base import DistributedTrainerMixin

WORLD, KEEP = 4, [0, 2]


class _HFEvaluationLoop:
    """``Trainer.evaluation_loop``'s contract with ``gather_function``: use it, then reset it."""

    def evaluation_loop(self, *args, **kwargs):
        self.gathers_used.append(self.gather_function)
        self.gather_function = self.accelerator.gather_for_metrics
        return "metrics"


class _Trainer(DistributedTrainerMixin, _HFEvaluationLoop):
    """The real composed mixin over a fake base — its ``evaluation_loop`` is what is under test."""

    def __init__(self, *, scoped: bool):
        self.accelerator = types.SimpleNamespace(gather_for_metrics=lambda data, **kwargs: data)
        self.gather_function = self.accelerator.gather_for_metrics
        self.gathers_used = []
        if scoped:
            self._dp_metric_gather_scope = (KEEP, WORLD)
            self.gather_function = self._dp_gather_for_metrics


def test_upstream_still_resets_the_gather_function():
    """The premise: transformers hands ``gather_function`` back to accelerate at the end of the loop."""
    source = inspect.getsource(Trainer.evaluation_loop)
    assert re.search(r"self\.gather_function\s*=\s*self\.accelerator\.gather_for_metrics", source)


def test_every_evaluation_loop_gathers_over_the_dp_replicas():
    trainer = _Trainer(scoped=True)

    for _ in range(3):
        trainer.evaluation_loop()

    assert trainer.gathers_used == [trainer._dp_gather_for_metrics] * 3


def test_a_world_scoped_trainer_is_left_on_accelerate_s_gather():
    """No scope means the loader is not DP-sharded — re-arming there would drop real rows."""
    trainer = _Trainer(scoped=False)

    trainer.evaluation_loop()
    trainer.evaluation_loop()

    assert trainer.gathers_used == [trainer.accelerator.gather_for_metrics] * 2


class _PerSplitEvaluate(_HFEvaluationLoop):
    """``Trainer.evaluate``'s shape for a dict ``eval_dataset``: it re-enters ``self.evaluate`` once
    per split, so the mixin's escape hatch nests inside itself."""

    def evaluate(self, *args, **kwargs):
        if self.splits:
            splits, self.splits = self.splits, 0
            for _ in range(splits):
                self.evaluate()
            return "metrics"
        return self.evaluation_loop()


class _NestingTrainer(DistributedTrainerMixin, _PerSplitEvaluate):
    """Two eval splits, the first of them ragged — the shape that nests one hatch inside another."""

    def __init__(self, hatch_verdicts):
        self.accelerator = types.SimpleNamespace(gather_for_metrics=lambda data, **kwargs: data)
        self.gather_function = self.accelerator.gather_for_metrics
        self.args = types.SimpleNamespace(top_entropy_quantile=1.0)
        self.gathers_used = []
        self.splits = 2
        self.hatch_verdicts = list(hatch_verdicts)
        self._dp_metric_gather_scope = (KEEP, WORLD)

    def _needs_custom_accelerator(self):
        return True

    def _eval_ranks_have_unequal_batches(self, eval_args, eval_kwargs):
        return self.hatch_verdicts.pop(0)


def test_the_escape_hatch_survives_the_per_split_recursion(monkeypatch):
    """``Trainer.evaluate`` re-enters itself once per split of a dict ``eval_dataset``, so a split's
    exit must not lift the OUTER hatch: the next split's loop would then re-arm the DP-scoped gather
    while the outer hatch's identity padding is still installed, on ranks whose eval batch counts do
    not match — the deadlock the hatch exists to avoid."""
    monkeypatch.setattr("src.trainers.mixins.base.dist", types.SimpleNamespace(is_initialized=lambda: True))
    monkeypatch.setattr("src.trainers.mixins.base.barrier", lambda *args, **kwargs: None)
    # The dict call and the first split are ragged; the second split is even, so it takes no hatch
    # of its own and must inherit the outer's.
    trainer = _NestingTrainer([True, True, False])

    trainer.evaluate()

    assert len(trainer.gathers_used) == 2, "one evaluation loop per split"
    assert trainer._dp_gather_for_metrics not in trainer.gathers_used, (
        "a split ran the DP-scoped gather inside the outer escape hatch, whose padding is the identity"
    )
    assert trainer._dp_metric_gather_suspended is False, "the hatch must be lifted once the outer exits"


def test_the_unequal_batch_escape_hatch_keeps_its_identity_gather():
    """``evaluate()`` swaps in an identity gather when ranks hold unequal eval batch counts; the
    re-arm must not undo it mid-loop, or that loop deadlocks on the very gather it removed."""
    trainer = _Trainer(scoped=True)
    identity = lambda data: data  # noqa: E731 — the escape hatch's own spelling
    trainer.gather_function = identity
    trainer._dp_metric_gather_suspended = True

    trainer.evaluation_loop()

    assert trainer.gathers_used == [identity]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
