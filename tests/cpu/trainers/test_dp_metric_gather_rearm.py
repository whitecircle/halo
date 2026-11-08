#!/usr/bin/env python
"""The toolkit's eval metric gather must survive more than one evaluation loop.
The HF loop resets ``gather_function``; without a re-arm later evaluates report duplicated or mis-trimmed metrics."""

import inspect
import re
import types

import pytest
from transformers import Trainer

from halo.trainers.distributed_trainer import DistributedTrainerMixin

WORLD, KEEP = 4, [0, 2]


class _HFEvaluationLoop:
    def evaluation_loop(self, *args, **kwargs):
        self.gathers_used.append(self.gather_function)
        self.gather_function = self.accelerator.gather_for_metrics
        return "metrics"


class _Trainer(DistributedTrainerMixin, _HFEvaluationLoop):
    def __init__(self, *, scoped: bool):
        self.accelerator = types.SimpleNamespace(gather_for_metrics=lambda data, **kwargs: data)
        self.gather_function = self.accelerator.gather_for_metrics
        self.gathers_used = []
        if scoped:
            self._dp_metric_gather_scope = (KEEP, WORLD)
        self.gather_function = self._dp_gather_for_metrics


def test_upstream_still_resets_the_gather_function():
    source = inspect.getsource(Trainer.evaluation_loop)
    assert re.search(r"self\.gather_function\s*=\s*self\.accelerator\.gather_for_metrics", source)


def test_every_evaluation_loop_gathers_over_the_dp_replicas():
    trainer = _Trainer(scoped=True)

    for _ in range(3):
        trainer.evaluation_loop()

    assert trainer.gathers_used == [trainer._dp_gather_for_metrics] * 3


def test_a_world_scoped_trainer_is_re_armed_too():
    """On accelerate's trim, a one-process or RepeatSampler loader is mis-trimmed from the second evaluate() on."""
    trainer = _Trainer(scoped=False)

    trainer.evaluation_loop()
    trainer.evaluation_loop()

    assert trainer.gathers_used == [trainer._dp_gather_for_metrics] * 2


class _PerSplitEvaluate(_HFEvaluationLoop):
    """``Trainer.evaluate`` re-enters itself per split of a dict ``eval_dataset``."""

    def evaluate(self, eval_dataset=None):
        if isinstance(eval_dataset, dict):
            return {name: self.evaluate(eval_dataset=split) for name, split in eval_dataset.items()}
        return self.evaluation_loop()


class _PerSplitTrainer(DistributedTrainerMixin, _PerSplitEvaluate):
    eval_dataset = None

    def __init__(self, hatch_verdicts):
        self.accelerator = types.SimpleNamespace(gather_for_metrics=lambda data, **kwargs: data)
        self.gather_function = self.accelerator.gather_for_metrics
        self.args = types.SimpleNamespace(top_entropy_quantile=1.0)
        self.gathers_used = []
        self.measured = []
        self.hatch_verdicts = list(hatch_verdicts)
        self._dp_metric_gather_scope = (KEEP, WORLD)

    def _needs_custom_accelerator(self):
        return True

    def _eval_ranks_have_unequal_batches(self, eval_dataset):
        self.measured.append(eval_dataset)
        return self.hatch_verdicts.pop(0)


def test_each_split_of_a_dict_takes_its_own_escape_hatch_verdict(monkeypatch):
    """The dict itself is never measured, so an even split's metrics stay global beside a ragged one's rank-0 metrics,
    and the ragged split's hatch is lifted before the next split runs."""
    monkeypatch.setattr("halo.trainers.loop.evaluation.has_process_group", lambda: True)
    monkeypatch.setattr("halo.trainers.loop.evaluation.barrier", lambda *args, **kwargs: None)
    trainer = _PerSplitTrainer([True, False])

    trainer.evaluate({"ragged": "ragged split", "even": "even split"})

    assert trainer.measured == ["ragged split", "even split"]
    ragged_gather, even_gather = trainer.gathers_used
    assert ragged_gather != trainer._dp_gather_for_metrics, "the ragged split lost the hatch's identity gather"
    assert even_gather == trainer._dp_gather_for_metrics, "the even split stayed on the ragged split's hatch"
    assert trainer._dp_metric_gather_suspended is False, "the hatch must be lifted once the split exits"


def test_the_unequal_batch_escape_hatch_keeps_its_identity_gather():
    """The re-arm must not undo the identity gather mid-loop, or that loop deadlocks on the gather it removed."""
    trainer = _Trainer(scoped=True)
    identity = lambda data: data  # noqa: E731 — the escape hatch's own spelling
    trainer.gather_function = identity
    trainer._dp_metric_gather_suspended = True

    trainer.evaluation_loop()

    assert trainer.gathers_used == [identity]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
