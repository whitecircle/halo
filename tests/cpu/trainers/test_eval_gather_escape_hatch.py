#!/usr/bin/env python
"""``evaluate``'s unequal-batch escape hatch: each rank scores its own shard, as the per-step gather would deadlock.
TRL's entropy threshold gathers per step: identity padding SIGABRTs and real padding deadlocks, so it is refused."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from datasets import Dataset

import halo.trainers.loop.evaluation as mixin_module
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.grpo.offline.group_sampler import GroupedLoaderMixin
from halo.trainers.loop.dataloaders import DataParallelLoaderMixin
from halo.trainers.loop.evaluation import EvaluationMixin

# Any quantile < 1.0 turns entropy masking on; the sentinel is what the stubbed super().evaluate() returns.
_MASKING_QUANTILE = 0.8
_SENTINEL_METRICS = {"eval_loss": 0.5}

_SENTINEL = object()


class _Accelerator:
    """``pad_across_processes`` is on the class, which makes restore by deletion vs re-assignment observable."""

    def pad_across_processes(self, tensor, *args, **kwargs):
        return ("class-padded", tensor)


class _InnerTrainer:
    """Records what the two seams did at the moment of the call: after the hatch, before the restore."""

    def evaluate(self, *args, **kwargs):
        self.observed.append((self.gather_function(_SENTINEL), self.accelerator.pad_across_processes(_SENTINEL)))
        return dict(_SENTINEL_METRICS)


class _StubTrainer(DistributedTrainerMixin, _InnerTrainer):
    """Real MRO so the mixin's zero-arg ``super().evaluate()`` resolves."""

    eval_dataset = None

    def __init__(self, *, top_entropy_quantile: float = 1.0, unequal: bool = True):
        self.args = SimpleNamespace(top_entropy_quantile=top_entropy_quantile)
        self.accelerator = _Accelerator()
        self.gather_function = lambda tensor: ("real-gather", tensor)
        self.observed: list = []
        self._unequal = unequal

    def _needs_custom_accelerator(self) -> bool:
        return True

    def _eval_ranks_have_unequal_batches(self, eval_dataset) -> bool:
        return self._unequal


@pytest.fixture
def distributed():
    """The branch is gated on a live process group; nothing below needs a real one."""
    with (
        patch.object(mixin_module.dist, "is_initialized", return_value=True),
        patch.object(mixin_module, "barrier", lambda *args, **kwargs: None),
        patch.object(mixin_module, "is_global_main_process", return_value=True),
    ):
        yield


def test_entropy_masking_is_refused_on_the_unequal_batch_branch(distributed):
    trainer = _StubTrainer(top_entropy_quantile=_MASKING_QUANTILE)

    with pytest.raises(ValueError, match="top_entropy_quantile") as excinfo:
        trainer.evaluate()

    assert "1.0" in str(excinfo.value), "the message must name the setting that unblocks the run"
    assert trainer.observed == [], "the inner evaluate must not have been reached"


def test_the_refusal_fires_before_anything_is_neutralized(distributed):
    """A raise from inside the ``try`` would leave the swap in place; the ``finally`` only covers what it wraps."""
    trainer = _StubTrainer(top_entropy_quantile=_MASKING_QUANTILE)

    with pytest.raises(ValueError):
        trainer.evaluate()

    assert trainer.gather_function(_SENTINEL) == ("real-gather", _SENTINEL)
    assert "pad_across_processes" not in trainer.accelerator.__dict__


def test_both_seams_are_identity_inside_the_hatch_and_restored_after(distributed):
    trainer = _StubTrainer()

    assert trainer.evaluate() == _SENTINEL_METRICS

    gathered, padded = trainer.observed[0]
    assert gathered is _SENTINEL, "gather_function must be identity while the inner loop runs"
    assert padded is _SENTINEL, "pad_across_processes must be identity too — evaluation_loop pads there"

    assert trainer.gather_function(_SENTINEL) == ("real-gather", _SENTINEL)
    assert trainer.accelerator.pad_across_processes(_SENTINEL) == ("class-padded", _SENTINEL)


def test_the_accelerator_is_restored_by_deleting_the_instance_attribute(distributed):
    """Re-assigning the bound method would pin an instance shadow, and later gathers would keep identity semantics."""
    trainer = _StubTrainer()
    trainer.evaluate()

    assert "pad_across_processes" not in trainer.accelerator.__dict__


def test_the_hatch_is_unwound_when_the_inner_evaluate_raises(distributed):
    trainer = _StubTrainer()
    with patch.object(_InnerTrainer, "evaluate", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            trainer.evaluate()

    assert trainer.gather_function(_SENTINEL) == ("real-gather", _SENTINEL)
    assert "pad_across_processes" not in trainer.accelerator.__dict__


def test_the_equal_batch_path_never_swaps_either_seam(distributed):
    """Identity gathering on equal batches would report rank 0's shard as the global metric."""
    trainer = _StubTrainer(top_entropy_quantile=_MASKING_QUANTILE, unequal=False)

    assert trainer.evaluate() == _SENTINEL_METRICS

    gathered, padded = trainer.observed[0]
    assert gathered == ("real-gather", _SENTINEL)
    assert padded == ("class-padded", _SENTINEL)


def test_the_hatch_warns_that_the_logged_metrics_are_one_ranks_shard(distributed):
    """Nothing in the logged numbers says they cover rank 0's shard alone; the warning is the only sign."""
    trainer = _StubTrainer()
    with patch.object(mixin_module.logger, "warning") as warning:
        trainer.evaluate(metric_key_prefix="test")
    (message,), _ = warning.call_args
    assert "'test_loss'" in message and "rank 0's own shard" in message


def test_the_equal_batch_path_does_not_warn(distributed):
    trainer = _StubTrainer(unequal=False)
    with patch.object(mixin_module.logger, "warning") as warning:
        trainer.evaluate()
    warning.assert_not_called()


def _raise(exc):
    raise exc


def test_only_a_length_less_loader_counts_as_unmeasurable():
    """``len()`` of an iterable loader is a ``TypeError``; anything else is a bug to surface."""
    unmeasurable = SimpleNamespace(get_eval_dataloader=lambda dataset: iter(()))
    assert DistributedTrainerMixin._eval_ranks_have_unequal_batches(unmeasurable, None) is True

    broken = SimpleNamespace(get_eval_dataloader=lambda dataset: _raise(AttributeError("loader bug")))
    with pytest.raises(AttributeError, match="loader bug"):
        DistributedTrainerMixin._eval_ranks_have_unequal_batches(broken, None)


class _ReachedBase:
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        return {"evaluated": eval_dataset}


class _OfflineLoaderOrder(GroupedLoaderMixin, EvaluationMixin, DataParallelLoaderMixin, _ReachedBase):
    """Offline GRPO's real order of these mixins, over a base recording what reached it."""

    eval_dataset = None

    def _needs_custom_accelerator(self) -> bool:
        return True


@pytest.mark.parametrize("given_as", ["positional", "keyword", "constructor"])
def test_a_dict_of_splits_reaches_the_base_unmeasured(distributed, given_as):
    """Nothing can measure a loader over a dict: offline GRPO's loader reads the dict's ``column_names`` and raises on
    every multi-rank run, the very call its constructor refusal points at."""
    splits = {"first": Dataset.from_dict({"group_id": [0, 1]}), "second": Dataset.from_dict({"group_id": [2, 3]})}
    trainer = _OfflineLoaderOrder()
    if given_as == "positional":
        metrics = trainer.evaluate(splits)
    elif given_as == "keyword":
        metrics = trainer.evaluate(eval_dataset=splits)
    else:
        trainer.eval_dataset = splits
        metrics = trainer.evaluate()
    assert metrics == {"evaluated": None if given_as == "constructor" else splits}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
