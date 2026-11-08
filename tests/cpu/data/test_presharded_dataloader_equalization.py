#!/usr/bin/env python
"""SMPO and Classification dataloaders equalize pre-sharded lengths: ranks running different step counts hang the
next gradient all-reduce. Both train and eval truncate to the global minimum via ``_equalize_presharded_length``,
and a zero-row eval rank raises."""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from datasets import Dataset

from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.on_policy import GenerationRoundMixin
from halo.trainers.grpo.online import DistributedOnlineGRPOTrainer
from halo.trainers.loop.dataloaders import DataParallelLoaderMixin
from halo.trainers.preference.smpo import SMPOTrainer
from halo.trainers.reward.classification import ClassificationTrainer


class _StopAfterEqualize(Exception):
    """Raised by the stubbed sampler to stop before real DataLoader construction."""


def _make_stub(trainer_cls, *, presharded: bool, tp: bool):
    """A bare trainer (no __init__) with enough state to run the dataloader prefix, recording each equalize call's
    split."""
    t = object.__new__(trainer_cls)
    # A TP group of 2 replicates each batch on two ranks.
    t.parallelism_config = SimpleNamespace(
        is_tp_mode=tp, is_cp_mode=False, is_pp_mode=False, non_dp_replication_factor=2 if tp else 1
    )
    t._dataset_presharded = presharded
    t.train_dataset = [{"x": 1}, {"x": 2}, {"x": 3}]
    t.eval_dataset = [{"x": 1}, {"x": 2}]
    t._train_batch_size = 1
    t.data_collator = lambda batch: batch
    t.args = SimpleNamespace(
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
        dataloader_drop_last=False,
        train_sampling_strategy="random",
        dataloader_prefetch_factor=None,
        dataloader_multiprocessing_context=None,
        dataloader_in_order=True,
        process_index=0,
        eval_batch_size=1,
    )
    calls = []
    t._equalize_presharded_length = lambda ds, split="train": calls.append(split) or ds

    def _stop_sampler(*args, **kwargs):
        raise _StopAfterEqualize()

    t._get_train_sampler = _stop_sampler
    t._get_eval_sampler = _stop_sampler
    return t, calls


@pytest.mark.parametrize("trainer_cls", [SMPOTrainer, ClassificationTrainer])
def test_equalizes_when_presharded(trainer_cls):
    t, calls = _make_stub(trainer_cls, presharded=True, tp=False)
    with pytest.raises(_StopAfterEqualize):
        trainer_cls.get_train_dataloader(t)
    assert len(calls) == 1, f"{trainer_cls.__name__}: _equalize_presharded_length not called for presharded dataset"


@pytest.mark.parametrize("trainer_cls", [SMPOTrainer, ClassificationTrainer])
def test_no_equalize_when_not_presharded(trainer_cls):
    # Custom path entered via TP, but the dataset is not pre-sharded, so no truncation.
    t, calls = _make_stub(trainer_cls, presharded=False, tp=True)
    with pytest.raises(_StopAfterEqualize):
        trainer_cls.get_train_dataloader(t)
    assert len(calls) == 0, f"{trainer_cls.__name__}: equalized a non-presharded dataset"


@pytest.mark.parametrize("trainer_cls", [SMPOTrainer, ClassificationTrainer])
def test_eval_equalizes_when_presharded(trainer_cls):
    t, calls = _make_stub(trainer_cls, presharded=True, tp=False)
    with pytest.raises(_StopAfterEqualize):
        trainer_cls.get_eval_dataloader(t)
    assert calls == ["eval"], f"{trainer_cls.__name__}: _equalize_presharded_length not called for presharded eval"


@pytest.mark.parametrize("trainer_cls", [SMPOTrainer, ClassificationTrainer])
def test_eval_no_equalize_when_not_presharded(trainer_cls):
    t, calls = _make_stub(trainer_cls, presharded=False, tp=True)
    with pytest.raises(_StopAfterEqualize):
        trainer_cls.get_eval_dataloader(t)
    assert len(calls) == 0, f"{trainer_cls.__name__}: equalized a non-presharded eval dataset"


def _equalize_with_forced_min(dataset, forced_min: int, split: str, forced_max: int = 0):
    """Run the real _equalize_presharded_length as global rank 0 of a world whose other ranks hold
    ``forced_min`` and ``forced_max`` rows."""
    with (
        patch(
            "halo.trainers.loop.dataloaders.rank_max",
            lambda values: [max(forced_max, values[0]), max(-forced_min, values[1])],
        ),
        # The shared multi-rank probe reads the real torch.distributed, which no CPU test initializes.
        patch("halo.trainers.loop.dataloaders.is_multi_rank_run", return_value=True),
        patch("halo.trainers.loop.dataloaders.is_global_main_process", return_value=True),
    ):
        return DataParallelLoaderMixin._equalize_presharded_length(None, dataset, split=split)


@pytest.mark.parametrize("split", ["eval", "test"])
def test_eval_zero_row_rank_raises(split):
    """Fewer non-empty shards than DP ranks must raise and name the split, not truncate to an empty eval or hang at
    the metrics gather."""
    ds = Dataset.from_dict({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match=rf"Pre-sharded {split} dataset: .* zero {split} examples") as raised:
        _equalize_with_forced_min(ds, forced_min=0, split=split)
    other = {"eval": "test", "test": "eval"}[split]
    assert other not in str(raised.value), f"the {split} error names the {other} split: {raised.value}"


def test_train_zero_row_rank_raises():
    ds = Dataset.from_dict({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="(?i)num.?shards|data_parallel"):
        _equalize_with_forced_min(ds, forced_min=0, split="train")


def test_eval_truncates_to_global_min():
    ds = Dataset.from_dict({"x": [1, 2, 3, 4, 5]})
    out = _equalize_with_forced_min(ds, forced_min=3, split="eval")
    assert len(out) == 3, f"expected truncation to global min 3, got {len(out)}"


def _data_loss_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING and "DATA LOSS" in r.getMessage()]


def test_rank_zero_warns_of_another_ranks_drop_when_it_holds_the_minimum(caplog):
    """Rank 0 truncating nothing still reports the drop the fullest rank takes; shard sizes differ by one row even
    when num_shards is a multiple of data_parallel_size."""
    ds = Dataset.from_dict({"x": list(range(5))})
    with caplog.at_level(logging.WARNING, logger="halo.trainers.loop.dataloaders"):
        out = _equalize_with_forced_min(ds, forced_min=5, forced_max=7, split="train")

    assert len(out) == 5
    (warning,) = _data_loss_warnings(caplog)
    assert "5 to 7 examples" in warning and "DROPS 2" in warning, warning
    assert "not a multiple of data_parallel_size" in warning and "one row per shard" in warning, warning


def test_equal_ranks_warn_of_nothing(caplog):
    ds = Dataset.from_dict({"x": list(range(4))})
    with caplog.at_level(logging.WARNING, logger="halo.trainers.loop.dataloaders"):
        out = _equalize_with_forced_min(ds, forced_min=4, forced_max=4, split="train")
    assert len(out) == 4
    assert _data_loss_warnings(caplog) == []


def test_grpo_trainers_inherit_mixin_eval_dataloader():
    """The GRPO mixin overrides only the train path; eval must resolve to DataParallelLoaderMixin through the MRO,
    ahead of TRL's GRPOTrainer."""
    assert "get_eval_dataloader" not in vars(GenerationRoundMixin)
    for cls in (DistributedOnlineGRPOTrainer, DistributedAsyncEnvironmentalGRPOTrainer):
        assert cls.get_eval_dataloader is DataParallelLoaderMixin.get_eval_dataloader, (
            f"{cls.__name__}.get_eval_dataloader does not resolve to the DP mixin's eval path"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
