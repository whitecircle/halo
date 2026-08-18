#!/usr/bin/env python
"""CPU test: SMPO / Classification dataloaders equalize pre-sharded lengths.

When a dataset is pre-sharded per DP rank, the shards can differ in length, so ranks would run a
different number of optimizer steps and the next gradient all-reduce hangs forever.
``DataParallelDataLoaderMixin.get_train_dataloader`` AND ``get_eval_dataloader`` (inherited by SMPO
and Classification) truncate every rank to the global minimum via ``_equalize_presharded_length``.
These tests fail if either call is dropped (the hang would return silently), if the eval zero-row
case stops raising, or if the GRPO trainers stop inheriting the eval path from the mixin.

    python tests/cpu/data/test_presharded_dataloader_equalization.py
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from accelerate import PartialState
from datasets import Dataset

# _equalize_presharded_length logs via the accelerate logger, which requires an initialized state.
PartialState()

from src.trainers.mixins.dataloader import DataParallelDataLoaderMixin
from src.trainers.preference.smpo import SmoothMarginPOTrainer
from src.trainers.reward.classification import ClassificationTrainer


class _StopAfterEqualize(Exception):
    """Sentinel raised by the stubbed sampler to stop before real DataLoader construction."""


def _make_stub(trainer_cls, *, presharded: bool, tp: bool):
    """Build a bare trainer instance (no __init__) with just enough state to run the
    get_train_dataloader/get_eval_dataloader prefix, recording each _equalize_presharded_length
    call's split."""
    t = object.__new__(trainer_cls)
    t.parallelism_config = SimpleNamespace(is_tp_mode=tp, is_cp_mode=False, is_expert_tp_mode=False, is_pp_mode=False)
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
        dataloader_prefetch_factor=None,
        process_index=0,
        eval_batch_size=1,
    )
    calls = []
    t._equalize_presharded_length = lambda ds, split="train": calls.append(split) or ds

    # Stop right after the equalization decision, before building a real DataLoader.
    def _stop_sampler(*args, **kwargs):
        raise _StopAfterEqualize()

    t._get_train_sampler = _stop_sampler
    t._get_eval_sampler = _stop_sampler
    return t, calls


@pytest.mark.parametrize("trainer_cls", [SmoothMarginPOTrainer, ClassificationTrainer])
def test_equalizes_when_presharded(trainer_cls):
    t, calls = _make_stub(trainer_cls, presharded=True, tp=False)
    with pytest.raises(_StopAfterEqualize):
        trainer_cls.get_train_dataloader(t)
    assert len(calls) == 1, f"{trainer_cls.__name__}: _equalize_presharded_length not called for presharded dataset"


@pytest.mark.parametrize("trainer_cls", [SmoothMarginPOTrainer, ClassificationTrainer])
def test_no_equalize_when_not_presharded(trainer_cls):
    # Custom path entered via TP, but dataset not pre-sharded → must NOT truncate.
    t, calls = _make_stub(trainer_cls, presharded=False, tp=True)
    with pytest.raises(_StopAfterEqualize):
        trainer_cls.get_train_dataloader(t)
    assert len(calls) == 0, f"{trainer_cls.__name__}: equalized a non-presharded dataset"


@pytest.mark.parametrize("trainer_cls", [SmoothMarginPOTrainer, ClassificationTrainer])
def test_eval_equalizes_when_presharded(trainer_cls):
    """Pre-sharded EVAL must be equalized too: unequal eval step counts hang on NCCL."""
    t, calls = _make_stub(trainer_cls, presharded=True, tp=False)
    with pytest.raises(_StopAfterEqualize):
        trainer_cls.get_eval_dataloader(t)
    assert calls == ["eval"], f"{trainer_cls.__name__}: _equalize_presharded_length not called for presharded eval"


@pytest.mark.parametrize("trainer_cls", [SmoothMarginPOTrainer, ClassificationTrainer])
def test_eval_no_equalize_when_not_presharded(trainer_cls):
    t, calls = _make_stub(trainer_cls, presharded=False, tp=True)
    with pytest.raises(_StopAfterEqualize):
        trainer_cls.get_eval_dataloader(t)
    assert len(calls) == 0, f"{trainer_cls.__name__}: equalized a non-presharded eval dataset"


class _FakeAllReduce:
    """Stand-in for torch.distributed that forces the global-min all_reduce result."""

    def __init__(self, forced_min: int):
        self.forced_min = forced_min
        self.ReduceOp = torch.distributed.ReduceOp

    def is_available(self):
        return True

    def is_initialized(self):
        return True

    def get_world_size(self):
        return 2

    def all_reduce(self, tensor, op=None):
        tensor.fill_(min(self.forced_min, int(tensor.item())))


def _equalize_with_forced_min(dataset, forced_min: int, split: str):
    """Run the real _equalize_presharded_length with a faked distributed MIN reduction."""
    with (
        patch("src.trainers.mixins.dataloader.dist", _FakeAllReduce(forced_min)),
        # The shared multi-rank probe reads the real torch.distributed, which no CPU test initializes.
        patch("src.trainers.mixins.dataloader.is_multi_rank_run", return_value=True),
        patch("src.trainers.mixins.dataloader.current_device", return_value=torch.device("cpu")),
    ):
        return DataParallelDataLoaderMixin._equalize_presharded_length(None, dataset, split=split)


def test_eval_zero_row_rank_raises():
    """A rank with zero eval rows (fewer non-empty test shards than DP ranks) must raise, not
    silently truncate to an empty eval or hang at the metrics gather."""
    ds = Dataset.from_dict({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="(?i)eval.*zero|zero.*eval"):
        _equalize_with_forced_min(ds, forced_min=0, split="eval")


def test_train_zero_row_rank_raises():
    ds = Dataset.from_dict({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="(?i)num.?shards|data_parallel"):
        _equalize_with_forced_min(ds, forced_min=0, split="train")


def test_eval_truncates_to_global_min():
    ds = Dataset.from_dict({"x": [1, 2, 3, 4, 5]})
    out = _equalize_with_forced_min(ds, forced_min=3, split="eval")
    assert len(out) == 3, f"expected truncation to global min 3, got {len(out)}"


def test_grpo_trainers_inherit_mixin_eval_dataloader():
    """The GRPO dataloader mixin overrides only the TRAIN path; eval must resolve to
    DataParallelDataLoaderMixin.get_eval_dataloader through the MRO (ahead of TRL's GRPOTrainer),
    so the pre-sharded eval equalization applies to online + environmental GRPO too."""
    from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer
    from src.trainers.grpo.mixins.dataloader import GRPOTrainDataLoaderMixin
    from src.trainers.grpo.online import DistributedGRPOTrainer

    assert "get_eval_dataloader" not in vars(GRPOTrainDataLoaderMixin)
    for cls in (DistributedGRPOTrainer, DistributedAsyncEnvironmentalGRPOTrainer):
        assert cls.get_eval_dataloader is DataParallelDataLoaderMixin.get_eval_dataloader, (
            f"{cls.__name__}.get_eval_dataloader does not resolve to the DP mixin's eval path"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
