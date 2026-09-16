#!/usr/bin/env python
"""Env-GRPO's eval loader draws a rollout round of ``eval_rollout_batch_size`` rows per rank.

An eval batch is rolled out synchronously, so its wall time is its slowest episode. The round size
reaches both loader paths through the mixin's ``_eval_loader_batch_size`` hook, ``per_device_eval_batch_size``
(the loss forward's chunk) is what the config set once the loader is built, and the round's geometry
is refused at construction where it would only fail after a full round of rollouts.

    python tests/cpu/grpo/test_env_eval_rollout_rounds.py
"""

import types
from unittest import mock

import pytest
from trl import GRPOTrainer

from src.configs.async_training_config import AsyncTrainingConfig
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer
from src.trainers.mixins.dataloader import DataParallelDataLoaderMixin

SENTINEL = object()


class _Args:
    """The two eval-batch readings a real ``TrainingArguments`` exposes."""

    def __init__(self, per_device: int = 4, drop_last: bool = False):
        self.per_device_eval_batch_size = per_device
        self.dataloader_drop_last = drop_last

    @property
    def eval_batch_size(self) -> int:
        return self.per_device_eval_batch_size


def _trainer(
    rows,
    *,
    custom_path: bool = False,
    drop_last: bool = False,
    num_generations_eval: int = 1,
    per_device: int = 4,
):
    host = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    host.args = _Args(per_device=per_device, drop_last=drop_last)
    host.async_config = AsyncTrainingConfig(eval_rollout_batch_size=rows)
    host.num_generations_eval = num_generations_eval
    host._needs_custom_dataloader = lambda: custom_path
    return host


def test_hook_reports_the_round_when_set_and_the_eval_batch_otherwise():
    assert _trainer(68)._eval_loader_batch_size() == 68
    assert _trainer(None)._eval_loader_batch_size() == 4


def test_plain_dp_path_builds_the_loader_at_the_round_and_restores_the_forward_chunk():
    seen = []

    def base_builder(self, eval_dataset=None):
        seen.append(self.args.per_device_eval_batch_size)
        return SENTINEL

    with mock.patch.object(GRPOTrainer, "get_eval_dataloader", base_builder, create=True):
        host = _trainer(68)
        assert host.get_eval_dataloader() is SENTINEL
    assert seen == [68], "the base Trainer reads args.eval_batch_size, so it must see the round there"
    assert host.args.per_device_eval_batch_size == 4, "the loss forward's chunk is what the config set"


def test_forward_chunk_is_restored_when_the_base_builder_raises():
    def failing(self, eval_dataset=None):
        raise RuntimeError("loader build failed")

    with mock.patch.object(GRPOTrainer, "get_eval_dataloader", failing, create=True):
        host = _trainer(68)
        with pytest.raises(RuntimeError, match="loader build failed"):
            host.get_eval_dataloader()
    assert host.args.per_device_eval_batch_size == 4


def test_custom_dp_path_sizes_its_loader_params_from_the_hook():
    seen = []

    def loader_params(self, dataset, batch_size, description):
        seen.append(batch_size)
        raise RuntimeError("stop after sizing")

    host = _trainer(68, custom_path=True)
    host._dataset_presharded = False
    host.eval_dataset = types.SimpleNamespace()
    with mock.patch.object(DataParallelDataLoaderMixin, "_loader_params", loader_params):
        host._cached_eval_dataloader = lambda eval_dataset, build: build(host.eval_dataset)
        with pytest.raises(RuntimeError, match="stop after sizing"):
            host.get_eval_dataloader()
    assert seen == [68]


def test_round_must_hold_whole_eval_groups_and_a_full_tail():
    _trainer(68)._validate_eval_round()
    _trainer(None, drop_last=True)._validate_eval_round()
    with pytest.raises(ValueError, match="divisible by num_generations_eval"):
        _trainer(70, num_generations_eval=4)._validate_eval_round()
    with pytest.raises(ValueError, match="dataloader_drop_last"):
        _trainer(68, drop_last=True)._validate_eval_round()


def test_the_default_round_must_hold_whole_groups_per_rank():
    """With no ``eval_rollout_batch_size`` the round IS ``per_device_eval_batch_size``, so that is the
    geometry the gate reads. TRL validates only the GLOBAL eval batch, so nothing else catches it."""
    with pytest.raises(ValueError, match=r"per_device_eval_batch_size \(6\) must be divisible"):
        _trainer(None, per_device=6, num_generations_eval=4)._validate_eval_round()
    _trainer(None, per_device=8, num_generations_eval=4)._validate_eval_round()


def test_an_explicit_round_is_the_geometry_that_is_checked():
    """With ``eval_rollout_batch_size`` set, the loader draws that many rows per rank and the eval batch
    is only the loss forward's chunk, so it is the round that must hold whole groups."""
    _trainer(8, per_device=6, num_generations_eval=4)._validate_eval_round()
    with pytest.raises(ValueError, match=r"eval_rollout_batch_size \(6\)"):
        _trainer(6, per_device=8, num_generations_eval=4)._validate_eval_round()


def test_round_may_not_exceed_the_in_flight_cap():
    _trainer(68)._check_eval_round_fits_cap(68)
    _trainer(None)._check_eval_round_fits_cap(4)
    with pytest.raises(ValueError, match="3 waves"):
        _trainer(68)._check_eval_round_fits_cap(32)


def test_config_refuses_a_non_positive_round():
    with pytest.raises(ValueError, match="eval_rollout_batch_size must be >= 1"):
        AsyncTrainingConfig(eval_rollout_batch_size=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
