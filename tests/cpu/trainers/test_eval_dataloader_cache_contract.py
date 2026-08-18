#!/usr/bin/env python
"""Every eval path must cache the SAME kind of object in ``_eval_dataloaders``.

Two paths write that dict on one MRO — the shared ``DataParallelDataLoaderMixin`` one and offline
GRPO's grouped-sampler one. The base ``Trainer`` contract is to store the PREPARED loader, which is
the whole point of ``dataloader_persistent_workers``: the next ``evaluate()`` reuses it instead of
forking a new worker pool. A path storing the pre-prepare loader under the same key either hands
back an unprepared loader or silently re-prepares one per evaluation.

    python tests/cpu/trainers/test_eval_dataloader_cache_contract.py
"""

from types import SimpleNamespace

import pytest
from torch.utils.data import SequentialSampler

from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.grpo.offline import OfflineGRPOTrainer
from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.mixins.dataloader import DataParallelDataLoaderMixin

_ROWS = [{"x": 1}, {"x": 2}]


class _Prepared:
    """Stand-in for an accelerate-prepared loader — distinguishable from the loader it wraps."""

    def __init__(self, raw):
        self.raw = raw


def _base_stub(trainer_cls):
    t = object.__new__(trainer_cls)
    t._eval_dataloaders = {}
    t._dataset_presharded = False
    t.eval_dataset = _ROWS
    t.data_collator = lambda batch: batch
    t.args = SimpleNamespace(
        dataloader_persistent_workers=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_drop_last=False,
        dataloader_prefetch_factor=None,
        eval_batch_size=1,
        per_device_eval_batch_size=1,
        data_seed=None,
        seed=0,
    )
    t.builds = []
    t._prepare_dataloader = lambda loader, **kwargs: _Prepared(loader)
    return t


def _mixin_stub():
    """The shared DP eval path, stopped just short of accelerate."""
    t = _base_stub(DataParallelDataLoaderMixin)
    t._needs_custom_dataloader = lambda: True
    t._get_eval_sampler = SequentialSampler

    def _loader_params(dataset, batch_size, description):
        t.builds.append(description)
        return dataset, {"batch_size": batch_size}

    t._loader_params = _loader_params
    return t, lambda: DataParallelDataLoaderMixin.get_eval_dataloader(t)


def _offline_stub():
    """Offline GRPO's own grouped-sampler eval path."""
    t = _base_stub(OfflineGRPOTrainer)
    t._cached_eval_group_ids = [0, 0]

    def _build_grouped_dataloader(dataset, group_ids, batch_size, shuffle):
        t.builds.append("evaluation")
        return object()

    t._build_grouped_dataloader = _build_grouped_dataloader
    return t, lambda: OfflineGRPOTrainer.get_eval_dataloader(t)


PATHS = {"shared-mixin": _mixin_stub, "offline-grpo": _offline_stub}


@pytest.mark.parametrize("path", list(PATHS))
def test_the_cached_loader_is_the_one_handed_back(path):
    """Cache the PREPARED loader: caching the raw one makes the cached value a different object from
    the return value, so the next evaluation reuses nothing the caller ever saw."""
    trainer, get_eval_dataloader = PATHS[path]()
    loader = get_eval_dataloader()

    assert isinstance(loader, _Prepared), f"{path} returned a loader it never prepared"
    assert trainer._eval_dataloaders["eval"] is loader, (
        f"{path} cached a different object than it returned — the two eval paths disagree on what "
        f"``_eval_dataloaders`` holds"
    )


@pytest.mark.parametrize("path", list(PATHS))
def test_a_second_evaluation_reuses_the_cached_loader(path):
    """The persistent-worker cache must actually short-circuit: rebuilding per ``evaluate()`` forks a
    new worker pool every time, which is the leak the cache exists to prevent."""
    trainer, get_eval_dataloader = PATHS[path]()
    first = get_eval_dataloader()
    second = get_eval_dataloader()

    assert second is first, f"{path} rebuilt the eval loader instead of returning the cached one"
    assert len(trainer.builds) == 1, f"{path} rebuilt the underlying loader {len(trainer.builds)} times"


class _InitStopped(Exception):
    """Carries control out of the shared init once it is past the declarations."""


class _InitHost:
    """A bare trainer host for the shared init, which must declare the cache before anything else."""

    def _configure_mixed_precision(self, kwargs, training_args):
        raise _InitStopped


def test_the_cache_is_declared_by_the_shared_init():
    """Both paths index ``_eval_dataloaders`` unguarded, so the shared init has to declare it: an
    attribute materialized lazily at first use lets the two paths hold different kinds under it."""
    host = _InitHost()

    with pytest.raises(_InitStopped):
        DistributedTrainerMixin._init_distributed_config(host, {"parallelism_config": ParallelismConfig()})

    assert host._eval_dataloaders == {}, "the shared init no longer declares the eval-loader cache"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
