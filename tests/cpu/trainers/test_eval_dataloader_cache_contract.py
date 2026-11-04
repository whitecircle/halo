#!/usr/bin/env python
"""Eval paths must cache the prepared loader in ``_eval_dataloaders`` so the next ``evaluate()`` reuses its workers.
Storing the pre-prepare loader hands back an unprepared one or silently re-prepares one per evaluation."""

from types import SimpleNamespace

import pytest
from torch.utils.data import SequentialSampler

import halo.trainers.dataloaders as dataloader_mod
from halo.distributed.parallelism_config import ParallelismConfig
from halo.runtime.consensus import RankConsensus
from halo.trainers.dataloaders import DataParallelLoaderMixin
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.grpo.offline.trainer import OfflineGRPOTrainer

_ROWS = [{"x": 1}, {"x": 2}]


class _Prepared:
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
    t = _base_stub(DataParallelLoaderMixin)
    t._needs_custom_dataloader = lambda: True
    t._get_eval_sampler = SequentialSampler

    def _loader_params(dataset, batch_size, description):
        t.builds.append(description)
        return dataset, {"batch_size": batch_size}

    t._loader_params = _loader_params
    return t, lambda: DataParallelLoaderMixin.get_eval_dataloader(t)


def _offline_stub():
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
    """A raw cached loader differs from the return value, so the next evaluation reuses nothing the caller saw."""
    trainer, get_eval_dataloader = PATHS[path]()
    loader = get_eval_dataloader()

    assert isinstance(loader, _Prepared), f"{path} returned a loader it never prepared"
    assert trainer._eval_dataloaders["eval"] is loader, (
        f"{path} cached a different object than it returned — the two eval paths disagree on what "
        f"``_eval_dataloaders`` holds"
    )


@pytest.mark.parametrize("path", list(PATHS))
def test_a_second_evaluation_reuses_the_cached_loader(path):
    """Rebuilding per ``evaluate()`` forks a new worker pool every time, the leak the cache exists to prevent."""
    trainer, get_eval_dataloader = PATHS[path]()
    first = get_eval_dataloader()
    second = get_eval_dataloader()

    assert second is first, f"{path} rebuilt the eval loader instead of returning the cached one"
    assert len(trainer.builds) == 1, f"{path} rebuilt the underlying loader {len(trainer.builds)} times"


@pytest.mark.parametrize("held", [True, False], ids=["rank-holding-the-key", "rank-that-missed-it"])
def test_a_split_cache_hit_rebuilds_in_lockstep_and_displaces_no_live_loader(monkeypatch, held):
    """A key hits only when every rank holds it; on a split every rank rebuilds, since the build issues collectives.
    A holder keeps its own loader; if replaced, accelerate would keep its worker pool alive for the rest of the run."""
    trainer, _ = _offline_stub()
    live = _Prepared("a loader whose persistent workers are running")
    if held:
        trainer._eval_dataloaders["eval/fingerprint"] = live
    monkeypatch.setattr(dataloader_mod, "rank_consensus", lambda local_ok: RankConsensus(False, True))
    built = []

    def build(dataset):
        built.append(_Prepared(dataset))
        return built[-1]

    loader = DataParallelLoaderMixin._cached_eval_dataloader(trainer, _ROWS, build, key="eval/fingerprint")

    assert len(built) == 1, "a rank skipped the rebuild its peers run in lockstep"
    expected = live if held else built[0]
    assert loader is expected and trainer._eval_dataloaders["eval/fingerprint"] is expected, (
        "the rank's cached loader was displaced, leaving its worker pool running"
        if held
        else "the rebuilt loader was not cached"
    )


class _InitStopped(Exception):
    """Carries control out of the shared init once it is past the declarations."""


class _InitHost:
    def _snapshot_model_config_state(self, model_config):
        raise _InitStopped


def test_the_cache_is_declared_by_the_shared_init():
    """Both paths index ``_eval_dataloaders`` unguarded; a lazy attribute lets them hold different kinds under it."""
    host = _InitHost()

    with pytest.raises(_InitStopped):
        DistributedTrainerMixin._init_distributed_config(host, {"parallelism_config": ParallelismConfig()})

    assert host._eval_dataloaders == {}, "the shared init no longer declares the eval-loader cache"


def test_the_test_loader_takes_the_eval_loaders_dp_sharded_body():
    """HF's own loader shards by world rank, handing TP/CP/PP siblings different rows under the DP-scoped gather."""
    trainer, get_eval_dataloader = _mixin_stub()
    eval_loader = get_eval_dataloader()

    test_loader = DataParallelLoaderMixin.get_test_dataloader(trainer, _ROWS)

    assert isinstance(test_loader, _Prepared), "the test loader skipped the DP-aware prepare"
    assert type(test_loader.raw.sampler) is type(eval_loader.raw.sampler), "the test loader drew another sampler"
    assert trainer.builds == ["evaluation", "test"], trainer.builds
    assert "test" not in trainer._eval_dataloaders, "the test loader must not enter the eval cache"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
