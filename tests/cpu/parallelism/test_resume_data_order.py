#!/usr/bin/env python
"""A mid-epoch resume emits the resumed epoch's data order. ``SkipBatchSampler`` hides epoch-seeded samplers from
accelerate's ``set_epoch`` probes; ``set_sampler_epoch`` walks the whole chain."""

import datasets
import pytest
import torch
from accelerate.data_loader import get_sampler, prepare_data_loader, skip_first_batches
from torch.utils.data import DataLoader, RandomSampler, Sampler

from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.loop.dataloaders import DataParallelLoaderMixin, set_sampler_epoch

_ROWS = 24
_BATCH_SIZE = 2
_RESUMED_AT_BATCH = 3
_DROPPED_ROWS = _RESUMED_AT_BATCH * _BATCH_SIZE
_DATA_SEED = 1234


class _EpochSeededSampler(Sampler):
    """The shape of a toolkit sampler (``MultiGroupSampler``): no ``RandomSampler``, so ``prepare_data_loader`` does
    not re-plant it."""

    def __init__(self, size: int):
        self.size = size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(_DATA_SEED + self.epoch)
        return iter(torch.randperm(self.size, generator=generator).tolist())

    def __len__(self) -> int:
        return self.size


def _dataset() -> datasets.Dataset:
    """So an emitted batch names the order that produced it."""
    return datasets.Dataset.from_dict({"idx": list(range(_ROWS))})


def _dp_sharded_loader() -> tuple[Sampler, DataLoader]:
    sampler = _EpochSeededSampler(_ROWS)
    loader = prepare_data_loader(
        DataLoader(_dataset(), batch_size=_BATCH_SIZE, sampler=sampler),
        device=torch.device("cpu"),
        num_processes=2,
        process_index=0,
        use_seedable_sampler=True,
        data_seed=_DATA_SEED,
    )
    return sampler, loader


def _dispatch_loader() -> tuple[Sampler, DataLoader]:
    """The dispatch loader's ``set_epoch`` probes a single level, so the skip wrapper hides even the re-planted
    sampler."""
    dataset = _dataset()
    loader = prepare_data_loader(
        DataLoader(dataset, batch_size=_BATCH_SIZE, sampler=RandomSampler(dataset)),
        device=torch.device("cpu"),
        num_processes=2,
        process_index=0,
        put_on_device=True,
        dispatch_batches=True,
        use_seedable_sampler=True,
        data_seed=_DATA_SEED,
    )
    return get_sampler(loader), loader


_SHAPES = {"dp_sharded": _dp_sharded_loader, "dispatched": _dispatch_loader}


def _emitted_order(loader: DataLoader) -> list[int]:
    return [int(index) for batch in loader for index in batch["idx"]]


def _order_at_epoch(build, epoch: int) -> list[int]:
    """``__iter__`` re-stamps from the loader's own counter, which a sampler-only stamp would leave behind."""
    _, loader = build()
    loader.set_epoch(epoch)
    return _emitted_order(loader)


@pytest.mark.parametrize("build", _SHAPES.values(), ids=_SHAPES)
def test_resumed_epoch_emits_the_resumed_epochs_order(build):
    epoch0 = _order_at_epoch(build, 0)
    epoch1 = _order_at_epoch(build, 1)
    assert epoch1 != epoch0, "epoch 1 draws epoch 0's order — the fixture cannot tell the two apart"

    # Premise: accelerate's own set_epoch does not reach through the skip wrapper.
    sampler, loader = build()
    native = skip_first_batches(loader, _RESUMED_AT_BATCH)
    native.set_epoch(1)
    assert sampler.epoch == 0
    assert _emitted_order(native) == epoch0[_DROPPED_ROWS:]

    sampler, loader = build()
    resumed = skip_first_batches(loader, _RESUMED_AT_BATCH)
    assert set_sampler_epoch(resumed, 1) == 1
    assert sampler.epoch == 1
    order = _emitted_order(resumed)
    assert order != epoch0[_DROPPED_ROWS:], "the resumed epoch still replays epoch 0's order"
    assert order == epoch1[_DROPPED_ROWS:], "the resumed epoch does not continue the epoch it resumed into"


class _ResumingBase:
    """The base skips batches and only then sets the epoch, so the stamp has to survive on the same sampler object."""

    def __init__(self, sampler: Sampler):
        self.sampler = sampler
        self.epoch_at_entry = None
        self.emitted = None
        self.forwarded = None

    def _run_epoch(self, model, epoch, train_dataloader, steps_in_epoch, **kwargs):
        self.epoch_at_entry = self.sampler.epoch
        resumed = skip_first_batches(train_dataloader, _RESUMED_AT_BATCH)
        resumed.set_epoch(epoch)
        self.emitted = _emitted_order(resumed)
        self.forwarded = (model, epoch, train_dataloader, steps_in_epoch, kwargs)
        return "base"


class _StubTrainer(DataParallelLoaderMixin, _ResumingBase):
    """The mixin ahead of a Trainer base."""


def test_run_epoch_stamps_the_epoch_the_base_then_resumes_into():
    epoch1 = _order_at_epoch(_dp_sharded_loader, 1)
    sampler, loader = _dp_sharded_loader()
    trainer = _StubTrainer(sampler)

    result = trainer._run_epoch(model="model", epoch=1, train_dataloader=loader, steps_in_epoch=6, trial=None)

    assert trainer.epoch_at_entry == 1, "the base ran its epoch with the sampler still at epoch 0"
    assert trainer.emitted == epoch1[_DROPPED_ROWS:], "the resumed epoch trained on another epoch's order"
    assert result == "base", "the base's return value did not come back to the caller"
    assert trainer.forwarded == ("model", 1, loader, 6, {"trial": None}), (
        "the base's call shape changed crossing the override"
    )


def test_every_trainer_reaches_the_stamp_through_the_composed_mixin():
    """Including the plain FSDP2/DP path, whose loader the custom dataloader seam never sees."""
    owner = next(base for base in DistributedTrainerMixin.__mro__ if "_run_epoch" in base.__dict__)
    assert owner is DataParallelLoaderMixin, (
        f"_run_epoch resolves to {owner.__name__} for the composed trainer mixin, not the dataloader mixin"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
