#!/usr/bin/env python
"""A mid-epoch resume must emit the RESUMED epoch's data order, not epoch 0's.

``skip_first_batches`` wraps the sampler chain in a ``SkipBatchSampler``, one level deeper than the
fixed shapes accelerate's ``set_epoch`` probes. The stock DP chain is rescued by the re-plant
``prepare_data_loader`` does on a ``RandomSampler``; the two shapes below are not — a sampler the
toolkit brings itself, and a dispatching loader whose probe is a single level. Their epoch-seeded
sampler stays at epoch 0, so the resumed epoch re-draws epoch 0's permutation (and every later
epoch shifts by one).
``set_sampler_epoch`` walks the whole chain instead, and ``DataParallelDataLoaderMixin._run_epoch``
runs it before the base's epoch body — the seam every trainer, custom loader or not, goes through.

Run: python tests/cpu/parallelism/test_resume_data_order.py
"""

import datasets
import pytest
import torch
from accelerate.data_loader import get_sampler, prepare_data_loader, skip_first_batches
from torch.utils.data import DataLoader, RandomSampler, Sampler

from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.mixins.dataloader import DataParallelDataLoaderMixin, set_sampler_epoch

_ROWS = 24
_BATCH_SIZE = 2
_RESUMED_AT_BATCH = 3
_DROPPED_ROWS = _RESUMED_AT_BATCH * _BATCH_SIZE
_DATA_SEED = 1234


class _EpochSeededSampler(Sampler):
    """Minimal ``set_epoch`` contract: the permutation is seeded by the epoch.

    The shape of any sampler the toolkit brings itself (offline GRPO's ``MultiGroupSampler``): not a
    ``RandomSampler``, so ``prepare_data_loader`` does not re-plant it onto the shard the way it does
    the seedable one, and the DP shard leaves it a level lower than that rescue reaches.
    """

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
    """Rows that carry their own index, so an emitted batch names the order that produced it."""
    return datasets.Dataset.from_dict({"idx": list(range(_ROWS))})


def _dp_sharded_loader() -> tuple[Sampler, DataLoader]:
    """DP > 1: ``BatchSamplerShard`` over an epoch-seeded sampler."""
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
    """Dispatched batches (``dispatch_batches``) over the stock ``RandomSampler``.

    The re-plant does happen here — accelerate installs its ``SeedableRandomSampler`` — but the
    dispatch loader's ``set_epoch`` probes a single level, so the skip wrapper hides even that one.
    """
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
    """The order an uninterrupted pass over ``epoch`` emits.

    Stamped through the loader, as the base ``Trainer`` does it: ``__iter__`` re-stamps the chain
    from the loader's own iteration counter, which a sampler-only stamp would leave behind. Without
    the skip wrapper accelerate's probes still reach, so this is the unbroken reference order.
    """
    _, loader = build()
    loader.set_epoch(epoch)
    return _emitted_order(loader)


@pytest.mark.parametrize("build", _SHAPES.values(), ids=_SHAPES)
def test_resumed_epoch_emits_the_resumed_epochs_order(build):
    epoch0 = _order_at_epoch(build, 0)
    epoch1 = _order_at_epoch(build, 1)
    assert epoch1 != epoch0, "epoch 1 draws epoch 0's order — the fixture cannot tell the two apart"

    # Premise: accelerate's own set_epoch does not reach through the skip wrapper, so a resumed
    # epoch replays epoch 0. This is what the toolkit helper below has to correct.
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
    """Stands in for ``transformers.Trainer._run_epoch``'s resume branch, and records what it saw.

    The base skips the trained batches and only then sets the epoch on the rebuilt loader, so this
    is also where the stamp has to survive: the skip wrapper keeps the same sampler object.
    """

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


class _StubTrainer(DataParallelDataLoaderMixin, _ResumingBase):
    """The shape every concrete trainer has: the mixin ahead of a Trainer base."""


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
    assert owner is DataParallelDataLoaderMixin, (
        f"_run_epoch resolves to {owner.__name__} for the composed trainer mixin, not the dataloader mixin"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
