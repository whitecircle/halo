"""A run resumed into a later epoch draws that epoch's ``group_by_length`` order. HF's length-grouped sampler takes
epoch ``e``'s order as its generator's ``e``-th draw and has no ``set_epoch``, so a resumed process, whose generator
is fresh, would train the first epoch's order again."""

from types import SimpleNamespace

import pytest
import torch
from accelerate.data_loader import prepare_data_loader, skip_first_batches
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import Trainer

from halo.trainers.loop.dataloaders import DataParallelLoaderMixin, set_sampler_epoch

ROWS = 24
BATCH_SIZE = 2
RESUMED_AT_BATCH = 3


class _Host(DataParallelLoaderMixin, Trainer):
    """The mixin over HF's own train sampler, with what the sampler build reads off a trainer."""

    def __init__(self):
        self.processing_class = None
        self.train_dataset = Dataset.from_dict(
            {"idx": list(range(ROWS)), "length": [1 + (7 * i) % 13 for i in range(ROWS)]}
        )
        self.args = SimpleNamespace(
            seed=42,
            data_seed=None,
            train_sampling_strategy="group_by_length",
            length_column_name="length",
            train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=1,
        )


def _loader() -> DataLoader:
    """Rank 0's loader of two data-parallel ranks."""
    host = _Host()
    return prepare_data_loader(
        DataLoader(host.train_dataset, batch_size=BATCH_SIZE, sampler=host._get_train_sampler()),
        device=torch.device("cpu"),
        num_processes=2,
        process_index=0,
    )


def _epoch_order(loader, epoch: int) -> list[int]:
    """The rows an epoch reads, stamped as the trainer stamps it: the mixin's walk, then the base's ``set_epoch``."""
    set_sampler_epoch(loader, epoch)
    loader.set_epoch(epoch)
    return [int(index) for batch in loader for index in batch["idx"]]


def test_a_run_resumed_into_a_later_epoch_draws_that_epochs_order():
    uninterrupted = _loader()
    epochs = [_epoch_order(uninterrupted, epoch) for epoch in range(2)]
    assert epochs[1] != epochs[0], "both epochs drew one order: the fixture cannot tell them apart"

    resumed = skip_first_batches(_loader(), RESUMED_AT_BATCH)
    assert _epoch_order(resumed, 1) == epochs[1][RESUMED_AT_BATCH * BATCH_SIZE :], (
        "the resumed epoch trains another epoch's order"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
