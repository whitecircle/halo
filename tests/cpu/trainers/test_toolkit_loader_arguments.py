#!/usr/bin/env python
"""The toolkit's DP-sharded train, eval and test loaders take HF's DataLoader arguments.
Without them in ``_loader_params``, ``dataloader_in_order: false`` and a multiprocessing context would be dropped."""

from types import SimpleNamespace

import pytest
from torch.utils.data import SequentialSampler

from halo.trainers.dataloaders import DataParallelLoaderMixin

_ROWS = list(range(8))


def _host() -> DataParallelLoaderMixin:
    """A TP-style rank on the toolkit's loader path; the prepare step is the identity."""
    host = object.__new__(DataParallelLoaderMixin)
    host.parallelism_config = SimpleNamespace(
        non_dp_replication_factor=2, is_cp_mode=False, is_pp_mode=False, get_data_parallel_rank=lambda: 0
    )
    host._dataset_presharded = False
    host.train_dataset = host.eval_dataset = _ROWS
    host._train_batch_size = 2
    host.data_collator = list
    host.args = SimpleNamespace(
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
        dataloader_multiprocessing_context="spawn",
        dataloader_prefetch_factor=3,
        dataloader_in_order=False,
        dataloader_drop_last=False,
        train_sampling_strategy="random",
        eval_batch_size=2,
        per_device_eval_batch_size=2,
    )
    host._get_collator_with_removed_columns = lambda collator, description: collator
    host._get_train_sampler = lambda: SequentialSampler(_ROWS)
    host._get_eval_sampler = SequentialSampler
    host._prepare_dataloader = lambda loader: loader
    return host


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda host: host.get_train_dataloader(), id="train"),
        pytest.param(lambda host: host.get_eval_dataloader(), id="eval"),
        pytest.param(lambda host: host.get_test_dataloader(_ROWS), id="test"),
    ],
)
def test_every_split_takes_hfs_worker_arguments(build):
    loader = build(_host())

    assert loader.multiprocessing_context.get_start_method() == "spawn"
    assert loader.prefetch_factor == 3
    assert loader.in_order is False


def test_a_process_sharded_batch_sampler_is_refused():
    """HF's ``batch_rebalance`` sampler shards by global process, so ranks sharing a batch draw different rows."""
    host = _host()
    host.args.train_sampling_strategy = "batch_rebalance"
    with pytest.raises(ValueError, match="batch_rebalance"):
        host.get_train_dataloader()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
