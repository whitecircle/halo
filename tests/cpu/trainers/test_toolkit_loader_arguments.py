#!/usr/bin/env python
"""The toolkit's DP-sharded train, eval and test loaders take HF's DataLoader arguments.

HF's ``_get_dataloader`` passes ``multiprocessing_context``, ``prefetch_factor`` and ``in_order`` on
every split. The toolkit's builders share one parameter set (``_loader_params``); without the first
and the last in it, a run setting ``dataloader_multiprocessing_context: spawn`` or
``dataloader_in_order: false`` gets neither once TP/CP/ETP/PP or a pre-sharded dataset puts it on
the toolkit's loader. The pin: each of the three builders hands the configured values to its loader.

    python tests/cpu/trainers/test_toolkit_loader_arguments.py
"""

from types import SimpleNamespace

import pytest
from torch.utils.data import SequentialSampler

from src.trainers.mixins.dataloader import DataParallelDataLoaderMixin

_ROWS = list(range(8))


def _host() -> DataParallelDataLoaderMixin:
    """A TP-style rank on the toolkit's loader path; the prepare step is the identity, so the built
    ``DataLoader`` is what the assertions read."""
    host = object.__new__(DataParallelDataLoaderMixin)
    host.parallelism_config = SimpleNamespace(non_dp_replication_factor=2, is_pp_mode=False)
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
        eval_batch_size=2,
        per_device_eval_batch_size=2,
    )
    host.get_data_parallel_rank = lambda: 0
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
