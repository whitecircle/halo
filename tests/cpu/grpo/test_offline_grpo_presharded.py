#!/usr/bin/env python
"""``ShardedDatasetLoader`` gives each DP rank a disjoint slice, so a further ``1/dp_size`` cut in the sampler
silently drops ``(dp-1)/dp`` of the data. Offline GRPO's loader must take the shared ``dp_shard_geometry`` decision."""

from functools import partial
from types import SimpleNamespace

import pytest

from halo.trainers.dataloaders import DataParallelLoaderMixin
from halo.trainers.grpo.offline.trainer import OfflineGRPOTrainer

GROUP_IDS = [0, 0, 1, 1, 2, 2, 3, 3]


def _sampler_span(presharded: bool, dp_size: int, dp_rank: int) -> int:
    """Rows the trainer's real loader iterates, built by ``_build_grouped_dataloader`` off ``dp_shard_geometry``."""
    stub = SimpleNamespace(
        _dataset_presharded=presharded,
        parallelism_config=SimpleNamespace(
            is_cp_mode=False, data_parallel_size=dp_size, get_data_parallel_rank=lambda: dp_rank
        ),
        data_collator=lambda rows: rows,
        args=SimpleNamespace(
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            dataloader_persistent_workers=False,
            dataloader_prefetch_factor=None,
            dataloader_drop_last=False,
            data_seed=None,
            seed=0,
        ),
    )
    stub.dp_shard_geometry = partial(DataParallelLoaderMixin.dp_shard_geometry, stub)
    stub._loader_collate_fn = partial(DataParallelLoaderMixin._loader_collate_fn, stub)

    loader = OfflineGRPOTrainer._build_grouped_dataloader(
        stub, [{"row": index} for index in GROUP_IDS], GROUP_IDS, batch_size=1, shuffle=False
    )
    return len(loader.batch_sampler.sampler)


def test_presharded_dataset_is_consumed_whole():
    """A second 1/dp slice taken out of this rank's own shard would cut 8 rows to 2."""
    assert _sampler_span(presharded=True, dp_size=4, dp_rank=1) == len(GROUP_IDS)


def test_non_presharded_dataset_is_still_sharded():
    spans = [_sampler_span(presharded=False, dp_size=4, dp_rank=r) for r in range(4)]
    assert sum(spans) == len(GROUP_IDS), spans
    assert all(span == 2 for span in spans), spans


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
