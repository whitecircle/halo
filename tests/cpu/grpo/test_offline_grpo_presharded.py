#!/usr/bin/env python
"""Offline GRPO must not re-shard an already-presharded dataset.

``ShardedDatasetLoader`` hands each DP rank a DISJOINT slice; the trainer's ``MultiGroupSampler``
then cuts a further ``1/dp_size`` out of whatever it is given. Applying both silently drops
``(dp-1)/dp`` of the training data — no warning, no error, just a run that sees a fraction of its
corpus. Every trainer takes this decision through one shared accessor
(``DataParallelDataLoaderMixin.dp_shard_geometry`` → ``(1, 0)`` when pre-sharded); this pins that
offline GRPO's own loader takes it too.

Run: pytest tests/cpu/grpo/test_offline_grpo_presharded.py
"""

from functools import partial
from types import SimpleNamespace

import pytest
from accelerate import PartialState

PartialState()  # MultiGroupSampler logs through accelerate's logger, which requires the state

from src.trainers.grpo.offline import OfflineGRPOTrainer
from src.trainers.mixins.dataloader import DataParallelDataLoaderMixin

# 4 groups x 2 completions, as the per-rank slice of an already-sharded dataset.
GROUP_IDS = [0, 0, 1, 1, 2, 2, 3, 3]


def _sampler_span(presharded: bool, dp_size: int, dp_rank: int) -> int:
    """Rows the loader the trainer really builds would iterate.

    Built by ``_build_grouped_dataloader`` itself, off the shared ``dp_shard_geometry``: a private
    re-derivation of dp size/rank inside the builder, or an accessor that stopped reading the
    presharded flag, changes this span and nothing else would notice.
    """
    stub = SimpleNamespace(
        _dataset_presharded=presharded,
        get_data_parallel_size=lambda: dp_size,
        get_data_parallel_rank=lambda: dp_rank,
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
    stub.dp_shard_geometry = partial(DataParallelDataLoaderMixin.dp_shard_geometry, stub)

    loader = OfflineGRPOTrainer._build_grouped_dataloader(
        stub, [{"row": index} for index in GROUP_IDS], GROUP_IDS, batch_size=1, shuffle=False
    )
    return len(loader.batch_sampler.sampler)


def test_presharded_dataset_is_consumed_whole():
    """A second 1/dp slice taken out of this rank's own shard would cut 8 rows to 2."""
    assert _sampler_span(presharded=True, dp_size=4, dp_rank=1) == len(GROUP_IDS)


def test_non_presharded_dataset_is_still_sharded():
    """The presharded branch must not disable sharding for the ordinary (whole-dataset-per-rank) case."""
    spans = [_sampler_span(presharded=False, dp_size=4, dp_rank=r) for r in range(4)]
    assert sum(spans) == len(GROUP_IDS), spans
    assert all(span == 2 for span in spans), spans


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
