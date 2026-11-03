#!/usr/bin/env python
"""``ShardedDatasetLoader`` shard assignment is an exact partition: a gap silently drops data and an overlap
silently double-trains it."""

import os
import shutil
import tempfile
from collections import Counter

import pytest
from datasets import Dataset

from halo.data.preprocessed.preparation import shard_dataset
from halo.data.preprocessed.shard_index import SHARD_INDEX_FILE, write_stamped_sidecar
from halo.data.sources.sharded_dataset import ShardedDatasetLoader


def _assignment(num_shards: int, world_size: int, rank: int) -> tuple[int, int]:
    """(start, end-exclusive) shard range for a rank, via the pure assignment math."""
    loader = ShardedDatasetLoader.__new__(ShardedDatasetLoader)
    loader.dp_rank = rank
    loader.dp_size = world_size
    return loader._compute_shard_assignment(num_shards)


@pytest.mark.parametrize("num_shards,world_size", [(10, 3), (8, 8), (7, 4), (1, 1)])
def test_exact_partition_no_overlap_no_gap(num_shards, world_size):
    coverage = Counter()
    for rank in range(world_size):
        start, end = _assignment(num_shards, world_size, rank)
        assert 0 <= start <= end <= num_shards, f"rank {rank} range out of bounds: ({start}, {end})"
        for shard_id in range(start, end):
            coverage[shard_id] += 1

    assert set(coverage.keys()) == set(range(num_shards)), (
        f"coverage gap: assigned {sorted(coverage.keys())} for {num_shards} shards"
    )
    assert all(c == 1 for c in coverage.values()), f"overlap: some shard assigned to >1 rank: {dict(coverage)}"


def test_num_shards_equals_world_size_one_each():
    world_size = 6
    seen = []
    for rank in range(world_size):
        start, end = _assignment(world_size, world_size, rank)
        assert end - start == 1, f"rank {rank} got {end - start} shards, expected exactly 1"
        seen.append(start)
    assert sorted(seen) == list(range(world_size)), f"shards not one-per-rank: {sorted(seen)}"


def _build_sharded(num_examples: int, num_shards: int, temp_dir: str, split: str = "train") -> None:
    dataset = Dataset.from_dict(
        {
            "input_ids": [[i, i + 1, i + 2] for i in range(num_examples)],
            "example_id": list(range(num_examples)),
        }
    )
    index = shard_dataset(dataset, output_dir=temp_dir, split_name=split, num_shards=num_shards)
    write_stamped_sidecar(os.path.join(temp_dir, split, SHARD_INDEX_FILE), index)


def test_writer_loader_roundtrip_exact_multiset():
    """100 examples in 7 shards over 4 ranks: the multiset of example_ids is exactly range(100)."""
    temp_dir = tempfile.mkdtemp()
    try:
        _build_sharded(100, num_shards=7, temp_dir=temp_dir)

        world_size = 4
        all_ids = Counter()
        for rank in range(world_size):
            loader = ShardedDatasetLoader(dataset_path=temp_dir, dp_rank=rank, dp_size=world_size)
            ds = loader.load_split("train")
            for ex in ds:
                all_ids[ex["example_id"]] += 1

        assert set(all_ids.keys()) == set(range(100)), "round-trip lost or invented example ids"
        duplicated = {k: v for k, v in all_ids.items() if v != 1}
        assert not duplicated, f"round-trip double-loaded example ids: {duplicated}"
    finally:
        shutil.rmtree(temp_dir)


def test_num_shards_equals_world_size_load_does_not_raise():
    """The well-shaped boundary: the undersharded guard only fires for 0 < num_shards < world_size."""
    temp_dir = tempfile.mkdtemp()
    try:
        _build_sharded(40, num_shards=4, temp_dir=temp_dir)
        for rank in range(4):
            loader = ShardedDatasetLoader(dataset_path=temp_dir, dp_rank=rank, dp_size=4)
            ds = loader.load_split("train")
            assert len(ds) > 0, f"rank {rank} got an empty assignment at num_shards==world_size"
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
