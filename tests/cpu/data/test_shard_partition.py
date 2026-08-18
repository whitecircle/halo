#!/usr/bin/env python
"""Exact-partition tests for ShardedDatasetLoader shard assignment.

Megatron-style sharding must give every rank a DISJOINT slice and together COVER all
shards exactly once — a gap silently drops data, an overlap silently double-trains it,
and neither surfaces as an error. These tests assert the partition exactly (Counter over
all ranks == every shard once) across several (num_shards, world_size) shapes, plus a
real writer->loader round-trip on example IDs and the boundary cases.

Run: pytest tests/cpu/data/test_shard_partition.py
"""

import os
import shutil
import tempfile
from collections import Counter

import pytest
from datasets import Dataset

from src.data.pipeline.preprocessing import shard_dataset
from src.data.shard_index import SHARD_INDEX_FILE
from src.data.sources.sharded_dataset import ShardedDatasetLoader


def _assignment(num_shards: int, world_size: int, rank: int) -> tuple[int, int]:
    """(start, end-exclusive) shard range for a rank, via the pure assignment math."""
    loader = ShardedDatasetLoader.__new__(ShardedDatasetLoader)
    loader.global_rank = rank
    loader.world_size = world_size
    return loader._compute_shard_assignment(num_shards)


@pytest.mark.parametrize("num_shards,world_size", [(10, 3), (8, 8), (7, 4), (1, 1)])
def test_exact_partition_no_overlap_no_gap(num_shards, world_size):
    """Union of all ranks' assignments == every shard id exactly once."""
    coverage = Counter()
    for rank in range(world_size):
        start, end = _assignment(num_shards, world_size, rank)
        assert 0 <= start <= end <= num_shards, f"rank {rank} range out of bounds: ({start}, {end})"
        for shard_id in range(start, end):
            coverage[shard_id] += 1

    # Every shard covered exactly once: no gap (all present) and no overlap (count==1).
    assert set(coverage.keys()) == set(range(num_shards)), (
        f"coverage gap: assigned {sorted(coverage.keys())} for {num_shards} shards"
    )
    assert all(c == 1 for c in coverage.values()), f"overlap: some shard assigned to >1 rank: {dict(coverage)}"


def test_num_shards_equals_world_size_one_each():
    """num_shards == world_size: each rank gets exactly one distinct shard (remainder=0 path)."""
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
    index.save(os.path.join(temp_dir, split, SHARD_INDEX_FILE))


def test_writer_loader_roundtrip_exact_multiset():
    """Shard 100 examples into 7, load across 4 ranks, concatenate: the multiset of
    example_ids must be exactly range(100) — every example once, no duplicates, no drops."""
    temp_dir = tempfile.mkdtemp()
    try:
        _build_sharded(100, num_shards=7, temp_dir=temp_dir)

        world_size = 4
        all_ids = Counter()
        for rank in range(world_size):
            loader = ShardedDatasetLoader(dataset_path=temp_dir, global_rank=rank, world_size=world_size)
            ds = loader.load_split("train")
            for ex in ds:
                all_ids[ex["example_id"]] += 1

        assert set(all_ids.keys()) == set(range(100)), "round-trip lost or invented example ids"
        duplicated = {k: v for k, v in all_ids.items() if v != 1}
        assert not duplicated, f"round-trip double-loaded example ids: {duplicated}"
    finally:
        shutil.rmtree(temp_dir)


def test_num_shards_equals_world_size_load_does_not_raise():
    """num_shards == world_size is the well-shaped boundary: load_split must NOT raise
    (the undersharded guard only fires for 0 < num_shards < world_size)."""
    temp_dir = tempfile.mkdtemp()
    try:
        _build_sharded(40, num_shards=4, temp_dir=temp_dir)
        # Every rank loads its single shard cleanly.
        for rank in range(4):
            loader = ShardedDatasetLoader(dataset_path=temp_dir, global_rank=rank, world_size=4)
            ds = loader.load_split("train")  # must not raise
            assert len(ds) > 0, f"rank {rank} got an empty assignment at num_shards==world_size"
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
