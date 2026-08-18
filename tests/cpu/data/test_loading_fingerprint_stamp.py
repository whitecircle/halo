#!/usr/bin/env python3
"""load_datasets' forced-fingerprint stamp: DP identity for sharded loads + content sensitivity.

Two contracts pinned here:
1. Presharded DP cache collision — equal-length per-rank shards stamping IDENTICAL forced
   fingerprints collide every downstream coordinated_map/filter cache, and non-writer ranks then
   load rank 0's mapped shard (all DP ranks silently training on one slice).
2. Content-free stamp — a key built only from path/sizes/config keeps serving the stale mapped
   cache when a dataset is re-pushed with the same shapes.

Usage:
    python tests/cpu/data/test_loading_fingerprint_stamp.py
"""

import os
import shutil
import sys
import tempfile

# Cache dir must be pinned before datasets/processing modules read it.
_CACHE_DIR = tempfile.mkdtemp(prefix="halo_test_hf_cache_")
os.environ["HF_DATASETS_CACHE"] = _CACHE_DIR

import pytest
from accelerate import PartialState
from datasets import Dataset, DatasetDict

PartialState()  # loading.py logs via the accelerate logger

from src.data.pipeline.preprocessing import shard_dataset
from src.data.pipeline.processing import _build_cache_file_name
from src.data.shard_index import SHARD_INDEX_FILE
from src.data.sources.loading import load_datasets


def _make_sharded_dataset(tmp_dir: str, num_rows: int = 100, num_shards: int = 2) -> str:
    """Two equal-length DISJOINT shards for train and test (the collision-triggering shape)."""
    train = Dataset.from_dict({"x": list(range(num_rows))})
    test = Dataset.from_dict({"x": list(range(1000, 1000 + num_shards * 2))})
    for split, ds in (("train", train), ("test", test)):
        index = shard_dataset(ds, output_dir=tmp_dir, split_name=split, num_shards=num_shards)
        index.save(os.path.join(tmp_dir, split, SHARD_INDEX_FILE))
    return tmp_dir


def _load(path: str, rank: int, size: int) -> DatasetDict:
    return load_datasets(
        path=path,
        test_size=None,
        dataset_ratio=1,
        conversation_field=None,
        data_parallel_rank=rank,
        data_parallel_size=size,
    )


def _map_fn(example):
    return {"y": example["x"] * 2}


def test_presharded_ranks_get_distinct_cache_keys():
    """Disjoint equal-length shards on two simulated DP ranks must produce DIFFERENT forced
    fingerprints/cache keys — identical keys made every rank load rank 0's mapped shard."""
    tmp = tempfile.mkdtemp()
    try:
        _make_sharded_dataset(tmp)
        ds0 = _load(tmp, rank=0, size=2)
        ds1 = _load(tmp, rank=1, size=2)

        # Preconditions: equal lengths, disjoint content (the collision-triggering shape).
        assert len(ds0["train"]) == len(ds1["train"]) == 50
        assert set(ds0["train"]["x"]).isdisjoint(ds1["train"]["x"])

        assert ds0["train"]._fingerprint != ds1["train"]._fingerprint, (
            "per-rank shard slices must not share a forced fingerprint"
        )
        assert ds0["train"]._toolkit_cache_key != ds1["train"]._toolkit_cache_key

        # End-to-end: the downstream coordinated_map cache file names must differ per rank.
        name0 = _build_cache_file_name("map", _map_fn, ds0["train"], "tokenizing", {})
        name1 = _build_cache_file_name("map", _map_fn, ds1["train"], "tokenizing", {})
        assert name0 != name1, "identical cache names collide: rank 1 would load rank 0's mapped shard"

        # Simulate the coordinated flow: each rank maps to its own deterministic cache file and
        # must get back ITS OWN slice.
        out0 = ds0["train"].map(_map_fn, cache_file_name=os.path.join(_CACHE_DIR, name0), load_from_cache_file=True)
        out1 = ds1["train"].map(_map_fn, cache_file_name=os.path.join(_CACHE_DIR, name1), load_from_cache_file=True)
        assert out0["y"] == [x * 2 for x in ds0["train"]["x"]]
        assert out1["y"] == [x * 2 for x in ds1["train"]["x"]], "rank 1 received rank 0's mapped data"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_replicated_load_keeps_shared_cache_key():
    """Non-sharded (replicated) loads must keep the SAME key across DP ranks — the single-writer
    coordinated cache depends on all ranks agreeing on one file."""
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "replicated")
        DatasetDict(
            {
                "train": Dataset.from_dict({"x": list(range(40))}),
                "test": Dataset.from_dict({"x": list(range(500, 510))}),
            }
        ).save_to_disk(path)

        ds0 = _load(path, rank=0, size=2)
        ds1 = _load(path, rank=1, size=2)
        assert ds0["train"]._fingerprint == ds1["train"]._fingerprint
        assert ds0["train"]._toolkit_cache_key == ds1["train"]._toolkit_cache_key
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_same_shape_different_content_changes_cache_key():
    """Re-pushing a dataset with identical shapes but different content must change the forced
    fingerprint — a content-free stamp served the stale mapped cache after every same-shape re-push."""
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "ds")

        def _save(offset: int):
            if os.path.exists(path):
                shutil.rmtree(path)
            DatasetDict(
                {
                    "train": Dataset.from_dict({"x": list(range(offset, offset + 40))}),
                    "test": Dataset.from_dict({"x": list(range(offset + 500, offset + 510))}),
                }
            ).save_to_disk(path)

        _save(0)
        key_a = _load(path, rank=0, size=1)["train"]._toolkit_cache_key
        # Stability: reloading the SAME content keeps the key (the shared cache still hits).
        assert _load(path, rank=0, size=1)["train"]._toolkit_cache_key == key_a

        _save(1)  # same shapes, different content
        key_b = _load(path, rank=0, size=1)["train"]._toolkit_cache_key
        assert key_a != key_b, "same-shape content change must invalidate the forced cache key"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
