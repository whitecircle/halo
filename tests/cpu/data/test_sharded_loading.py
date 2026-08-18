#!/usr/bin/env python3
"""
Tests for sharded dataset loading.

These tests verify the distributed shard loading functionality
similar to Megatron-LM's approach.

Usage:
    python tests/data/test_sharded_loading.py
"""

import os
import shutil
import sys
import tempfile
from unittest.mock import patch

import pytest
from accelerate import PartialState
from botocore.exceptions import ClientError, NoCredentialsError
from datasets import Dataset, DatasetDict, load_from_disk

# load_preprocessed_dataset logs via the accelerate logger, which requires an initialized state.
PartialState()

from src.data.pipeline.preprocessing import shard_dataset
from src.data.shard_index import SHARD_INDEX_FILE, IncompatiblePreprocessedDataset, ShardIndex, ShardInfo
from src.data.sources.loading import load_datasets
from src.data.sources.sharded_dataset import ShardedDatasetLoader


def create_test_sharded_dataset(num_examples: int, num_shards: int, temp_dir: str) -> str:
    """Helper to create a test sharded dataset."""
    dataset = Dataset.from_dict(
        {
            "input_ids": [[i, i + 1, i + 2] for i in range(num_examples)],
            "attention_mask": [[1, 1, 1]] * num_examples,
            "example_id": list(range(num_examples)),
        }
    )

    index = shard_dataset(dataset, output_dir=temp_dir, split_name="train", num_shards=num_shards)

    index_path = os.path.join(temp_dir, "train", SHARD_INDEX_FILE)
    index.save(index_path)

    return temp_dir


def test_shard_assignment_multi_rank():
    """Test shard assignment for multiple ranks."""
    print("Testing shard assignment (multi rank)...")

    temp_dir = tempfile.mkdtemp()

    try:
        create_test_sharded_dataset(200, 8, temp_dir)

        # 8 shards / 4 ranks → contiguous blocks of 2: rank r gets shards [2r, 2r+1].
        expected_ids = {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7]}
        for rank in range(4):
            loader = ShardedDatasetLoader(
                dataset_path=temp_dir,
                global_rank=rank,
                world_size=4,
            )

            assigned = loader.get_assigned_shards("train")
            assert [s.id for s in assigned] == expected_ids[rank], (
                f"Rank {rank} got shard ids {[s.id for s in assigned]}, expected {expected_ids[rank]}"
            )

    finally:
        shutil.rmtree(temp_dir)

    print("  shard assignment (multi rank): PASSED")


def test_shard_assignment_remainder():
    """Test shard assignment with remainder shards."""
    print("Testing shard assignment (remainder)...")

    temp_dir = tempfile.mkdtemp()

    try:
        create_test_sharded_dataset(300, 10, temp_dir)

        # Remainder (10 % 3 == 1) goes to rank 0; contiguous, each shard assigned exactly once.
        expected_ids = {0: [0, 1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9]}
        all_assigned = []
        for rank in range(3):
            loader = ShardedDatasetLoader(
                dataset_path=temp_dir,
                global_rank=rank,
                world_size=3,
            )
            assigned = [s.id for s in loader.get_assigned_shards("train")]
            assert assigned == expected_ids[rank], f"Rank {rank}: {assigned} != {expected_ids[rank]}"
            all_assigned.extend(assigned)

        assert sorted(all_assigned) == list(range(10))

    finally:
        shutil.rmtree(temp_dir)

    print("  shard assignment (remainder): PASSED")


def test_load_split():
    """Test loading a dataset split."""
    print("Testing load_split...")

    temp_dir = tempfile.mkdtemp()

    try:
        create_test_sharded_dataset(100, 4, temp_dir)

        loader = ShardedDatasetLoader(
            dataset_path=temp_dir,
            global_rank=0,
            world_size=1,
        )

        dataset = loader.load_split("train")

        assert len(dataset) == 100, f"Expected 100 examples, got {len(dataset)}"
        assert "input_ids" in dataset.column_names
        assert "example_id" in dataset.column_names

    finally:
        shutil.rmtree(temp_dir)

    print("  load_split: PASSED")


def test_load_split_distributed():
    """Test loading split for distributed training simulation."""
    print("Testing load_split (distributed)...")

    temp_dir = tempfile.mkdtemp()

    try:
        create_test_sharded_dataset(100, 4, temp_dir)

        from collections import Counter

        all_example_ids = Counter()

        for rank in range(2):
            loader = ShardedDatasetLoader(
                dataset_path=temp_dir,
                global_rank=rank,
                world_size=2,
            )

            dataset = loader.load_split("train")

            assert len(dataset) > 0, f"Rank {rank} got 0 examples"

            for example in dataset:
                all_example_ids[example["example_id"]] += 1

        assert set(all_example_ids.keys()) == set(range(100)), (
            f"Expected coverage of all 100 example ids, got {len(all_example_ids)} distinct"
        )
        # A shard handed to two ranks would double-train those examples silently.
        duplicated = {k: v for k, v in all_example_ids.items() if v != 1}
        assert not duplicated, f"example ids loaded by more than one rank (overlap): {duplicated}"

    finally:
        shutil.rmtree(temp_dir)

    print("  load_split (distributed): PASSED")


def test_get_total_examples():
    """Test getting total example count."""
    print("Testing get_total_examples...")

    temp_dir = tempfile.mkdtemp()

    try:
        create_test_sharded_dataset(150, 5, temp_dir)

        loader = ShardedDatasetLoader(
            dataset_path=temp_dir,
            global_rank=0,
            world_size=1,
        )

        total = loader.get_total_examples("train")
        assert total == 150, f"Expected 150 total examples, got {total}"

    finally:
        shutil.rmtree(temp_dir)

    print("  get_total_examples: PASSED")


def test_is_sharded_dataset():
    """Test ShardedDatasetLoader.is_sharded_dataset static method."""
    print("Testing is_sharded_dataset...")

    temp_dir = tempfile.mkdtemp()

    try:
        assert ShardedDatasetLoader.is_sharded_dataset(temp_dir) is False

        create_test_sharded_dataset(50, 2, temp_dir)

        assert ShardedDatasetLoader.is_sharded_dataset(temp_dir) is True

    finally:
        shutil.rmtree(temp_dir)

    print("  is_sharded_dataset: PASSED")


def test_is_sharded_dataset_s3_probe_failure_returns_false(tmp_path):
    """Any S3 probe failure (403 AccessDenied, throttling — not just missing credentials) must
    return False when no local mirror exists, NOT raise: the caller reconciles the probe across
    ranks with an all_reduce, and a raise on one rank skips that collective and hangs the peers
    until the NCCL watchdog. (With a mirror the probe answers True offline —
    test_s3_control_json_cache.py.)

    Catching only credential errors lets any other ClientError propagate.
    """
    print("Testing is_sharded_dataset S3 probe failure...")

    for error in (
        ClientError({"Error": {"Code": "AccessDenied", "Message": "403"}}, "HeadObject"),
        ClientError({"Error": {"Code": "SlowDown", "Message": "throttled"}}, "HeadObject"),
        OSError("connection reset"),
    ):
        with patch("src.data.sources.s3_client.HALO_S3_DATASET_CACHE_DIR", str(tmp_path)):
            with patch("src.data.sources.s3_client.read_json_from_s3", side_effect=error):
                assert ShardedDatasetLoader.is_sharded_dataset("s3://bucket/dataset") is False, (
                    f"probe must err toward non-sharded on {type(error).__name__}, not raise"
                )

    print("  is_sharded_dataset_s3_probe_failure_returns_false: PASSED")


def test_load_single_shard():
    """Test loading a dataset with only 1 shard (num_shards=1)."""
    print("Testing load single shard...")

    temp_dir = tempfile.mkdtemp()

    try:
        create_test_sharded_dataset(50, 1, temp_dir)

        loader = ShardedDatasetLoader(
            dataset_path=temp_dir,
            global_rank=0,
            world_size=1,
        )

        assigned = loader.get_assigned_shards("train")
        assert len(assigned) == 1, f"Expected 1 shard, got {len(assigned)}"

        dataset = loader.load_split("train")
        assert len(dataset) == 50, f"Expected 50 examples, got {len(dataset)}"
        assert "input_ids" in dataset.column_names

    finally:
        shutil.rmtree(temp_dir)

    print("  load single shard: PASSED")


def test_empty_shard_assignment_for_high_rank():
    """A rank beyond num_shards gets an empty shard assignment."""
    temp_dir = tempfile.mkdtemp()
    try:
        create_test_sharded_dataset(20, 2, temp_dir)
        loader = ShardedDatasetLoader(dataset_path=temp_dir, global_rank=3, world_size=4)
        assert loader.get_assigned_shards("train") == []
    finally:
        shutil.rmtree(temp_dir)


def test_undersharded_train_split_fails_fast():
    """num_shards < world_size on the TRAIN split must fail loudly, not silently
    truncate every rank to zero examples.

    The length-equalizer downstream all_reduce(MIN)s rank lengths, so one
    zero-length rank zeroes the whole batch; load_split raises with an
    actionable re-preprocess hint instead.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        create_test_sharded_dataset(20, 2, temp_dir)
        loader = ShardedDatasetLoader(dataset_path=temp_dir, global_rank=0, world_size=4)
        with pytest.raises(ValueError, match="(?i)num.?shards|data_parallel"):
            loader.load_split("train")
    finally:
        shutil.rmtree(temp_dir)


def _one_shard_index_payload() -> dict:
    """A well-formed single-shard train index, as ``shard_index.json`` carries it."""
    index = ShardIndex(
        split="train",
        num_shards=1,
        total_examples=4,
        shards=[ShardInfo(id=0, path="train/shard_00000", num_examples=4)],
    )
    return index.to_dict()


def test_shard_index_version_mismatch_raises():
    """The index stamp is COMPARED, not merely carried: read with this build's field meanings, a
    diverged build's index hands ranks the wrong shards. The ``metadata.json`` twin refuses the same
    way; without the comparison the shard half accepts anything JSON-shaped."""
    payload = _one_shard_index_payload() | {"version": "0.9"}
    with pytest.raises(IncompatiblePreprocessedDataset, match="0.9"):
        ShardIndex.from_dict(payload)


def test_shard_index_unknown_field_raises():
    """A field this build does not know, at a version stamp it does: a diverged writer, whose extra
    field would be dropped in silence."""
    payload = _one_shard_index_payload() | {"invented_by_a_future_build": True}
    with pytest.raises(IncompatiblePreprocessedDataset, match="invented_by_a_future_build"):
        ShardIndex.from_dict(payload)


def test_zero_shard_train_index_raises(tmp_path):
    """An index claiming ZERO shards for train must raise, not hand every rank an empty Dataset.

    The undersharded guard reads ``0 < num_shards < world_size``, so zero slipped past it and every
    rank trained on an empty split behind a per-rank warning.
    """
    (tmp_path / "train").mkdir()
    ShardIndex(split="train", num_shards=0, total_examples=0, shards=[]).save(
        str(tmp_path / "train" / SHARD_INDEX_FILE)
    )
    loader = ShardedDatasetLoader(dataset_path=str(tmp_path), global_rank=0, world_size=1)
    with pytest.raises(ValueError, match="(?i)no shards"):
        loader.load_split("train")


def test_load_datasets_list_entry_missing_test_split_fails_loud():
    """A dataset LIST entry gets the same split guard as a single path.

    The list branch discards the sharded flag and the loader skips a missing split, so without the
    guard a train-only sharded entry surfaces as a bare ``KeyError('test')`` inside the per-entry
    filtering. The raise must name the offending entry and the remedy.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        create_test_sharded_dataset(20, 2, temp_dir)  # writes only the train split
        with pytest.raises(ValueError, match="missing the .*test.*split") as excinfo:
            load_datasets([temp_dir], test_size=None, dataset_ratio=1, conversation_field=None)
        assert temp_dir in str(excinfo.value), str(excinfo.value)
    finally:
        shutil.rmtree(temp_dir)


def test_load_preprocessed_dataset_empty_test_split_raises():
    """A GLOBALLY empty test split (zero examples in the shard index) must fail loud at load.

    Regression guard: without the guard the empty split flowed through and either KeyError'd
    downstream or hung distributed eval. Mirrors load_datasets' empty-test guard. (Per-rank-only
    emptiness — fewer non-empty shards than DP ranks — is the trainer-side equalize-raise case.)
    """
    from src.data.sources.loading import load_preprocessed_dataset

    temp_dir = tempfile.mkdtemp()
    try:
        create_test_sharded_dataset(20, 2, temp_dir)  # non-empty train
        empty = Dataset.from_dict({"input_ids": [], "attention_mask": [], "example_id": []})
        index = shard_dataset(empty, output_dir=temp_dir, split_name="test", num_shards=2)
        assert index.total_examples == 0
        index.save(os.path.join(temp_dir, "test", SHARD_INDEX_FILE))

        with pytest.raises(ValueError, match="(?i)no test data|empty"):
            load_preprocessed_dataset(temp_dir, data_parallel_rank=0, data_parallel_size=1)
    finally:
        shutil.rmtree(temp_dir)


def test_load_preprocessed_dataset_empty_train_split_raises(monkeypatch):
    """The train split gets the same emptiness verdict as the test split.

    A pre-processed artifact whose filters dropped every training row is not a degraded run, it is
    no run at all — and unguarded it reaches the trainer as a zero-step schedule (sharded loads die
    later in the length equalizer, with a message that names neither the split nor the cause).
    The empty split is handed over in memory: the pinned ``datasets`` writes no shard for a zero-row
    split, so ``save_to_disk`` cannot even produce the artifact this guard is for.
    """
    import src.data.sources.loading as loading

    columns = ["input_ids", "attention_mask", "example_id"]
    ds = DatasetDict(
        {
            "train": Dataset.from_dict(dict.fromkeys(columns, [])),
            "test": Dataset.from_dict({"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]], "example_id": [0]}),
        }
    )
    monkeypatch.setattr(loading, "_load_dataset_from_path", lambda path, test_size: (ds, None))
    monkeypatch.setattr(loading, "is_sharded_dataset_coordinated", lambda path: False)

    with pytest.raises(ValueError, match="(?i)train"):
        loading.load_preprocessed_dataset("unused", data_parallel_rank=0, data_parallel_size=1)


def test_load_preprocessed_dataset_missing_test_split_fails_loud():
    """A preprocessed dataset with NO test split at all must raise the clear re-prepare error at the
    loader seam (mirroring load_datasets' guard), not surface later as a bare KeyError('test') at the
    consumer's unconditional ds["test"]."""
    from src.data.sources.loading import load_preprocessed_dataset

    temp_dir = tempfile.mkdtemp()
    try:
        create_test_sharded_dataset(20, 2, temp_dir)  # writes only the train split
        with pytest.raises(ValueError, match="missing the .*test.*split"):
            load_preprocessed_dataset(temp_dir, data_parallel_rank=0, data_parallel_size=1)
    finally:
        shutil.rmtree(temp_dir)


def test_load_preprocessed_dataset_nonempty_test_split_loads():
    """Positive control for the empty-test guard: a non-empty test split loads normally."""
    from src.data.sources.loading import load_preprocessed_dataset

    temp_dir = tempfile.mkdtemp()
    try:
        create_test_sharded_dataset(20, 2, temp_dir)
        test_ds = Dataset.from_dict(
            {
                "input_ids": [[i, i + 1, i + 2] for i in range(6)],
                "attention_mask": [[1, 1, 1]] * 6,
                "example_id": list(range(6)),
            }
        )
        index = shard_dataset(test_ds, output_dir=temp_dir, split_name="test", num_shards=2)
        index.save(os.path.join(temp_dir, "test", SHARD_INDEX_FILE))

        ds = load_preprocessed_dataset(temp_dir, data_parallel_rank=0, data_parallel_size=1)
        assert len(ds["train"]) == 20
        assert len(ds["test"]) == 6
    finally:
        shutil.rmtree(temp_dir)


def test_undersharded_test_split_warns_not_fails():
    """A small TEST split with fewer shards than ranks is tolerated (warn-only).

    Ranks beyond num_shards get an empty test dataset rather than a hard error,
    since a tiny eval split is a legitimate case.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        dataset = Dataset.from_dict(
            {
                "input_ids": [[i, i + 1, i + 2] for i in range(20)],
                "attention_mask": [[1, 1, 1]] * 20,
                "example_id": list(range(20)),
            }
        )
        index = shard_dataset(dataset, output_dir=temp_dir, split_name="test", num_shards=2)
        index.save(os.path.join(temp_dir, "test", SHARD_INDEX_FILE))

        loader0 = ShardedDatasetLoader(dataset_path=temp_dir, global_rank=0, world_size=4)
        ds0 = loader0.load_split("test")
        assert len(ds0) > 0

        loader3 = ShardedDatasetLoader(dataset_path=temp_dir, global_rank=3, world_size=4)
        ds3 = loader3.load_split("test")  # warns, does not raise
        assert len(ds3) == 0
    finally:
        shutil.rmtree(temp_dir)


def test_load_datasets_sharded_missing_test_split_fails_loud():
    """A raw sharded dataset carrying only a train split is the one load path that can return a
    partial DatasetDict — load_datasets must raise the clear re-shard ValueError, not an opaque
    KeyError downstream."""
    temp_dir = tempfile.mkdtemp()
    try:
        create_test_sharded_dataset(20, 2, temp_dir)  # writes only the train split
        with pytest.raises(ValueError, match="missing the .*test.*split"):
            load_datasets(temp_dir, test_size=None, dataset_ratio=1, conversation_field=None)
    finally:
        shutil.rmtree(temp_dir)


def test_warm_shard_cache_serves_without_aws_credentials(tmp_path):
    """A relaunch with a complete shard cache and no AWS credentials must serve that cache.

    The staleness probe is best-effort end to end, so BUILDING its S3 client — which resolves
    credentials and a region — has to sit inside the guard too; outside it, the offline relaunch
    the cache exists for dies before the first probe.
    """
    source = tmp_path / "shard_source"
    Dataset.from_dict({"input_ids": [[1, 2, 3]] * 4}).save_to_disk(str(source))
    real_load_from_disk = load_from_disk
    shard = ShardInfo(id=0, path="train/shard_00000", num_examples=4)

    def _load(path, *args, **kwargs):
        # Stands in for the S3 fetch only; the cache read must go through the real loader.
        return real_load_from_disk(str(source)) if str(path).startswith("s3://") else real_load_from_disk(path)

    with (
        patch("src.data.sources.sharded_dataset.HALO_S3_DATASET_CACHE_DIR", str(tmp_path / "cache")),
        patch("src.data.sources.sharded_dataset.load_from_disk", side_effect=_load),
        patch("src.data.sources.sharded_dataset.S3Client", side_effect=NoCredentialsError()),
    ):
        ShardedDatasetLoader("s3://bucket/dataset", global_rank=0, world_size=1)._load_shard_from_s3(shard)
        # A fresh loader is the relaunch: the first one has already latched the probe off.
        relaunched = ShardedDatasetLoader("s3://bucket/dataset", global_rank=0, world_size=1)
        served = relaunched._load_shard_from_s3(shard)

    assert len(served) == 4
    assert relaunched._s3_fingerprint_unavailable, "an unusable client must latch the probe off for the process"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
