#!/usr/bin/env python3
import os
import shutil
import tempfile
from collections import Counter
from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError, NoCredentialsError
from datasets import Dataset, DatasetDict

import halo.data.sources.loading as loading
from halo.data.preprocessed.preparation import shard_dataset
from halo.data.preprocessed.shard_index import (
    SHARD_INDEX_FILE,
    IncompatiblePreprocessedDataset,
    ShardIndex,
    ShardInfo,
    write_stamped_sidecar,
)
from halo.data.sources.loading import load_datasets, load_preprocessed_dataset
from halo.data.sources.sharded_dataset import ShardedDatasetLoader


def create_test_sharded_dataset(num_examples: int, num_shards: int, temp_dir: str) -> str:
    dataset = Dataset.from_dict(
        {
            "input_ids": [[i, i + 1, i + 2] for i in range(num_examples)],
            "attention_mask": [[1, 1, 1]] * num_examples,
            "example_id": list(range(num_examples)),
        }
    )

    index = shard_dataset(dataset, output_dir=temp_dir, split_name="train", num_shards=num_shards)

    index_path = os.path.join(temp_dir, "train", SHARD_INDEX_FILE)
    write_stamped_sidecar(index_path, index)

    return temp_dir


def test_shard_assignment_multi_rank():
    print("Testing shard assignment (multi rank)...")

    temp_dir = tempfile.mkdtemp()

    try:
        create_test_sharded_dataset(200, 8, temp_dir)

        # 8 shards / 4 ranks → contiguous blocks of 2: rank r gets shards [2r, 2r+1].
        expected_ids = {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7]}
        for rank in range(4):
            loader = ShardedDatasetLoader(
                dataset_path=temp_dir,
                dp_rank=rank,
                dp_size=4,
            )

            assigned = loader.get_assigned_shards("train")
            assert [s.id for s in assigned] == expected_ids[rank], (
                f"Rank {rank} got shard ids {[s.id for s in assigned]}, expected {expected_ids[rank]}"
            )

    finally:
        shutil.rmtree(temp_dir)

    print("  shard assignment (multi rank): PASSED")


def test_shard_assignment_remainder():
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
                dp_rank=rank,
                dp_size=3,
            )
            assigned = [s.id for s in loader.get_assigned_shards("train")]
            assert assigned == expected_ids[rank], f"Rank {rank}: {assigned} != {expected_ids[rank]}"
            all_assigned.extend(assigned)

        assert sorted(all_assigned) == list(range(10))

    finally:
        shutil.rmtree(temp_dir)

    print("  shard assignment (remainder): PASSED")


def test_load_split():
    print("Testing load_split...")

    temp_dir = tempfile.mkdtemp()

    try:
        create_test_sharded_dataset(100, 4, temp_dir)

        loader = ShardedDatasetLoader(
            dataset_path=temp_dir,
            dp_rank=0,
            dp_size=1,
        )

        dataset = loader.load_split("train")

        assert len(dataset) == 100, f"Expected 100 examples, got {len(dataset)}"
        assert "input_ids" in dataset.column_names
        assert "example_id" in dataset.column_names

    finally:
        shutil.rmtree(temp_dir)

    print("  load_split: PASSED")


def test_load_split_distributed():
    print("Testing load_split (distributed)...")

    temp_dir = tempfile.mkdtemp()

    try:
        create_test_sharded_dataset(100, 4, temp_dir)

        all_example_ids = Counter()

        for rank in range(2):
            loader = ShardedDatasetLoader(
                dataset_path=temp_dir,
                dp_rank=rank,
                dp_size=2,
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


def test_total_examples():
    print("Testing total_examples...")

    temp_dir = tempfile.mkdtemp()

    try:
        create_test_sharded_dataset(150, 5, temp_dir)

        loader = ShardedDatasetLoader(
            dataset_path=temp_dir,
            dp_rank=0,
            dp_size=1,
        )

        total = loader.total_examples("train")
        assert total == 150, f"Expected 150 total examples, got {total}"

    finally:
        shutil.rmtree(temp_dir)

    print("  total_examples: PASSED")


def test_is_sharded_dataset():
    print("Testing is_sharded_dataset...")

    temp_dir = tempfile.mkdtemp()

    try:
        assert ShardedDatasetLoader.is_sharded_dataset(temp_dir) is False

        create_test_sharded_dataset(50, 2, temp_dir)

        assert ShardedDatasetLoader.is_sharded_dataset(temp_dir) is True

    finally:
        shutil.rmtree(temp_dir)

    print("  is_sharded_dataset: PASSED")


def test_a_file_uri_loads_shards_as_its_plain_path_does(tmp_path):
    source = create_test_sharded_dataset(50, 2, str(tmp_path))
    uri = f"file://{source}"
    assert ShardedDatasetLoader.is_sharded_dataset(uri) is True

    loader = ShardedDatasetLoader(dataset_path=uri, dp_rank=0, dp_size=2)
    rows = loader.load_split("train")["example_id"]
    expected = ShardedDatasetLoader(dataset_path=source, dp_rank=0, dp_size=2).load_split("train")["example_id"]

    assert list(rows) == list(expected)
    assert len(rows) == 25


def test_is_sharded_dataset_s3_probe_failure_returns_false(tmp_path):
    """With no local mirror any S3 probe failure returns False: the caller reconciles across ranks with an
    all_reduce, and a raise on one rank hangs the peers until the NCCL watchdog."""
    print("Testing is_sharded_dataset S3 probe failure...")

    for error in (
        ClientError({"Error": {"Code": "AccessDenied", "Message": "403"}}, "HeadObject"),
        ClientError({"Error": {"Code": "SlowDown", "Message": "throttled"}}, "HeadObject"),
        OSError("connection reset"),
    ):
        with patch("halo.data.sources.s3_dataset_store.HALO_S3_DATASET_CACHE_DIR", str(tmp_path)):
            with patch("halo.data.sources.s3_dataset_store.read_json_from_s3", side_effect=error):
                assert ShardedDatasetLoader.is_sharded_dataset("s3://bucket/dataset") is False, (
                    f"probe must err toward non-sharded on {type(error).__name__}, not raise"
                )

    print("  is_sharded_dataset_s3_probe_failure_returns_false: PASSED")


def test_load_single_shard():
    print("Testing load single shard...")

    temp_dir = tempfile.mkdtemp()

    try:
        create_test_sharded_dataset(50, 1, temp_dir)

        loader = ShardedDatasetLoader(
            dataset_path=temp_dir,
            dp_rank=0,
            dp_size=1,
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
    temp_dir = tempfile.mkdtemp()
    try:
        create_test_sharded_dataset(20, 2, temp_dir)
        loader = ShardedDatasetLoader(dataset_path=temp_dir, dp_rank=3, dp_size=4)
        assert loader.get_assigned_shards("train") == []
    finally:
        shutil.rmtree(temp_dir)


def test_undersharded_train_split_fails_fast():
    """The length equalizer all_reduce(MIN)s rank lengths, so one zero-length rank would zero the batch; load_split
    raises with a re-preprocess hint."""
    temp_dir = tempfile.mkdtemp()
    try:
        create_test_sharded_dataset(20, 2, temp_dir)
        loader = ShardedDatasetLoader(dataset_path=temp_dir, dp_rank=0, dp_size=4)
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
    """Read with this build's field meanings, a diverged build's index hands ranks the wrong shards."""
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
    """The undersharded guard reads ``0 < num_shards < world_size``, so zero would slip past it and every rank would
    train on an empty split."""
    (tmp_path / "train").mkdir()
    write_stamped_sidecar(
        str(tmp_path / "train" / SHARD_INDEX_FILE),
        ShardIndex(split="train", num_shards=0, total_examples=0, shards=[]),
    )
    loader = ShardedDatasetLoader(dataset_path=str(tmp_path), dp_rank=0, dp_size=1)
    with pytest.raises(ValueError, match="(?i)no shards"):
        loader.load_split("train")


def test_load_datasets_list_entry_missing_test_split_fails_loud():
    """The list branch discards the sharded flag and the loader skips a missing split, so the raise names the entry
    and the remedy instead of a bare ``KeyError('test')``."""
    temp_dir = tempfile.mkdtemp()
    try:
        create_test_sharded_dataset(20, 2, temp_dir)  # writes only the train split
        with pytest.raises(ValueError, match="missing the .*test.*split") as excinfo:
            load_datasets([temp_dir], test_size=None, dataset_ratio=1, conversation_field=None)
        assert temp_dir in str(excinfo.value), str(excinfo.value)
    finally:
        shutil.rmtree(temp_dir)


def test_load_preprocessed_dataset_empty_test_split_raises():
    """The empty split would KeyError downstream or hang distributed eval. Per-rank-only emptiness is the
    trainer-side equalize-raise case."""
    temp_dir = tempfile.mkdtemp()
    try:
        create_test_sharded_dataset(20, 2, temp_dir)  # non-empty train
        empty = Dataset.from_dict({"input_ids": [], "attention_mask": [], "example_id": []})
        index = shard_dataset(empty, output_dir=temp_dir, split_name="test", num_shards=2)
        assert index.total_examples == 0
        write_stamped_sidecar(os.path.join(temp_dir, "test", SHARD_INDEX_FILE), index)

        with pytest.raises(ValueError, match="(?i)no test data|empty"):
            load_preprocessed_dataset(temp_dir, data_parallel_rank=0, data_parallel_size=1)
    finally:
        shutil.rmtree(temp_dir)


def test_load_preprocessed_dataset_empty_train_split_raises(monkeypatch):
    """Unguarded it reaches the trainer as a zero-step schedule. The empty split is handed over in memory, since the
    pinned ``datasets`` writes no shard for a zero-row split."""
    columns = ["input_ids", "attention_mask", "example_id"]
    ds = DatasetDict(
        {
            "train": Dataset.from_dict(dict.fromkeys(columns, [])),
            "test": Dataset.from_dict({"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]], "example_id": [0]}),
        }
    )
    monkeypatch.setattr(loading, "_load_replicated", lambda path, test_size, placeholder_test: ds)
    monkeypatch.setattr(loading, "is_sharded_dataset_across_ranks", lambda path: False)

    with pytest.raises(ValueError, match="(?i)train"):
        loading.load_preprocessed_dataset("unused", data_parallel_rank=0, data_parallel_size=1)


def test_load_preprocessed_dataset_missing_test_split_fails_loud():
    """Raises the re-prepare error at the loader seam, not a bare KeyError('test') at the consumer."""
    temp_dir = tempfile.mkdtemp()
    try:
        create_test_sharded_dataset(20, 2, temp_dir)  # writes only the train split
        with pytest.raises(ValueError, match="missing the .*test.*split"):
            load_preprocessed_dataset(temp_dir, data_parallel_rank=0, data_parallel_size=1)
    finally:
        shutil.rmtree(temp_dir)


def test_load_preprocessed_dataset_nonempty_test_split_loads():
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
        write_stamped_sidecar(os.path.join(temp_dir, "test", SHARD_INDEX_FILE), index)

        ds = load_preprocessed_dataset(temp_dir, data_parallel_rank=0, data_parallel_size=1)
        assert len(ds["train"]) == 20
        assert len(ds["test"]) == 6
    finally:
        shutil.rmtree(temp_dir)


def test_undersharded_test_split_warns_not_fails():
    """Ranks beyond num_shards get an empty test dataset, since a tiny eval split is legitimate."""
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
        write_stamped_sidecar(os.path.join(temp_dir, "test", SHARD_INDEX_FILE), index)

        loader0 = ShardedDatasetLoader(dataset_path=temp_dir, dp_rank=0, dp_size=4)
        ds0 = loader0.load_split("test")
        assert len(ds0) > 0

        loader3 = ShardedDatasetLoader(dataset_path=temp_dir, dp_rank=3, dp_size=4)
        ds3 = loader3.load_split("test")  # warns, does not raise
        assert len(ds3) == 0
    finally:
        shutil.rmtree(temp_dir)


def test_load_datasets_sharded_missing_test_split_fails_loud():
    """A raw sharded dataset with only train is the one load path that can return a partial DatasetDict."""
    temp_dir = tempfile.mkdtemp()
    try:
        create_test_sharded_dataset(20, 2, temp_dir)  # writes only the train split
        with pytest.raises(ValueError, match="missing the .*test.*split"):
            load_datasets(temp_dir, test_size=None, dataset_ratio=1, conversation_field=None)
    finally:
        shutil.rmtree(temp_dir)


def test_warm_shard_cache_serves_without_aws_credentials(tmp_path):
    """Building the staleness probe's S3 client resolves credentials and a region, so it sits inside the best-effort
    guard."""
    source = tmp_path / "shard_source"
    Dataset.from_dict({"input_ids": [[1, 2, 3]] * 4}).save_to_disk(str(source))
    shard = ShardInfo(id=0, path="train/shard_00000", num_examples=4)
    cache_dir = patch("halo.data.sources.sharded_dataset.HALO_S3_DATASET_CACHE_DIR", str(tmp_path / "cache"))

    # The first run, with credentials, fills the cache; its fetch stands in for the S3 download.
    with (
        cache_dir,
        patch("halo.data.sources.sharded_dataset.S3Client"),
        patch(
            "halo.data.sources.sharded_dataset.download_dataset_tree",
            side_effect=lambda client, key, path: shutil.copytree(source, path),
        ),
    ):
        ShardedDatasetLoader("s3://bucket/dataset", dp_rank=0, dp_size=1)._load_shard_from_s3(shard)
    with cache_dir, patch("halo.data.sources.sharded_dataset.S3Client", side_effect=NoCredentialsError()):
        relaunched = ShardedDatasetLoader("s3://bucket/dataset", dp_rank=0, dp_size=1)
        served = relaunched._load_shard_from_s3(shard)

    assert len(served) == 4
    assert relaunched._s3_fingerprint_unavailable, "an unusable client must latch the probe off for the process"


def test_load_raises_on_a_shard_its_index_lists_but_the_source_lacks(tmp_path):
    """A missing shard is a broken dataset, not a missing split: ``load`` must raise, not drop train."""
    create_test_sharded_dataset(40, 2, str(tmp_path))
    index = ShardIndex.load(os.path.join(tmp_path, "train", SHARD_INDEX_FILE))
    shutil.rmtree(os.path.join(tmp_path, index.shards[1].path))

    loader = ShardedDatasetLoader(dataset_path=str(tmp_path), dp_rank=0, dp_size=1)
    with pytest.raises(FileNotFoundError):
        loader.load()


def test_load_skips_a_split_without_an_index(tmp_path):
    create_test_sharded_dataset(40, 2, str(tmp_path))

    loaded = ShardedDatasetLoader(dataset_path=str(tmp_path), dp_rank=0, dp_size=1).load()

    assert list(loaded) == ["train"]
    assert len(loaded["train"]) == 40


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
