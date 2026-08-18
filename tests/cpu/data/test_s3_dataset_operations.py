#!/usr/bin/env python3
"""
Tests for S3 dataset operations using datasets native s3fs support.

These tests validate the S3 dataset loading/pushing that uses HuggingFace
datasets' native S3 support via s3fs rather than manual boto3
download/upload code.

All S3 calls are mocked — no real AWS access needed.

Usage:
    python tests/cpu/data/test_s3_dataset_operations.py
"""

import logging
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from scripts.before_training.s3_datasets import main as s3_cli_main
from src.data.sources.dataset_cache import (
    _read_marker_fingerprint,
    _write_download_marker,
    compute_etag_fingerprint,
)
from src.data.sources.s3_client import DEFAULT_BUCKET, S3Client, build_s3_uri
from src.data.sources.s3_client import logger as s3_module_logger


def test_get_storage_options_default():
    """Default S3Client (no custom creds) returns empty storage_options."""
    print("Testing get_storage_options (default)...")

    with patch.object(S3Client, "__post_init__"):
        client = S3Client()
    opts = client.get_storage_options()

    assert opts == {}, f"Expected empty dict, got {opts}"
    print("  get_storage_options default: PASSED")


def test_get_storage_options_with_credentials():
    """Custom credentials are translated to s3fs format."""
    print("Testing get_storage_options (with credentials)...")

    with patch.object(S3Client, "__post_init__"):
        client = S3Client(
            aws_access_key_id="AKID",
            aws_secret_access_key="SECRET",
            endpoint_url="http://minio:9000",
            region_name="us-west-2",
        )
    opts = client.get_storage_options()

    assert opts["key"] == "AKID", f"Expected 'AKID', got {opts.get('key')}"
    assert opts["secret"] == "SECRET", f"Expected 'SECRET', got {opts.get('secret')}"
    assert opts["client_kwargs"]["endpoint_url"] == "http://minio:9000"
    assert opts["client_kwargs"]["region_name"] == "us-west-2"
    print("  get_storage_options with credentials: PASSED")


def test_get_storage_options_partial():
    """Only set fields appear in storage_options."""
    print("Testing get_storage_options (partial)...")

    with patch.object(S3Client, "__post_init__"):
        client = S3Client(region_name="eu-west-1")
    opts = client.get_storage_options()

    assert "key" not in opts, "key should not be set"
    assert "secret" not in opts, "secret should not be set"
    assert opts["client_kwargs"]["region_name"] == "eu-west-1"
    print("  get_storage_options partial: PASSED")


def test_push_dataset_saves_to_a_staging_sibling():
    """push_dataset writes to a unique dot-prefixed staging sibling, then promotes — never straight
    onto the destination (the staged-push protocol; crash matrix in test_s3_staged_push.py)."""
    print("Testing push_dataset (staged save_to_disk)...")

    mock_dataset = MagicMock(spec=Dataset)
    mock_dataset.__len__ = MagicMock(return_value=100)

    with patch.object(S3Client, "__post_init__"):
        client = S3Client(bucket="test-bucket")

    client.exists = MagicMock(return_value=False)
    client._client = MagicMock()
    client._promote_staged_dataset = MagicMock()
    client.push_dataset(mock_dataset, "my/dataset")

    mock_dataset.save_to_disk.assert_called_once()
    client._promote_staged_dataset.assert_called_once()
    call_args = mock_dataset.save_to_disk.call_args
    assert call_args[0][0].startswith("s3://test-bucket/my/.staging-dataset-"), (
        f"Expected a staging-sibling target, got {call_args[0][0]}"
    )
    assert "storage_options" in call_args[1]
    print("  push_dataset staged save_to_disk: PASSED")


def test_push_dataset_overwrite_never_deletes_the_destination_first():
    """The pre-staging protocol deleted the destination before uploading — a crash in that window
    silently erased the only copy. Overwrite must reach save_to_disk with zero destination deletes."""
    print("Testing push_dataset (overwrite is delete-free before save)...")

    mock_dataset = MagicMock(spec=Dataset)
    mock_dataset.__len__ = MagicMock(return_value=50)

    with patch.object(S3Client, "__post_init__"):
        client = S3Client(bucket="test-bucket")

    client.exists = MagicMock(return_value=True)
    client.delete = MagicMock()
    client._client = MagicMock()
    deletes_before_save = []
    mock_dataset.save_to_disk = MagicMock(
        side_effect=lambda *a, **k: deletes_before_save.extend(client._client.delete_object.call_args_list)
    )
    client._promote_staged_dataset = MagicMock()

    client.push_dataset(mock_dataset, "existing/dataset", overwrite=True)

    client.delete.assert_not_called()
    assert deletes_before_save == [], "no destination object may be deleted before the staging save"
    mock_dataset.save_to_disk.assert_called_once()
    client._promote_staged_dataset.assert_called_once()
    print("  push_dataset overwrite delete-free: PASSED")


def test_push_dataset_no_overwrite_raises_on_existing():
    """overwrite=False against an existing dataset raises FileExistsError (no clobber)."""
    mock_dataset = MagicMock(spec=Dataset)
    mock_dataset.__len__ = MagicMock(return_value=50)

    with patch.object(S3Client, "__post_init__"):
        client = S3Client(bucket="test-bucket")
    client.exists = MagicMock(return_value=True)
    client.delete = MagicMock()

    with pytest.raises(FileExistsError, match="(?i)already exists"):
        client.push_dataset(mock_dataset, "existing/dataset", overwrite=False)

    client.delete.assert_not_called()
    mock_dataset.save_to_disk.assert_not_called()


def test_load_dataset_cache_hit():
    """When cache is valid, load_dataset reads from local cache without S3 call."""
    print("Testing load_dataset (cache hit)...")

    with patch.object(S3Client, "__post_init__"):
        client = S3Client(bucket="test-bucket")

    client.exists = MagicMock(return_value=True)

    with tempfile.TemporaryDirectory() as cache_root:
        with patch("src.data.sources.s3_client.HALO_S3_DATASET_CACHE_DIR", cache_root):
            import hashlib

            cache_key = hashlib.md5(b"test-bucket/my/dataset").hexdigest()
            cache_dir = os.path.join(cache_root, cache_key)
            cache_path = os.path.join(cache_dir, "dataset")
            os.makedirs(cache_path, exist_ok=True)

            with open(os.path.join(cache_path, ".download_complete"), "w") as f:
                f.write("s3://test-bucket/my/dataset")

            ds = Dataset.from_dict({"text": ["hello", "world"]})
            ds.save_to_disk(cache_path)
            # save_to_disk may wipe the directory, so the marker is rewritten.
            with open(os.path.join(cache_path, ".download_complete"), "w") as f:
                f.write("s3://test-bucket/my/dataset")

            with patch("src.data.sources.s3_client.load_from_disk") as mock_load:
                mock_load.return_value = ds
                client.load_dataset("my/dataset", use_cache=True)

                call_args = mock_load.call_args[0][0]
                assert call_args == cache_path, f"Expected load from cache {cache_path}, got {call_args}"

    print("  load_dataset cache hit: PASSED")


def test_load_dataset_cache_miss():
    """When cache is cold, load_dataset downloads from S3 URI then caches locally."""
    print("Testing load_dataset (cache miss)...")

    with patch.object(S3Client, "__post_init__"):
        client = S3Client(bucket="test-bucket")

    client.exists = MagicMock(return_value=True)

    mock_ds = MagicMock(spec=Dataset)
    mock_ds.__len__ = MagicMock(return_value=10)

    with tempfile.TemporaryDirectory() as cache_root:
        # The cache dir must exist for .download_complete to be written next to it.
        def fake_save_to_disk(path, **kwargs):
            os.makedirs(path, exist_ok=True)

        mock_ds.save_to_disk = MagicMock(side_effect=fake_save_to_disk)

        with patch("src.data.sources.s3_client.HALO_S3_DATASET_CACHE_DIR", cache_root):
            with patch("src.data.sources.s3_client.load_from_disk") as mock_load:
                mock_load.return_value = mock_ds

                client.load_dataset("my/dataset", use_cache=True)

                assert mock_load.call_count >= 2, f"Expected >= 2 calls to load_from_disk, got {mock_load.call_count}"

                first_call = mock_load.call_args_list[0]
                assert first_call[0][0] == "s3://test-bucket/my/dataset", (
                    f"Expected S3 URI in first call, got {first_call[0][0]}"
                )
                assert "storage_options" in first_call[1]

    print("  load_dataset cache miss: PASSED")


# Cache staleness: an in-place S3 overwrite must not serve stale data.


def _make_cached_dataset(cache_root: str, fingerprint: str | None) -> tuple[str, str]:
    """Create a complete local cache for s3://test-bucket/my/dataset; returns (cache_path, marker).

    ``fingerprint=None`` writes a legacy marker (URI only, no content identity).
    """
    import hashlib

    cache_key = hashlib.md5(b"test-bucket/my/dataset").hexdigest()
    cache_path = os.path.join(cache_root, cache_key, "dataset")
    os.makedirs(cache_path, exist_ok=True)
    Dataset.from_dict({"text": ["hello", "world"]}).save_to_disk(cache_path)
    marker = os.path.join(cache_path, ".download_complete")
    _write_download_marker(marker, "s3://test-bucket/my/dataset", fingerprint)
    return cache_path, marker


def _make_validating_client(live_fingerprint):
    """S3Client stub whose content_fingerprint returns a canned live value (None = unreachable)."""
    with patch.object(S3Client, "__post_init__"):
        client = S3Client(bucket="test-bucket")
    client.exists = MagicMock(return_value=True)
    client.content_fingerprint = MagicMock(return_value=live_fingerprint)
    return client


def test_compute_etag_fingerprint():
    """Order-insensitive, content-sensitive, None on empty listing."""
    a = compute_etag_fingerprint([("k1", '"e1"', 10), ("k2", '"e2"', 20)])
    b = compute_etag_fingerprint([("k2", '"e2"', 20), ("k1", '"e1"', 10)])
    c = compute_etag_fingerprint([("k1", '"e1-changed"', 10), ("k2", '"e2"', 20)])
    assert a == b, "fingerprint must be listing-order-insensitive"
    assert a != c, "fingerprint must change when an ETag changes"
    assert compute_etag_fingerprint([]) is None


def test_content_fingerprint_none_when_unreachable():
    """Any listing error (no creds, network outage) yields None — cache stays servable."""
    with patch.object(S3Client, "__post_init__"):
        client = S3Client(bucket="test-bucket")
    client._client = MagicMock()
    client._client.get_paginator.side_effect = RuntimeError("no credentials")
    assert client.content_fingerprint("my/dataset") is None


def test_content_fingerprint_lists_prefix():
    """content_fingerprint aggregates (Key, ETag, Size) of every object under the prefix."""
    with patch.object(S3Client, "__post_init__"):
        client = S3Client(bucket="test-bucket")
    client._client = MagicMock()
    pages = [{"Contents": [{"Key": "my/dataset/a", "ETag": '"e1"', "Size": 1}]}]
    client._client.get_paginator.return_value.paginate.return_value = pages
    expected = compute_etag_fingerprint([("my/dataset/a", '"e1"', 1)])
    assert client.content_fingerprint("my/dataset") == expected


def test_load_dataset_stale_cache_redownloads():
    """An in-place overwrite of the same S3 key must invalidate the local cache.

    The marker records the old content fingerprint; live S3 reports a new one → load_dataset must
    re-download instead of silently serving OLD data, and record the new fingerprint.
    """
    client = _make_validating_client(live_fingerprint="newfp")

    mock_ds = MagicMock(spec=Dataset)
    mock_ds.__len__ = MagicMock(return_value=10)

    with tempfile.TemporaryDirectory() as cache_root:
        _cache_path, marker = _make_cached_dataset(cache_root, fingerprint="oldfp")

        def fake_save_to_disk(path, **kwargs):
            os.makedirs(path, exist_ok=True)

        mock_ds.save_to_disk = MagicMock(side_effect=fake_save_to_disk)

        with patch("src.data.sources.s3_client.HALO_S3_DATASET_CACHE_DIR", cache_root):
            with patch("src.data.sources.s3_client.load_from_disk") as mock_load:
                mock_load.return_value = mock_ds
                client.load_dataset("my/dataset", use_cache=True)

                first_call = mock_load.call_args_list[0]
                assert first_call[0][0] == "s3://test-bucket/my/dataset", (
                    f"stale cache was served instead of re-downloaded: first load was {first_call[0][0]}"
                )
        assert _read_marker_fingerprint(marker) == "newfp"


def test_load_dataset_offline_serves_cache():
    """S3 unreachable (fingerprint probe returns None) → the complete cache is served as-is."""
    client = _make_validating_client(live_fingerprint=None)

    with tempfile.TemporaryDirectory() as cache_root:
        cache_path, marker = _make_cached_dataset(cache_root, fingerprint="oldfp")

        with patch("src.data.sources.s3_client.HALO_S3_DATASET_CACHE_DIR", cache_root):
            with patch("src.data.sources.s3_client.load_from_disk") as mock_load:
                mock_load.return_value = Dataset.from_dict({"text": ["hello", "world"]})
                client.load_dataset("my/dataset", use_cache=True)

                mock_load.assert_called_once()
                assert mock_load.call_args[0][0] == cache_path, "offline load must come from the local cache"
        assert _read_marker_fingerprint(marker) == "oldfp", "offline load must not rewrite the marker"


def test_load_dataset_matching_fingerprint_serves_cache():
    """Unchanged live content (fingerprints match) → cache served, no re-download."""
    client = _make_validating_client(live_fingerprint="fp")

    with tempfile.TemporaryDirectory() as cache_root:
        cache_path, _marker = _make_cached_dataset(cache_root, fingerprint="fp")

        with patch("src.data.sources.s3_client.HALO_S3_DATASET_CACHE_DIR", cache_root):
            with patch("src.data.sources.s3_client.load_from_disk") as mock_load:
                mock_load.return_value = Dataset.from_dict({"text": ["hello", "world"]})
                client.load_dataset("my/dataset", use_cache=True)

                mock_load.assert_called_once()
                assert mock_load.call_args[0][0] == cache_path


def test_load_dataset_legacy_marker_served_and_upgraded():
    """A pre-fingerprint marker stays valid (backward compatible) and is upgraded in place when
    live S3 is reachable, so the NEXT load can detect an overwrite."""
    client = _make_validating_client(live_fingerprint="livefp")

    with tempfile.TemporaryDirectory() as cache_root:
        cache_path, marker = _make_cached_dataset(cache_root, fingerprint=None)  # legacy marker

        with patch("src.data.sources.s3_client.HALO_S3_DATASET_CACHE_DIR", cache_root):
            with patch("src.data.sources.s3_client.load_from_disk") as mock_load:
                mock_load.return_value = Dataset.from_dict({"text": ["hello", "world"]})
                client.load_dataset("my/dataset", use_cache=True)

                mock_load.assert_called_once()
                assert mock_load.call_args[0][0] == cache_path, "legacy marker must still serve the cache"
        assert _read_marker_fingerprint(marker) == "livefp", "legacy marker was not upgraded with the live identity"


def _make_s3_loader():
    """Build a ShardedDatasetLoader instance bypassing __init__ with an S3 source."""
    from src.data.sources.sharded_dataset import ShardedDatasetLoader

    with patch.object(ShardedDatasetLoader, "__init__", lambda self, **kw: None):
        loader = ShardedDatasetLoader.__new__(ShardedDatasetLoader)
    loader.bucket = "test-bucket"
    loader.key = "preprocessed/dataset"
    loader.source_type = "s3"
    loader._s3_fingerprint_unavailable = False
    loader._s3_client = None
    return loader


def test_load_shard_from_s3_caches_locally():
    """The shard is downloaded once into a local md5-keyed cache, marked complete, and
    re-read from disk on subsequent loads.

    The per-shard FileLock + completion-marker cache lets disjoint DP ranks
    download in parallel and TP/CP siblings (or reruns) reuse the cache without
    re-streaming S3.
    """
    from src.data.shard_index import ShardInfo

    real_ds = Dataset.from_dict({"x": [1, 2, 3]})
    shard = ShardInfo(id=0, path="train/shard_0000", num_examples=100, byte_size=1024)

    with tempfile.TemporaryDirectory() as cache_root:
        with patch("src.data.sources.sharded_dataset.HALO_S3_DATASET_CACHE_DIR", cache_root):
            with patch("src.data.sources.sharded_dataset.load_from_disk") as mock_load:
                mock_load.return_value = real_ds

                # The cache dir must exist for the completion marker to be written next to it.
                def fake_save(path, **kw):
                    os.makedirs(path, exist_ok=True)

                real_ds.save_to_disk = MagicMock(side_effect=fake_save)

                loader = _make_s3_loader()
                loader._shard_fingerprint = MagicMock(return_value="fp0")  # hermetic: no live S3 probe
                loader._load_shard_from_s3(shard)

                assert mock_load.call_args_list[0][0][0] == ("s3://test-bucket/preprocessed/dataset/train/shard_0000")
                import hashlib

                key = hashlib.md5(b"test-bucket/preprocessed/dataset/train/shard_0000").hexdigest()
                marker = os.path.join(cache_root, "shards", key, "shard", ".download_complete")
                assert os.path.exists(marker), "download completion marker not written"
                assert mock_load.call_args_list[-1][0][0].endswith(os.path.join(key, "shard"))


def _make_cached_shard(cache_root: str, fingerprint: str | None) -> tuple[str, str]:
    """Create a complete per-shard cache for train/shard_0000; returns (cache_path, marker)."""
    import hashlib

    key = hashlib.md5(b"test-bucket/preprocessed/dataset/train/shard_0000").hexdigest()
    cache_path = os.path.join(cache_root, "shards", key, "shard")
    os.makedirs(cache_path, exist_ok=True)
    Dataset.from_dict({"x": [1, 2, 3]}).save_to_disk(cache_path)
    marker = os.path.join(cache_path, ".download_complete")
    _write_download_marker(marker, "s3://test-bucket/preprocessed/dataset/train/shard_0000", fingerprint)
    return cache_path, marker


def test_load_shard_from_s3_stale_cache_redownloads():
    """Re-preprocessing to the same S3 URI must invalidate the per-shard cache.

    Marker holds the old fingerprint, live S3 reports a new one → the shard is re-downloaded
    instead of silently training on OLD data.
    """
    from src.data.shard_index import ShardInfo

    shard = ShardInfo(id=0, path="train/shard_0000", num_examples=100, byte_size=1024)
    real_ds = Dataset.from_dict({"x": [1, 2, 3]})

    with tempfile.TemporaryDirectory() as cache_root:
        _cache_path, marker = _make_cached_shard(cache_root, fingerprint="oldfp")

        with patch("src.data.sources.sharded_dataset.HALO_S3_DATASET_CACHE_DIR", cache_root):
            with patch("src.data.sources.sharded_dataset.load_from_disk") as mock_load:
                mock_load.return_value = real_ds

                def fake_save(path, **kw):
                    os.makedirs(path, exist_ok=True)

                real_ds.save_to_disk = MagicMock(side_effect=fake_save)

                loader = _make_s3_loader()
                loader._shard_fingerprint = MagicMock(return_value="newfp")
                loader._load_shard_from_s3(shard)

                first_call = mock_load.call_args_list[0][0][0]
                assert first_call == "s3://test-bucket/preprocessed/dataset/train/shard_0000", (
                    f"stale shard cache was served instead of re-downloaded: first load was {first_call}"
                )
        assert _read_marker_fingerprint(marker) == "newfp"


def test_load_shard_from_s3_offline_serves_cache():
    """S3 unreachable (shard fingerprint probe returns None) → the complete shard cache is served."""
    from src.data.shard_index import ShardInfo

    shard = ShardInfo(id=0, path="train/shard_0000", num_examples=100, byte_size=1024)

    with tempfile.TemporaryDirectory() as cache_root:
        cache_path, marker = _make_cached_shard(cache_root, fingerprint="oldfp")

        with patch("src.data.sources.sharded_dataset.HALO_S3_DATASET_CACHE_DIR", cache_root):
            with patch("src.data.sources.sharded_dataset.load_from_disk") as mock_load:
                mock_load.return_value = Dataset.from_dict({"x": [1, 2, 3]})

                loader = _make_s3_loader()
                loader._shard_fingerprint = MagicMock(return_value=None)
                loader._load_shard_from_s3(shard)

                mock_load.assert_called_once()
                assert mock_load.call_args[0][0] == cache_path, "offline shard load must come from the local cache"
        assert _read_marker_fingerprint(marker) == "oldfp", "offline shard load must not rewrite the marker"


def test_shard_fingerprint_skips_after_unreachable_s3(caplog):
    """After one unreachable-S3 probe, later shards must not retry (per-shard timeout cost), and the
    give-up must be logged — an unannounced latch serves every later shard unvalidated in silence.

    The probe lists through ``S3Client.content_entries`` — the same listing the whole-dataset cache
    validates with, so a shard cache and a dataset cache compare byte-identical fingerprints.
    """
    loader = _make_s3_loader()

    with patch("src.data.sources.sharded_dataset.S3Client") as mock_client_cls:
        mock_client_cls.return_value.content_entries = MagicMock(side_effect=OSError("connection reset"))
        with caplog.at_level(logging.WARNING, logger="src.data.sources.sharded_dataset"):
            assert loader._shard_fingerprint("preprocessed/dataset/train/shard_0000") is None
            assert loader._shard_fingerprint("preprocessed/dataset/train/shard_0001") is None
        assert mock_client_cls.return_value.content_entries.call_count == 1, (
            "second probe must be skipped after an unreachable S3"
        )

    latched = [r for r in caplog.records if "skipping the fingerprint probe" in r.getMessage()]
    assert len(latched) == 1, f"the latch must warn exactly once, naming the reason; got {len(latched)} records"
    assert "connection reset" in latched[0].getMessage(), "the warning must name the failure that disarmed the probe"


def test_shard_fingerprint_empty_listing_does_not_latch():
    """An EMPTY prefix listing is a real absence, not a transport fault: it yields None for THAT
    shard (nothing to fingerprint) but must leave the next shard probeable — latching on it would
    serve a whole run's shards unvalidated after one legitimately empty prefix."""
    loader = _make_s3_loader()

    with patch("src.data.sources.sharded_dataset.S3Client") as mock_client_cls:
        mock_client_cls.return_value.content_entries = MagicMock(return_value=[])
        assert loader._shard_fingerprint("preprocessed/dataset/train/shard_0000") is None
        assert loader._shard_fingerprint("preprocessed/dataset/train/shard_0001") is None
        assert mock_client_cls.return_value.content_entries.call_count == 2, (
            "an empty listing must not disarm fingerprinting for the rest of the process"
        )
    assert loader._s3_fingerprint_unavailable is False


def test_load_shard_index_from_s3():
    """_load_shard_index_from_s3 reads JSON from S3 via s3fs."""
    print("Testing ShardedDatasetLoader._load_shard_index_from_s3...")

    from src.data.sources.sharded_dataset import ShardedDatasetLoader

    index_data = {
        "version": "1.0",
        "num_shards": 2,
        "total_examples": 200,
        "shards": [
            {"id": 0, "path": "train/shard_0000", "num_examples": 100, "byte_size": 1024},
            {"id": 1, "path": "train/shard_0001", "num_examples": 100, "byte_size": 1024},
        ],
    }

    mock_fs = MagicMock()
    mock_file = MagicMock()
    mock_file.__enter__ = MagicMock(return_value=mock_file)
    mock_file.__exit__ = MagicMock(return_value=False)
    mock_fs.open.return_value = mock_file

    with patch.object(ShardedDatasetLoader, "__init__", lambda self, **kw: None):
        loader = ShardedDatasetLoader.__new__(ShardedDatasetLoader)
        loader.bucket = "test-bucket"
        loader.key = "preprocessed/dataset"
        loader.source_type = "s3"
        loader._shard_indices = {}

    with patch("src.data.sources.s3_client.s3fs.S3FileSystem", return_value=mock_fs):
        with patch("src.data.sources.s3_client.json.load", return_value=index_data):
            index = loader._load_shard_index_from_s3("train")

    assert index.num_shards == 2
    assert index.total_examples == 200
    assert len(index.shards) == 2
    mock_fs.open.assert_called_once()
    print("  _load_shard_index_from_s3: PASSED")


def test_load_preprocessed_metadata_from_s3():
    """load_preprocessed_metadata reads JSON from S3 via s3fs."""
    print("Testing load_preprocessed_metadata (S3)...")

    from src.data.pipeline.preprocessed_metadata import load_preprocessed_metadata
    from src.data.shard_index import PREPROCESSING_VERSION

    metadata_dict = {
        # Every written metadata.json carries the stamp (dataclass default) and the load compares it.
        "version": PREPROCESSING_VERSION,
        "preprocessed": True,
        "model_name": "Qwen/Qwen3-8B",
        "max_length": 8192,
        "total_train_examples": 1000,
        "num_shards": 4,
        "packed": True,
        "packing_strategy": "greedy",
        "created_at": "2024-01-01T00:00:00",
    }

    mock_fs = MagicMock()
    mock_file = MagicMock()
    mock_file.__enter__ = MagicMock(return_value=mock_file)
    mock_file.__exit__ = MagicMock(return_value=False)
    mock_fs.open.return_value = mock_file

    with patch("src.data.sources.s3_client.s3fs.S3FileSystem", return_value=mock_fs):
        with patch("src.data.sources.s3_client.json.load", return_value=metadata_dict):
            metadata = load_preprocessed_metadata("s3://test-bucket/preprocessed/dataset")

    assert metadata.preprocessed is True
    assert metadata.model_name == "Qwen/Qwen3-8B"
    assert metadata.max_length == 8192
    mock_fs.open.assert_called_once()
    print("  load_preprocessed_metadata S3: PASSED")


def test_build_s3_uri():
    """Test the build_s3_uri helper."""
    print("Testing build_s3_uri helper...")

    uri = build_s3_uri("my_dataset", "datasets")
    assert uri == f"s3://{DEFAULT_BUCKET}/datasets/my_dataset", f"Got {uri}"

    uri = build_s3_uri("my_dataset", None)
    assert uri == f"s3://{DEFAULT_BUCKET}/my_dataset", f"Got {uri}"

    uri = build_s3_uri("nested/path/data")
    assert uri == f"s3://{DEFAULT_BUCKET}/nested/path/data", f"Got {uri}"

    print("  build_s3_uri: PASSED")


def test_s3_cache_dir_derives_from_scratch_root():
    """No explicit S3 cache override → the cache dir derives from HALO_DATA_ROOT (one scratch knob);
    an explicit HALO_S3_DATASET_CACHE_DIR still wins."""
    import importlib

    import src.data.sources.dataset_cache as s3_mod
    import src.env as env_mod

    keys = ("HALO_S3_DATASET_CACHE_DIR", "HALO_DATA_ROOT")
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        os.environ["HALO_DATA_ROOT"] = "/scratch/halo-test"
        importlib.reload(env_mod)
        s3r = importlib.reload(s3_mod)
        assert s3r.HALO_S3_DATASET_CACHE_DIR == "/scratch/halo-test/s3_datasets"

        os.environ["HALO_S3_DATASET_CACHE_DIR"] = "/explicit/cache"
        importlib.reload(env_mod)
        s3r = importlib.reload(s3_mod)
        assert s3r.HALO_S3_DATASET_CACHE_DIR == "/explicit/cache"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(env_mod)
        importlib.reload(s3_mod)


# CLI `delete` tests: argument parsing → the delete() call; no S3 is ever touched.


def _run_cli(
    argv: list[str], *, answer: str | None = None, exists: bool = True, object_exists: bool = True
) -> MagicMock:
    """Run the s3 CLI offline with ``delete`` mocked out; returns the mock for call assertions.

    The handlers act through the ``S3Client`` ``main`` passes them (never a module-level default
    client), so the client's own methods are the seam: ``__post_init__`` is stubbed so no boto3
    session is built, and the URI/existence probes are canned. ``exists``/``object_exists`` model
    what is actually in the bucket: the default is an object at the exact key; ``object_exists=False``
    with ``exists=True`` is a folder-like prefix. ``answer`` feeds the confirmation prompt (None =
    never prompted, i.e. ``--yes`` was passed).
    """
    # Patched by dotted path, not on the imported class object: the cache-dir test above reloads
    # this module, so the name bound at import time is a STALE class the CLI no longer instantiates.
    with (
        patch("src.data.sources.s3_client.S3Client.__post_init__"),
        patch("src.data.sources.s3_client.S3Client.delete", return_value=True) as mock_delete,
        patch("src.data.sources.s3_client.S3Client._get_s3_uri", return_value="s3://test-bucket/my_folder"),
        patch("src.data.sources.s3_client.S3Client.exists", return_value=exists),
        patch("src.data.sources.s3_client.S3Client.object_exists", return_value=object_exists),
        patch("builtins.input", return_value=answer),
        patch.object(sys, "argv", ["s3", *argv]),
    ):
        s3_cli_main()
    return mock_delete


def test_module_info_records_survive_the_root_warning_pin():
    """``src`` pins the ROOT logger to WARNING, so the module's own level is what admits its INFO
    lines — the run's only record of what data moved (downloads, cache hits, stale-object sweeps).
    Inheriting the pin emits them nowhere while every call still reports success."""
    captured = []

    class _Capture(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    root = logging.getLogger()
    handler = _Capture()
    saved_level, saved_handlers = root.level, list(root.handlers)
    root.handlers = [handler]
    root.setLevel(logging.WARNING)
    try:
        s3_module_logger.info("downloaded a dataset")
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)

    assert captured == ["downloaded a dataset"], "the module's INFO record never reached a handler"


def test_cli_owns_the_root_handler_it_inherited():
    """The CLI's own verbosity must actually take effect: importing ``src`` already installed a root
    handler and pinned the level, against which a non-forcing ``basicConfig`` is a silent no-op."""
    root = logging.getLogger()
    saved_level, saved_handlers = root.level, list(root.handlers)
    inherited = logging.NullHandler()
    root.handlers = [inherited]
    root.setLevel(logging.WARNING)
    try:
        _run_cli(["exists", "my_folder"])
        cli_level, cli_handlers = root.level, list(root.handlers)
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)

    assert cli_level == logging.INFO, "the CLI's basicConfig did not set its own level"
    assert inherited not in cli_handlers, "the inherited handler survived, so the CLI configured nothing"


@pytest.mark.parametrize("raw", ["0", "-1"])
def test_non_positive_cache_lock_timeout_warns_and_falls_back(raw, caplog):
    """``filelock`` reads the extremes as its own modes — 0 fails immediately, a negative waits
    forever — so a non-positive override must be REFUSED out loud and fall back to the store budget,
    never silently redefine the peer-download wait. A positive value is still honoured verbatim."""
    import importlib

    import src.data.sources.dataset_cache as s3_mod
    import src.env as env_mod

    saved = os.environ.get("HALO_S3_CACHE_LOCK_TIMEOUT_SECONDS")
    try:
        os.environ["HALO_S3_CACHE_LOCK_TIMEOUT_SECONDS"] = raw
        importlib.reload(env_mod)
        with caplog.at_level(logging.WARNING):
            s3r = importlib.reload(s3_mod)
        derived_default = s3r.resolve_store_timeout_hours() * 3600
        assert derived_default == s3r._CACHE_LOCK_TIMEOUT_SECONDS
        assert any("HALO_S3_CACHE_LOCK_TIMEOUT_SECONDS" in message for message in caplog.messages), (
            f"{raw} was swallowed silently instead of being refused"
        )

        os.environ["HALO_S3_CACHE_LOCK_TIMEOUT_SECONDS"] = "120"
        importlib.reload(env_mod)
        s3r = importlib.reload(s3_mod)
        assert s3r._CACHE_LOCK_TIMEOUT_SECONDS == 120
    finally:
        if saved is None:
            os.environ.pop("HALO_S3_CACHE_LOCK_TIMEOUT_SECONDS", None)
        else:
            os.environ["HALO_S3_CACHE_LOCK_TIMEOUT_SECONDS"] = saved
        importlib.reload(env_mod)
        importlib.reload(s3_mod)


def test_cli_delete_is_not_recursive_by_default():
    """A bare `delete <key>` must delete ONE object, never the whole prefix.

    `store_true` can only ever set True, so a `default=True` on `--recursive` makes the flag inert
    and turns every CLI delete into a prefix-wipe the user has no way to opt out of.
    """
    mock_delete = _run_cli(["delete", "my_folder", "--yes"])

    mock_delete.assert_called_once()
    assert mock_delete.call_args.kwargs["recursive"] is False, (
        "bare `delete <key>` deleted the whole prefix: --recursive must be opt-in"
    )


@pytest.mark.parametrize("flag", ["--recursive", "-r"])
def test_cli_delete_recursive_opts_in(flag):
    """The user's intent must still be expressible: --recursive/-r reaches delete() as True."""
    mock_delete = _run_cli(["delete", "my_folder", flag, "--yes"])

    mock_delete.assert_called_once()
    assert mock_delete.call_args.kwargs["recursive"] is True, f"{flag} did not reach delete()"


def test_cli_delete_refuses_a_prefix_without_recursive():
    """A non-recursive delete aimed at a PREFIX must refuse, not report success having done nothing.

    S3's DeleteObject is idempotent: against a key that is only a prefix it returns 204, so
    ``delete()`` reports True and the CLI would print "Deleted" while the data is still there.
    """
    mock_delete = _run_cli(["delete", "my_folder", "--yes"], exists=True, object_exists=False)

    mock_delete.assert_not_called()


def test_cli_delete_reports_a_missing_key():
    """Nothing at the key at all: still no delete call, and no success message."""
    mock_delete = _run_cli(["delete", "my_folder", "--yes"], exists=False, object_exists=False)

    mock_delete.assert_not_called()


def test_cli_delete_recursive_still_deletes_a_prefix():
    """Anti-vacuity for the guard: with --recursive the same prefix IS deleted."""
    mock_delete = _run_cli(["delete", "my_folder", "-r", "--yes"], exists=True, object_exists=False)

    mock_delete.assert_called_once()
    assert mock_delete.call_args.kwargs["recursive"] is True


def test_recursive_delete_reports_false_when_the_prefix_matched_nothing():
    """A prefix sweep that deleted nothing must not report success.

    ``--recursive`` aimed at a single object sweeps ``<key>/``, which has no children, so an
    unconditional ``return True`` prints "Deleted" over data that is still there — the mirror of the
    non-recursive-on-a-prefix no-op.
    """
    with patch.object(S3Client, "__post_init__"):
        client = S3Client(bucket="test-bucket")
    paginator = MagicMock()
    paginator.paginate.return_value = [{}]  # no Contents -> nothing matched
    client._client = MagicMock()
    client._client.get_paginator.return_value = paginator

    assert client.delete("my_folder", recursive=True) is False
    client._client.delete_objects.assert_not_called()


def test_cli_delete_declines_confirmation():
    """Without --yes, anything but 'y' at the prompt must leave S3 untouched."""
    mock_delete = _run_cli(["delete", "my_folder", "--recursive"], answer="n")

    mock_delete.assert_not_called()


def test_cli_delete_confirmed_deletes():
    """A 'y' at the prompt proceeds — the guard blocks on the answer, not on being unreachable."""
    mock_delete = _run_cli(["delete", "my_folder", "--recursive"], answer="y")

    mock_delete.assert_called_once()
    assert mock_delete.call_args.kwargs["recursive"] is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
