#!/usr/bin/env python3
"""A push writes through datasets' s3fs support, a load downloads the tree straight into its local cache
(``download_dataset_tree``). All S3 calls are mocked."""

import hashlib
import importlib
import logging
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

import halo.data.sources.dataset_cache as s3_mod
import halo.env as env_mod
from halo.data.preprocessed.metadata import load_preprocessed_metadata
from halo.data.preprocessed.shard_index import PREPROCESSING_VERSION, ShardInfo
from halo.data.sources import s3_client as s3_client_mod
from halo.data.sources import s3_dataset_store
from halo.data.sources.dataset_cache import (
    _read_marker_fingerprint,
    _write_download_marker,
    compute_etag_fingerprint,
)
from halo.data.sources.s3_client import S3Client, build_s3_uri
from halo.data.sources.s3_client import logger as s3_module_logger
from halo.data.sources.s3_commands import build_command_client
from halo.data.sources.sharded_dataset import ShardedDatasetLoader
from scripts.before_training import s3 as s3_cli_mod
from scripts.before_training.s3 import main as s3_cli_main


def test_get_storage_options_default():
    print("Testing get_storage_options (default)...")

    with patch.object(S3Client, "__post_init__"):
        client = S3Client()
    opts = client.get_storage_options()

    assert opts == {}, f"Expected empty dict, got {opts}"
    print("  get_storage_options default: PASSED")


def test_get_storage_options_with_credentials():
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
    print("Testing get_storage_options (partial)...")

    with patch.object(S3Client, "__post_init__"):
        client = S3Client(region_name="eu-west-1")
    opts = client.get_storage_options()

    assert "key" not in opts, "key should not be set"
    assert "secret" not in opts, "secret should not be set"
    assert opts["client_kwargs"]["region_name"] == "eu-west-1"
    print("  get_storage_options partial: PASSED")


def test_push_dataset_saves_to_a_staging_sibling():
    """Never straight onto the destination; crash matrix in test_s3_staged_push.py."""
    print("Testing push_dataset (staged save_to_disk)...")

    mock_dataset = MagicMock(spec=Dataset)
    mock_dataset.__len__ = MagicMock(return_value=100)

    with patch.object(S3Client, "__post_init__"):
        client = S3Client(bucket="test-bucket")

    client.exists = MagicMock(return_value=False)
    client.boto_client = MagicMock()
    with patch("halo.data.sources.s3_dataset_store._promote_pushed_dataset") as promote:
        s3_dataset_store.push_dataset(client, mock_dataset, "my/dataset")

    mock_dataset.save_to_disk.assert_called_once()
    promote.assert_called_once()
    call_args = mock_dataset.save_to_disk.call_args
    assert call_args[0][0].startswith("s3://test-bucket/my/.staging-dataset-"), (
        f"Expected a staging-sibling target, got {call_args[0][0]}"
    )
    assert "storage_options" in call_args[1]
    print("  push_dataset staged save_to_disk: PASSED")


def test_push_dataset_overwrite_never_deletes_the_destination_first():
    """Deleting first leaves a crash window that erases the only copy."""
    print("Testing push_dataset (overwrite is delete-free before save)...")

    mock_dataset = MagicMock(spec=Dataset)
    mock_dataset.__len__ = MagicMock(return_value=50)

    with patch.object(S3Client, "__post_init__"):
        client = S3Client(bucket="test-bucket")

    client.exists = MagicMock(return_value=True)
    client.delete = MagicMock()
    client.boto_client = MagicMock()
    deletes_before_save = []
    mock_dataset.save_to_disk = MagicMock(
        side_effect=lambda *a, **k: deletes_before_save.extend(client.boto_client.delete_object.call_args_list)
    )
    with patch("halo.data.sources.s3_dataset_store._promote_pushed_dataset") as promote:
        s3_dataset_store.push_dataset(client, mock_dataset, "existing/dataset", overwrite=True)

    client.delete.assert_not_called()
    assert deletes_before_save == [], "no destination object may be deleted before the staging save"
    mock_dataset.save_to_disk.assert_called_once()
    promote.assert_called_once()
    print("  push_dataset overwrite delete-free: PASSED")


def test_push_dataset_no_overwrite_raises_on_existing():
    mock_dataset = MagicMock(spec=Dataset)
    mock_dataset.__len__ = MagicMock(return_value=50)

    with patch.object(S3Client, "__post_init__"):
        client = S3Client(bucket="test-bucket")
    client.exists = MagicMock(return_value=True)
    client.delete = MagicMock()

    with pytest.raises(FileExistsError, match="(?i)already exists"):
        s3_dataset_store.push_dataset(client, mock_dataset, "existing/dataset", overwrite=False)

    client.delete.assert_not_called()
    mock_dataset.save_to_disk.assert_not_called()


def test_load_dataset_cache_hit():
    print("Testing load_dataset (cache hit)...")

    with patch.object(S3Client, "__post_init__"):
        client = S3Client(bucket="test-bucket")

    client.exists = MagicMock(return_value=True)

    with tempfile.TemporaryDirectory() as cache_root:
        with patch("halo.data.sources.s3_dataset_store.HALO_S3_DATASET_CACHE_DIR", cache_root):
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

            with patch("halo.data.sources.s3_dataset_store.load_from_disk") as mock_load:
                mock_load.return_value = ds
                s3_dataset_store.load_dataset(client, "my/dataset", use_cache=True)

                call_args = mock_load.call_args[0][0]
                assert call_args == cache_path, f"Expected load from cache {cache_path}, got {call_args}"

    print("  load_dataset cache hit: PASSED")


def test_load_dataset_cache_miss():
    print("Testing load_dataset (cache miss)...")

    with patch.object(S3Client, "__post_init__"):
        client = S3Client(bucket="test-bucket")

    client.exists = MagicMock(return_value=True)

    with tempfile.TemporaryDirectory() as cache_root:
        with (
            patch("halo.data.sources.s3_dataset_store.HALO_S3_DATASET_CACHE_DIR", cache_root),
            patch(
                "halo.data.sources.s3_dataset_store.download_dataset_tree", side_effect=_fake_download(["a", "b"])
            ) as download,
        ):
            loaded = s3_dataset_store.load_dataset(client, "my/dataset", use_cache=True)

        (call,) = download.call_args_list
        fetched_client, fetched_key, tmp_path = call.args
        assert (fetched_client, fetched_key) == (client, "my/dataset")
        cache_path = os.path.join(cache_root, hashlib.md5(b"test-bucket/my/dataset").hexdigest(), "dataset")
        assert tmp_path.startswith(f"{cache_path}.tmp-"), "the fetch must land beside the cache it publishes"
        assert os.path.exists(os.path.join(cache_path, ".download_complete"))
        assert loaded["text"] == ["a", "b"]

    print("  load_dataset cache miss: PASSED")


def _make_cached_dataset(cache_root: str, fingerprint: str | None) -> tuple[str, str]:
    """A complete local cache for s3://test-bucket/my/dataset; returns (cache_path, marker). ``fingerprint=None``
    writes a marker with the URI alone."""
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


def _fake_download(rows: list):
    """A ``download_dataset_tree`` stand-in that writes a real one-column tree of ``rows`` into the target."""

    def _download(client, full_key, local_path):
        Dataset.from_dict({"text": rows}).save_to_disk(local_path)

    return _download


def test_compute_etag_fingerprint():
    """Order-insensitive, content-sensitive, None on empty listing."""
    a = compute_etag_fingerprint([("k1", '"e1"', 10), ("k2", '"e2"', 20)])
    b = compute_etag_fingerprint([("k2", '"e2"', 20), ("k1", '"e1"', 10)])
    c = compute_etag_fingerprint([("k1", '"e1-changed"', 10), ("k2", '"e2"', 20)])
    assert a == b, "fingerprint must be listing-order-insensitive"
    assert a != c, "fingerprint must change when an ETag changes"
    assert compute_etag_fingerprint([]) is None


def test_content_fingerprint_none_when_unreachable():
    """Any listing error yields None, so the cache stays servable."""
    with patch.object(S3Client, "__post_init__"):
        client = S3Client(bucket="test-bucket")
    client.boto_client = MagicMock()
    client.boto_client.get_paginator.side_effect = RuntimeError("no credentials")
    assert client.content_fingerprint("my/dataset") is None


def test_content_fingerprint_lists_prefix():
    with patch.object(S3Client, "__post_init__"):
        client = S3Client(bucket="test-bucket")
    client.boto_client = MagicMock()
    pages = [{"Contents": [{"Key": "my/dataset/a", "ETag": '"e1"', "Size": 1}]}]
    client.boto_client.get_paginator.return_value.paginate.return_value = pages
    expected = compute_etag_fingerprint([("my/dataset/a", '"e1"', 1)])
    assert client.content_fingerprint("my/dataset") == expected


def test_load_dataset_stale_cache_redownloads():
    """The marker records the old fingerprint and live S3 reports a new one, so the load re-downloads and records
    it."""
    client = _make_validating_client(live_fingerprint="newfp")

    with tempfile.TemporaryDirectory() as cache_root:
        _cache_path, marker = _make_cached_dataset(cache_root, fingerprint="oldfp")

        with (
            patch("halo.data.sources.s3_dataset_store.HALO_S3_DATASET_CACHE_DIR", cache_root),
            patch(
                "halo.data.sources.s3_dataset_store.download_dataset_tree", side_effect=_fake_download(["new"])
            ) as download,
        ):
            loaded = s3_dataset_store.load_dataset(client, "my/dataset", use_cache=True)

        assert download.call_count == 1, "stale cache was served instead of re-downloaded"
        assert loaded["text"] == ["new"]
        assert _read_marker_fingerprint(marker) == "newfp"


def test_load_dataset_offline_serves_cache():
    """S3 unreachable (fingerprint probe returns None) → the complete cache is served as-is."""
    client = _make_validating_client(live_fingerprint=None)

    with tempfile.TemporaryDirectory() as cache_root:
        cache_path, marker = _make_cached_dataset(cache_root, fingerprint="oldfp")

        with patch("halo.data.sources.s3_dataset_store.HALO_S3_DATASET_CACHE_DIR", cache_root):
            with patch("halo.data.sources.s3_dataset_store.load_from_disk") as mock_load:
                mock_load.return_value = Dataset.from_dict({"text": ["hello", "world"]})
                s3_dataset_store.load_dataset(client, "my/dataset", use_cache=True)

                mock_load.assert_called_once()
                assert mock_load.call_args[0][0] == cache_path, "offline load must come from the local cache"
        assert _read_marker_fingerprint(marker) == "oldfp", "offline load must not rewrite the marker"


def test_load_dataset_matching_fingerprint_serves_cache():
    client = _make_validating_client(live_fingerprint="fp")

    with tempfile.TemporaryDirectory() as cache_root:
        cache_path, _marker = _make_cached_dataset(cache_root, fingerprint="fp")

        with patch("halo.data.sources.s3_dataset_store.HALO_S3_DATASET_CACHE_DIR", cache_root):
            with patch("halo.data.sources.s3_dataset_store.load_from_disk") as mock_load:
                mock_load.return_value = Dataset.from_dict({"text": ["hello", "world"]})
                s3_dataset_store.load_dataset(client, "my/dataset", use_cache=True)

                mock_load.assert_called_once()
                assert mock_load.call_args[0][0] == cache_path


def test_load_dataset_legacy_marker_served_and_upgraded():
    """A marker with no fingerprint stays valid and is upgraded in place when S3 is reachable, so the next load can
    detect an overwrite."""
    client = _make_validating_client(live_fingerprint="livefp")

    with tempfile.TemporaryDirectory() as cache_root:
        cache_path, marker = _make_cached_dataset(cache_root, fingerprint=None)

        with patch("halo.data.sources.s3_dataset_store.HALO_S3_DATASET_CACHE_DIR", cache_root):
            with patch("halo.data.sources.s3_dataset_store.load_from_disk") as mock_load:
                mock_load.return_value = Dataset.from_dict({"text": ["hello", "world"]})
                s3_dataset_store.load_dataset(client, "my/dataset", use_cache=True)

                mock_load.assert_called_once()
                assert mock_load.call_args[0][0] == cache_path, "legacy marker must still serve the cache"
        assert _read_marker_fingerprint(marker) == "livefp", "legacy marker was not upgraded with the live identity"


def _make_s3_loader():
    """Build a ShardedDatasetLoader instance bypassing __init__ with an S3 source."""
    with patch.object(ShardedDatasetLoader, "__init__", lambda self, **kw: None):
        loader = ShardedDatasetLoader.__new__(ShardedDatasetLoader)
    loader.bucket = "test-bucket"
    loader.key = "preprocessed/dataset"
    loader.source_type = "s3"
    loader._s3_fingerprint_unavailable = False
    loader._s3_client = None
    return loader


def test_load_shard_from_s3_caches_locally():
    """The per-shard FileLock and completion-marker cache let disjoint DP ranks download in parallel and TP/CP
    siblings reuse the cache."""
    shard = ShardInfo(id=0, path="train/shard_0000", num_examples=100, byte_size=1024)

    with tempfile.TemporaryDirectory() as cache_root:
        with (
            patch("halo.data.sources.sharded_dataset.HALO_S3_DATASET_CACHE_DIR", cache_root),
            patch("halo.data.sources.sharded_dataset.download_dataset_tree", side_effect=_fake_download([1, 2])) as dl,
        ):
            loader = _make_s3_loader()
            loader._s3_client = MagicMock()
            loader._shard_fingerprint = MagicMock(return_value="fp0")  # hermetic: no live S3 probe
            first = loader._load_shard_from_s3(shard)
            again = loader._load_shard_from_s3(shard)

        assert dl.call_count == 1, "a complete shard cache must be re-read from disk, not re-downloaded"
        fetched_client, fetched_key, tmp_path = dl.call_args.args
        assert (fetched_client, fetched_key) == (loader._s3_client, "preprocessed/dataset/train/shard_0000")
        key = hashlib.md5(b"test-bucket/preprocessed/dataset/train/shard_0000").hexdigest()
        cache_path = os.path.join(cache_root, "shards", key, "shard")
        assert tmp_path.startswith(f"{cache_path}.tmp-"), "the fetch must land beside the cache it publishes"
        assert os.path.exists(os.path.join(cache_path, ".download_complete")), "download completion marker not written"
        assert first["text"] == again["text"] == [1, 2]


def _make_cached_shard(cache_root: str, fingerprint: str | None) -> tuple[str, str]:
    key = hashlib.md5(b"test-bucket/preprocessed/dataset/train/shard_0000").hexdigest()
    cache_path = os.path.join(cache_root, "shards", key, "shard")
    os.makedirs(cache_path, exist_ok=True)
    Dataset.from_dict({"x": [1, 2, 3]}).save_to_disk(cache_path)
    marker = os.path.join(cache_path, ".download_complete")
    _write_download_marker(marker, "s3://test-bucket/preprocessed/dataset/train/shard_0000", fingerprint)
    return cache_path, marker


def test_load_shard_from_s3_stale_cache_redownloads():
    shard = ShardInfo(id=0, path="train/shard_0000", num_examples=100, byte_size=1024)

    with tempfile.TemporaryDirectory() as cache_root:
        _cache_path, marker = _make_cached_shard(cache_root, fingerprint="oldfp")

        with (
            patch("halo.data.sources.sharded_dataset.HALO_S3_DATASET_CACHE_DIR", cache_root),
            patch("halo.data.sources.sharded_dataset.download_dataset_tree", side_effect=_fake_download([9])) as dl,
        ):
            loader = _make_s3_loader()
            loader._s3_client = MagicMock()
            loader._shard_fingerprint = MagicMock(return_value="newfp")
            loaded = loader._load_shard_from_s3(shard)

        assert dl.call_count == 1, "stale shard cache was served instead of re-downloaded"
        assert loaded["text"] == [9]
        assert _read_marker_fingerprint(marker) == "newfp"


def test_load_shard_from_s3_offline_serves_cache():
    """S3 unreachable (shard fingerprint probe returns None) → the complete shard cache is served."""
    shard = ShardInfo(id=0, path="train/shard_0000", num_examples=100, byte_size=1024)

    with tempfile.TemporaryDirectory() as cache_root:
        cache_path, marker = _make_cached_shard(cache_root, fingerprint="oldfp")

        with patch("halo.data.sources.sharded_dataset.HALO_S3_DATASET_CACHE_DIR", cache_root):
            with patch("halo.data.sources.sharded_dataset.load_from_disk") as mock_load:
                mock_load.return_value = Dataset.from_dict({"x": [1, 2, 3]})

                loader = _make_s3_loader()
                loader._shard_fingerprint = MagicMock(return_value=None)
                loader._load_shard_from_s3(shard)

                mock_load.assert_called_once()
                assert mock_load.call_args[0][0] == cache_path, "offline shard load must come from the local cache"
        assert _read_marker_fingerprint(marker) == "oldfp", "offline shard load must not rewrite the marker"


def test_shard_fingerprint_skips_after_unreachable_s3(caplog):
    """An unannounced latch serves every later shard unvalidated in silence. The probe lists through
    ``S3Client.content_entries``, so shard and dataset caches compare identical fingerprints."""
    loader = _make_s3_loader()

    with patch("halo.data.sources.sharded_dataset.S3Client") as mock_client_cls:
        mock_client_cls.return_value.content_entries = MagicMock(side_effect=OSError("connection reset"))
        with caplog.at_level(logging.WARNING, logger="halo.data.sources.sharded_dataset"):
            assert loader._shard_fingerprint("preprocessed/dataset/train/shard_0000") is None
            assert loader._shard_fingerprint("preprocessed/dataset/train/shard_0001") is None
        assert mock_client_cls.return_value.content_entries.call_count == 1, (
            "second probe must be skipped after an unreachable S3"
        )

    latched = [r for r in caplog.records if "skipping the fingerprint probe" in r.getMessage()]
    assert len(latched) == 1, f"the latch must warn exactly once, naming the reason; got {len(latched)} records"
    assert "connection reset" in latched[0].getMessage(), "the warning must name the failure that disarmed the probe"


def test_shard_fingerprint_empty_listing_does_not_latch():
    """An empty listing is a real absence: latching on it would serve a whole run's shards unvalidated."""
    loader = _make_s3_loader()

    with patch("halo.data.sources.sharded_dataset.S3Client") as mock_client_cls:
        mock_client_cls.return_value.content_entries = MagicMock(return_value=[])
        assert loader._shard_fingerprint("preprocessed/dataset/train/shard_0000") is None
        assert loader._shard_fingerprint("preprocessed/dataset/train/shard_0001") is None
        assert mock_client_cls.return_value.content_entries.call_count == 2, (
            "an empty listing must not disarm fingerprinting for the rest of the process"
        )
    assert loader._s3_fingerprint_unavailable is False


def test_load_shard_index_from_s3():
    print("Testing ShardedDatasetLoader._load_shard_index (S3)...")

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

    with patch("halo.data.sources.s3_dataset_store.s3fs.S3FileSystem", return_value=mock_fs):
        with patch("halo.data.sources.s3_dataset_store.json.load", return_value=index_data):
            index = loader._load_shard_index("train")

    assert index.num_shards == 2
    assert index.total_examples == 200
    assert len(index.shards) == 2
    mock_fs.open.assert_called_once()
    print("  _load_shard_index (S3): PASSED")


def test_load_preprocessed_metadata_from_s3():
    print("Testing load_preprocessed_metadata (S3)...")

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

    with patch("halo.data.sources.s3_dataset_store.s3fs.S3FileSystem", return_value=mock_fs):
        with patch("halo.data.sources.s3_dataset_store.json.load", return_value=metadata_dict):
            metadata = load_preprocessed_metadata("s3://test-bucket/preprocessed/dataset")

    assert metadata.preprocessed is True
    assert metadata.model_name == "Qwen/Qwen3-8B"
    assert metadata.max_length == 8192
    mock_fs.open.assert_called_once()
    print("  load_preprocessed_metadata S3: PASSED")


def test_build_s3_uri(monkeypatch):
    monkeypatch.setattr(s3_client_mod, "DEFAULT_BUCKET", "team-bucket")

    def build(key, subfolder=None):
        return build_s3_uri(key, subfolder, given_as="--dataset")

    assert build("my_dataset", "datasets") == "s3://team-bucket/datasets/my_dataset"
    assert build("my_dataset", None) == "s3://team-bucket/my_dataset"
    assert build("/my_dataset/", "") == "s3://team-bucket/my_dataset"
    assert build("my_dataset", "/datasets/") == S3Client(bucket="team-bucket").s3_uri("my_dataset", "datasets")
    assert build("nested/path/data") == "s3://team-bucket/nested/path/data"


@pytest.mark.parametrize("default", ["team-bucket", None], ids=["default_set", "default_unset"])
def test_a_full_s3_uri_resolves_to_itself(monkeypatch, default):
    """Neither the default bucket nor the subfolder is prefixed onto a full URI, and an unset default bucket is no
    reason to refuse it."""
    monkeypatch.setattr(s3_client_mod, "DEFAULT_BUCKET", default)

    for subfolder in ("datasets", None):
        assert build_s3_uri("s3://my-bucket/prompts", subfolder, given_as="--dataset") == "s3://my-bucket/prompts"
    with pytest.raises(ValueError, match="Invalid S3 URI"):
        build_s3_uri("s3://my-bucket", "datasets", given_as="--dataset")


def test_a_key_only_path_without_a_configured_bucket_is_refused(monkeypatch):
    """A placeholder name would read and write a real bucket the user does not own, so both entry points refuse,
    naming the input and the way to give one."""
    monkeypatch.setattr(s3_client_mod, "DEFAULT_BUCKET", None)

    with pytest.raises(ValueError) as excinfo:
        build_s3_uri("my_dataset", given_as="--output_path")
    message = str(excinfo.value)
    assert "--output_path 'my_dataset'" in message
    assert "s3://<bucket>/<key>" in message and "HALO_S3_DEFAULT_BUCKET" in message
    assert "--bucket" not in message, "the generation CLIs take no --bucket flag"
    with pytest.raises(ValueError, match=r"S3Client: pass bucket=.*HALO_S3_DEFAULT_BUCKET"):
        S3Client()


def test_the_s3_cli_requires_a_bucket_only_for_a_bare_key_without_a_default(monkeypatch):
    """With the default unset the CLI refuses naming ``--bucket``, while a full ``s3://`` URI names its own."""

    def client_for(argv):
        return build_command_client(s3_cli_mod.build_parser().parse_args(argv))

    with patch.object(S3Client, "__post_init__"):
        monkeypatch.setattr(s3_client_mod, "DEFAULT_BUCKET", "team-bucket")
        assert client_for(["list"]).bucket == "team-bucket"

        monkeypatch.setattr(s3_client_mod, "DEFAULT_BUCKET", None)
        with pytest.raises(ValueError, match="pass --bucket"):
            client_for(["list"])
        assert client_for(["list", "--bucket", "my-bucket"]).bucket == "my-bucket"
        assert client_for(["list", "s3://example-bucket/runs"]).bucket == "example-bucket"


def test_s3_cache_dir_derives_from_scratch_root():
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


def _run_cli(
    argv: list[str], *, answer: str | None = None, exists: bool = True, object_exists: bool = True
) -> MagicMock:
    """Run the s3 CLI offline with ``delete`` mocked and return the mock. ``exists``/``object_exists`` model the
    bucket: ``object_exists=False`` with ``exists=True`` is a folder-like prefix. ``answer`` feeds the
    confirmation prompt (None: never prompted)."""
    # Patched by dotted path: the cache-dir test above reloads this module, so the name bound at import is a stale
    # class.
    with (
        patch("halo.data.sources.s3_client.S3Client.__post_init__"),
        patch("halo.data.sources.s3_client.S3Client.delete", return_value=True) as mock_delete,
        patch("halo.data.sources.s3_client.S3Client.s3_uri", return_value="s3://test-bucket/my_folder"),
        patch("halo.data.sources.s3_client.S3Client.exists", return_value=exists),
        patch("halo.data.sources.s3_client.S3Client.object_exists", return_value=object_exists),
        patch("builtins.input", return_value=answer),
        patch.object(sys, "argv", ["s3", *argv, "--bucket", "test-bucket"]),
    ):
        s3_cli_main()
    return mock_delete


def test_module_info_records_survive_the_root_warning_pin():
    """``halo`` pins the root to WARNING, so the module's own level admits its INFO lines, the run's only record of
    what data moved."""
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
    """Importing ``halo`` installed a root handler and pinned the level, against which a non-forcing ``basicConfig``
    is a silent no-op."""
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
    """``filelock`` reads 0 as fail immediately and a negative as wait forever, so a non-positive override is
    refused out loud."""
    saved = os.environ.get("HALO_S3_CACHE_LOCK_TIMEOUT_SECONDS")
    try:
        os.environ["HALO_S3_CACHE_LOCK_TIMEOUT_SECONDS"] = raw
        importlib.reload(env_mod)
        with caplog.at_level(logging.WARNING):
            s3r = importlib.reload(s3_mod)
        derived_default = int(s3r.store_timeout().total_seconds())
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
    """``store_true`` can only set True, so ``default=True`` on ``--recursive`` would make every CLI delete a
    prefix-wipe."""
    mock_delete = _run_cli(["delete", "my_folder", "--yes"])

    mock_delete.assert_called_once()
    assert mock_delete.call_args.kwargs["recursive"] is False, (
        "bare `delete <key>` deleted the whole prefix: --recursive must be opt-in"
    )


@pytest.mark.parametrize("flag", ["--recursive", "-r"])
def test_cli_delete_recursive_opts_in(flag):
    mock_delete = _run_cli(["delete", "my_folder", flag, "--yes"])

    mock_delete.assert_called_once()
    assert mock_delete.call_args.kwargs["recursive"] is True, f"{flag} did not reach delete()"


def test_cli_delete_refuses_a_prefix_without_recursive():
    """S3's DeleteObject is idempotent, returning 204 on a key that is only a prefix, so the CLI would print Deleted
    with the data still there."""
    with pytest.raises(SystemExit, match="is a prefix, not an object"):
        _run_cli(["delete", "my_folder", "--yes"], exists=True, object_exists=False)


def test_cli_delete_reports_a_missing_key():
    mock_delete = _run_cli(["delete", "my_folder", "--yes"], exists=False, object_exists=False)

    mock_delete.assert_not_called()


def test_cli_delete_recursive_still_deletes_a_prefix():
    mock_delete = _run_cli(["delete", "my_folder", "-r", "--yes"], exists=True, object_exists=False)

    mock_delete.assert_called_once()
    assert mock_delete.call_args.kwargs["recursive"] is True


def test_recursive_delete_reports_false_when_the_prefix_matched_nothing():
    """``--recursive`` on a single object sweeps ``<key>/``, which has no children, so an unconditional True prints
    Deleted over data still there."""
    with patch.object(S3Client, "__post_init__"):
        client = S3Client(bucket="test-bucket")
    paginator = MagicMock()
    paginator.paginate.return_value = [{}]  # no Contents -> nothing matched
    client.boto_client = MagicMock()
    client.boto_client.get_paginator.return_value = paginator

    assert client.delete("my_folder", recursive=True) is False
    client.boto_client.delete_objects.assert_not_called()


def test_cli_delete_declines_confirmation():
    mock_delete = _run_cli(["delete", "my_folder", "--recursive"], answer="n")

    mock_delete.assert_not_called()


def test_cli_delete_confirmed_deletes():
    mock_delete = _run_cli(["delete", "my_folder", "--recursive"], answer="y")

    mock_delete.assert_called_once()
    assert mock_delete.call_args.kwargs["recursive"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
