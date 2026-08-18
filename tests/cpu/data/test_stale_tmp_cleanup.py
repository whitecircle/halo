"""CPU tests for stale cache publish-temp cleanup (crashed-writer orphans).

The S3 dataset cache and the sharded-dataset shard cache publish downloads via a unique
``<cache_path>.tmp-<uuid>`` dir + atomic rename. A writer that dies between ``save_to_disk`` and the
rename leaks its temp dir forever. ``_cleanup_stale_publish_tmp_dirs`` reclaims those at publish time:
only siblings matching the exact temp pattern AND older than the age threshold are removed — a fresh
temp dir may belong to a live writer and must be kept.
"""

import hashlib
import os
import sys
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from src.data.shard_index import ShardInfo
from src.data.sources.dataset_cache import _cleanup_stale_publish_tmp_dirs, _write_download_marker
from src.data.sources.s3_client import S3Client
from src.data.sources.sharded_dataset import ShardedDatasetLoader

_DAY = 24 * 60 * 60


def _make_tmp_dir(cache_path: str, *, age_seconds: float, name: str | None = None) -> str:
    """Create a publish-temp sibling of cache_path with the given mtime age."""
    path = name if name is not None else f"{cache_path}.tmp-{uuid.uuid4().hex}"
    os.makedirs(path)
    with open(os.path.join(path, "data.arrow"), "w") as f:
        f.write("partial download")
    stamp = time.time() - age_seconds
    os.utime(path, (stamp, stamp))
    return path


def test_stale_temp_dir_removed_fresh_kept(tmp_path):
    cache_path = os.path.join(tmp_path, "dataset")
    stale = _make_tmp_dir(cache_path, age_seconds=25 * 3600)
    fresh = _make_tmp_dir(cache_path, age_seconds=60)

    removed = _cleanup_stale_publish_tmp_dirs(cache_path)

    assert removed == [stale]
    assert not os.path.exists(stale)
    assert os.path.exists(fresh), "a fresh temp dir may belong to a live writer and must be kept"


def test_non_matching_names_kept(tmp_path):
    """Only the exact ``<basename>.tmp-<32 hex>`` pattern is eligible — never the published cache,
    other bases, or malformed temp names."""
    cache_path = os.path.join(tmp_path, "dataset")
    published = _make_tmp_dir(cache_path, age_seconds=2 * _DAY, name=cache_path)
    other_base = _make_tmp_dir(cache_path, age_seconds=2 * _DAY, name=f"{cache_path}_v2.tmp-{uuid.uuid4().hex}")
    malformed = _make_tmp_dir(cache_path, age_seconds=2 * _DAY, name=f"{cache_path}.tmp-notahexuuid")
    prefixed = _make_tmp_dir(cache_path, age_seconds=2 * _DAY, name=f"{cache_path}.tmp-{uuid.uuid4().hex}-extra")

    removed = _cleanup_stale_publish_tmp_dirs(cache_path)

    assert removed == []
    for path in (published, other_base, malformed, prefixed):
        assert os.path.exists(path)


def test_stale_matching_file_kept(tmp_path):
    """The pattern targets directories; a stray matching FILE is left alone."""
    cache_path = os.path.join(tmp_path, "dataset")
    file_path = f"{cache_path}.tmp-{uuid.uuid4().hex}"
    with open(file_path, "w") as f:
        f.write("x")
    stamp = time.time() - 2 * _DAY
    os.utime(file_path, (stamp, stamp))

    assert _cleanup_stale_publish_tmp_dirs(cache_path) == []
    assert os.path.exists(file_path)


def test_missing_parent_is_noop(tmp_path):
    assert _cleanup_stale_publish_tmp_dirs(os.path.join(tmp_path, "nope", "dataset")) == []


def test_custom_age_threshold(tmp_path):
    cache_path = os.path.join(tmp_path, "dataset")
    temp = _make_tmp_dir(cache_path, age_seconds=2 * 3600)

    assert _cleanup_stale_publish_tmp_dirs(cache_path, max_age_seconds=_DAY) == []
    assert os.path.exists(temp)
    assert _cleanup_stale_publish_tmp_dirs(cache_path, max_age_seconds=3600) == [temp]
    assert not os.path.exists(temp)


# Both cache loaders reclaim orphans while holding their publish lock.


def test_s3_load_dataset_reclaims_stale_temp(tmp_path):
    """S3Client.load_dataset removes a crashed writer's stale temp sibling even on a cache hit."""
    with patch.object(S3Client, "__post_init__"):
        client = S3Client(bucket="test-bucket")
    client.exists = MagicMock(return_value=True)
    client.content_fingerprint = MagicMock(return_value=None)  # offline → serve cache as-is

    cache_path = os.path.join(tmp_path, hashlib.md5(b"test-bucket/my/dataset").hexdigest(), "dataset")
    os.makedirs(cache_path)
    ds = Dataset.from_dict({"text": ["hello"]})
    ds.save_to_disk(cache_path)
    _write_download_marker(os.path.join(cache_path, ".download_complete"), "s3://test-bucket/my/dataset", None)
    stale = _make_tmp_dir(cache_path, age_seconds=2 * _DAY)
    fresh = _make_tmp_dir(cache_path, age_seconds=60)

    with patch("src.data.sources.s3_client.HALO_S3_DATASET_CACHE_DIR", str(tmp_path)):
        with patch("src.data.sources.s3_client.load_from_disk", return_value=ds):
            client.load_dataset("my/dataset")

    assert not os.path.exists(stale)
    assert os.path.exists(fresh)


def test_sharded_shard_load_reclaims_stale_temp(tmp_path, monkeypatch):
    """ShardedDatasetLoader._load_shard_from_s3 removes a crashed writer's stale temp sibling."""
    loader = ShardedDatasetLoader("s3://test-bucket/prepped", global_rank=0, world_size=1)
    monkeypatch.setattr(ShardedDatasetLoader, "_shard_fingerprint", lambda self, uri: None)

    shard = ShardInfo(id=0, path="train/shard_00000", num_examples=1)
    cache_key = hashlib.md5(f"test-bucket/prepped/{shard.path}".encode()).hexdigest()
    cache_path = os.path.join(tmp_path, "shards", cache_key, "shard")
    os.makedirs(cache_path)
    ds = Dataset.from_dict({"text": ["hello"]})
    ds.save_to_disk(cache_path)
    _write_download_marker(os.path.join(cache_path, ".download_complete"), "s3://test-bucket/prepped", None)
    stale = _make_tmp_dir(cache_path, age_seconds=2 * _DAY)
    fresh = _make_tmp_dir(cache_path, age_seconds=60)

    monkeypatch.setattr("src.data.sources.sharded_dataset.HALO_S3_DATASET_CACHE_DIR", str(tmp_path))
    with patch("src.data.sources.sharded_dataset.load_from_disk", return_value=ds) as mock_load:
        loaded = loader._load_shard_from_s3(shard)

    assert len(loaded) == 1
    assert mock_load.call_args[0][0] == cache_path  # served from the local cache
    assert not os.path.exists(stale)
    assert os.path.exists(fresh)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
