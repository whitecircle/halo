#!/usr/bin/env python
"""Mirrored S3 control-file reads — sharded/preprocessed datasets must survive an S3 outage warm.

The dataset BYTES are offline-servable (completion-marker caches); the small control files training
reads first — ``shard_index.json``, ``metadata.json`` — must be too. Live-only, a warm-cache relaunch
of a sharded run during an SSO outage dies on the index fetch, and a preprocessed dataset degrades to
"raw" and dies on its stripped source columns, both against the documented offline
contract. ``read_control_json_with_cache`` closes that: live wins and refreshes a local mirror; the
mirror serves when S3 is unreachable; a live 404 is authoritative — the mirror is dropped so a
re-pushed non-sharded/raw dataset cannot keep its old classification offline.

    python tests/cpu/data/test_s3_control_json_cache.py
"""

import json
from unittest.mock import patch

import pytest

from src.data.pipeline.preprocessed_metadata import is_preprocessed_dataset
from src.data.sources.s3_client import has_control_json_mirror, read_control_json_with_cache
from src.data.sources.sharded_dataset import ShardedDatasetLoader

PAYLOAD = {"num_shards": 4, "shards": []}
OUTAGE = OSError("Could not connect to the endpoint URL")


@pytest.fixture()
def cache_root(tmp_path):
    with patch("src.data.sources.s3_client.HALO_S3_DATASET_CACHE_DIR", str(tmp_path)):
        yield tmp_path


def test_live_read_returns_payload_and_writes_the_mirror(cache_root):
    with patch("src.data.sources.s3_client.read_json_from_s3", return_value=PAYLOAD):
        assert read_control_json_with_cache("b", "d/train/shard_index.json") == PAYLOAD
    assert has_control_json_mirror("b", "d/train/shard_index.json")
    mirror_files = list((cache_root / "control").glob("*.json"))
    assert len(mirror_files) == 1
    assert json.loads(mirror_files[0].read_text()) == PAYLOAD
    assert not list((cache_root / "control").glob("*.tmp-*")), "the atomic publish must not leak temp files"


def test_outage_serves_the_mirror(cache_root):
    with patch("src.data.sources.s3_client.read_json_from_s3", return_value=PAYLOAD):
        read_control_json_with_cache("b", "d/metadata.json")
    with patch("src.data.sources.s3_client.read_json_from_s3", side_effect=OUTAGE):
        assert read_control_json_with_cache("b", "d/metadata.json") == PAYLOAD


def test_outage_without_a_mirror_reraises(cache_root):
    with patch("src.data.sources.s3_client.read_json_from_s3", side_effect=OUTAGE):
        with pytest.raises(OSError, match="endpoint"):
            read_control_json_with_cache("b", "cold/metadata.json")


def test_live_wins_over_a_stale_mirror(cache_root):
    """When S3 answers, the payload is re-fetched and the mirror refreshed — the mirror must never
    shadow a live re-push (same staleness contract as the fingerprint probes)."""
    with patch("src.data.sources.s3_client.read_json_from_s3", return_value={"v": 1}):
        read_control_json_with_cache("b", "d/x.json")
    with patch("src.data.sources.s3_client.read_json_from_s3", return_value={"v": 2}):
        assert read_control_json_with_cache("b", "d/x.json") == {"v": 2}
    with patch("src.data.sources.s3_client.read_json_from_s3", side_effect=OUTAGE):
        assert read_control_json_with_cache("b", "d/x.json") == {"v": 2}, "the mirror must hold the refresh"


def test_authoritative_absence_drops_the_mirror_and_raises(cache_root):
    """A live 404 says the control file is GONE (dataset re-pushed without it); serving the old
    mirror on the next outage would classify a now-raw dataset as preprocessed/sharded."""
    with patch("src.data.sources.s3_client.read_json_from_s3", return_value=PAYLOAD):
        read_control_json_with_cache("b", "d/metadata.json")
    with patch("src.data.sources.s3_client.read_json_from_s3", side_effect=FileNotFoundError("404")):
        with pytest.raises(FileNotFoundError):
            read_control_json_with_cache("b", "d/metadata.json")
    assert not has_control_json_mirror("b", "d/metadata.json")
    with patch("src.data.sources.s3_client.read_json_from_s3", side_effect=OUTAGE):
        with pytest.raises(OSError, match="endpoint"):
            read_control_json_with_cache("b", "d/metadata.json")


def test_unwritable_mirror_does_not_fail_the_live_read(cache_root, monkeypatch):
    """A full/read-only cache volume must cost the mirror, not the training run's control read."""
    monkeypatch.setattr("src.data.sources.s3_client.os.replace", _raise_enospc)
    with patch("src.data.sources.s3_client.read_json_from_s3", return_value=PAYLOAD):
        assert read_control_json_with_cache("b", "d/x.json") == PAYLOAD


def _raise_enospc(src, dst):
    raise OSError(28, "No space left on device")


# The consumers: the sharded probe and the preprocessed-metadata probe.


def test_sharded_probe_true_from_mirror_when_offline(cache_root):
    with patch("src.data.sources.s3_client.read_json_from_s3", return_value=PAYLOAD):
        assert ShardedDatasetLoader.is_sharded_dataset("s3://b/prepped") is True
    with patch("src.data.sources.s3_client.read_json_from_s3", side_effect=OUTAGE):
        assert ShardedDatasetLoader.is_sharded_dataset("s3://b/prepped") is True, (
            "a warm run relaunched during an outage must keep its sharded classification"
        )


def test_sharded_probe_false_and_silent_on_live_absence(cache_root):
    with patch("src.data.sources.s3_client.read_json_from_s3", side_effect=FileNotFoundError("404")):
        assert ShardedDatasetLoader.is_sharded_dataset("s3://b/raw") is False


def test_sharded_probe_false_when_offline_and_cold(cache_root):
    with patch("src.data.sources.s3_client.read_json_from_s3", side_effect=OUTAGE):
        assert ShardedDatasetLoader.is_sharded_dataset("s3://b/never-seen") is False


def test_shard_index_load_serves_the_mirror_offline(cache_root):
    index_payload = {
        "version": "1.0",
        "num_shards": 1,
        "total_examples": 3,
        "shards": [{"id": 0, "path": "train/shard_00000", "num_examples": 3}],
    }
    loader = ShardedDatasetLoader("s3://b/prepped", global_rank=0, world_size=1)
    with patch("src.data.sources.s3_client.read_json_from_s3", return_value=index_payload):
        assert loader.get_total_examples("train") == 3
    offline_loader = ShardedDatasetLoader("s3://b/prepped", global_rank=0, world_size=1)
    with patch("src.data.sources.s3_client.read_json_from_s3", side_effect=OUTAGE):
        assert offline_loader.get_total_examples("train") == 3
        assert [s.num_examples for s in offline_loader.get_assigned_shards("train")] == [3]


def test_preprocessed_probe_survives_an_outage_from_the_mirror(cache_root):
    metadata = {"preprocessed": True, "version": "1.0", "tokenizer_name": "t"}
    with patch("src.data.sources.s3_client.read_json_from_s3", return_value=metadata):
        assert is_preprocessed_dataset("s3://b/prepped") is True
    with patch("src.data.sources.s3_client.read_json_from_s3", side_effect=OUTAGE):
        assert is_preprocessed_dataset("s3://b/prepped") is True, (
            "offline the dataset must not degrade to raw — its source columns were stripped at prep"
        )


def test_preprocessed_probe_false_and_quiet_on_live_absence(cache_root):
    """An absent metadata.json is the NORMAL raw-dataset case — no mirror, no warning-tone failure."""
    with patch("src.data.sources.s3_client.read_json_from_s3", side_effect=FileNotFoundError("404")):
        assert is_preprocessed_dataset("s3://b/raw") is False
    assert not has_control_json_mirror("b", "raw/metadata.json")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
