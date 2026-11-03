#!/usr/bin/env python
"""The staged-push protocol, crash point by crash point on an in-memory fake S3: a complete copy exists at the
destination or in a sealed staging tree at every instant, an interrupted promote leaves the destination loudly
unreadable, and ``load_s3_dataset`` recovers from the newest sealed tree."""

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError
from datasets import Dataset, DatasetDict

from halo.data.preprocessed.shard_index import ShardInfo
from halo.data.sources import s3_dataset_store
from halo.data.sources.dataset_cache import DOWNLOAD_COMPLETE_MARKER
from halo.data.sources.s3_client import S3Client
from halo.data.sources.s3_dataset_store import _PUSH_COMPLETE_MARKER, _PUSH_INFIX
from halo.data.sources.sharded_dataset import ShardedDatasetLoader

BUCKET = "test-bucket"
KEY = "datasets/results"


class FakeBoto:
    """Minimal in-memory S3 client: put/copy/delete/list, with an operation log for ordering asserts."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self._clock = 0
        self.mtimes: dict[str, int] = {}
        self.ops: list[tuple[str, str]] = []

    def _write(self, key: str, body: bytes) -> None:
        self._clock += 1
        self.objects[key] = body
        self.mtimes[key] = self._clock

    def put_object(self, Bucket, Key, Body=b""):
        self.ops.append(("put", Key))
        self._write(Key, Body if isinstance(Body, bytes) else Body.encode())

    def copy(self, CopySource, Bucket, Key, Config=None):
        self.ops.append(("copy", Key))
        self._write(Key, self.objects[CopySource["Key"]])

    def delete_object(self, Bucket, Key):
        self.ops.append(("delete", Key))
        self.objects.pop(Key, None)
        self.mtimes.pop(Key, None)

    def delete_objects(self, Bucket, Delete):
        for entry in Delete["Objects"]:
            self.delete_object(Bucket, entry["Key"])

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        fake = self

        class _Paginator:
            def paginate(self, Bucket, Prefix="", Delimiter=None, **kwargs):
                matching = sorted(key for key in fake.objects if key.startswith(Prefix))
                if Delimiter is None:
                    contents = [
                        {
                            "Key": key,
                            "Size": len(fake.objects[key]),
                            "ETag": f"e{fake.mtimes[key]}",
                            "LastModified": fake.mtimes[key],
                        }
                        for key in matching
                    ]
                    return [{"Contents": contents, "KeyCount": len(contents)}] if contents else [{"KeyCount": 0}]
                files, prefixes = [], []
                for key in matching:
                    rest = key[len(Prefix) :]
                    if Delimiter in rest:
                        common = Prefix + rest.split(Delimiter, 1)[0] + Delimiter
                        if common not in prefixes:
                            prefixes.append(common)
                    else:
                        files.append(
                            {
                                "Key": key,
                                "Size": len(fake.objects[key]),
                                "ETag": f"e{fake.mtimes[key]}",
                                "LastModified": fake.mtimes[key],
                            }
                        )
                return [
                    {
                        "Contents": files,
                        "CommonPrefixes": [{"Prefix": p} for p in prefixes],
                        "KeyCount": len(files) + len(prefixes),
                    }
                ]

        return _Paginator()

    def download_file(self, Bucket, Key, Filename, Config=None):
        self.ops.append(("download", Key))
        Path(Filename).write_bytes(self.objects[Key])

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404", "Message": "NotFound"}}, "HeadObject")

    def list_objects_v2(self, Bucket, Prefix="", MaxKeys=1000):
        matching = [k for k in self.objects if k.startswith(Prefix)]
        return {"KeyCount": len(matching)}


def _new_push_key(full_key: str) -> str:
    return f"{s3_dataset_store._push_scan_prefix(full_key)}{uuid.uuid4().hex[:8]}"


def _client() -> tuple[S3Client, FakeBoto]:
    with patch.object(S3Client, "__post_init__"):
        client = S3Client(bucket=BUCKET)
    fake = FakeBoto()
    client.boto_client = fake
    client.transfer_config = None
    return client, fake


def _fake_dataset_dict_save(fake: FakeBoto, rows_tag: bytes = b"new"):
    """A save_to_disk side effect writing a DatasetDict-shaped tree into the fake store."""

    def _save(uri, storage_options=None, **kwargs):
        prefix = uri[len(f"s3://{BUCKET}/") :]
        for rel in (
            "train/data-00000-of-00001.arrow",
            "train/state.json",
            "train/dataset_info.json",
            "test/data-00000-of-00001.arrow",
            "test/state.json",
            "test/dataset_info.json",
            "dataset_dict.json",
        ):
            fake._write(f"{prefix}/{rel}", rows_tag + b":" + rel.encode())

    return _save


def _seed_old_copy(fake: FakeBoto):
    for rel in (
        "train/data-00000-of-00002.arrow",
        "train/data-00001-of-00002.arrow",  # stale under the new 1-shard tree, so it must be swept
        "train/state.json",
        "train/dataset_info.json",
        "test/data-00000-of-00001.arrow",
        "test/state.json",
        "test/dataset_info.json",
        "dataset_dict.json",
    ):
        fake._write(f"{KEY}/{rel}", b"old:" + rel.encode())


def _mock_dataset(save_side_effect) -> MagicMock:
    ds = MagicMock(spec=Dataset)
    ds.__len__ = MagicMock(return_value=3)
    ds.save_to_disk = MagicMock(side_effect=save_side_effect)
    return ds


def _upload_tree(fake: FakeBoto, local_dir: Path, key: str, *, skip=lambda rel: False) -> None:
    """Write a real ``save_to_disk`` tree into the fake store under ``key``, leaving out what ``skip`` names."""
    for path in sorted(local_dir.rglob("*")):
        rel = path.relative_to(local_dir).as_posix()
        if path.is_file() and not skip(rel):
            fake._write(f"{key}/{rel}", path.read_bytes())


def _saved_dataset_dict(directory: Path, train: list[int]) -> Path:
    DatasetDict({"train": Dataset.from_dict({"x": train}), "test": Dataset.from_dict({"x": [0]})}).save_to_disk(
        directory
    )
    return directory


def _published_markers(cache_root: Path) -> list[Path]:
    return list(cache_root.rglob(DOWNLOAD_COMPLETE_MARKER))


def _dest_keys(fake: FakeBoto) -> set[str]:
    return {k[len(KEY) + 1 :] for k in fake.objects if k.startswith(f"{KEY}/")}


def _staging_keys(fake: FakeBoto) -> set[str]:
    return {k for k in fake.objects if _PUSH_INFIX in k}


def test_successful_push_replaces_the_tree_and_sweeps_stale_and_staging():
    client, fake = _client()
    _seed_old_copy(fake)
    s3_dataset_store.push_dataset(client, _mock_dataset(_fake_dataset_dict_save(fake)), KEY)

    assert _dest_keys(fake) == {
        "train/data-00000-of-00001.arrow",
        "train/state.json",
        "train/dataset_info.json",
        "test/data-00000-of-00001.arrow",
        "test/state.json",
        "test/dataset_info.json",
        "dataset_dict.json",
    }
    assert all(body.startswith(b"new:") for k, body in fake.objects.items() if k.startswith(f"{KEY}/")), (
        "a swept push must leave no old bytes behind"
    )
    assert not _staging_keys(fake), "a completed promote must remove every staging tree"


def test_promote_ordering_gates_last_and_sweep_after_gates():
    client, fake = _client()
    _seed_old_copy(fake)
    s3_dataset_store.push_dataset(client, _mock_dataset(_fake_dataset_dict_save(fake)), KEY)

    dest_ops = [(op, key[len(KEY) + 1 :]) for op, key in fake.ops if key.startswith(f"{KEY}/")]
    gate_deletes = [
        i
        for i, (op, rel) in enumerate(dest_ops)
        if op == "delete" and rel.endswith(("state.json", "dataset_dict.json"))
    ]
    bulk_copies = [
        i
        for i, (op, rel) in enumerate(dest_ops)
        if op == "copy" and not rel.endswith(("state.json", "dataset_dict.json"))
    ]
    state_copies = [i for i, (op, rel) in enumerate(dest_ops) if op == "copy" and rel.endswith("state.json")]
    dict_copy = [i for i, (op, rel) in enumerate(dest_ops) if op == "copy" and rel == "dataset_dict.json"]
    sweep_deletes = [
        i for i, (op, rel) in enumerate(dest_ops) if op == "delete" and rel == "train/data-00001-of-00002.arrow"
    ]

    assert max(gate_deletes) < min(bulk_copies), "every destination gate must be deleted before the first copy"
    assert max(bulk_copies) < min(state_copies), "state.json must copy only after the whole bulk"
    assert dict_copy and max(state_copies) < dict_copy[0], "dataset_dict.json must be the last gate to land"
    assert sweep_deletes and dict_copy[0] < min(sweep_deletes), "the stale sweep must wait for the gates"


def test_crash_during_staging_upload_leaves_the_destination_untouched():
    client, fake = _client()
    _seed_old_copy(fake)
    before = dict(fake.objects)

    def _boom(uri, storage_options=None, **kwargs):
        raise OSError("connection reset mid-upload")

    with pytest.raises(OSError, match="mid-upload"):
        s3_dataset_store.push_dataset(client, _mock_dataset(_boom), KEY)

    assert {k: v for k, v in fake.objects.items() if k.startswith(f"{KEY}/")} == before, (
        "the old copy must survive a staging-upload crash byte-for-byte"
    )
    assert not any(k.endswith(_PUSH_COMPLETE_MARKER) for k in fake.objects), (
        "an unfinished staging tree must not be sealed"
    )


def test_crash_mid_promote_keeps_a_sealed_staging_tree_and_a_loud_destination():
    client, fake = _client()
    _seed_old_copy(fake)

    original_copy = fake.copy
    calls = {"n": 0}

    def _failing_copy(CopySource, Bucket, Key, Config=None):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("connection reset mid-promote")
        original_copy(CopySource, Bucket, Key, Config)

    fake.copy = _failing_copy
    with pytest.raises(OSError, match="mid-promote"):
        s3_dataset_store.push_dataset(client, _mock_dataset(_fake_dataset_dict_save(fake)), KEY)

    assert "dataset_dict.json" not in _dest_keys(fake), (
        "an interrupted promote must leave the destination UNREADABLE (gate missing), not old-mixed-new"
    )
    sealed = [k for k in fake.objects if k.endswith(_PUSH_COMPLETE_MARKER)]
    assert sealed, "the sealed staging tree must survive as the complete copy"
    assert client.exists(KEY), (
        "the destination prefix must never empty out — exists()=False is the silent-regeneration path"
    )
    assert "train/data-00001-of-00002.arrow" in _dest_keys(fake), "the sweep must not have run"


def test_next_push_after_a_crash_heals_and_clears_orphans():
    client, fake = _client()
    _seed_old_copy(fake)
    original_copy = fake.copy
    calls = {"n": 0}

    def _failing_copy(CopySource, Bucket, Key, Config=None):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("mid-promote")
        original_copy(CopySource, Bucket, Key, Config)

    fake.copy = _failing_copy
    with pytest.raises(OSError):
        s3_dataset_store.push_dataset(client, _mock_dataset(_fake_dataset_dict_save(fake)), KEY)
    fake.copy = original_copy

    s3_dataset_store.push_dataset(client, _mock_dataset(_fake_dataset_dict_save(fake, b"new2")), KEY)
    assert not _staging_keys(fake), "the healing push must clear the crashed push's staging orphan too"
    assert all(body.startswith(b"new2:") for k, body in fake.objects.items() if k.startswith(f"{KEY}/"))


def test_no_overwrite_raises_before_any_write():
    client, fake = _client()
    _seed_old_copy(fake)
    ds = _mock_dataset(_fake_dataset_dict_save(fake))
    with pytest.raises(FileExistsError, match="(?i)already exists"):
        s3_dataset_store.push_dataset(client, ds, KEY, overwrite=False)
    ds.save_to_disk.assert_not_called()
    assert not _staging_keys(fake)


def test_staging_sibling_never_pollutes_the_destination_namespace():
    """The staging tree lives outside the destination's anchored prefix: listings, exists(), the fingerprint and
    recursive delete are blind to it."""
    client, fake = _client()
    _seed_old_copy(fake)
    fingerprint_before = client.content_fingerprint(KEY)

    staging_key = _new_push_key(KEY)
    _fake_dataset_dict_save(fake)(f"s3://{BUCKET}/{staging_key}")
    fake.put_object(Bucket=BUCKET, Key=f"{staging_key}/{_PUSH_COMPLETE_MARKER}", Body=KEY.encode())

    assert client.content_fingerprint(KEY) == fingerprint_before
    listed = {obj["FullKey"] for obj in client.list_objects(KEY)}
    assert listed and not any(_PUSH_INFIX in key for key in listed), (
        f"an anchored listing of {KEY} surfaced staging objects: {sorted(listed)}"
    )
    client.delete(KEY, recursive=True)
    assert not client.exists(KEY)
    assert _staging_keys(fake), "deleting the dataset must not touch its staging sibling"


def test_sibling_dataset_staging_is_never_swept_or_recovered_across():
    """The scan prefix for ``.../results`` is a string prefix of sibling ``.../results-2``'s staging trees.
    Unanchored, a push sweeps the sibling's sealed tree and recovery can serve its rows."""
    client, fake = _client()
    _seed_old_copy(fake)
    sibling_key = f"{KEY}-2"
    sibling_staging = _new_push_key(sibling_key)
    _fake_dataset_dict_save(fake, b"sib")(f"s3://{BUCKET}/{sibling_staging}")
    fake.put_object(Bucket=BUCKET, Key=f"{sibling_staging}/{_PUSH_COMPLETE_MARKER}", Body=sibling_key.encode())

    # The sibling's sealed tree is the only sealed staging around, and recovery for KEY must not take it.
    assert s3_dataset_store._latest_complete_push(client, KEY) is None
    assert s3_dataset_store._latest_complete_push(client, sibling_key) == sibling_staging

    s3_dataset_store.push_dataset(client, _mock_dataset(_fake_dataset_dict_save(fake)), KEY)
    assert f"{sibling_staging}/{_PUSH_COMPLETE_MARKER}" in fake.objects, (
        "a push of one dataset must never sweep a hyphen-sibling dataset's staging trees"
    )


def test_latest_complete_push_picks_the_newest_sealed_tree():
    client, fake = _client()
    older = _new_push_key(KEY)
    _fake_dataset_dict_save(fake)(f"s3://{BUCKET}/{older}")
    fake.put_object(Bucket=BUCKET, Key=f"{older}/{_PUSH_COMPLETE_MARKER}", Body=b"")
    unsealed = _new_push_key(KEY)
    _fake_dataset_dict_save(fake)(f"s3://{BUCKET}/{unsealed}")
    newer = _new_push_key(KEY)
    _fake_dataset_dict_save(fake)(f"s3://{BUCKET}/{newer}")
    fake.put_object(Bucket=BUCKET, Key=f"{newer}/{_PUSH_COMPLETE_MARKER}", Body=b"")

    assert s3_dataset_store._latest_complete_push(client, KEY) == newer


def test_recover_from_pending_push_loads_the_sealed_tree_and_guards_recursion(monkeypatch):
    client, fake = _client()
    sealed = _new_push_key(KEY)
    fake._write(f"{KEY}/train/data-00000-of-00001.arrow", b"torn")  # destination torn: data, no gates
    _fake_dataset_dict_save(fake)(f"s3://{BUCKET}/{sealed}")
    fake.put_object(Bucket=BUCKET, Key=f"{sealed}/{_PUSH_COMPLETE_MARKER}", Body=b"")

    recovered = MagicMock(spec=Dataset)
    loaded_keys: list[str] = []

    def _spy_load(client, key, subfolder=None, keep_in_memory=None, use_cache=True):
        loaded_keys.append(key)
        return recovered

    monkeypatch.setattr(s3_dataset_store, "load_s3_dataset", _spy_load)
    cause = FileNotFoundError("no state.json")
    assert s3_dataset_store._recover_from_pending_push(client, KEY, None, True, cause=cause) is recovered
    assert loaded_keys == [sealed], "recovery must load exactly the sealed staging tree"

    # A load already targeting a staging tree must not recover into another: one level only.
    assert s3_dataset_store._recover_from_pending_push(client, sealed, None, True, cause=cause) is None
    fake.delete_object(Bucket=BUCKET, Key=f"{sealed}/{_PUSH_COMPLETE_MARKER}")
    assert s3_dataset_store._recover_from_pending_push(client, KEY, None, True, cause=cause) is None


def test_load_dataset_falls_back_to_staging_on_a_torn_destination(tmp_path):
    client, fake = _client()
    fake._write(f"{KEY}/train/data-00000-of-00001.arrow", b"torn")
    recovered = MagicMock(spec=Dataset)
    torn = FileNotFoundError("no state.json")
    with (
        patch("halo.data.sources.s3_dataset_store._recover_from_pending_push", return_value=recovered),
        patch("halo.data.sources.s3_dataset_store.HALO_S3_DATASET_CACHE_DIR", str(tmp_path)),
        patch("halo.data.sources.dataset_cache.publish_cached_download", side_effect=torn),
    ):
        assert s3_dataset_store.load_s3_dataset(client, KEY) is recovered

    with (
        patch("halo.data.sources.s3_dataset_store._recover_from_pending_push", return_value=None),
        patch("halo.data.sources.s3_dataset_store.HALO_S3_DATASET_CACHE_DIR", str(tmp_path)),
        patch("halo.data.sources.dataset_cache.publish_cached_download", side_effect=torn),
    ):
        with pytest.raises(FileNotFoundError, match="state.json"):
            s3_dataset_store.load_s3_dataset(client, KEY)


def test_load_dataset_downloads_the_tree_straight_into_the_cache(tmp_path):
    """The fetch is the client's own download into the cache's publish temp, every object once, with no
    datasets-side copy."""
    client, fake = _client()
    _upload_tree(fake, _saved_dataset_dict(tmp_path / "source", train=[1, 2, 3]), KEY)
    cache_root = tmp_path / "cache"
    with patch("halo.data.sources.s3_dataset_store.HALO_S3_DATASET_CACHE_DIR", str(cache_root)):
        loaded = s3_dataset_store.load_s3_dataset(client, KEY)

    assert loaded["train"]["x"] == [1, 2, 3]
    assert sorted(key for op, key in fake.ops if op == "download") == sorted(fake.objects)
    (marker,) = _published_markers(cache_root)
    assert Path(loaded["train"].cache_files[0]["filename"]).is_relative_to(marker.parent)


def test_folder_marker_objects_are_not_downloaded_as_files(tmp_path):
    """The S3 console's Create folder leaves zero-byte keys ending in /, which written as files would fail the
    download."""
    client, fake = _client()
    _upload_tree(fake, _saved_dataset_dict(tmp_path / "source", train=[4]), KEY)
    for marker in (f"{KEY}/", f"{KEY}/train/"):
        fake._write(marker, b"")
    with patch("halo.data.sources.s3_dataset_store.HALO_S3_DATASET_CACHE_DIR", str(tmp_path / "cache")):
        assert s3_dataset_store.load_s3_dataset(client, KEY)["train"]["x"] == [4]


def test_a_torn_destination_is_never_published_and_recovers_from_the_sealed_push(tmp_path):
    """A destination missing its load gates must raise inside the fetch, before the cache marks it complete,
    leave no partial download behind, and hand the load to the newest sealed push."""
    client, fake = _client()
    source = _saved_dataset_dict(tmp_path / "source", train=[7, 8])
    gate_names = {"dataset_dict.json", "state.json"}
    _upload_tree(fake, source, KEY, skip=lambda rel: rel.rsplit("/", 1)[-1] in gate_names)
    cache_root = tmp_path / "cache"

    with patch("halo.data.sources.s3_dataset_store.HALO_S3_DATASET_CACHE_DIR", str(cache_root)):
        with pytest.raises(FileNotFoundError):
            s3_dataset_store.load_s3_dataset(client, KEY)
        assert _published_markers(cache_root) == []
        assert [p for p in cache_root.rglob("*") if ".tmp-" in p.name] == [], "a failed fetch left its partial tree"

        sealed = _new_push_key(KEY)
        _upload_tree(fake, source, sealed)
        fake.put_object(Bucket=BUCKET, Key=f"{sealed}/{_PUSH_COMPLETE_MARKER}", Body=KEY.encode())
        recovered = s3_dataset_store.load_s3_dataset(client, KEY)

    assert recovered["train"]["x"] == [7, 8]


def test_a_shard_downloads_straight_into_its_own_cache(tmp_path):
    client, fake = _client()
    shard = ShardInfo(id=0, path="train/shard_0000", num_examples=2)
    Dataset.from_dict({"x": [5, 6]}).save_to_disk(tmp_path / "shard")
    _upload_tree(fake, tmp_path / "shard", f"{KEY}/{shard.path}")
    loader = ShardedDatasetLoader(f"s3://{BUCKET}/{KEY}", dp_rank=0, dp_size=1)
    loader._s3_client = client
    cache_root = tmp_path / "cache"
    with patch("halo.data.sources.sharded_dataset.HALO_S3_DATASET_CACHE_DIR", str(cache_root)):
        loaded = loader._load_shard_from_s3(shard)

    assert loaded["x"] == [5, 6]
    assert sorted(key for op, key in fake.ops if op == "download") == sorted(fake.objects)
    (marker,) = _published_markers(cache_root)
    assert Path(loaded.cache_files[0]["filename"]).is_relative_to(marker.parent)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
