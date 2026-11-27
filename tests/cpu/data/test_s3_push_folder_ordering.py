#!/usr/bin/env python
"""``S3Client.push_folder`` overwrite protocol: delete the marker first, upload the bulk, sweep stale objects, upload
the marker last. Deleting the prefix first loses the old dataset on a mid-upload failure."""

import pytest

from halo.data.sources.s3_client import S3Client


class _FakePaginator:
    """``list_objects_v2`` paginator over a fixed key set, paging in twos so a truncated view of the listing would
    keep stale objects."""

    _PAGE_SIZE = 2

    def __init__(self, keys: list[str]):
        self._keys = keys

    def paginate(self, Bucket, Prefix):
        matching = [{"Key": key} for key in self._keys if key.startswith(Prefix)]
        if not matching:
            yield {}  # an empty page carries no "Contents" key at all
            return
        for start in range(0, len(matching), self._PAGE_SIZE):
            yield {"Contents": matching[start : start + self._PAGE_SIZE]}


class _FakeBoto:
    """Records the operation order push_folder drives against the raw client."""

    def __init__(self, existing_keys=()):
        self.ops: list[tuple[str, str]] = []
        self._keys = sorted(existing_keys)

    def upload_file(self, local_file, bucket, key, Config=None):
        self.ops.append(("upload", key))

    def delete_object(self, Bucket, Key):
        self.ops.append(("delete", Key))

    def delete_objects(self, Bucket, Delete):
        self.ops.append(("delete_batch", tuple(entry["Key"] for entry in Delete["Objects"])))
        return {"Deleted": Delete["Objects"]}

    def get_paginator(self, operation_name):
        assert operation_name == "list_objects_v2", operation_name
        return _FakePaginator(self._keys)


def _client_with(existing_keys):
    client = object.__new__(S3Client)
    client.bucket = "b"
    client.boto_client = _FakeBoto(existing_keys)
    client.transfer_config = None
    client.exists = lambda key, subfolder=None: bool(existing_keys)
    client.full_key = lambda key, subfolder=None: key
    return client


def _local_dataset(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "shard_0.arrow").write_bytes(b"x")
    (tmp_path / "metadata.json").write_text("{}")
    return str(tmp_path)


def test_overwrite_deletes_marker_first_and_uploads_it_last(tmp_path):
    existing = {"ds/metadata.json", "ds/train/shard_0.arrow", "ds/train/stale_9.arrow"}
    client = _client_with(existing)

    client.push_folder(_local_dataset(tmp_path), "ds")

    ops = client.boto_client.ops
    assert ops[0] == ("delete", "ds/metadata.json"), "the marker must be invalidated before any byte uploads"
    assert ops[-1] == ("upload", "ds/metadata.json"), "the marker must be re-written last"
    # The stale sweep runs after the bulk upload, in one request, touching nothing the new tree carries.
    assert ops[-2] == ("delete_batch", ("ds/train/stale_9.arrow",)), ops
    assert ("upload", "ds/train/shard_0.arrow") in ops[1:-2]


def test_a_push_killed_during_the_sweep_leaves_no_marker_over_the_old_rows(tmp_path):
    """Uploaded before the sweep, the marker would stand beside the old shard index if a kill hits the first stale
    delete, and a download of the prefix would load the old rows."""
    client = _client_with({"ds/metadata.json", "ds/train/shard_0.arrow", "ds/train/shard_index.json"})

    def killed(Bucket, Delete):
        raise KeyboardInterrupt

    client.boto_client.delete_objects = killed
    with pytest.raises(KeyboardInterrupt):
        client.push_folder(_local_dataset(tmp_path), "ds")
    assert ("upload", "ds/metadata.json") not in client.boto_client.ops


def test_fresh_destination_uploads_without_deletes(tmp_path):
    client = _client_with(set())

    client.push_folder(_local_dataset(tmp_path), "ds")

    ops = client.boto_client.ops
    assert all(op == "upload" for op, _ in ops), ops
    assert [key for _, key in ops][-1] == "ds/metadata.json"


def test_existing_destination_without_overwrite_refuses(tmp_path):
    client = _client_with({"ds/metadata.json"})
    with pytest.raises(FileExistsError):
        client.push_folder(_local_dataset(tmp_path), "ds", overwrite=False)
    assert client.boto_client.ops == [], "a refusal must not have touched the destination"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
