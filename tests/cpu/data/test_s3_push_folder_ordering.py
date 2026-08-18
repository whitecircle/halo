#!/usr/bin/env python
"""``S3Client.push_folder`` overwrite protocol: never destroy the only copy.

Deleting the whole prefix BEFORE uploading loses the old dataset on a mid-upload failure and can
leave a metadata.json-bearing partial that the preprocessed probe accepts. The protocol: delete the
marker first, upload the bulk, upload the marker LAST, sweep stale objects only after the upload
completes.

    python tests/cpu/data/test_s3_push_folder_ordering.py
"""

import pytest

from src.data.sources.s3_client import S3Client


class _FakePaginator:
    """botocore's ``list_objects_v2`` paginator over a fixed key set.

    Pages in twos so a reader that consumed only the first page (or capped the listing) would miss
    stale objects — the sweep below deletes what the listing does NOT contain, so a truncated view
    silently keeps them.
    """

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

    def get_paginator(self, operation_name):
        assert operation_name == "list_objects_v2", operation_name
        return _FakePaginator(self._keys)


def _client_with(existing_keys):
    client = object.__new__(S3Client)
    client.bucket = "b"
    client._client = _FakeBoto(existing_keys)
    client._transfer_config = None
    client.exists = lambda key, subfolder=None: bool(existing_keys)
    client._get_full_key = lambda key, subfolder=None: key
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

    ops = client._client.ops
    assert ops[0] == ("delete", "ds/metadata.json"), "the marker must be invalidated before any byte uploads"
    uploads = [key for op, key in ops if op == "upload"]
    assert uploads[-1] == "ds/metadata.json", "the marker must be re-written only after the bulk completed"
    # The stale sweep runs only AFTER the full upload, and touches nothing the new tree carries.
    tail = ops[ops.index(("upload", "ds/metadata.json")) + 1 :]
    assert tail == [("delete", "ds/train/stale_9.arrow")]


def test_fresh_destination_uploads_without_deletes(tmp_path):
    client = _client_with(set())

    client.push_folder(_local_dataset(tmp_path), "ds")

    ops = client._client.ops
    assert all(op == "upload" for op, _ in ops), ops
    assert [key for _, key in ops][-1] == "ds/metadata.json"


def test_existing_destination_without_overwrite_refuses(tmp_path):
    client = _client_with({"ds/metadata.json"})
    with pytest.raises(FileExistsError):
        client.push_folder(_local_dataset(tmp_path), "ds", overwrite=False)
    assert client._client.ops == [], "a refusal must not have touched the destination"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
