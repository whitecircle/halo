#!/usr/bin/env python
"""``S3Client.exists`` must not swallow every ``ClientError``.

Expired credentials or a 403 answering "does not exist" makes an overwrite check pass, a load
report the dataset missing, and a delete claim there was nothing there. Only an authoritative
absence may read as False; a ``NoSuchBucket`` is authoritative about the bucket, not the key.

Run: python tests/cpu/data/test_s3_exists_error_semantics.py
"""

from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from src.data.sources.s3_client import S3Client


def _client_raising(code: str) -> S3Client:
    """S3Client whose head_object 404s (so the prefix listing is reached) and whose listing fails."""
    client = S3Client.__new__(S3Client)
    client.bucket = "test-bucket"
    client._client = MagicMock()
    client._client.head_object.side_effect = ClientError({"Error": {"Code": "404"}}, "HeadObject")
    client._client.list_objects_v2.side_effect = ClientError({"Error": {"Code": code}}, "ListObjectsV2")
    return client


def test_exists_reraises_a_credentials_error():
    """A 403/expired-SSO listing failure must raise, not report 'does not exist'."""
    with pytest.raises(ClientError):
        _client_raising("AccessDenied").exists("some/dataset")


def test_exists_returns_false_on_an_authoritative_absence():
    """A missing key is a real answer: False."""
    assert _client_raising("NoSuchKey").exists("some/dataset") is False


def test_exists_names_a_missing_bucket_rather_than_an_absent_key():
    """``NoSuchBucket`` is authoritative about the BUCKET, not the key. Read as "does not exist" it
    reaches the operator as ``Dataset not found: s3://…``, sending them after a key in a bucket that
    is not there — a misspelled bucket or the wrong region."""
    with pytest.raises(FileNotFoundError, match="test-bucket"):
        _client_raising("NoSuchBucket").exists("some/dataset")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
