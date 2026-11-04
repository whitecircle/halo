#!/usr/bin/env python
"""``S3Client.exists`` does not swallow every ``ClientError``: expired credentials read as absent would pass an
overwrite check and make a delete claim nothing was there. Only an authoritative absence reads as False."""

from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from halo.data.sources.s3_client import S3Client


def _client_raising(code: str) -> S3Client:
    """S3Client whose head_object 404s (so the prefix listing is reached) and whose listing fails."""
    client = S3Client.__new__(S3Client)
    client.bucket = "test-bucket"
    client.boto_client = MagicMock()
    client.boto_client.head_object.side_effect = ClientError({"Error": {"Code": "404"}}, "HeadObject")
    client.boto_client.list_objects_v2.side_effect = ClientError({"Error": {"Code": code}}, "ListObjectsV2")
    return client


def test_exists_reraises_a_credentials_error():
    with pytest.raises(ClientError):
        _client_raising("AccessDenied").exists("some/dataset")


def test_exists_returns_false_on_an_authoritative_absence():
    assert _client_raising("NoSuchKey").exists("some/dataset") is False


def test_exists_names_a_missing_bucket_rather_than_an_absent_key():
    """Authoritative about the bucket, not the key; read as absent it sends the operator after a key in a bucket
    that is not there."""
    with pytest.raises(FileNotFoundError, match="test-bucket"):
        _client_raising("NoSuchBucket").exists("some/dataset")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
