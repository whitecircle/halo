#!/usr/bin/env python
"""The ``s3.py delete`` CLI exits non-zero whenever the data it was asked to delete is still there, so a ``&&``
chain cannot proceed as if it were gone. The client is stubbed at its own methods (no boto3 session), patched by
dotted path."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.before_training.s3 import main as s3_cli_main

_URI = "s3://test-bucket/my_folder/a.json"


def _run_delete(delete_result: bool, *, is_object: bool = True, recursive: bool = False) -> None:
    argv = ["s3", "delete", "my_folder/a.json", "--yes", "--bucket", "test-bucket", *(["-r"] if recursive else [])]
    with (
        patch("halo.data.sources.s3_client.S3Client.__post_init__"),
        patch("halo.data.sources.s3_client.S3Client.delete", return_value=delete_result),
        patch("halo.data.sources.s3_client.S3Client.s3_uri", return_value=_URI),
        patch("halo.data.sources.s3_client.S3Client.exists", return_value=True),
        patch("halo.data.sources.s3_client.S3Client.object_exists", return_value=is_object),
        patch.object(sys, "argv", argv),
    ):
        s3_cli_main()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"delete_result": False}, f"Failed to delete {_URI}"),
        ({"delete_result": True, "is_object": False}, "is a prefix, not an object"),
        ({"delete_result": False, "recursive": True}, "that key is a single object"),
    ],
    ids=["refused_delete", "prefix_without_recursive", "object_with_recursive"],
)
def test_a_delete_that_leaves_the_data_exits_non_zero(kwargs, message):
    with pytest.raises(SystemExit) as excinfo:
        _run_delete(**kwargs)
    assert excinfo.value.code not in (0, None), "the data is still there, so the exit status must not be zero"
    assert message in str(excinfo.value.code)


def test_a_successful_delete_returns_normally():
    _run_delete(delete_result=True)


class _RefusingBoto:
    """Lists two objects under the prefix and refuses to delete either, as ``DeleteObjects`` does: success with an
    error list."""

    def get_paginator(self, name):
        return SimpleNamespace(
            paginate=lambda **kwargs: [{"Contents": [{"Key": "my_folder/a"}, {"Key": "my_folder/b"}]}]
        )

    def delete_objects(self, Bucket, Delete):
        return {"Errors": [{"Key": e["Key"], "Code": "AccessDenied", "Message": "denied"} for e in Delete["Objects"]]}


def test_a_recursive_delete_s3_refuses_raises_instead_of_reporting_success():
    def post_init(client):
        client.boto_client = _RefusingBoto()

    argv = ["s3", "delete", "my_folder", "--recursive", "--yes", "--bucket", "test-bucket"]
    with (
        patch("halo.data.sources.s3_client.S3Client.__post_init__", post_init),
        patch("halo.data.sources.s3_client.S3Client.exists", return_value=True),
        patch.object(sys, "argv", argv),
        pytest.raises(PermissionError, match=r"refused to delete 2 of 2 .* my_folder/a \(AccessDenied"),
    ):
        s3_cli_main()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
