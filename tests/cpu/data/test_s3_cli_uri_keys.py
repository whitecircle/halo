#!/usr/bin/env python
"""``s3.py`` takes a full ``s3://bucket/key`` URI wherever it takes a key; read as a bare key it would run against
``s3://<--bucket>/s3://bucket/key``. ``--subfolder`` or a different ``--bucket`` beside one is refused. The
client is stubbed at its own methods, patched by dotted path."""

import sys
from unittest.mock import patch

import pytest

from halo.data.sources import s3_client
from scripts.before_training.s3 import main as s3_cli_main


def _run(argv: list[str], method: str, **return_value) -> tuple[list[str], dict]:
    buckets = []
    with (
        patch.object(s3_client.S3Client, "__post_init__", lambda self: buckets.append(self.bucket)),
        patch(f"halo.data.sources.s3_client.S3Client.{method}", **return_value) as called,
        patch.object(sys, "argv", ["s3", *argv]),
    ):
        s3_cli_main()
    return buckets, called.call_args.kwargs


@pytest.mark.parametrize(
    ("argv", "method", "key_kwarg", "key"),
    [
        (["download", "s3://test-bucket/runs/a/", "local"], "download_folder", "key", "runs/a"),
        (["push", "local", "s3://test-bucket/runs/a"], "push_folder", "key", "runs/a"),
        (["list", "s3://test-bucket/"], "list_objects", "prefix", ""),
        (["list", "s3://test-bucket/runs/"], "list_objects", "prefix", "runs"),
    ],
    ids=["download", "push", "list-bucket-root", "list-prefix"],
)
def test_a_uri_key_runs_against_the_bucket_it_names(argv, method, key_kwarg, key):
    buckets, kwargs = _run(argv, method, return_value=[] if method == "list_objects" else "s3://test-bucket/runs/a")
    assert buckets == ["test-bucket"]
    assert kwargs[key_kwarg] == key and kwargs["subfolder"] is None


def test_a_bare_key_still_reads_bucket():
    buckets, kwargs = _run(["download", "runs/a", "local", "--bucket", "b2"], "download_folder", return_value="x")
    assert buckets == ["b2"] and kwargs["key"] == "runs/a"


@pytest.mark.parametrize(
    "extra", [["--subfolder", "exp"], ["--bucket", "other-bucket"]], ids=["subfolder", "another-bucket"]
)
def test_a_uri_key_refuses_a_second_location(extra):
    with pytest.raises(SystemExit, match="names its bucket and key itself"):
        _run(["download", "s3://test-bucket/runs/a", "local", *extra], "download_folder", return_value="x")


def test_a_uri_without_a_key_is_refused_where_a_key_is_needed():
    with pytest.raises(SystemExit, match="is not an s3://<bucket>/<key> URI"):
        _run(["delete", "s3://test-bucket/", "--yes"], "delete", return_value=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
