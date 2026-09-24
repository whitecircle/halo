#!/usr/bin/env python
"""The ``s3_datasets.py delete`` CLI exits non-zero whenever the data it was asked to delete is still there.

``S3Client.delete`` logs the ``ClientError`` and returns ``False`` for a single-object delete, and the
two ``--recursive`` mismatches (a prefix without it, a single object with it) refuse before deleting
anything; a CLI that only printed and exited 0 would let a ``&&`` chain proceed as if the data were
gone. The client is stubbed at its own methods (no boto3 session), patched by dotted path like the
sibling CLI tests.

Run: pytest tests/cpu/data/test_s3_cli_delete_exit_status.py
"""

import sys
from unittest.mock import patch

import pytest

from scripts.before_training.s3_datasets import main as s3_cli_main

_URI = "s3://test-bucket/my_folder/a.json"


def _run_delete(delete_result: bool, *, is_object: bool = True, recursive: bool = False) -> None:
    argv = ["s3", "delete", "my_folder/a.json", "--yes", *(["--recursive"] if recursive else [])]
    with (
        patch("src.data.sources.s3_client.S3Client.__post_init__"),
        patch("src.data.sources.s3_client.S3Client.delete", return_value=delete_result),
        patch("src.data.sources.s3_client.S3Client._get_s3_uri", return_value=_URI),
        patch("src.data.sources.s3_client.S3Client.exists", return_value=True),
        patch("src.data.sources.s3_client.S3Client.object_exists", return_value=is_object),
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
