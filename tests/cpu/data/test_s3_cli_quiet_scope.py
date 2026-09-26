"""``s3_datasets.py --quiet`` exists only on the commands that draw a transfer progress bar.

``push`` and ``download`` forward it as ``show_progress=False``; ``list`` / ``exists`` / ``delete`` draw
no bar, so they refuse the flag instead of accepting it and doing nothing. The client is stubbed at its
own methods (no boto3 session), patched by dotted path like the sibling CLI tests.
"""

import sys
from unittest.mock import patch

import pytest

from scripts.before_training.s3_datasets import main as s3_cli_main


def _run(argv: list[str]) -> None:
    with patch("src.data.sources.s3_client.S3Client.__post_init__"), patch.object(sys, "argv", ["s3", *argv]):
        s3_cli_main()


@pytest.mark.parametrize(
    ("argv", "method"),
    [(["push", "local_dir", "key"], "push_folder"), (["download", "key", "local_dir"], "download_folder")],
    ids=["push", "download"],
)
@pytest.mark.parametrize("quiet", [False, True], ids=["default", "quiet"])
def test_quiet_turns_off_the_transfer_progress_bar(argv, method, quiet):
    with patch(f"src.data.sources.s3_client.S3Client.{method}", return_value="s3://b/key") as transfer:
        _run([*argv, *(["-q"] if quiet else [])])
    assert transfer.call_args.kwargs["show_progress"] is not quiet


@pytest.mark.parametrize(
    "argv",
    [["list", "-q"], ["exists", "key", "-q"], ["delete", "key", "--yes", "-q"]],
    ids=["list", "exists", "delete"],
)
def test_quiet_is_refused_where_no_progress_bar_is_drawn(argv, capsys):
    with pytest.raises(SystemExit) as excinfo:
        _run(argv)
    assert excinfo.value.code == 2, "argparse must reject the flag as unrecognized"
    assert "unrecognized arguments: -q" in capsys.readouterr().err


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
