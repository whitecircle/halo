#!/usr/bin/env python
"""CLI over the toolkit's S3 transport: push, download, list, exists and delete S3 folders."""

import argparse
import logging

from halo.data.sources import dataset_cache, s3_client, s3_dataset_store
from halo.data.sources.s3_commands import (
    build_command_client,
    delete_command,
    download_command,
    exists_command,
    list_command,
    push_command,
)
from halo.log import configure_cli_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="S3 utilities for uploading/downloading folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload a folder
  python scripts/before_training/s3.py push ./my_folder my_folder

  # Download a folder
  python scripts/before_training/s3.py download my_folder ./local_folder

  # List objects
  python scripts/before_training/s3.py list
  python scripts/before_training/s3.py list my_project/

  # Check if exists
  python scripts/before_training/s3.py exists my_folder

  # A full URI names its own bucket
  python scripts/before_training/s3.py download s3://my-bucket/my_folder ./local_folder

  # Delete a single object, or a whole prefix with --recursive
  python scripts/before_training/s3.py delete my_folder/data.json --yes
  python scripts/before_training/s3.py delete my_folder --recursive --yes
        """,
    )

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--subfolder", "-s", default=None, help="Optional S3 subfolder prefix (default: none)")
    common_parser.add_argument(
        "--bucket",
        "-b",
        default=None,
        help="S3 bucket of a bare key (default: $HALO_S3_DEFAULT_BUCKET); a full s3:// URI names its own",
    )
    common_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    transfer_parser = argparse.ArgumentParser(add_help=False)
    transfer_parser.add_argument("--quiet", "-q", action="store_true", help="Disable progress bar")

    subparsers = parser.add_subparsers(dest="command", required=True)

    push_parser = subparsers.add_parser(
        "push", parents=[common_parser, transfer_parser], help="Upload a local folder to S3"
    )
    push_parser.add_argument("local_path", help="Local folder path to upload")
    push_parser.add_argument("s3_key", help="S3 key/path, or s3:// URI, for the folder")
    push_parser.add_argument("--no_overwrite", action="store_true", help="Don't overwrite if exists")
    push_parser.set_defaults(func=push_command)

    download_parser = subparsers.add_parser(
        "download", parents=[common_parser, transfer_parser], help="Download a folder from S3"
    )
    download_parser.add_argument("s3_key", help="S3 key/path, or s3:// URI, of the folder")
    download_parser.add_argument("local_path", help="Local path to download to")
    download_parser.add_argument("--no_overwrite", action="store_true", help="Don't overwrite if local exists")
    download_parser.set_defaults(func=download_command)

    list_parser = subparsers.add_parser("list", parents=[common_parser], help="List objects in S3")
    list_parser.add_argument("prefix", nargs="?", default="", help="Prefix, or s3:// URI, to filter (optional)")
    list_parser.add_argument("--recursive", "-r", action="store_true", help="List recursively")
    list_parser.add_argument(
        "--max_keys", "-n", type=int, default=100, help="Maximum number of items to list (default: 100)"
    )
    list_parser.set_defaults(func=list_command)

    delete_parser = subparsers.add_parser("delete", parents=[common_parser], help="Delete an object or folder from S3")
    delete_parser.add_argument("s3_key", help="S3 key/path, or s3:// URI, to delete")
    delete_parser.add_argument("--recursive", "-r", action="store_true", help="Delete every object under the prefix")
    delete_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    delete_parser.set_defaults(func=delete_command)

    exists_parser = subparsers.add_parser(
        "exists", parents=[common_parser], help="Check if an object or folder exists"
    )
    exists_parser.add_argument("s3_key", help="S3 key/path, or s3:// URI, to check")
    exists_parser.set_defaults(func=exists_command)
    return parser


def main():
    args = build_parser().parse_args()

    configure_cli_logging(verbose=args.verbose)
    # Undo the library modules' INFO pins so ``-v`` widens the S3 code as well as its dependencies.
    level = logging.DEBUG if args.verbose else logging.INFO
    for module_logger in (s3_client.logger, s3_dataset_store.logger, dataset_cache.logger):
        module_logger.setLevel(level)

    print(args.func(args, build_command_client(args)))


if __name__ == "__main__":
    main()
