#!/usr/bin/env python
"""Push, download, list and delete S3 folders — the CLI over the toolkit's S3 dataset transport.

Every handler acts through an :class:`~src.data.sources.s3_client.S3Client` built for the
invocation's ``--bucket``, never the module default: rebinding that singleton would hand this
invocation's bucket to every other caller in the process.

Usage:
    python scripts/before_training/s3_datasets.py <push|download|list|exists|delete> ...
"""

import argparse
import logging
import sys

from src.data.sources import dataset_cache, s3_client
from src.data.sources.s3_client import DEFAULT_BUCKET, S3Client


def _cli_push(args, client: S3Client):
    result = client.push_folder(
        local_path=args.local_path,
        key=args.s3_key,
        subfolder=args.subfolder,
        overwrite=not args.no_overwrite,
        show_progress=not args.quiet,
    )
    print(f"✓ Uploaded to {result}")


def _cli_download(args, client: S3Client):
    result = client.download_folder(
        key=args.s3_key,
        local_path=args.local_path,
        subfolder=args.subfolder,
        overwrite=not args.no_overwrite,
        show_progress=not args.quiet,
    )
    print(f"✓ Downloaded to {result}")


def _cli_list(args, client: S3Client):
    objects = client.list_objects(
        prefix=args.prefix or "",
        subfolder=args.subfolder,
        recursive=args.recursive,
        max_keys=args.max_keys,
    )

    if not objects:
        print("No objects found.")
        return

    for obj in objects:
        if obj["Type"] == "directory":
            print(f"📁 {obj['Key']}/")
        else:
            size_kb = obj["Size"] / 1024
            size_str = f"{size_kb / 1024:.1f} MB" if size_kb > 1024 else f"{size_kb:.1f} KB"
            print(f"   {obj['Key']} ({size_str})")


def _cli_delete(args, client: S3Client):
    s3_uri = client._get_s3_uri(args.s3_key, args.subfolder)

    # Both no-op deletes report success at the API: DeleteObject on a prefix is idempotent (204), and
    # the recursive path returns True when zero objects matched. Reporting "Deleted" for either would
    # tell the user their data is gone when it is not, so resolve what is actually there first.
    if not client.exists(args.s3_key, args.subfolder):
        print(f"✗ Nothing to delete at {s3_uri}")
        return
    if not args.recursive and not client.object_exists(args.s3_key, args.subfolder):
        print(f"✗ {s3_uri} is a prefix, not an object — pass --recursive to delete everything under it.")
        return

    if not args.yes:
        target = f"everything under {s3_uri}/" if args.recursive else s3_uri
        confirm = input(f"Delete {target}? [y/N]: ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return

    result = client.delete(
        key=args.s3_key,
        subfolder=args.subfolder,
        recursive=args.recursive,
    )
    if result:
        print(f"✓ Deleted {s3_uri}")
    elif args.recursive:
        # exists() passed above, so the key resolves — it is a single object with nothing beneath it.
        print(f"✗ Nothing under {s3_uri}/ — that key is a single object; drop --recursive to delete it.")
    else:
        print(f"✗ Failed to delete {s3_uri}")


def _cli_exists(args, client: S3Client):
    s3_uri = client._get_s3_uri(args.s3_key, args.subfolder)

    if client.exists(args.s3_key, args.subfolder):
        print(f"✓ {s3_uri} exists")
    else:
        print(f"✗ {s3_uri} does not exist")


def main():
    parser = argparse.ArgumentParser(
        description="S3 utilities for uploading/downloading folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload a folder
  python scripts/before_training/s3_datasets.py push ./my_folder my_folder

  # Download a folder
  python scripts/before_training/s3_datasets.py download my_folder ./local_folder

  # List objects
  python scripts/before_training/s3_datasets.py list
  python scripts/before_training/s3_datasets.py list my_project/

  # Check if exists
  python scripts/before_training/s3_datasets.py exists my_folder

  # Delete a single object, or a whole prefix with --recursive
  python scripts/before_training/s3_datasets.py delete my_folder/data.json --yes
  python scripts/before_training/s3_datasets.py delete my_folder --recursive --yes
        """,
    )

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--subfolder", "-s", default=None, help="Optional S3 subfolder prefix (default: none)")
    common_parser.add_argument(
        "--bucket", "-b", default=DEFAULT_BUCKET, help=f"S3 bucket name (default: {DEFAULT_BUCKET})"
    )
    common_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    common_parser.add_argument("--quiet", "-q", action="store_true", help="Disable progress bar")

    subparsers = parser.add_subparsers(dest="command", required=True)

    push_parser = subparsers.add_parser("push", parents=[common_parser], help="Upload a local folder to S3")
    push_parser.add_argument("local_path", help="Local folder path to upload")
    push_parser.add_argument("s3_key", help="S3 key/path for the folder")
    push_parser.add_argument("--no-overwrite", action="store_true", help="Don't overwrite if exists")
    push_parser.set_defaults(func=_cli_push)

    download_parser = subparsers.add_parser("download", parents=[common_parser], help="Download a folder from S3")
    download_parser.add_argument("s3_key", help="S3 key/path of the folder")
    download_parser.add_argument("local_path", help="Local path to download to")
    download_parser.add_argument("--no-overwrite", action="store_true", help="Don't overwrite if local exists")
    download_parser.set_defaults(func=_cli_download)

    list_parser = subparsers.add_parser("list", parents=[common_parser], help="List objects in S3")
    list_parser.add_argument("prefix", nargs="?", default="", help="Prefix to filter (optional)")
    list_parser.add_argument("--recursive", "-r", action="store_true", help="List recursively")
    list_parser.add_argument(
        "--max-keys", "-n", type=int, default=100, help="Maximum number of items to list (default: 100)"
    )
    list_parser.set_defaults(func=_cli_list)

    delete_parser = subparsers.add_parser("delete", parents=[common_parser], help="Delete an object or folder from S3")
    delete_parser.add_argument("s3_key", help="S3 key/path to delete")
    delete_parser.add_argument("--recursive", "-r", action="store_true", help="Delete every object under the prefix")
    delete_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    delete_parser.set_defaults(func=_cli_delete)

    exists_parser = subparsers.add_parser(
        "exists", parents=[common_parser], help="Check if an object or folder exists"
    )
    exists_parser.add_argument("s3_key", help="S3 key/path to check")
    exists_parser.set_defaults(func=_cli_exists)

    args = parser.parse_args()

    # The CLI is a process entry point, so it owns the root handler; as a library this module only
    # emits to its own logger and lets the training scripts configure logging. ``force`` because
    # importing ``src`` already installed one, against which a plain basicConfig is a silent no-op.
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(stream=sys.stderr, level=level, force=True)
    # Undo the library modules' INFO pins so ``-v`` genuinely widens the S3 code that emits the
    # transfer records, not only its dependencies.
    for module_logger in (s3_client.logger, dataset_cache.logger):
        module_logger.setLevel(level)

    # The client the handlers act through — passed, never installed as the module default: rebinding
    # that singleton makes every unrelated caller in the process inherit this invocation's --bucket.
    args.func(args, S3Client(bucket=args.bucket))


if __name__ == "__main__":
    main()
