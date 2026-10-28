"""The S3 folder commands ``scripts/before_training/s3.py`` runs — push, download, list, exists and delete — each
through the :class:`~halo.data.sources.s3_client.S3Client` built for ``--bucket`` and returning the report the CLI
prints."""

from halo.data.sources.s3_client import S3Client


def push_command(args, client: S3Client) -> str:
    """Upload ``args.local_path`` to ``args.s3_key``."""
    result = client.push_folder(
        local_path=args.local_path,
        key=args.s3_key,
        subfolder=args.subfolder,
        overwrite=not args.no_overwrite,
        show_progress=not args.quiet,
    )
    return f"✓ Uploaded to {result}"


def download_command(args, client: S3Client) -> str:
    """Download ``args.s3_key`` to ``args.local_path``."""
    result = client.download_folder(
        key=args.s3_key,
        local_path=args.local_path,
        subfolder=args.subfolder,
        overwrite=not args.no_overwrite,
        show_progress=not args.quiet,
    )
    return f"✓ Downloaded to {result}"


def list_command(args, client: S3Client) -> str:
    """The objects and directories under ``args.prefix``, one per line."""
    objects = client.list_objects(
        prefix=args.prefix,
        subfolder=args.subfolder,
        recursive=args.recursive,
        max_keys=args.max_keys,
    )

    if not objects:
        return "No objects found."

    lines = []
    for obj in objects:
        if obj["Type"] == "directory":
            lines.append(f"📁 {obj['Key']}/")
        else:
            size_kb = obj["Size"] / 1024
            size_str = f"{size_kb / 1024:.1f} MB" if size_kb > 1024 else f"{size_kb:.1f} KB"
            lines.append(f"   {obj['Key']} ({size_str})")
    return "\n".join(lines)


def delete_command(args, client: S3Client) -> str:
    """Delete the object at ``args.s3_key``, or with ``args.recursive`` everything under it, confirming first unless
    ``args.yes``; a refused or failed delete exits non-zero."""
    s3_uri = client.s3_uri(args.s3_key, args.subfolder)

    # A no-op delete reports success at the API, so resolve what is actually there first.
    if not client.exists(args.s3_key, args.subfolder):
        return f"✗ Nothing to delete at {s3_uri}"
    if not args.recursive and not client.object_exists(args.s3_key, args.subfolder):
        raise SystemExit(f"✗ {s3_uri} is a prefix, not an object — pass --recursive to delete everything under it.")

    if not args.yes:
        target = f"everything under {s3_uri}/" if args.recursive else s3_uri
        confirm = input(f"Delete {target}? [y/N]: ")
        if confirm.lower() != "y":
            return "Cancelled."

    result = client.delete(
        key=args.s3_key,
        subfolder=args.subfolder,
        recursive=args.recursive,
    )
    if result:
        return f"✓ Deleted {s3_uri}"
    if args.recursive:
        # exists() passed above, so the key resolves to a single object with nothing beneath it.
        raise SystemExit(f"✗ Nothing under {s3_uri}/ — that key is a single object; drop --recursive to delete it.")
    # The client logged the S3 error.
    raise SystemExit(f"✗ Failed to delete {s3_uri}")


def exists_command(args, client: S3Client) -> str:
    """Whether anything exists at ``args.s3_key``."""
    s3_uri = client.s3_uri(args.s3_key, args.subfolder)

    if client.exists(args.s3_key, args.subfolder):
        return f"✓ {s3_uri} exists"
    return f"✗ {s3_uri} does not exist"
