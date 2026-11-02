"""The S3 folder commands ``scripts/before_training/s3.py`` runs — push, download, list, exists and delete — each
through the :class:`~halo.data.sources.s3_client.S3Client` built for ``--bucket``, or for the bucket a full
``s3://`` URI key names, and returning the report the CLI prints."""

from halo.data.sources.s3_client import S3Client, default_bucket


def build_command_client(args) -> S3Client:
    """The client a command runs through: for ``--bucket``, or, when the command's key (``s3_key``, or ``prefix``
    for ``list``) is a full ``s3://bucket/key`` URI, for that bucket, the key then replacing the URI in ``args``. A
    URI beside ``--subfolder`` or another ``--bucket`` raises, since it names its whole location itself."""
    field = "prefix" if hasattr(args, "prefix") else "s3_key"
    value = getattr(args, field)
    if not value.startswith("s3://"):
        return S3Client(bucket=args.bucket or default_bucket(f"the key {value!r}", "pass --bucket or an s3:// URI"))
    bucket, _, key = value.removeprefix("s3://").partition("/")
    if not bucket or not (key.strip("/") or field == "prefix"):
        raise SystemExit(f"✗ {value!r} is not an s3://<bucket>/<key> URI.")
    if args.subfolder or args.bucket not in (None, bucket):
        raise SystemExit(
            f"✗ {value!r} names its bucket and key itself; drop --bucket/--subfolder, or pass the bare key with them."
        )
    setattr(args, field, key.strip("/"))
    return S3Client(bucket=bucket)


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
