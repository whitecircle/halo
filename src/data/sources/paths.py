"""Dataset path classification: which of S3 / HuggingFace Hub / local a source or destination
spells, and the on-disk names that verdict depends on.

Pure string rules, no boto3/s3fs, so the loader, the preprocessing pipeline and the standalone
scripts can classify a path without pulling in an S3 stack they never call.
"""

# A local path ending in one of these is a data file, not a saved-dataset directory. The classifier
# below and the loader's builder choice read the same mapping, so a newly accepted extension cannot
# classify as local and then fail to load.
DATA_FILE_BUILDERS = {
    ".jsonl": "json",
    ".json": "json",
    ".parquet": "parquet",
    ".arrow": "arrow",
    ".csv": "csv",
}

# The preprocessed-dataset stamp, shared by the pipeline and the uploader: ``push_folder`` orders it
# delete-first/upload-last so a torn upload cannot present as a valid preprocessed dataset, and a
# second spelling would break that ordering.
METADATA_FILE = "metadata.json"


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse "s3://bucket/key" into (bucket, key); trailing slashes on key are stripped.

    Raises ValueError if the URI is malformed (missing scheme, bucket, or key).
    """
    if not uri.startswith("s3://"):
        raise ValueError(
            f"Invalid S3 URI: '{uri}'. S3 URIs must start with 's3://'. Example: s3://bucket-name/path/to/dataset"
        )

    path = uri[5:]

    if "/" not in path:
        raise ValueError(
            f"Invalid S3 URI: '{uri}'. S3 URIs must include both bucket and key. "
            f"Example: s3://bucket-name/path/to/dataset"
        )

    bucket, key = path.split("/", 1)

    if not bucket:
        raise ValueError(f"Invalid S3 URI: '{uri}'. Bucket name cannot be empty.")

    if not key:
        raise ValueError(f"Invalid S3 URI: '{uri}'. Key/path cannot be empty.")

    key = key.rstrip("/")

    return bucket, key


def parse_dataset_source(path: str) -> tuple[str, str | None, str]:
    """Classify a dataset path, returning (source_type, bucket_or_none, key_or_path).

    source_type is 's3', 'hf_hub', or 'local'. A bare "org/dataset" (exactly one slash,
    no dot-prefixed parts) is treated as HF Hub; anything else falls back to local.
    """
    if path.startswith("s3://"):
        bucket, key = parse_s3_uri(path)
        return ("s3", bucket, key)

    if path.startswith("file://"):
        local_path = path[7:]
        return ("local", None, local_path)

    if path.startswith("/") or (len(path) > 1 and path[1] == ":"):
        return ("local", None, path)

    if path.startswith("./") or path.startswith("../"):
        return ("local", None, path)

    if path.endswith(tuple(DATA_FILE_BUILDERS)):
        return ("local", None, path)

    if "/" in path:
        parts = path.split("/")
        if len(parts) == 2 and not any(p.startswith(".") for p in parts):
            return ("hf_hub", None, path)

    return ("local", None, path)


def parse_hub_spec(path: str) -> tuple[str, str | None, str | None]:
    """Split a Hub dataset spec ``org/name[:config][@split]`` into ``(repo_id, config, split)``.

    ``@`` selects the split and ``:`` the config, in that order. A caller that re-derives it the
    other way round loads a config as a split name.
    """
    repo_id, split = path.rsplit("@", 1) if "@" in path else (path, None)
    repo_id, config = repo_id.rsplit(":", 1) if ":" in repo_id else (repo_id, None)
    return repo_id, config, split


def hub_repo_id(path: str) -> str:
    """The bare ``org/name`` of a Hub dataset spec, dropping the ``:config`` / ``@split`` suffixes
    the loader accepts. Hub file APIs address the repo, not a split of it."""
    return parse_hub_spec(path)[0]


def parse_dataset_destination(path: str) -> tuple[str, str | None, str]:
    """Classify a dataset output path, returning (dest_type, bucket_or_none, key_or_path).

    dest_type is 's3', 'hf_hub', or 'local'. Unlike :func:`parse_dataset_source`, a bare "org/name"
    is local: inferring the Hub is acceptable for a read, but a write must not upload anywhere the
    caller did not name, so only an explicit "hf://org/name" targets the Hub.
    """
    if path.startswith("s3://"):
        bucket, key = parse_s3_uri(path)
        return ("s3", bucket, key)

    if path.startswith("hf://"):
        repo_id = path[5:]
        if repo_id.count("/") != 1 or not all(repo_id.split("/")):
            raise ValueError(f"Invalid HF Hub destination '{path}'. Expected 'hf://org/name'.")
        return ("hf_hub", None, repo_id)

    return ("local", None, path[7:] if path.startswith("file://") else path)
