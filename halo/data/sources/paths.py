"""Dataset path classification: which of S3 / HuggingFace Hub / local a source or destination spells,
the on-disk names that verdict depends on, which split of a source holds its held-out rows, and the
refusals of a source missing one or spelled ambiguously. String rules plus that one directory probe, so a
caller classifies a path without importing an S3 stack it never calls."""

import os
from collections.abc import Mapping

# A local path ending in one of these is a data file read by that ``datasets`` builder, not a saved dataset.
DATA_FILE_BUILDERS = {
    ".jsonl": "json",
    ".json": "json",
    ".parquet": "parquet",
    ".arrow": "arrow",
    ".csv": "csv",
}

# The preprocessed-dataset stamp ``push_folder`` deletes first and uploads last, so a torn upload reads as none.
METADATA_FILE = "metadata.json"

# The split pair every trainer reads (``ds["train"]``, ``ds["test"]``): what a loader renders and
# checks, and what a preprocessed artifact writes.
TRAIN_TEST_SPLITS = ("train", "test")

# A source's held-out split, in lookup order: one that ships no ``test`` split evaluates on (and
# bakes as ``test``) its ``validation`` split.
EVAL_SPLIT_NAMES = ("test", "validation")


def data_file_builder(path: str) -> str | None:
    """The ``datasets`` builder that reads ``path`` as a data file (:data:`DATA_FILE_BUILDERS`), or
    ``None`` when its extension names none."""
    return next((builder for ext, builder in DATA_FILE_BUILDERS.items() if path.endswith(ext)), None)


def eval_split_name(splits) -> str | None:
    """The split of ``splits`` that holds the source's held-out rows, first in :data:`EVAL_SPLIT_NAMES`."""
    return next((name for name in EVAL_SPLIT_NAMES if name in splits), None)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """``(bucket, key)`` of an ``s3://bucket/key`` URI, the key's trailing slashes stripped; raises on a
    malformed one."""
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
    return bucket, key.rstrip("/")


def parse_dataset_source(path: str) -> tuple[str, str | None, str]:
    """``(source_type, bucket_or_none, key_or_path)`` of a dataset path, ``source_type`` one of 's3',
    'hf_hub' (a bare "org/dataset": one slash, no dot-prefixed part) and 'local' (everything else)."""
    if path.startswith("s3://"):
        return ("s3", *parse_s3_uri(path))
    if path.startswith("file://"):
        return ("local", None, path[7:])
    if (
        path.startswith(("/", "./", "../"))
        or (len(path) > 1 and path[1] == ":")
        or data_file_builder(path) is not None
    ):
        return ("local", None, path)
    parts = path.split("/")
    if len(parts) == 2 and not any(p.startswith(".") for p in parts):
        return ("hf_hub", None, path)
    return ("local", None, path)


def parse_hub_spec(path: str) -> tuple[str, str | None, str | None]:
    """``(repo_id, config, split)`` of a Hub dataset spec ``org/name[:config][@split]``."""
    repo_id, split = path.rsplit("@", 1) if "@" in path else (path, None)
    repo_id, config = repo_id.rsplit(":", 1) if ":" in repo_id else (repo_id, None)
    return repo_id, config, split


def hub_repo_id(path: str) -> str:
    """The bare ``org/name`` of a Hub dataset spec, dropping the ``:config`` / ``@split`` suffixes
    the loader accepts. Hub file APIs address the repo, not a split of it."""
    return parse_hub_spec(path)[0]


def reject_ambiguous_hub_source(path: str) -> None:
    """Refuse a Hub-shaped source (:func:`parse_dataset_source`) that also names a local directory: either reading
    is a guess, and the wrong one fails far from the cause (the Hub builder reads a ``save_to_disk`` tree's
    sidecar files as rows)."""
    if parse_dataset_source(path)[0] == "hf_hub" and os.path.isdir(directory := hub_repo_id(path)):
        raise ValueError(
            f"Dataset path '{path}' is spelled as a HuggingFace Hub id, but a local directory of that name "
            f"exists here. Write './{directory}' or an absolute path to load the directory (a local path takes no "
            f"':config' or '@split' suffix: name a saved split by its own directory); to load the Hub dataset, "
            f"run where no such directory exists."
        )


def parse_dataset_destination(path: str) -> tuple[str, str | None, str]:
    """:func:`parse_dataset_source` for an output path, where only an explicit "hf://org/name" targets the
    Hub: a write must not upload anywhere the caller did not name."""
    if path.startswith("s3://"):
        return ("s3", *parse_s3_uri(path))
    if path.startswith("hf://"):
        repo_id = path[5:]
        if repo_id.count("/") != 1 or not all(repo_id.split("/")):
            raise ValueError(f"Invalid HF Hub destination '{path}'. Expected 'hf://org/name'.")
        return ("hf_hub", None, repo_id)
    return ("local", None, path[7:] if path.startswith("file://") else path)


def reject_partial_splits(ds: Mapping, path: str) -> None:
    """Fail loud on a partial DatasetDict: sharded/preprocessed loads skip missing splits, and the
    trainers index ``ds["test"]`` unconditionally — a bare ``KeyError`` far from the cause."""
    missing = [split for split in TRAIN_TEST_SPLITS if split not in ds]
    if missing:
        raise ValueError(
            f"Dataset {path} is missing the {missing} split(s). Trainers require both 'train' and "
            f"'test' — re-prepare the dataset with a test split "
            f"(scripts/before_training/prepare_dataset.py writes one when --test_size is set)."
        )
