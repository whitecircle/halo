"""Shard-index contract of a preprocessed dataset directory (``<split>/shard_index.json``).

Torch-free leaf: the sharded loader reads the index of an artifact it never produces, and reaching
it through the writer (:mod:`src.data.pipeline.preprocessing`) would drag torch, transformers and
the collators into a plain JSON read.
"""

import json
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from typing import Any

# Stamped into both halves of a preprocessed artifact — this index and the ``metadata.json`` stamp —
# and compared on load, so the two can never claim different format versions for one directory.
PREPROCESSING_VERSION = "1.0"

SHARD_INDEX_FILE = "shard_index.json"


class IncompatiblePreprocessedDataset(ValueError):
    """A preprocessed-dataset payload this build cannot read (stamp version or schema).

    Raised for both halves of the artifact: the index decides WHICH rows each rank loads and a
    mis-read metadata stamp silently re-tokenizes tokenized rows, so reading either under a diverged
    build's field meanings is worse than stopping. The remedy for both is to re-prepare the dataset.
    """


def write_stamped_sidecar(path: str, payload: dict[str, Any]) -> None:
    """Write one half of a preprocessed artifact's stamped sidecar (this index, the metadata stamp)."""
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def reject_incompatible_stamp(cls, data: dict[str, Any], artifact: str) -> None:
    """Refuse a stamped payload this build cannot read: a bumped version, or a field ``cls`` does not
    declare (a retired spelling, or a diverged build at the same stamp).

    ``artifact`` names the half of the dataset in the message. Every rank reads its own copy and
    reaches the same verdict, so the refusal cannot skew a collective.
    """
    version = data.get("version")
    if version != PREPROCESSING_VERSION:
        raise IncompatiblePreprocessedDataset(
            f"{artifact} is version {version!r}, but this build reads {PREPROCESSING_VERSION!r}. "
            f"Re-run scripts/before_training/prepare_dataset.py with this version of the toolkit."
        )
    unknown = sorted(set(data) - {f.name for f in dataclass_fields(cls)})
    if unknown:
        raise IncompatiblePreprocessedDataset(
            f"{artifact} carries field(s) {unknown} that this build does not know, at the same "
            f"version stamp {version!r} — a retired spelling or a diverged build. Re-run "
            f"scripts/before_training/prepare_dataset.py with this version of the toolkit."
        )


@dataclass
class ShardInfo:
    """Information about a single shard."""

    id: int
    path: str
    num_examples: int
    byte_size: int = 0


@dataclass
class ShardIndex:
    """Index for sharded dataset."""

    version: str = PREPROCESSING_VERSION
    split: str = "train"
    num_shards: int = 0
    total_examples: int = 0
    shards: list[ShardInfo] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (``asdict`` so a new field cannot be silently dropped)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShardIndex":
        """Create from a ``shard_index.json`` payload, refusing a stamp this build cannot read."""
        reject_incompatible_stamp(cls, data, "Shard index")
        return cls(
            version=data["version"],
            split=data.get("split", "train"),
            num_shards=data.get("num_shards", 0),
            total_examples=data.get("total_examples", 0),
            shards=[ShardInfo(**s) for s in data.get("shards", [])],
        )

    def save(self, path: str) -> None:
        """Save shard index to JSON file."""
        write_stamped_sidecar(path, self.to_dict())

    @classmethod
    def load(cls, path: str) -> "ShardIndex":
        """Load shard index from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
