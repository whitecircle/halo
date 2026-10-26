"""The ``<split>/shard_index.json`` contract of a sharded preprocessed dataset, and the stamp both halves of an
artifact carry."""

import json
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from typing import Any

# The format version both halves of an artifact (this index and ``metadata.json``) stamp and are read against.
PREPROCESSING_VERSION = "1.0"

SHARD_INDEX_FILE = "shard_index.json"


class IncompatiblePreprocessedDataset(ValueError):
    """A preprocessed-dataset payload (index or metadata stamp) this build cannot read; the remedy is to
    re-prepare the dataset."""


def write_stamped_sidecar(path: str, stamp: Any) -> None:
    """Write one stamped half of a preprocessed artifact, a :class:`ShardIndex` or the metadata stamp."""
    with open(path, "w") as f:
        json.dump(asdict(stamp), f, indent=2)


def reject_incompatible_stamp(cls, data: dict[str, Any], artifact: str) -> None:
    """Refuse a stamped payload this build cannot read: another version, or a field ``cls`` does not declare;
    ``artifact`` names the half in the message."""
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
    """One shard of a split."""

    id: int
    path: str
    num_examples: int
    byte_size: int = 0


@dataclass
class ShardIndex:
    """The index of a sharded split."""

    version: str = PREPROCESSING_VERSION
    split: str = "train"
    num_shards: int = 0
    total_examples: int = 0
    shards: list[ShardInfo] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShardIndex":
        """Read a ``shard_index.json`` payload, refusing a stamp this build cannot read."""
        reject_incompatible_stamp(cls, data, "Shard index")
        return cls(
            version=data["version"],
            split=data.get("split", "train"),
            num_shards=data.get("num_shards", 0),
            total_examples=data.get("total_examples", 0),
            shards=[ShardInfo(**s) for s in data.get("shards", [])],
        )

    @classmethod
    def load(cls, path: str) -> "ShardIndex":
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
