#!/usr/bin/env python
"""The split identity the holders of one data replica compare digests values, not their ``repr``.

``_split_identity`` samples evenly spaced rows of a split and digests them; ranks whose copies of the
source differ then refuse to train. Proven on real ``datasets`` splits:

1. the same rows digest alike however the copy is laid out — one Arrow chunk, several, a multi-file
   save, an indices mapping — across string, binary, nested-list, struct and nullable columns;
2. a change to any sampled value, a list split differently over the same flat values, or a null where
   a copy holds an empty list, changes the digest;
3. a binary column of large values (a preprocessed VLM split's ``pixel_values``) reaches the hash
   one value at a time, never as one string spelling every sampled row.

    python tests/cpu/data/test_split_identity.py
"""

import hashlib
import types

import pytest
from datasets import Dataset, Features, Sequence, Value, concatenate_datasets, load_from_disk

from src.data.sources import loading
from src.data.sources.loading import _split_identity

ROWS = 20
FEATURES = Features(
    {
        "text": Value("string"),
        "ids": Sequence(Value("int64")),
        "pixel_values": Value("binary"),
        "nested": Sequence(Sequence(Value("float32"))),
        "meta": {"turn": Value("int64"), "role": Value("string")},
        "maybe": Sequence(Value("int64")),
    }
)
# A preprocessed VLM row's pixel buffer is megabytes; the bound the digest's working set must keep.
LARGE_VALUE_BYTES = 1 << 20


def _rows() -> dict[str, list]:
    return {
        "text": [f"row {i}" for i in range(ROWS)],
        "ids": [list(range(i % 5)) for i in range(ROWS)],
        "pixel_values": [bytes([i]) * (i + 3) for i in range(ROWS)],
        "nested": [[[float(i)], [float(i) + 0.5, 1.0]] for i in range(ROWS)],
        "meta": [{"turn": i, "role": "user" if i % 2 else "assistant"} for i in range(ROWS)],
        "maybe": [None if i % 3 == 0 else [i] for i in range(ROWS)],
    }


def _split(rows: dict[str, list] | None = None) -> Dataset:
    return Dataset.from_dict(rows or _rows(), features=FEATURES)


def _edited(column: str, value) -> Dataset:
    """The split with row 0 — always sampled — holding ``value`` in ``column``."""
    rows = _rows()
    rows[column][0] = value
    return _split(rows)


def test_the_same_rows_digest_alike_whatever_the_copy_layout(tmp_path):
    base = _split()
    chunked = concatenate_datasets(
        [base.select(range(7)).flatten_indices(), base.select(range(7, ROWS)).flatten_indices()]
    )
    reversed_order = list(reversed(range(ROWS)))
    mapped = base.select(reversed_order).select(reversed_order)
    base.save_to_disk(str(tmp_path / "saved"), num_shards=3)
    saved = load_from_disk(str(tmp_path / "saved"))
    assert mapped._indices is not None, "the indices-mapped copy must read its rows through a mapping"
    assert len(saved.data.to_batches()) > 1, "the saved copy must span several Arrow batches"

    identity = _split_identity(base)
    for name, copy in {"chunked": chunked, "index-mapped": mapped, "saved in 3 shards": saved}.items():
        assert _split_identity(copy) == identity, f"the {name} copy of the same rows digested differently"


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("text", "row 0 (re-pushed)"),
        ("ids", [7]),
        ("pixel_values", b"\x00\x00\x01"),
        ("nested", [[0.0, 0.5], [1.0]]),
        ("meta", {"turn": 0, "role": "user"}),
        ("maybe", []),
    ],
    ids=["string", "list", "binary", "list-boundaries", "struct-field", "null-vs-empty"],
)
def test_a_changed_sampled_value_changes_the_digest(column, value):
    assert _split_identity(_edited(column, value)) != _split_identity(_split())


def test_large_binary_values_reach_the_hash_one_value_at_a_time(monkeypatch):
    split = Dataset.from_dict(
        {"pixel_values": [bytes([i]) * LARGE_VALUE_BYTES for i in range(16)]},
        features=Features({"pixel_values": Value("binary")}),
    )
    fed: list[int] = []

    def recording_blake2b(data=b"", **kwargs):
        hasher = hashlib.blake2b(data, **kwargs)
        fed.append(len(data))

        class Recording:
            def update(self, chunk):
                fed.append(memoryview(chunk).nbytes)
                hasher.update(chunk)

            def hexdigest(self):
                return hasher.hexdigest()

        return Recording()

    monkeypatch.setattr(loading, "hashlib", types.SimpleNamespace(blake2b=recording_blake2b))

    identity = _split_identity(split)

    assert identity.startswith("16 rows, content "), identity
    assert max(fed) <= LARGE_VALUE_BYTES, f"one update of {max(fed)} bytes: the sampled rows were spelled out whole"
    assert sum(fed) < 2 * 16 * LARGE_VALUE_BYTES, f"{sum(fed)} bytes fed for 16 values of {LARGE_VALUE_BYTES}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
