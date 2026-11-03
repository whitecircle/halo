#!/usr/bin/env python3
"""Which rows a fractional ``dataset_ratio`` keeps is training signal the dataset fingerprint never stamps, so the
exact selection is pinned: an RNG change must be a deliberate edit here. The draw may not consult numpy's global
state, since every rank must reach the same rows with no exchange."""

import numpy as np
import pytest
from datasets import Dataset, DatasetDict

from halo.data.sources.loading import _DEFAULT_DATA_SEED, load_datasets

# Rows kept from a 20-row split at ratio 0.5. Train draws with ``_DEFAULT_DATA_SEED``, test with
# ``+ 1`` so the two subsets are not the same row positions.
_PINNED_TRAIN_ROWS = [0, 1, 3, 6, 8, 9, 12, 13, 14, 19]
_PINNED_TEST_ROWS = [0, 4, 5, 7, 8, 11, 12, 15, 17, 18]


def _numbered_dataset(num_rows: int) -> Dataset:
    return Dataset.from_dict({"text": [f"row {i}" for i in range(num_rows)], "row_id": list(range(num_rows))})


def _save_raw_dataset(directory: str, num_train: int = 20, num_test: int = 20) -> str:
    DatasetDict({"train": _numbered_dataset(num_train), "test": _numbered_dataset(num_test)}).save_to_disk(directory)
    return directory


def _kept(ds, split: str) -> list[int]:
    return sorted(ds[split]["row_id"])


def test_ratio_subset_keeps_the_pinned_rows(tmp_path):
    path = _save_raw_dataset(str(tmp_path / "ds"))

    ds = load_datasets(path, test_size=None, dataset_ratio=0.5, conversation_field=None)

    assert _kept(ds, "train") == _PINNED_TRAIN_ROWS
    assert _kept(ds, "test") == _PINNED_TEST_ROWS


def test_train_and_test_draw_different_positions(tmp_path):
    """The ``seed + 1`` offset on the test split: identical splits must not keep identical positions."""
    path = _save_raw_dataset(str(tmp_path / "ds"))

    ds = load_datasets(path, test_size=None, dataset_ratio=0.5, conversation_field=None)

    assert _kept(ds, "train") != _kept(ds, "test")


def test_selection_ignores_global_numpy_state(tmp_path):
    """A draw off ``np.random``'s global stream would diverge once one rank consumed a different count, handing
    ranks disjoint corpora with no error."""
    path = _save_raw_dataset(str(tmp_path / "ds"))

    np.random.seed(0)
    first = _kept(load_datasets(path, test_size=None, dataset_ratio=0.5, conversation_field=None), "train")
    np.random.seed(12345)
    np.random.random(97)
    second = _kept(load_datasets(path, test_size=None, dataset_ratio=0.5, conversation_field=None), "train")

    assert first == second == _PINNED_TRAIN_ROWS


def test_list_entries_draw_distinct_rows(tmp_path):
    """Concatenated sources step the seed per entry, so two identical datasets do not contribute the
    same rows twice."""
    first = _save_raw_dataset(str(tmp_path / "a"))
    second = _save_raw_dataset(str(tmp_path / "b"))

    ds = load_datasets([first, second], test_size=None, dataset_ratio=[0.5, 0.5], conversation_field=None)

    kept = ds["train"]["row_id"]
    assert len(kept) == 20
    assert sorted(kept[:10]) == _PINNED_TRAIN_ROWS
    assert sorted(kept[10:]) != _PINNED_TRAIN_ROWS


def test_default_seed_is_the_documented_one():
    """The pinned row sets above are only meaningful against the shipped default."""
    assert _DEFAULT_DATA_SEED == 42


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
