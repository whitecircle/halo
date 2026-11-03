#!/usr/bin/env python
"""``dataset_content_fingerprint`` keys off on-disk Arrow paths, since HF's ``Dataset._fingerprint`` is not stable
across processes and every fresh run would miss its cache and leave another set of shards."""

from __future__ import annotations

import tempfile

import pytest
from datasets import Dataset, DatasetDict

from halo.data.pipeline.map_cache_key import dataset_content_fingerprint


def _on_disk_dataset(dirpath: str, data: dict, fingerprint: str) -> Dataset:
    """Save and reload a Dataset, forcing ``_fingerprint`` to simulate the per-process recomputation."""
    ds = Dataset.from_dict(data)
    ds.save_to_disk(dirpath)
    reloaded = Dataset.load_from_disk(dirpath)
    reloaded._fingerprint = fingerprint
    return reloaded


def test_run_to_run_stability_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ds_run1 = _on_disk_dataset(tmp, {"x": list(range(16))}, fingerprint="run1-fp")
        ds_run2 = _on_disk_dataset(tmp, {"x": list(range(16))}, fingerprint="run2-fp")
        assert ds_run1._fingerprint != ds_run2._fingerprint
        assert dataset_content_fingerprint(ds_run1) == dataset_content_fingerprint(ds_run2), (
            f"{dataset_content_fingerprint(ds_run1)!r} vs {dataset_content_fingerprint(ds_run2)!r}"
        )


def test_run_to_run_stability_datasetdict() -> None:
    with tempfile.TemporaryDirectory() as tmp_train, tempfile.TemporaryDirectory() as tmp_test:
        train1 = _on_disk_dataset(tmp_train, {"x": list(range(16))}, fingerprint="t1")
        test1 = _on_disk_dataset(tmp_test, {"x": list(range(4))}, fingerprint="e1")
        train2 = _on_disk_dataset(tmp_train, {"x": list(range(16))}, fingerprint="t2")
        test2 = _on_disk_dataset(tmp_test, {"x": list(range(4))}, fingerprint="e2")
        dd1 = DatasetDict({"train": train1, "test": test1})
        dd2 = DatasetDict({"train": train2, "test": test2})
        assert dataset_content_fingerprint(dd1) == dataset_content_fingerprint(dd2), (
            f"{dataset_content_fingerprint(dd1)!r} vs {dataset_content_fingerprint(dd2)!r}"
        )


def test_different_content_different_fingerprint() -> None:
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        small = _on_disk_dataset(a, {"x": list(range(16))}, fingerprint="fp")
        large = _on_disk_dataset(b, {"x": list(range(128))}, fingerprint="fp")
        assert dataset_content_fingerprint(small) != dataset_content_fingerprint(large)


def test_in_memory_fallback_uses_fingerprint() -> None:
    """Without the fallback, identical-shape in-memory datasets with different content would collide."""
    ds_a = Dataset.from_dict({"x": [1, 2, 3]})
    ds_b = Dataset.from_dict({"x": [1, 2, 3]})
    ds_a._fingerprint = "marker-a"
    ds_b._fingerprint = "marker-b"
    assert dataset_content_fingerprint(ds_a) != dataset_content_fingerprint(ds_b)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
