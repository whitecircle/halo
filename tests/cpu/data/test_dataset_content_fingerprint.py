#!/usr/bin/env python
"""Unit tests for :func:`src.data.pipeline.processing._dataset_content_fingerprint`.

HF's ``Dataset._fingerprint`` is *not* stable across Python processes, so
using it as the dataset half of the cache key built by
:func:`_build_cache_file_name` means every fresh ``python`` recomputes it
differently and no run hits a previous run's cache — each rerun of the same
config leaves another full set of arrow shards behind. The dataset half
therefore routes through
:func:`_rank_stable_dataset_id`, which keys off the dataset's on-disk Arrow
paths (written with deterministic md5 names upstream in
:func:`coordinated_map` / :func:`coordinated_filter`).

These tests pin that contract: as long as the on-disk shards are unchanged,
the content fingerprint is *byte-identical* across simulated processes
(different ``_fingerprint`` values).

Run::

    python tests/cpu/data/test_dataset_content_fingerprint.py
"""

from __future__ import annotations

import tempfile

import pytest
from datasets import Dataset, DatasetDict

from src.data.pipeline.processing import _dataset_content_fingerprint


def _on_disk_dataset(dirpath: str, data: dict, fingerprint: str) -> Dataset:
    """Save a Dataset to disk and reload it; force a specific ``_fingerprint`` value
    to simulate the per-process recomputation that HF datasets exhibits in practice."""
    ds = Dataset.from_dict(data)
    ds.save_to_disk(dirpath)
    reloaded = Dataset.load_from_disk(dirpath)
    reloaded._fingerprint = fingerprint
    return reloaded


def test_run_to_run_stability_dataset() -> None:
    """A single Dataset whose ``_fingerprint`` differs between simulated runs
    must still produce the same content fingerprint."""
    with tempfile.TemporaryDirectory() as tmp:
        ds_run1 = _on_disk_dataset(tmp, {"x": list(range(16))}, fingerprint="run1-fp")
        ds_run2 = _on_disk_dataset(tmp, {"x": list(range(16))}, fingerprint="run2-fp")
        assert ds_run1._fingerprint != ds_run2._fingerprint
        assert _dataset_content_fingerprint(ds_run1) == _dataset_content_fingerprint(ds_run2), (
            f"{_dataset_content_fingerprint(ds_run1)!r} vs {_dataset_content_fingerprint(ds_run2)!r}"
        )


def test_run_to_run_stability_datasetdict() -> None:
    """A DatasetDict (train + test) with per-process-unstable per-split
    ``_fingerprint`` values must still produce the same content fingerprint."""
    with tempfile.TemporaryDirectory() as tmp_train, tempfile.TemporaryDirectory() as tmp_test:
        train1 = _on_disk_dataset(tmp_train, {"x": list(range(16))}, fingerprint="t1")
        test1 = _on_disk_dataset(tmp_test, {"x": list(range(4))}, fingerprint="e1")
        train2 = _on_disk_dataset(tmp_train, {"x": list(range(16))}, fingerprint="t2")
        test2 = _on_disk_dataset(tmp_test, {"x": list(range(4))}, fingerprint="e2")
        dd1 = DatasetDict({"train": train1, "test": test1})
        dd2 = DatasetDict({"train": train2, "test": test2})
        assert _dataset_content_fingerprint(dd1) == _dataset_content_fingerprint(dd2), (
            f"{_dataset_content_fingerprint(dd1)!r} vs {_dataset_content_fingerprint(dd2)!r}"
        )


def test_different_content_different_fingerprint() -> None:
    """A change in dataset size or on-disk shards must change the fingerprint —
    otherwise we'd risk loading a stale cache against fresh inputs."""
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        small = _on_disk_dataset(a, {"x": list(range(16))}, fingerprint="fp")
        large = _on_disk_dataset(b, {"x": list(range(128))}, fingerprint="fp")
        assert _dataset_content_fingerprint(small) != _dataset_content_fingerprint(large)


def test_in_memory_fallback_uses_fingerprint() -> None:
    """A purely in-memory Dataset (no ``cache_files``) falls back to
    ``_fingerprint`` — without it we'd collide on identical-shape in-memory
    datasets that actually carry different content."""
    ds_a = Dataset.from_dict({"x": [1, 2, 3]})
    ds_b = Dataset.from_dict({"x": [1, 2, 3]})
    ds_a._fingerprint = "marker-a"
    ds_b._fingerprint = "marker-b"
    assert _dataset_content_fingerprint(ds_a) != _dataset_content_fingerprint(ds_b)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
