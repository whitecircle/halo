#!/usr/bin/env python
"""Unit tests for :func:`src.data.pipeline.processing.pack_dataset_coordinated`
and the rank-stable cache identifier it uses.

Pins the behavior the helper exists for: even when HF's ``Dataset._fingerprint``
diverges across ranks (it does after the upstream map+filter chain, so the
per-rank cache file names diverge and the helper silently writes one copy per
rank instead of deduplicating), the cache file name must stay rank-stable as
long as the inputs come from the same on-disk Arrow shards.

Run::

    python tests/cpu/data/test_pack_dataset_coordinated.py
"""

from __future__ import annotations

import tempfile

import pytest
from datasets import Dataset

from src.data.pipeline.processing import _rank_stable_dataset_id


def _on_disk_dataset(dirpath: str, data: dict, fingerprint: str) -> Dataset:
    """Build a Dataset, save it under ``dirpath``, then reload so it has cache_files."""
    ds = Dataset.from_dict(data)
    ds.save_to_disk(dirpath)
    reloaded = Dataset.load_from_disk(dirpath)
    # Force a different `_fingerprint` to mimic the rank-divergence we observed:
    # two ranks reading the same on-disk shards but ending up with different
    # `_fingerprint` attributes after upstream operations.
    reloaded._fingerprint = fingerprint
    return reloaded


def test_same_cache_files_same_id_across_fingerprints() -> None:
    """Two datasets backed by the same on-disk shards must produce the same id
    even when ``_fingerprint`` differs — this is the rank-divergence case."""
    with tempfile.TemporaryDirectory() as tmp:
        rank0 = _on_disk_dataset(tmp, {"x": list(range(16))}, fingerprint="rank0-fp")
        rank1 = _on_disk_dataset(tmp, {"x": list(range(16))}, fingerprint="rank1-fp")
        # Sanity: the empirical condition we are guarding against.
        assert rank0._fingerprint != rank1._fingerprint, f"{rank0._fingerprint!r} vs {rank1._fingerprint!r}"
        assert _rank_stable_dataset_id(rank0) == _rank_stable_dataset_id(rank1), (
            f"{_rank_stable_dataset_id(rank0)!r} vs {_rank_stable_dataset_id(rank1)!r}"
        )


def test_different_cache_files_different_id() -> None:
    """Datasets backed by different on-disk shards must produce different ids
    (so packs of different inputs do not collide on the same cache file)."""
    with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
        ds_a = _on_disk_dataset(tmp_a, {"x": list(range(16))}, fingerprint="fp")
        ds_b = _on_disk_dataset(tmp_b, {"x": list(range(32))}, fingerprint="fp")
        assert _rank_stable_dataset_id(ds_a) != _rank_stable_dataset_id(ds_b), "different shards → different ids"


def test_in_memory_fallback() -> None:
    """A Dataset with no on-disk shards must still produce a stable id —
    fall back to ``_fingerprint``."""
    ds = Dataset.from_dict({"x": [1, 2, 3]})
    ds._fingerprint = "manual-fp"
    assert _rank_stable_dataset_id(ds) == "manual-fp", "in-memory fallback uses _fingerprint"


def test_in_memory_fallback_default() -> None:
    """Missing ``_fingerprint`` falls back to the sentinel string."""

    class _Bare:
        cache_files = []

    assert _rank_stable_dataset_id(_Bare()) == "nofp", "missing _fingerprint → 'nofp'"


def test_toolkit_stamp_wins_over_cache_files() -> None:
    """When the toolkit has stamped ``_toolkit_cache_key`` on the dataset, it
    must override both ``cache_files`` and ``_fingerprint``. This is the case
    that matters in production: HF's ``cache_files`` representation differs
    between the rank that *wrote* a ``.map(num_proc=64)`` cache (sharded paths)
    and the ranks that *loaded* it (consolidated paths), even when the on-disk
    shards are identical — defeating any inference-based scheme. The stamp is
    the single source of truth set inside :func:`coordinated_dataset_operation`.
    """
    with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
        # Two datasets with *different* on-disk shards and different fingerprints,
        # but with the SAME explicit toolkit cache key — the stamp must win.
        rank_writer = _on_disk_dataset(tmp_a, {"x": list(range(16))}, fingerprint="writer-fp")
        rank_loader = _on_disk_dataset(tmp_b, {"x": list(range(16))}, fingerprint="loader-fp")
        rank_writer._toolkit_cache_key = "cache-deadbeefcafebabe.arrow"
        rank_loader._toolkit_cache_key = "cache-deadbeefcafebabe.arrow"
        # Sanity: divergent inferred ids (would mean different cache files in the
        # absence of the stamp).
        writer_files = sorted(cf["filename"] for cf in rank_writer.cache_files)
        loader_files = sorted(cf["filename"] for cf in rank_loader.cache_files)
        assert writer_files != loader_files, "test setup: cache_files diverge between simulated writer/loader"
        assert _rank_stable_dataset_id(rank_writer) == _rank_stable_dataset_id(rank_loader), (
            "toolkit stamp overrides cache_files + _fingerprint"
        )
        assert _rank_stable_dataset_id(rank_writer) == "cache-deadbeefcafebabe.arrow", (
            "stamp value is the cache_file_name itself"
        )


def test_toolkit_stamp_different_keys_different_ids() -> None:
    """Different stamps must still produce different ids (no collisions when the
    upstream operation legitimately produced different caches)."""
    ds_a = Dataset.from_dict({"x": [1, 2, 3]})
    ds_b = Dataset.from_dict({"x": [1, 2, 3]})
    ds_a._toolkit_cache_key = "cache-aaaa.arrow"
    ds_b._toolkit_cache_key = "cache-bbbb.arrow"
    assert _rank_stable_dataset_id(ds_a) != _rank_stable_dataset_id(ds_b), "different stamps → different ids"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
