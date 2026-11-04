#!/usr/bin/env python
"""``coordinated_pack`` and its rank-stable cache id (:func:`halo.data.pipeline.coordinated_map.coordinated_pack`).
Its rows cannot depend on the worker count, since TRL packs each 1000-row batch alone; its file name stays stable
though HF's ``_fingerprint`` diverges across ranks after the upstream map+filter chain."""

from __future__ import annotations

import os
import tempfile
from typing import get_args

import numpy as np
import pytest
from datasets import Dataset
from trl.data_utils import pack_dataset

from halo.data.pipeline.coordinated_map import _TRL_PACKERS, coordinated_pack
from halo.data.pipeline.map_cache_key import _RENDER_LIBRARY_VERSIONS, rank_stable_dataset_id, stamp_cache_key
from halo.data.preprocessed.metadata import _PackingStrategy

# Not a multiple of the 1000-row batch, so any contiguous worker split lands inside a batch.
_DOCUMENTS = 2_503
_SEQ_LENGTH = 64


def _on_disk_dataset(dirpath: str, data: dict, fingerprint: str) -> Dataset:
    ds = Dataset.from_dict(data)
    ds.save_to_disk(dirpath)
    reloaded = Dataset.load_from_disk(dirpath)
    # Force another ``_fingerprint`` to mimic ranks that read the same shards and diverge after upstream operations.
    reloaded._fingerprint = fingerprint
    return reloaded


def test_same_cache_files_same_id_across_fingerprints() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        rank0 = _on_disk_dataset(tmp, {"x": list(range(16))}, fingerprint="rank0-fp")
        rank1 = _on_disk_dataset(tmp, {"x": list(range(16))}, fingerprint="rank1-fp")
        assert rank0._fingerprint != rank1._fingerprint, f"{rank0._fingerprint!r} vs {rank1._fingerprint!r}"
        assert rank_stable_dataset_id(rank0) == rank_stable_dataset_id(rank1), (
            f"{rank_stable_dataset_id(rank0)!r} vs {rank_stable_dataset_id(rank1)!r}"
        )


def test_different_cache_files_different_id() -> None:
    with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
        ds_a = _on_disk_dataset(tmp_a, {"x": list(range(16))}, fingerprint="fp")
        ds_b = _on_disk_dataset(tmp_b, {"x": list(range(32))}, fingerprint="fp")
        assert rank_stable_dataset_id(ds_a) != rank_stable_dataset_id(ds_b), "different shards → different ids"


def test_in_memory_fallback() -> None:
    ds = Dataset.from_dict({"x": [1, 2, 3]})
    ds._fingerprint = "manual-fp"
    assert rank_stable_dataset_id(ds) == "manual-fp", "in-memory fallback uses _fingerprint"


def test_in_memory_fallback_default() -> None:
    """Missing ``_fingerprint`` falls back to the sentinel string."""

    class _Bare:
        cache_files = []

    assert rank_stable_dataset_id(_Bare()) == "nofp", "missing _fingerprint → 'nofp'"


def test_toolkit_stamp_wins_over_cache_files() -> None:
    """HF's ``cache_files`` differs between the rank that wrote a ``.map(num_proc=64)`` cache and those that loaded
    it, defeating inference. The stamp set inside :func:`coordinated_dataset_operation` is the source of truth."""
    with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
        # Different shards and fingerprints with the same stamp: the stamp wins.
        rank_writer = _on_disk_dataset(tmp_a, {"x": list(range(16))}, fingerprint="writer-fp")
        rank_loader = _on_disk_dataset(tmp_b, {"x": list(range(16))}, fingerprint="loader-fp")
        stamp_cache_key(rank_writer, "cache-deadbeefcafebabe.arrow")
        stamp_cache_key(rank_loader, "cache-deadbeefcafebabe.arrow")
        writer_files = sorted(cf["filename"] for cf in rank_writer.cache_files)
        loader_files = sorted(cf["filename"] for cf in rank_loader.cache_files)
        assert writer_files != loader_files, "test setup: cache_files diverge between simulated writer/loader"
        assert rank_stable_dataset_id(rank_writer) == rank_stable_dataset_id(rank_loader), (
            "toolkit stamp overrides cache_files + _fingerprint"
        )
        assert rank_stable_dataset_id(rank_writer) == "cache-deadbeefcafebabe.arrow", (
            "stamp value is the cache_file_name itself"
        )


def test_toolkit_stamp_different_keys_different_ids() -> None:
    ds_a = Dataset.from_dict({"x": [1, 2, 3]})
    ds_b = Dataset.from_dict({"x": [1, 2, 3]})
    stamp_cache_key(ds_a, "cache-aaaa.arrow")
    stamp_cache_key(ds_b, "cache-bbbb.arrow")
    assert rank_stable_dataset_id(ds_a) != rank_stable_dataset_id(ds_b), "different stamps → different ids"


@pytest.fixture
def documents(tmp_path, monkeypatch) -> Dataset:
    """On-disk token rows of mixed lengths: empty documents, short ones and ones longer than a packed row."""
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "cache"))
    rng = np.random.default_rng(0)
    long_documents = rng.random(_DOCUMENTS) < 0.1
    lengths = np.where(
        long_documents, rng.integers(_SEQ_LENGTH, 3 * _SEQ_LENGTH, _DOCUMENTS), rng.integers(0, 40, _DOCUMENTS)
    )
    ids = [rng.integers(1, 1000, n).tolist() for n in lengths]
    Dataset.from_dict({"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]}).save_to_disk(
        str(tmp_path / "documents")
    )
    return Dataset.load_from_disk(str(tmp_path / "documents"))


def _rows(dataset: Dataset) -> list[dict]:
    return dataset.with_format(None)[:]


@pytest.mark.parametrize("strategy", get_args(_PackingStrategy))
def test_the_pack_is_trls_single_process_pack_at_any_worker_count(documents, tmp_path, monkeypatch, strategy):
    """Rows, features and format equal TRL's one-process pack at 1, 2 and 3 workers, less the attention_mask the
    collator would drop. TRL's own map at 3 workers packs other rows on this input, so a forwarded worker count
    would fail."""
    uncached = {"keep_in_memory": True}
    reference = pack_dataset(documents, _SEQ_LENGTH, strategy, map_kwargs=uncached).remove_columns("attention_mask")
    parallel_trl = pack_dataset(documents, _SEQ_LENGTH, strategy, map_kwargs={**uncached, "num_proc": 3})
    assert _rows(parallel_trl.remove_columns("attention_mask")) != _rows(reference), (
        "control: the input must be batch-boundary sensitive"
    )

    for num_proc in (1, 2, 3):
        monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / f"cache-{strategy}-{num_proc}"))
        packed = coordinated_pack(documents, _SEQ_LENGTH, strategy, num_proc=num_proc)
        assert "attention_mask" not in packed.column_names, f"{strategy}: packed rows store an all-ones mask"
        assert _rows(packed) == _rows(reference), f"{strategy} at num_proc={num_proc} packed other rows than TRL"
        assert packed.features == reference.features, f"{strategy} at num_proc={num_proc}: {packed.features}"
        assert packed.format == reference.format, f"{strategy} at num_proc={num_proc}: {packed.format}"


def test_a_follower_at_another_worker_count_reads_the_writers_cache(documents, tmp_path):
    """The load rank's pack at 3 workers is what a follower at 1 or 2 workers loads: no new cache file."""
    cache = tmp_path / "cache"
    written = coordinated_pack(documents, _SEQ_LENGTH, "bfd_split", num_proc=3)
    files = set(os.listdir(cache))
    for num_proc in (1, 2):
        follower = coordinated_pack(documents, _SEQ_LENGTH, "bfd_split", num_proc=num_proc)
        assert set(os.listdir(cache)) == files, f"num_proc={num_proc} repacked: {set(os.listdir(cache)) - files}"
        assert _rows(follower) == _rows(written)
        assert rank_stable_dataset_id(follower) == rank_stable_dataset_id(written)


def test_a_cached_pack_that_stored_the_mask_is_not_served(documents, tmp_path):
    """The shared cache holds what a pack that stored the ``attention_mask`` wrote for this input, under that pack's
    name (no column-set term): the pack writes its own rows rather than serve that layout."""
    cache = tmp_path / "cache"
    os.makedirs(cache)
    stored_mask = (
        cache / f"packed_bfd_{_SEQ_LENGTH}_{rank_stable_dataset_id(documents)}_train_{_RENDER_LIBRARY_VERSIONS}.arrow"
    )
    pack_dataset(documents, _SEQ_LENGTH, "bfd", map_kwargs={"cache_file_name": str(stored_mask)})
    assert stored_mask.exists(), "control: the mask-storing pack must be in the cache"

    packed = coordinated_pack(documents, _SEQ_LENGTH, "bfd", num_proc=1)

    assert "attention_mask" not in packed.column_names, "served a cached pack that stored the mask"
    reference = pack_dataset(documents, _SEQ_LENGTH, "bfd", map_kwargs={"keep_in_memory": True})
    assert _rows(packed) == _rows(reference.remove_columns("attention_mask"))


def test_an_empty_split_is_refused_by_name(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "cache"))
    empty = Dataset.from_dict({"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}).select([])
    with pytest.raises(ValueError, match="Cannot pack the eval split: it has no rows"):
        coordinated_pack(empty, _SEQ_LENGTH, "wrapped", split="eval", num_proc=2)


def test_the_pack_runs_exactly_the_strategies_a_bake_accepts():
    assert sorted(_TRL_PACKERS) == sorted(get_args(_PackingStrategy))


def test_an_unknown_strategy_is_refused_by_name(documents):
    with pytest.raises(ValueError, match="best_fit"):
        coordinated_pack(documents, _SEQ_LENGTH, "best_fit")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
