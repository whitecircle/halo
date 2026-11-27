#!/usr/bin/env python
"""``halo/data/deduplication.py``: row normalization, the seeded passes that decide which member of a near-duplicate
group survives, the FAISS search's surviving indices and the staged output."""

import argparse
import gzip
import json
import logging

import faiss
import numpy as np
import pytest

from halo.data import deduplication
from halo.data.deduplication import _normalize_embeddings, deduplicate_blocks, deduplicate_embeddings

# Four orthogonal unit rows, each repeated once: rows i and i + 4 are exact duplicates.
DUPLICATED_BASIS = np.concatenate([np.eye(4, dtype=np.float32)] * 2)


def test_normalize_scales_each_row_to_unit_norm():
    emb = np.array([[3.0, 4.0], [0.0, 5.0], [-1.0, -1.0]])
    norms = np.linalg.norm(_normalize_embeddings(emb), axis=1)
    np.testing.assert_allclose(norms, np.ones(3), atol=1e-7)


def test_normalize_preserves_direction():
    # 3-4-5 triangle: [6, 8] / 10 = [0.6, 0.8].
    np.testing.assert_allclose(_normalize_embeddings(np.array([[6.0, 8.0]])), [[0.6, 0.8]], atol=1e-7)


def _run_args(tmp_path, embeddings: np.ndarray, **overrides) -> argparse.Namespace:
    """``run_deduplication``'s arguments over a text file of one row per embedding, the embeddings loaded from disk."""
    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(json.dumps({"text": f"row {i}"}) + "\n" for i in range(len(embeddings))))
    np.save(tmp_path / "embeddings.npy", embeddings)
    args = {
        "dataset": str(rows),
        "text_field": "text",
        "config": None,
        "split": "train",
        "sample_size": None,
        "load_embeddings": str(tmp_path / "embeddings.npy"),
        "device": "cpu",
        "similarity_threshold": 0.9,
        "dedup_batch_size": 8,
        "dedup_steps": 1,
        "max_workers": None,
        "dry_run": False,
        "output_path": str(tmp_path / "out" / "rows.jsonl"),
        "output_format": "jsonl",
        "keep_fields": None,
    }
    return argparse.Namespace(**{**args, **overrides})


def test_single_pass_keeps_the_first_row_of_each_duplicate_group():
    survivors = deduplicate_blocks(DUPLICATED_BASIS, np.arange(8), batch_size=8, similarity_threshold=0.9)
    assert survivors.tolist() == [0, 1, 2, 3]


def test_single_pass_returns_global_indices_and_dedups_within_a_batch_only():
    # Duplicates sit in different batches here, so every row survives, under its global index.
    survivors = deduplicate_blocks(DUPLICATED_BASIS, np.arange(8), batch_size=4, similarity_threshold=0.9)
    assert survivors.tolist() == list(range(8))


def test_identical_rows_are_searched_once_not_against_each_other(monkeypatch):
    """A search returns every pair over the threshold, so K identical rows queried whole return K squared pairs; rows a
    kept one already dropped are left out of the later queries."""
    queried = []

    class _RecordingIndex(faiss.IndexFlatIP):
        def range_search(self, x, thresh):
            queried.append(len(x))
            return super().range_search(x, thresh)

    monkeypatch.setattr(deduplication.faiss, "IndexFlatIP", _RecordingIndex)
    monkeypatch.setattr(deduplication, "_SEARCH_QUERY_ROWS", 2)
    rows = np.concatenate([np.tile(np.eye(3, dtype=np.float32)[:1], (6, 1)), np.eye(3, dtype=np.float32)[1:]])

    survivors = deduplicate_blocks(rows, np.arange(8), batch_size=8, similarity_threshold=0.9)

    assert survivors.tolist() == [0, 6, 7]
    assert queried == [2, 2], "the dropped copies were queried, or the block was queried whole"


def test_multistep_maps_survivors_back_to_the_input_rows():
    survivors, sizes = deduplicate_embeddings(DUPLICATED_BASIS, steps_count=3, batch_size=8, similarity_threshold=0.9)
    assert sizes == [8, 4, 4, 4]
    assert sorted(i % 4 for i in survivors.tolist()) == [0, 1, 2, 3], "one survivor per duplicate pair"
    np.testing.assert_array_equal(np.sort(DUPLICATED_BASIS[survivors].argmax(axis=1)), [0, 1, 2, 3])


def test_multistep_survivors_come_back_in_input_order():
    """The output file keeps the input's row order, not the last pass's shuffle."""
    survivors, _ = deduplicate_embeddings(
        np.eye(16, dtype=np.float32), steps_count=2, batch_size=4, similarity_threshold=0.9
    )
    assert survivors.tolist() == list(range(16))


def test_a_seeded_run_keeps_the_same_rows():
    """The shuffles decide which member of a near-duplicate group survives."""
    groups = np.repeat(np.eye(8, dtype=np.float32), 4, axis=0)
    runs = [deduplicate_embeddings(groups, steps_count=3, batch_size=6, similarity_threshold=0.9)[0] for _ in range(2)]
    np.testing.assert_array_equal(*runs)
    other_seed = deduplicate_embeddings(groups, steps_count=3, batch_size=6, similarity_threshold=0.9, seed=7)[0]
    assert other_seed.tolist() != runs[0].tolist(), "premise: the seed decides the survivors"


def test_the_passes_search_the_callers_matrix_not_copies(monkeypatch):
    """A copy of the matrix per pass would hold several copies of it at once."""
    matrix = np.concatenate([DUPLICATED_BASIS] * 4)
    searched = []
    real_blocks = deduplication.deduplicate_blocks

    def recording_blocks(embeddings, rows, **kwargs):
        searched.append(embeddings is matrix)
        return real_blocks(embeddings, rows, **kwargs)

    monkeypatch.setattr(deduplication, "deduplicate_blocks", recording_blocks)
    deduplication.select_unique_rows(matrix, similarity_threshold=0.9, batch_size=8, steps=3)
    assert searched == [True, True, True]


def test_the_expected_share_of_compared_pairs_is_logged(caplog):
    """Rows in different blocks never meet, so the share is what a run over a large corpus can catch."""
    with caplog.at_level(logging.WARNING, logger=deduplication.__name__):
        deduplication.select_unique_rows(DUPLICATED_BASIS, similarity_threshold=0.9, batch_size=4, steps=2)
        deduplication.select_unique_rows(DUPLICATED_BASIS, similarity_threshold=0.9, batch_size=8, steps=2)
    (warned,) = [r.getMessage() for r in caplog.records if "compared only within blocks" in r.getMessage()]
    assert "about 75%" in warned, warned


def test_the_search_takes_the_usable_cpu_count_not_the_hosts(monkeypatch, tmp_path):
    """faiss sizes its OpenMP pool by the host's cores, so under a container CPU quota every search oversubscribes."""
    monkeypatch.setattr(deduplication, "available_cpu_count", lambda: 3)
    held = faiss.omp_get_max_threads()
    try:
        faiss.omp_set_num_threads(held + 1)
        assert deduplication.run_deduplication(_run_args(tmp_path, DUPLICATED_BASIS, dry_run=True)) == 0
        assert faiss.omp_get_max_threads() == 3
        assert deduplication.run_deduplication(_run_args(tmp_path, DUPLICATED_BASIS, dry_run=True, max_workers=2)) == 0
        assert faiss.omp_get_max_threads() == 2
    finally:
        faiss.omp_set_num_threads(held)


def test_a_failed_write_leaves_neither_a_partial_output_nor_its_staging(monkeypatch, tmp_path):
    """A run killed mid-write would leave whole lines that load as a complete, smaller dataset."""
    args = _run_args(tmp_path, DUPLICATED_BASIS)
    output = tmp_path / "out" / "rows.jsonl"
    output.parent.mkdir()
    output.write_text("previous run\n")

    def partial_writer(dataset, path):
        with open(path, "w") as f:
            f.write(json.dumps({"text": "row 0"}) + "\n")
        raise OSError("killed mid-write")

    monkeypatch.setitem(deduplication.OUTPUT_WRITERS, "jsonl", partial_writer)
    assert deduplication.run_deduplication(args) == 1
    assert output.read_text() == "previous run\n"
    assert sorted(p.name for p in output.parent.iterdir()) == ["rows.jsonl"], "the staging was left behind"


def test_the_output_keeps_the_compression_its_name_asks_for(tmp_path):
    """The writer infers compression from the file name it is handed, so the staged file carries the output's."""
    args = _run_args(tmp_path, DUPLICATED_BASIS, output_path=str(tmp_path / "out" / "rows.jsonl.gz"))
    assert deduplication.run_deduplication(args) == 0
    with gzip.open(args.output_path, "rt") as f:
        assert [json.loads(line)["text"] for line in f] == ["row 0", "row 1", "row 2", "row 3"]


def test_the_row_sample_is_drawn_with_the_runs_fixed_seed(monkeypatch, tmp_path):
    """``--sample_size`` decides which rows are deduplicated at all, so it follows the seed the passes use."""
    data = tmp_path / "rows.jsonl"
    data.write_text("".join(json.dumps({"text": f"row {i}"}) + "\n" for i in range(64)))
    monkeypatch.setattr(deduplication, "DEDUP_SHUFFLE_SEED", 7)
    sample = deduplication.load_dedup_input(str(data), "text", None, "train", sample_size=8)["text"]
    whole = deduplication.load_dedup_input(str(data), "text", None, "train", sample_size=None)
    assert sample == whole.shuffle(seed=7).select(range(8))["text"]
    assert sample != whole.shuffle(seed=42).select(range(8))["text"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
