#!/usr/bin/env python
"""``halo/data/deduplication.py``: row normalization, the seeded shuffle that decides which member of
a near-duplicate group survives, and the FAISS passes' surviving indices."""

import time

import numpy as np
import pytest

from halo.data import deduplication
from halo.data.deduplication import (
    DEDUP_SHUFFLE_SEED,
    _normalize_embeddings,
    _shuffle_matrix_with_mapping,
    deduplicate_blocks,
    deduplicate_embeddings,
)

# Four orthogonal unit rows, each repeated once: rows i and i + 4 are exact duplicates.
DUPLICATED_BASIS = np.concatenate([np.eye(4, dtype=np.float32)] * 2)

# _normalize_embeddings


def test_normalize_unit_vectors_unchanged():
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(_normalize_embeddings(emb), emb, atol=1e-7)


def test_normalize_scales_each_row_to_unit_norm():
    emb = np.array([[3.0, 4.0], [0.0, 5.0], [-1.0, -1.0]])
    norms = np.linalg.norm(_normalize_embeddings(emb), axis=1)
    np.testing.assert_allclose(norms, np.ones(3), atol=1e-7)


def test_normalize_preserves_direction():
    # 3-4-5 triangle: [6, 8] / 10 = [0.6, 0.8].
    np.testing.assert_allclose(_normalize_embeddings(np.array([[6.0, 8.0]])), [[0.6, 0.8]], atol=1e-7)


def test_normalize_is_idempotent():
    rng = np.random.RandomState(1)
    emb = rng.randn(8, 32)
    once = _normalize_embeddings(emb)
    twice = _normalize_embeddings(once)
    np.testing.assert_allclose(once, twice, atol=1e-6)


# _shuffle_matrix_with_mapping


def test_shuffle_recoverable_via_mapping():
    rng = np.random.RandomState(7)
    matrix = rng.randn(20, 4)
    shuffled, mapping = _shuffle_matrix_with_mapping(matrix)
    np.testing.assert_array_equal(shuffled, matrix[mapping])


def test_shuffle_mapping_is_a_permutation():
    _, mapping = _shuffle_matrix_with_mapping(np.arange(15).reshape(5, 3))
    assert sorted(mapping.tolist()) == list(range(5))


def test_shuffle_default_rng_is_reproducible_across_calls():
    # The default generator is seeded from DEDUP_SHUFFLE_SEED, NOT numpy's global RNG: the shuffle
    # decides which member of a near-duplicate group dedup keeps, so two runs must keep the same rows.
    matrix = np.arange(50).reshape(10, 5)
    _, m1 = _shuffle_matrix_with_mapping(matrix)
    _, m2 = _shuffle_matrix_with_mapping(matrix)
    np.testing.assert_array_equal(m1, m2)


def test_shuffle_threaded_rng_permutes_differently_each_step():
    # deduplicate_embeddings threads ONE generator through its steps: each step must permute
    # differently (that is what re-batches near-duplicates across batch boundaries) while the run as a
    # whole replays from the seed. A step that re-seeded per call would repeat one permutation.
    matrix = np.arange(200).reshape(40, 5)
    rng = np.random.default_rng(DEDUP_SHUFFLE_SEED)
    mappings = [_shuffle_matrix_with_mapping(matrix, rng)[1] for _ in range(4)]

    assert len({m.tobytes() for m in mappings}) == len(mappings)

    replay_rng = np.random.default_rng(DEDUP_SHUFFLE_SEED)
    for step, mapping in enumerate(mappings):
        np.testing.assert_array_equal(_shuffle_matrix_with_mapping(matrix, replay_rng)[1], mapping, err_msg=f"{step=}")


# deduplicate_blocks / deduplicate_embeddings


def test_single_pass_keeps_the_first_row_of_each_duplicate_group():
    survivors = deduplicate_blocks(DUPLICATED_BASIS, max_workers=1, batch_size=8, similarity_threshold=0.9)
    assert sorted(survivors.tolist()) == [0, 1, 2, 3]


def test_single_pass_returns_global_indices_and_dedups_within_a_batch_only():
    # Duplicates sit in different batches here, so every row survives — under its GLOBAL index.
    survivors = deduplicate_blocks(DUPLICATED_BASIS, max_workers=2, batch_size=4, similarity_threshold=0.9)
    assert sorted(survivors.tolist()) == list(range(8))


def test_survivors_come_back_in_batch_order_whatever_order_the_batches_finish(monkeypatch):
    """The next pass shuffles the survivors with the seeded generator and the script selects rows in
    this order, so a thread-completion order would make a seeded run keep different rows. The batches
    are made to finish last-first here."""
    real_single = deduplication._deduplicate_block

    def last_batch_finishes_first(batch, similarity_threshold):
        time.sleep(0.1 * (3 - int(batch[0].argmax()) // 2))
        return real_single(batch, similarity_threshold)

    monkeypatch.setattr(deduplication, "_deduplicate_block", last_batch_finishes_first)
    survivors = deduplicate_blocks(np.eye(8, dtype=np.float32), max_workers=4, batch_size=2, similarity_threshold=0.9)
    assert survivors.tolist() == list(range(8))


def test_multistep_maps_survivors_back_to_the_input_rows():
    survivors, sizes = deduplicate_embeddings(
        DUPLICATED_BASIS, steps_count=3, max_workers=1, batch_size=8, similarity_threshold=0.9
    )
    assert sizes == [8, 4, 4, 4]
    assert sorted(i % 4 for i in survivors.tolist()) == [0, 1, 2, 3], "one survivor per duplicate pair"
    np.testing.assert_array_equal(np.sort(DUPLICATED_BASIS[survivors].argmax(axis=1)), [0, 1, 2, 3])


def test_unset_max_workers_reads_the_usable_cpu_count_at_call_time(monkeypatch):
    """The default follows available_cpu_count when the pass runs, so a container CPU quota caps the threads,
    not the host count frozen at import."""
    seen = []
    real_executor = deduplication.ThreadPoolExecutor

    def recording_executor(max_workers):
        seen.append(max_workers)
        return real_executor(max_workers=max_workers)

    monkeypatch.setattr(deduplication, "available_cpu_count", lambda: 3)
    monkeypatch.setattr(deduplication, "ThreadPoolExecutor", recording_executor)
    deduplicate_embeddings(DUPLICATED_BASIS, steps_count=2, batch_size=8, similarity_threshold=0.9)
    assert seen == [3, 3]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
