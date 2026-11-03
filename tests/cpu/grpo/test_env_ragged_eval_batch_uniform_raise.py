#!/usr/bin/env python
"""A ragged eval batch must fail env-GRPO on every rank. A rank-local raise on the ranks that got the remainder leaves
their peers in the batch fence's gather (``BatchBuildFence.reject``) until ``DIST_NCCL_TIMEOUT_MINUTES``."""

from __future__ import annotations

import datetime
import os
import types

import pytest

from halo.trainers.grpo.environmental.step_batch import BatchBuildFence
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 2
NUM_GENERATIONS_EVAL = 2

# Far above the work it bounds, because the suite runs 8-way under xdist. Gloo aborts a blocked collective at this
# timeout, so a stuck rank surfaces as a failed assertion.
PG_TIMEOUT = datetime.timedelta(seconds=90)


def _eval_trainer():
    host = types.SimpleNamespace(
        _group_random_effort=True,
        _batch_errors=BatchBuildFence(),
        num_generations=4,
        num_generations_eval=NUM_GENERATIONS_EVAL,
        model=types.SimpleNamespace(training=False),
    )
    cls = DistributedAsyncEnvironmentalGRPOTrainer
    host._stamp_group_efforts = cls._stamp_group_efforts.__get__(host)
    host._extract_prompts_and_contexts = cls._extract_prompts_and_contexts.__get__(host)
    return host


def _worker(rank: int, tmp_dir: str, ragged: bool) -> None:
    """Writes its verdict to a file: the failure under test is a rank that never returns."""
    rows = 3 if (ragged and rank == 0) else 4
    host = _eval_trainer()
    try:
        host._extract_prompts_and_contexts([{"prompt": "solve it"} for _ in range(rows)])
        host._batch_errors.reject()
        result = "NO RAISE"
    except Exception as e:
        result = f"{type(e).__name__}: {e}"
    with open(os.path.join(tmp_dir, f"result_{rank}.txt"), "w") as fh:
        fh.write(result)


def _run_ranks(tmp_path, ragged: bool) -> dict[int, str]:
    run_gloo_ranks(_worker, WORLD_SIZE, str(tmp_path), ragged, pg_timeout=PG_TIMEOUT)
    results = {}
    for rank in range(WORLD_SIZE):
        path = tmp_path / f"result_{rank}.txt"
        results[rank] = path.read_text() if path.exists() else "NO RESULT (the rank never returned)"
    return results


def test_a_ragged_eval_tail_raises_on_every_rank(tmp_path):
    """Rank 1 can only reach the error if rank 0 recorded the failure and still entered the gather. Every rank names
    the failing rank and its cause."""
    results = _run_ranks(tmp_path, ragged=True)

    for rank, result in results.items():
        assert result != "NO RAISE", f"rank {rank} sailed past a ragged eval batch: {results}"
        assert result.startswith("ValueError"), f"rank {rank} raised the wrong error: {results}"
        assert "multiple of the group size" in result, f"rank {rank} was not told the cause: {results}"
        assert "rank 0" in result, f"rank {rank} was not told which rank failed: {results}"


def test_an_aligned_eval_batch_still_stamps_and_returns(tmp_path):
    assert _run_ranks(tmp_path, ragged=False) == {0: "NO RAISE", 1: "NO RAISE"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
