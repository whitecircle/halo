#!/usr/bin/env python
"""A ragged eval batch must fail env-GRPO on EVERY rank, never on the subset that received it.

``_stamp_group_efforts`` refuses a context count that is not a whole number of generation groups.
Train mode cannot produce one (the loader gate rejects the shape at construction), but eval can:
``dataloader_drop_last`` defaults to ``False``, so the final eval batch is split unevenly and the
remainder lands on a SUBSET of the DP ranks. A rank-local raise there leaves its peers in the
``_raise_batch_error_uniformly`` all-reduce until ``DIST_NCCL_TIMEOUT_MINUTES``, with the explaining
traceback only on the ranks that died — at 512 GPUs, 500 NCCL timeouts and 12 real tracebacks.

Proven on a real 2-rank gloo group with an asymmetric batch, so the regression manifests as the hang
it is: rank 1 can only be told the cause if rank 0 reached the all-reduce instead of raising out of
it. The group timeout and the join are both bounded, so a regression fails this suite rather than
wedging it.

    python tests/cpu/grpo/test_env_ragged_eval_batch_uniform_raise.py
"""

from __future__ import annotations

import contextlib
import datetime
import os
import time
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer
from tests.common.ports import free_port

WORLD_SIZE = 2
NUM_GENERATIONS_EVAL = 2

# A stuck rank must surface as a failed assertion, not a wedged worker: gloo aborts the blocked
# collective at the group timeout and the bounded join covers the process itself. Both are far above
# the work they bound, because the suite runs 8-way under xdist.
PG_TIMEOUT = datetime.timedelta(seconds=90)
JOIN_TIMEOUT_S = 420.0


def _eval_trainer():
    """Minimal stand-in exposing exactly what the batch-build path reads, in EVAL mode."""
    host = types.SimpleNamespace(
        _group_random_effort=True,
        _batch_build_error=None,
        num_generations=4,
        num_generations_eval=NUM_GENERATIONS_EVAL,
        model=types.SimpleNamespace(training=False),
    )
    cls = DistributedAsyncEnvironmentalGRPOTrainer
    host._stamp_group_efforts = cls._stamp_group_efforts.__get__(host)
    host._extract_prompts_and_contexts = cls._extract_prompts_and_contexts.__get__(host)
    host._raise_batch_error_uniformly = cls._raise_batch_error_uniformly.__get__(host)
    return host


def _worker(rank: int, tmp_dir: str, port: int, ragged: bool) -> None:
    """One rank of the batch build. Writes its verdict to a file — the failure under test is a rank
    that never returns at all, so an exception here cannot be the signal."""
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE, timeout=PG_TIMEOUT)
    try:
        # The ragged tail: rank 0 gets an odd row count, its peer a whole number of groups.
        rows = 3 if (ragged and rank == 0) else 4
        host = _eval_trainer()
        try:
            host._extract_prompts_and_contexts([{"prompt": "solve it"} for _ in range(rows)])
            host._raise_batch_error_uniformly(torch.device("cpu"))
            result = "NO RAISE"
        except Exception as e:
            result = f"{type(e).__name__}: {e}"
        with open(os.path.join(tmp_dir, f"result_{rank}.txt"), "w") as fh:
            fh.write(result)
    finally:
        with contextlib.suppress(Exception):
            dist.destroy_process_group()


def _run_ranks(tmp_path, ragged: bool) -> dict[int, str]:
    ctx = mp.start_processes(
        _worker, args=(str(tmp_path), free_port(), ragged), nprocs=WORLD_SIZE, join=False, start_method="spawn"
    )
    deadline = time.monotonic() + JOIN_TIMEOUT_S
    while not ctx.join(timeout=max(0.1, deadline - time.monotonic())):
        if time.monotonic() >= deadline:
            for process in ctx.processes:
                process.terminate()
            pytest.fail(f"the ranks did not finish within {JOIN_TIMEOUT_S}s — a rank is stuck in a collective")

    results = {}
    for rank in range(WORLD_SIZE):
        path = tmp_path / f"result_{rank}.txt"
        results[rank] = path.read_text() if path.exists() else "NO RESULT (the rank never returned)"
    return results


def test_a_ragged_eval_tail_raises_on_every_rank(tmp_path):
    """The P1: rank 0's eval batch does not divide by ``num_generations_eval``, rank 1's does.

    Rank 1 reaching the error is the whole proof — it can only get there if rank 0 recorded the
    failure and still entered the all-reduce. Raising rank-locally instead leaves rank 1 blocked in
    that all-reduce until the watchdog.
    """
    results = _run_ranks(tmp_path, ragged=True)

    for rank, result in results.items():
        assert result != "NO RAISE", f"rank {rank} sailed past a ragged eval batch: {results}"
        assert result.startswith("ValueError"), f"rank {rank} raised the wrong error: {results}"
    assert "multiple of the group size" in results[0], f"the offending rank did not name the cause: {results}"
    assert "peer rank" in results[1], f"the healthy rank was not told a peer failed: {results}"


def test_an_aligned_eval_batch_still_stamps_and_returns(tmp_path):
    """Anti-vacuity: the uniform raise must not fire on a batch that divides evenly on both ranks."""
    assert _run_ranks(tmp_path, ragged=False) == {0: "NO RAISE", 1: "NO RAISE"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
