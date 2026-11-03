#!/usr/bin/env python
"""The optimizer-shard resume reads under the node's load throttle, on every rank, joining the world over the store
on exit. Unthrottled, a node peaks at ``local_world_size`` x (own + writer) shards."""

from __future__ import annotations

import datetime
import os
import time

import pytest
import torch.distributed as dist
import torch.nn as nn

from halo.checkpoint.parallel.context import CheckpointLoadContext
from halo.checkpoint.parallel.optimizer import OptimizerShardStore
from tests.common.checkpoint_io import load_context
from tests.common.gloo import run_gloo_ranks
from tests.common.parallelism import make_parallelism_config

WORLD_SIZE = 2
# Long enough that concurrent reads overlap unmistakably, short enough to keep the suite fast.
READ_SECONDS = 0.4
PG_TIMEOUT_SEC = 60
# Twice the gloo timeout the join test runs under.
SLOW_READ_SECONDS = 6


def _context(model: nn.Module) -> CheckpointLoadContext:
    config = make_parallelism_config(world_size=WORLD_SIZE, gpus_per_node=WORLD_SIZE, max_concurrent_loading=1)
    return load_context(model, config, fsdp_wrapped=True)


def _worker(rank: int, tmp_dir: str) -> None:
    store = OptimizerShardStore(_context(nn.Linear(2, 2)))
    window = []

    def read(_path):
        window.append(time.monotonic())
        time.sleep(READ_SECONDS)
        window.append(time.monotonic())
        return {"state": {}, "param_groups": []}, None

    store._read_shard = read
    _osd, failure = store._read_local_state(tmp_dir, os.path.join(tmp_dir, "optimizer_shard_00000.pt"))
    # The store server lives in rank 0's process, so no rank may leave while a peer has store work.
    dist.barrier()

    with open(os.path.join(tmp_dir, f"window_{rank}.txt"), "w") as fh:
        fh.write(f"{failure}|{window[0]}|{window[1]}")


def _slow_or_failing_last_batch(rank: int, tmp_dir: str, mode: str) -> None:
    """Slowly past the gloo timeout, or raising out of the read."""
    store = OptimizerShardStore(_context(nn.Linear(2, 2)))

    def read(_path):
        if rank == 1 and mode == "slow":
            time.sleep(SLOW_READ_SECONDS)
        if rank == 1 and mode == "raises":
            raise ValueError("shard decoder bug on rank 1")
        return {"state": {}, "param_groups": []}, None

    store._read_shard = read
    try:
        store._read_local_state(tmp_dir, os.path.join(tmp_dir, "optimizer_shard_00000.pt"))
        dist.barrier()
        outcome = "joined"
    except Exception as e:  # The verdict file must be written whatever the rank raised.
        outcome = f"{type(e).__name__}: {e}"
    with open(os.path.join(tmp_dir, f"join_{rank}.txt"), "w") as fh:
        fh.write(outcome)


def test_the_first_batch_waits_for_the_last_over_the_store(tmp_path):
    run_gloo_ranks(
        _slow_or_failing_last_batch, WORLD_SIZE, str(tmp_path), "slow", pg_timeout=datetime.timedelta(seconds=3)
    )
    for rank in range(WORLD_SIZE):
        outcome = (tmp_path / f"join_{rank}.txt").read_text()
        assert outcome == "joined", f"rank {rank} waited for the slow batch under the collective watchdog: {outcome}"


def test_an_exception_escaping_one_ranks_read_raises_on_every_rank(tmp_path):
    run_gloo_ranks(
        _slow_or_failing_last_batch, WORLD_SIZE, str(tmp_path), "raises", pg_timeout=datetime.timedelta(seconds=30)
    )
    assert (tmp_path / "join_1.txt").read_text() == "ValueError: shard decoder bug on rank 1"
    peer = (tmp_path / "join_0.txt").read_text()
    assert peer.startswith("RuntimeError") and "shard decoder bug on rank 1" in peer, peer
    assert "timed out" not in peer.lower(), peer


def _windows(tmp_path) -> list[tuple[float, float]]:
    # The runner exports LOCAL_RANK/LOCAL_WORLD_SIZE, so both ranks share one node's throttle.
    run_gloo_ranks(_worker, WORLD_SIZE, str(tmp_path), pg_timeout=datetime.timedelta(seconds=PG_TIMEOUT_SEC))
    windows = []
    for rank in range(WORLD_SIZE):
        with open(os.path.join(str(tmp_path), f"window_{rank}.txt")) as fh:
            failure, start, end = fh.read().split("|")
        assert failure == "None", f"rank {rank} did not complete its read: {failure}"
        windows.append((float(start), float(end)))
    return windows


def test_concurrent_ranks_do_not_read_their_shards_at_the_same_time(tmp_path):
    """Bounds per-node host RAM and the fan-in on the replica writer's single file."""
    (first, second) = sorted(_windows(tmp_path))
    assert first[1] <= second[0], (
        f"the two ranks' shard reads overlapped ({first} vs {second}): the resume read is outside "
        f"the node load throttle, so every local rank materializes its shard (and, under multi-group "
        f"EP, its replica writer's) at once."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
