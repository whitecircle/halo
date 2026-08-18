#!/usr/bin/env python
"""The optimizer-shard resume reads under the node's load throttle, on every rank.

A resume materializes each rank's shard in host RAM (``torch.load(map_location="cpu")``), and under
multi-group EP every follower of a replica group additionally reads its writer's shard — the SAME
file for all of them. Unthrottled, a node peaks at ``local_world_size`` × (own + writer) shards at
once and one writer's file is opened by one reader per follower (63 of them at ``ep8`` on 512 GPUs).
The throttle is the toolkit's existing answer to "many ranks reading big files at once", and it is a
store phase: its participants are the node's local ranks, so entering it from a rank-dependent branch
would leave a peer waiting out the whole store timeout on a key nobody writes.

Two ranks, real gloo, ``max_concurrent_loading=1``: the reads must not overlap, which is what the
throttle buys and what an unwrapped ``torch.load`` cannot deliver. A dense model is used on purpose —
the throttle is entered by every rank, not only the ones with a replica writer to read.

    python tests/cpu/checkpoint/test_optimizer_shard_read_throttle.py
"""

from __future__ import annotations

import contextlib
import datetime
import os
import time

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from src.distributed.checkpoint.context import CheckpointLoadContext
from src.distributed.checkpoint.optimizer import OptimizerShardStore
from tests.common.ports import free_port

WORLD_SIZE = 2
# The read each rank simulates. Long enough that concurrent reads would overlap unmistakably, short
# enough to keep the suite fast.
READ_SECONDS = 0.4
PG_TIMEOUT_SEC = 60


class _Config:
    """The one field the store reads off the parallelism config here."""

    max_concurrent_loading = 1


def _context(model: nn.Module) -> CheckpointLoadContext:
    return CheckpointLoadContext(
        model=model,
        optimizer=None,
        lr_scheduler=None,
        parallelism_config=_Config(),
        is_pp_mode=False,
        is_cp_mode=False,
        is_tp_mode=False,
        has_ep_layers=False,
        fsdp_wrapped=True,
        tp_rank=0,
        tp_size=1,
        super_load_from_checkpoint=lambda *a, **k: None,
        super_load_optimizer_and_scheduler=lambda *a, **k: None,
    )


def _worker(rank: int, tmp_dir: str, port: str) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=port,
        RANK=str(rank),
        WORLD_SIZE=str(WORLD_SIZE),
        LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE=str(WORLD_SIZE),
    )
    dist.init_process_group(
        "gloo", rank=rank, world_size=WORLD_SIZE, timeout=datetime.timedelta(seconds=PG_TIMEOUT_SEC)
    )
    store = OptimizerShardStore(_context(nn.Linear(2, 2)))
    window = []

    def read(_path):
        window.append(time.monotonic())
        time.sleep(READ_SECONDS)
        window.append(time.monotonic())
        return {"state": {}, "param_groups": []}, True

    store._read_shard = read
    _osd, ok = store._read_local_state(tmp_dir, os.path.join(tmp_dir, "optimizer_shard_00000.pt"))
    # As ``load`` does: the store server lives in rank 0's process, so no rank may leave while a
    # peer still has store work (the throttle's own bookkeeping) to do.
    dist.barrier()

    with open(os.path.join(tmp_dir, f"window_{rank}.txt"), "w") as fh:
        fh.write(f"{ok}|{window[0]}|{window[1]}")
    with contextlib.suppress(Exception):
        dist.destroy_process_group()


def _windows(tmp_path) -> list[tuple[float, float]]:
    port = str(free_port())
    mp.start_processes(_worker, args=(str(tmp_path), port), nprocs=WORLD_SIZE, join=True, start_method="spawn")
    windows = []
    for rank in range(WORLD_SIZE):
        with open(os.path.join(str(tmp_path), f"window_{rank}.txt")) as fh:
            ok, start, end = fh.read().split("|")
        assert ok == "True", f"rank {rank} did not complete its read"
        windows.append((float(start), float(end)))
    return windows


def test_concurrent_ranks_do_not_read_their_shards_at_the_same_time(tmp_path):
    """``max_concurrent_loading=1`` means one reader per node at a time — the peak this bounds is
    per-node host RAM, and the fan-in it bounds is the replica writer's single file."""
    (first, second) = sorted(_windows(tmp_path))
    assert first[1] <= second[0], (
        f"the two ranks' shard reads overlapped ({first} vs {second}): the resume read is outside "
        f"the node load throttle, so every local rank materializes its shard (and, under multi-group "
        f"EP, its replica writer's) at once."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
