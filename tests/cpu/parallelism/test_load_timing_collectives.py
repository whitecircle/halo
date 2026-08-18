#!/usr/bin/env python
"""``log_global_load_duration_seconds`` — the load-timing report costs exactly two all_reduces.

Every parallelism mode's loader ends in this call, so whatever it does runs once per model load on
every rank. It reduces the earliest start and the latest end and reports the span; reading the
reduced value already blocks until every rank has contributed, so a barrier on top of it buys no
ordering and only adds a world-wide collective — at 512 ranks that is the whole cost of the report.

Proven on a real gloo group: the span is the cross-rank ``max(end) - min(start)`` (not this rank's
own interval), and no barrier is issued.

    python tests/cpu/parallelism/test_load_timing_collectives.py
"""

import datetime
import os
import sys

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from src.distributed.runtime import log_global_load_duration_seconds
from tests.common.ports import free_port

WORLD_SIZE = 2

# Bounded so a regression fails the suite instead of stalling it (mp.start_processes has no timeout).
PG_TIMEOUT_SEC = 60

# Deliberately skewed intervals: rank 0 starts first, rank 1 finishes last, and NEITHER rank's own
# interval (10 s / 25 s) equals the global span (30 s) — a local-only report cannot pass.
RANK_INTERVALS = {0: (100.0, 110.0), 1: (105.0, 130.0)}
GLOBAL_SPAN_SEC = 30.0


def _worker(rank: int, tmp_dir: str, port: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE))
    dist.init_process_group(
        "gloo", rank=rank, world_size=WORLD_SIZE, timeout=datetime.timedelta(seconds=PG_TIMEOUT_SEC)
    )
    barriers = 0
    real_barrier = dist.barrier

    def counting_barrier(*args, **kwargs):
        nonlocal barriers
        barriers += 1
        return real_barrier(*args, **kwargs)

    dist.barrier = counting_barrier
    try:
        start, end = RANK_INTERVALS[rank]
        span = log_global_load_duration_seconds(tag="unit", method="unit", t_start_wall=start, t_end_wall=end)
        result = f"{span:.3f}|{barriers}"
    except Exception as e:
        result = f"{type(e).__name__}: {e}"
    finally:
        dist.barrier = real_barrier
        dist.destroy_process_group()
    with open(os.path.join(tmp_dir, f"result_{rank}.txt"), "w") as fh:
        fh.write(result)


def test_the_span_is_global_and_costs_no_barrier(tmp_path):
    mp.start_processes(_worker, args=(str(tmp_path), free_port()), nprocs=WORLD_SIZE, join=True, start_method="spawn")
    for rank in range(WORLD_SIZE):
        with open(tmp_path / f"result_{rank}.txt") as fh:
            result = fh.read()
        assert result == f"{GLOBAL_SPAN_SEC:.3f}|0", f"rank {rank}: {result}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
