#!/usr/bin/env python
"""``_log_global_load_duration_seconds`` costs exactly two all_reduces, on a real gloo group.
Reading the reduced value already blocks, so a barrier only adds a world-wide collective."""

import datetime
import os

import pytest
import torch.distributed as dist

from halo.runtime.load_coordination import _log_global_load_duration_seconds
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 2

# So its gloo error lands in the verdict before the join deadline.
PG_TIMEOUT_SEC = 60

# Neither rank's own interval (10 s / 25 s) equals the global span (30 s), so a local report cannot pass.
RANK_INTERVALS = {0: (100.0, 110.0), 1: (105.0, 130.0)}
GLOBAL_SPAN_SEC = 30.0


def _worker(rank: int, tmp_dir: str) -> None:
    barriers = 0
    real_barrier = dist.barrier

    def counting_barrier(*args, **kwargs):
        nonlocal barriers
        barriers += 1
        return real_barrier(*args, **kwargs)

    dist.barrier = counting_barrier
    try:
        start, end = RANK_INTERVALS[rank]
        span = _log_global_load_duration_seconds(tag="unit", method="unit", t_start_wall=start, t_end_wall=end)
        result = f"{span:.3f}|{barriers}"
    except Exception as e:
        result = f"{type(e).__name__}: {e}"
    finally:
        dist.barrier = real_barrier
    with open(os.path.join(tmp_dir, f"result_{rank}.txt"), "w") as fh:
        fh.write(result)


def test_the_span_is_global_and_costs_no_barrier(tmp_path):
    run_gloo_ranks(_worker, WORLD_SIZE, str(tmp_path), pg_timeout=datetime.timedelta(seconds=PG_TIMEOUT_SEC))
    for rank in range(WORLD_SIZE):
        with open(tmp_path / f"result_{rank}.txt") as fh:
            result = fh.read()
        assert result == f"{GLOBAL_SPAN_SEC:.3f}|0", f"rank {rank}: {result}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
