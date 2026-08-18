#!/usr/bin/env python
"""The Ulysses all-to-all primitive refuses a scatter dim that cp_size does not divide.

``UlyssesAllToAll.forward`` narrows ``world_size`` equal chunks out of the scatter dim; on an
indivisible input the ranged narrows silently drop the tail elements — wrong attention with no
error. The wrappers validate head/sequence divisibility before ever reaching the primitive, so this
pins the primitive's own guard (every rank raises before the collective, so nothing hangs).

    python tests/cpu/parallelism/test_ulysses_all_to_all_divisibility.py
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.distributed.context_parallel.autograd import UlyssesAllToAll
from tests.common.ports import free_port

WORLD_SIZE = 2


def _worker(rank: int, tmp: str, port: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
    try:
        failures = []

        # Indivisible scatter dim (5 % 2 != 0) raises on EVERY rank, ahead of the collective.
        try:
            UlyssesAllToAll.apply(torch.randn(1, 5, 4), dist.group.WORLD, 1, 2)
            failures.append("indivisible scatter dim did not raise")
        except ValueError as e:
            if "not divisible" not in str(e):
                failures.append(f"raised the wrong error: {e}")

        # A divisible input must get PAST the guard, all the way to the backend collective — gloo
        # then rejects alltoall with its own RuntimeError, which proves the guard did not over-fire
        # (the real collective path is exercised by the CP GPU suites).
        try:
            UlyssesAllToAll.apply(torch.randn(1, 4, 6), dist.group.WORLD, 1, 2)
        except ValueError:
            failures.append("divisible scatter dim tripped the divisibility guard")
        except RuntimeError:
            pass

        result = "PASS" if not failures else "FAIL: " + "; ".join(failures)
        if rank == 0:
            with open(tmp, "w") as fh:
                fh.write(result)
    finally:
        dist.destroy_process_group()


def test_indivisible_scatter_dim_raises_divisible_roundtrips(tmp_path):
    out = str(tmp_path / "result.txt")
    mp.start_processes(_worker, args=(out, free_port()), nprocs=WORLD_SIZE, join=True, start_method="spawn")
    with open(out) as fh:
        result = fh.read()
    assert result == "PASS", result


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
