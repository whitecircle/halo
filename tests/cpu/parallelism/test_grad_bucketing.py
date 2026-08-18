#!/usr/bin/env python
"""Bucketed gradient reduction equals per-tensor reduction (numerical equivalence).

``reduce_grads_bucketed`` coalesces many gradients sharing one (op, group, divisor) into a single
collective per dtype, replacing a chain of latency-bound all-reduces used by the post-backward
deferred cross-node DP sweep and the TP-replicated grad sync. This test proves it produces
BIT-IDENTICAL results to calling ``reduce_grad`` on each tensor, across a real 2-rank gloo group and
for every reduction kind the callers use (SUM+divisor, AVG, fp32-reduce), including mixed dtypes and
non-contiguous grads. A bug in the flatten/scatter offset math or the dtype bucketing would diverge.

    python tests/cpu/parallelism/test_grad_bucketing.py
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import src.distributed.grad_reduce as grad_reduce
from src.distributed.grad_reduce import reduce_grad, reduce_grads_bucketed
from tests.common.ports import free_port

WORLD_SIZE = 2


def _make_grads(rank: int) -> list[torch.Tensor]:
    """Distinct per-rank grads of assorted shapes/dtypes, incl. a non-contiguous one."""
    g = torch.manual_seed(1000 + rank)
    tensors = [
        torch.randn(4, 5, generator=g, dtype=torch.float32),
        torch.randn(7, generator=g, dtype=torch.float32),
        torch.randn(3, 3, generator=g, dtype=torch.bfloat16),
        torch.randn(2, 6, generator=g, dtype=torch.bfloat16).t(),  # non-contiguous
        torch.randn(1, generator=g, dtype=torch.float32),
    ]
    return tensors


def _worker(rank: int, tmp: str, port: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
    try:
        failures = []
        # cap=huge → single chunk per dtype; cap=32 bytes → forces the multi-chunk path (boundaries
        # must still align across ranks and the scatter offsets must stay correct).
        for cap in (grad_reduce.BUCKET_MAX_BYTES, 32):
            grad_reduce.BUCKET_MAX_BYTES = cap
            for op, divisor, fp32 in [
                (dist.ReduceOp.SUM, WORLD_SIZE, False),
                (dist.ReduceOp.AVG, None, False),
                (dist.ReduceOp.SUM, WORLD_SIZE, True),  # fp32 reduce path (affects the bf16 tensors)
            ]:
                reference = _make_grads(rank)
                for t in reference:
                    reduce_grad(t, op=op, divisor=divisor, group=None, fp32=fp32)

                bucketed = _make_grads(rank)
                reduce_grads_bucketed(bucketed, op=op, divisor=divisor, group=None, fp32=fp32)

                for i, (r, b) in enumerate(zip(reference, bucketed, strict=True)):
                    if not torch.equal(r, b):
                        failures.append(f"cap={cap} op={op} fp32={fp32} tensor[{i}] max|Δ|={(r - b).abs().max()}")

        # Empty input is a no-op (must not hang — no collective issued).
        reduce_grads_bucketed([], op=dist.ReduceOp.AVG, group=None)

        result = "PASS" if not failures else "FAIL: " + "; ".join(failures)
        if rank == 0:
            with open(tmp, "w") as fh:
                fh.write(result)
    finally:
        dist.destroy_process_group()


def test_bucketed_equals_per_tensor(tmp_path):
    out = str(tmp_path / "result.txt")
    mp.start_processes(_worker, args=(out, free_port()), nprocs=WORLD_SIZE, join=True, start_method="spawn")
    with open(out) as fh:
        result = fh.read()
    assert result == "PASS", result


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
