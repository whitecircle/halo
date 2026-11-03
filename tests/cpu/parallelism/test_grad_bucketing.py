#!/usr/bin/env python
"""Bucketed gradient reduction is bit-identical to per-tensor reduction on a real 2-rank gloo group, for every
reduction kind callers use."""

import pytest
import torch
import torch.distributed as dist

import halo.distributed.grad_reduce as grad_reduce
from halo.distributed.grad_reduce import reduce_grad, reduce_grads_bucketed
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 2


def _make_grads(rank: int) -> list[torch.Tensor]:
    """Incl. a non-contiguous one."""
    g = torch.manual_seed(1000 + rank)
    tensors = [
        torch.randn(4, 5, generator=g, dtype=torch.float32),
        torch.randn(7, generator=g, dtype=torch.float32),
        torch.randn(3, 3, generator=g, dtype=torch.bfloat16),
        torch.randn(2, 6, generator=g, dtype=torch.bfloat16).t(),
        torch.randn(1, generator=g, dtype=torch.float32),
    ]
    return tensors


def _worker(rank: int, tmp: str) -> None:
    failures = []
    # cap=32 bytes forces the multi-chunk path, whose boundaries must align across ranks.
    for cap in (grad_reduce.BUCKET_MAX_BYTES, 32):
        grad_reduce.BUCKET_MAX_BYTES = cap
        for op, divisor, fp32 in [
            (dist.ReduceOp.SUM, WORLD_SIZE, False),
            (dist.ReduceOp.AVG, None, False),
            (dist.ReduceOp.SUM, WORLD_SIZE, True),  # fp32 reduce path (affects the bf16 tensors).
        ]:
            reference = _make_grads(rank)
            for t in reference:
                reduce_grad(t, op=op, divisor=divisor, group=None, fp32=fp32)

            bucketed = _make_grads(rank)
            reduce_grads_bucketed(bucketed, op=op, divisor=divisor, group=None, fp32=fp32)

            for i, (r, b) in enumerate(zip(reference, bucketed, strict=True)):
                if not torch.equal(r, b):
                    failures.append(f"cap={cap} op={op} fp32={fp32} tensor[{i}] max|Δ|={(r - b).abs().max()}")

    # Must not hang: no collective issued.
    reduce_grads_bucketed([], op=dist.ReduceOp.AVG, group=None)

    result = "PASS" if not failures else "FAIL: " + "; ".join(failures)
    if rank == 0:
        with open(tmp, "w") as fh:
            fh.write(result)


def test_bucketed_equals_per_tensor(tmp_path):
    out = str(tmp_path / "result.txt")
    run_gloo_ranks(_worker, WORLD_SIZE, out)
    with open(out) as fh:
        result = fh.read()
    assert result == "PASS", result


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
