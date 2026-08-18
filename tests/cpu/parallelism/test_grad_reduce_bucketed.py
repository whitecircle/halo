#!/usr/bin/env python
"""CPU tests for ``reduce_grads_bucketed`` (``src/distributed/grad_reduce.py``) on a one-rank gloo group.

A one-rank all-reduce is the identity, so the numerics pin the framing around the collective: the
chunking, the divisor, the fp32 upcast + write-back, and the scatter-back into every gradient in
place. The lone-chunk case is load-bearing at scale — every fused expert gradient of a 100B+ MoE is
above the bucket cap and forms a chunk of its own — so it must be reduced in its own storage rather
than through a flat copy and back.

Run: ``pytest tests/cpu/parallelism/test_grad_reduce_bucketed.py``.
"""

import pytest
import torch
import torch.distributed as dist

from src.distributed import grad_reduce
from src.distributed.grad_reduce import _launch_flat_chunk, reduce_grads_bucketed


@pytest.fixture
def one_rank_group(tmp_path):
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.fixture
def tiny_bucket(monkeypatch):
    """A 1 KiB bucket cap so a handful of small tensors exercise every chunking branch."""
    monkeypatch.setattr(grad_reduce, "BUCKET_MAX_BYTES", 1024)


def _grads(dtype: torch.dtype) -> list[torch.Tensor]:
    torch.manual_seed(0)
    # 16 + 16 elements fit one 1 KiB chunk (bf16: 64 B; fp32: 128 B); the 4096-element tensor is a
    # chunk of its own on either dtype; the trailing pair opens a third chunk.
    shapes = [(4, 4), (2, 8), (64, 64), (3, 5), (5, 3)]
    return [torch.randn(*shape, dtype=dtype) for shape in shapes]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("fp32", [False, True])
def test_reduce_scales_every_grad_in_place(one_rank_group, tiny_bucket, dtype, fp32):
    grads = _grads(dtype)
    originals = [g.clone() for g in grads]
    ptrs = [g.data_ptr() for g in grads]

    reduce_grads_bucketed(grads, op=dist.ReduceOp.SUM, divisor=4, fp32=fp32)

    for got, want, ptr in zip(grads, originals, ptrs, strict=True):
        assert got.data_ptr() == ptr, "the reduce must write back into the gradient's own storage"
        torch.testing.assert_close(got, (want.float() / 4).to(dtype), rtol=0, atol=0)


def test_mixed_dtypes_are_reduced_in_separate_buckets(one_rank_group, tiny_bucket):
    grads = [torch.ones(8, dtype=torch.bfloat16), torch.ones(8), torch.full((8,), 2.0, dtype=torch.bfloat16)]
    reduce_grads_bucketed(grads, op=dist.ReduceOp.SUM, divisor=2)
    assert grads[0].dtype is torch.bfloat16 and grads[1].dtype is torch.float32
    torch.testing.assert_close(grads[0], torch.full((8,), 0.5, dtype=torch.bfloat16))
    torch.testing.assert_close(grads[1], torch.full((8,), 0.5))
    torch.testing.assert_close(grads[2], torch.ones(8, dtype=torch.bfloat16))


def test_lone_chunk_is_reduced_in_its_own_storage(one_rank_group):
    grad = torch.randn(64, 64, dtype=torch.bfloat16)
    bucket, flat, buf, work, scatter = _launch_flat_chunk([grad], op=dist.ReduceOp.SUM, group=None, fp32=False)
    work.wait()
    assert flat.data_ptr() == grad.data_ptr(), "a single contiguous grad must not be copied into a flat buffer"
    assert buf is flat and not scatter


def test_lone_chunk_fp32_upcast_writes_back_to_the_grad(one_rank_group):
    grad = torch.randn(64, 64, dtype=torch.bfloat16)
    want = (grad.float() / 3).to(torch.bfloat16)
    bucket, flat, buf, work, scatter = _launch_flat_chunk([grad], op=dist.ReduceOp.SUM, group=None, fp32=True)
    assert buf.dtype is torch.float32 and buf is not flat and flat.data_ptr() == grad.data_ptr()
    grad_reduce._finish_flat_chunk(bucket, flat, buf, work, scatter, divisor=3)
    torch.testing.assert_close(grad, want, rtol=0, atol=0)


def test_multi_tensor_chunk_is_flattened_and_scattered_back(one_rank_group):
    grads = [torch.randn(4, 4, dtype=torch.bfloat16), torch.randn(2, 8, dtype=torch.bfloat16)]
    bucket, flat, buf, work, scatter = _launch_flat_chunk(grads, op=dist.ReduceOp.SUM, group=None, fp32=False)
    assert scatter and flat.numel() == 32
    assert flat.data_ptr() not in {g.data_ptr() for g in grads}
    flat.mul_(2)  # stands in for the peers' contributions
    grad_reduce._finish_flat_chunk(bucket, flat, buf, work, scatter, divisor=None)
    torch.testing.assert_close(torch.cat([g.reshape(-1) for g in grads]), flat)


def test_non_contiguous_lone_grad_still_takes_the_flat_path(one_rank_group):
    base = torch.randn(8, 8, dtype=torch.bfloat16)
    grad = base.t()  # non-contiguous view: reducing it "in place" through a flat view is impossible
    bucket, flat, buf, work, scatter = _launch_flat_chunk([grad], op=dist.ReduceOp.SUM, group=None, fp32=False)
    assert scatter and flat.is_contiguous() and flat.data_ptr() != base.data_ptr()
    grad_reduce._finish_flat_chunk(bucket, flat, buf, work, scatter, divisor=2)
    torch.testing.assert_close(base, (base * 1).to(torch.bfloat16))  # base was written through the view
    torch.testing.assert_close(grad, flat.view_as(grad))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
