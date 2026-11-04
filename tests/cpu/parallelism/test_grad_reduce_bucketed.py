#!/usr/bin/env python
"""``reduce_grads_bucketed`` (``halo/distributed/grad_reduce.py``) framing on a one-rank gloo group. Every fused
expert grad of a 100B+ MoE is its own chunk, reduced in its own storage."""

import pytest
import torch
import torch.distributed as dist

from halo.distributed import grad_reduce
from halo.distributed.grad_reduce import _launch_flat_chunk, reduce_grads_bucketed


@pytest.fixture
def one_rank_group(tmp_path):
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.fixture
def tiny_bucket(monkeypatch):
    """Small tensors exercise every chunking branch."""
    monkeypatch.setattr(grad_reduce, "BUCKET_MAX_BYTES", 1024)


def _grads(dtype: torch.dtype) -> list[torch.Tensor]:
    torch.manual_seed(0)
    # 16 + 16 elements fit one 1 KiB chunk; the 4096-element tensor is its own; the trailing pair opens a third.
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
    pending = _launch_flat_chunk([grad], op=dist.ReduceOp.SUM, group=None, fp32=False)
    pending.work.wait()
    assert pending.flat.data_ptr() == grad.data_ptr(), "a single contiguous grad must not be copied into a flat buffer"
    assert pending.buf is pending.flat and not pending.scatter


def test_lone_chunk_fp32_upcast_writes_back_to_the_grad(one_rank_group):
    grad = torch.randn(64, 64, dtype=torch.bfloat16)
    want = (grad.float() / 3).to(torch.bfloat16)
    pending = _launch_flat_chunk([grad], op=dist.ReduceOp.SUM, group=None, fp32=True)
    assert (
        pending.buf.dtype is torch.float32
        and pending.buf is not pending.flat
        and pending.flat.data_ptr() == grad.data_ptr()
    )
    grad_reduce._finish_flat_chunk(pending, divisor=3)
    torch.testing.assert_close(grad, want, rtol=0, atol=0)


def test_multi_tensor_chunk_is_flattened_and_scattered_back(one_rank_group):
    grads = [torch.randn(4, 4, dtype=torch.bfloat16), torch.randn(2, 8, dtype=torch.bfloat16)]
    pending = _launch_flat_chunk(grads, op=dist.ReduceOp.SUM, group=None, fp32=False)
    assert pending.scatter and pending.flat.numel() == 32
    assert pending.flat.data_ptr() not in {g.data_ptr() for g in grads}
    pending.flat.mul_(2)  # Stands in for the peers' contributions.
    grad_reduce._finish_flat_chunk(pending, divisor=None)
    torch.testing.assert_close(torch.cat([g.reshape(-1) for g in grads]), pending.flat)


def test_non_contiguous_lone_grad_still_takes_the_flat_path(one_rank_group):
    base = torch.randn(8, 8, dtype=torch.bfloat16)
    grad = base.t()  # A non-contiguous view cannot be reduced in place through a flat view.
    pending = _launch_flat_chunk([grad], op=dist.ReduceOp.SUM, group=None, fp32=False)
    assert pending.scatter and pending.flat.is_contiguous() and pending.flat.data_ptr() != base.data_ptr()
    grad_reduce._finish_flat_chunk(pending, divisor=2)
    torch.testing.assert_close(base, (base * 1).to(torch.bfloat16))  # base was written through the view.
    torch.testing.assert_close(grad, pending.flat.view_as(grad))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
