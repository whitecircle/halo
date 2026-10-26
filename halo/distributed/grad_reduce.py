"""Gradient all-reduce primitives for every post-backward sync path. A leaf (torch, :mod:`halo.env` and
:mod:`halo.runtime.collectives` only), so any package can reduce gradients without importing a
parallelism package.
"""

from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple

import torch
import torch.distributed as dist

from halo.env import env_int
from halo.runtime.collectives import collective_device

# Flat-buffer cap for bucketed reduction: bounds the transient cat allocation while keeping collectives few.
GRAD_BUCKET_MB = env_int("HALO_GRAD_BUCKET_MB", 256)
BUCKET_MAX_BYTES = GRAD_BUCKET_MB * 1024 * 1024
# Buckets in flight: after backward the only latency to hide is the sweep's own framing (cat, fp32
# upcast, scatter-back). Peak transient is this many flat buffers, 3x under ``fp32`` (bf16 flat + upcast).
BUCKET_MAX_INFLIGHT = env_int("HALO_GRAD_BUCKET_MAX_INFLIGHT", 2)


class _PendingChunk(NamedTuple):
    """One chunk's in-flight all-reduce: ``buf`` is ``flat`` or its fp32 upcast; ``scatter`` marks a
    concatenated ``flat`` that must be copied back into ``bucket``."""

    bucket: list[torch.Tensor]
    flat: torch.Tensor
    buf: torch.Tensor
    work: dist.Work
    scatter: bool


def _scale_and_write_back(buf: torch.Tensor, target: torch.Tensor, divisor: float | None) -> None:
    """Divide the reduced ``buf`` by ``divisor`` if given, then copy it into ``target`` if separate."""
    if divisor is not None:
        buf.div_(divisor)
    if buf is not target:
        target.copy_(buf)


def reduce_grad(
    grad: torch.Tensor,
    *,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    divisor: float | None = None,
    group: dist.ProcessGroup | None = None,
    fp32: bool = False,
) -> None:
    """All-reduce a gradient in place over ``group`` with ``op``, then scale by ``1 / divisor`` if given.

    ``fp32`` reduces and scales an fp32 copy and writes it back in the original dtype (FSDP2's
    ``reduce_dtype=fp32``: a bf16 reduce loses ~10^4x precision).
    """
    t = grad if grad.is_contiguous() else grad.contiguous()
    buf = t.float() if fp32 and t.dtype != torch.float32 else t
    dist.all_reduce(buf, op=op, group=group)
    _scale_and_write_back(buf, t, divisor)
    if t is not grad:
        grad.copy_(t)


class SumGradAcrossGroup(torch.autograd.Function):
    """Identity forward; SUM all-reduce of the gradient over ``group`` in backward, for an activation
    whose consumers are sharded over ``group`` (expert-TP's MoE input, TP's MLA rope rows). At the
    activation it runs once per backward; a hook on the weight would re-reduce ``.grad`` per micro-step.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        reduce_grad(grad, group=ctx.group)
        return grad, None


def agree_grad_presence(
    params: list[torch.nn.Parameter], group: dist.ProcessGroup | None = None
) -> list[torch.nn.Parameter]:
    """COLLECTIVE over ``group``: the ``params`` some rank produced a gradient for, each now holding one.

    Grad presence is rank-local (a router tie-break, a text-only batch), so a sweep reducing only its
    own grads would skip a collective its peers enter. A param some rank touched is zero-filled where
    absent; one no rank touched keeps ``grad is None``, so the optimizer leaves it alone. ``params``
    must be the same list on every rank.
    """
    if not params:
        return []
    local = [p.grad is not None for p in params]
    device = collective_device()
    full = all(local)
    present = (
        torch.ones(len(params), dtype=torch.uint8, device=device)
        if full
        else torch.tensor(local, dtype=torch.uint8, device=device)
    )
    dist.all_reduce(present, op=dist.ReduceOp.MAX, group=group)
    if full:
        return list(params)
    entering = []
    for param, any_rank_has_grad in zip(params, present.tolist(), strict=True):
        if any_rank_has_grad:
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            entering.append(param)
    return entering


def _launch_flat_chunk(
    bucket: list[torch.Tensor], *, op: dist.ReduceOp, group: dist.ProcessGroup | None, fp32: bool
) -> _PendingChunk:
    """Flatten one same-dtype chunk and start its all-reduce.

    A lone contiguous gradient (a fused expert tensor above the bucket cap) is reduced in its own
    storage: ``torch.cat`` always allocates, so flattening it would cost two full copies.
    """
    if len(bucket) == 1 and bucket[0].is_contiguous():
        flat, scatter = bucket[0].view(-1), False
    else:
        flat, scatter = torch.cat([g.reshape(-1) for g in bucket]), True
    buf = flat.float() if fp32 and flat.dtype != torch.float32 else flat
    work = dist.all_reduce(buf, op=op, group=group, async_op=True)
    return _PendingChunk(bucket, flat, buf, work, scatter)


def _finish_flat_chunk(pending: _PendingChunk, *, divisor: float | None) -> None:
    """Wait for one chunk's all-reduce, then scale and scatter it back in place."""
    pending.work.wait()
    _scale_and_write_back(pending.buf, pending.flat, divisor)
    if not pending.scatter:
        return
    offset = 0
    for g in pending.bucket:
        n = g.numel()
        g.copy_(pending.flat[offset : offset + n].view_as(g))
        offset += n


def reduce_grads_bucketed(
    grads: list[torch.Tensor],
    *,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    divisor: float | None = None,
    group: dist.ProcessGroup | None = None,
    fp32: bool = False,
) -> None:
    """Reduce many gradients sharing one (``op``, ``group``, ``divisor``) in few collectives per dtype.

    Numerically identical to :func:`reduce_grad` per tensor. Every rank in ``group`` must pass grads
    of matching shapes in the same order so chunk boundaries line up.
    """
    if not grads:
        return
    by_dtype: dict[torch.dtype, list[torch.Tensor]] = defaultdict(list)
    for grad in grads:
        by_dtype[grad.dtype].append(grad)
    chunks: list[list[torch.Tensor]] = []
    for dtype in sorted(by_dtype, key=str):
        itemsize = torch.empty(0, dtype=dtype).element_size()
        max_numel = max(1, BUCKET_MAX_BYTES // itemsize)
        chunk: list[torch.Tensor] = []
        chunk_numel = 0
        for grad in by_dtype[dtype]:
            # A param larger than the cap forms its own chunk (never split — offsets must align).
            if chunk and chunk_numel + grad.numel() > max_numel:
                chunks.append(chunk)
                chunk, chunk_numel = [], 0
            chunk.append(grad)
            chunk_numel += grad.numel()
        if chunk:
            chunks.append(chunk)

    with torch.profiler.record_function("grad_sync.reduce_bucketed"):
        inflight: list[_PendingChunk] = []
        for chunk in chunks:
            inflight.append(_launch_flat_chunk(chunk, op=op, group=group, fp32=fp32))
            if len(inflight) >= BUCKET_MAX_INFLIGHT:
                _finish_flat_chunk(inflight.pop(0), divisor=divisor)
        for pending in inflight:
            _finish_flat_chunk(pending, divisor=divisor)
