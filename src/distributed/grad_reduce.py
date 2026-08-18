"""Gradient all-reduce primitives shared by every post-backward sync path.

:func:`reduce_grad` reduces one gradient in place; :func:`reduce_grads_bucketed` coalesces many
gradients sharing a reduction into a few flat collectives. Callers are the deferred cross-replica EP
sweep, the TP replicated / per-head-norm sweep, the QLoRA sweep and the EP router grad hook.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.distributed as dist

from src.env import env_int

# Flat-buffer cap for bucketed reduction: bounds the transient cat allocation while keeping collectives few.
GRAD_BUCKET_MB = env_int("HALO_GRAD_BUCKET_MB", 256)
BUCKET_MAX_BYTES = GRAD_BUCKET_MB * 1024 * 1024
# Buckets reduced concurrently. Peak transient is this many flat buffers under a same-dtype reduce,
# and 3x that under ``fp32``: each in-flight chunk holds the bf16 flat buffer (needed for the
# scatter-back) alongside its fp32 upcast.
BUCKET_MAX_INFLIGHT = env_int("HALO_GRAD_BUCKET_MAX_INFLIGHT", 2)


def reduce_grad(grad, *, op=dist.ReduceOp.SUM, divisor=None, group=None, fp32=False):
    """All-reduce a gradient in place over ``group`` with ``op``, then scale by ``1 / divisor`` if given.

    With ``fp32`` the collective+scaling run in fp32 and the result is written back in the original dtype
    (FSDP2 ``reduce_dtype=fp32`` semantics: precise reduce, low-precision storage; a bf16 reduce loses
    ~10^4x precision).
    """
    t = grad if grad.is_contiguous() else grad.contiguous()
    if fp32 and t.dtype != torch.float32:
        g = t.float()
        dist.all_reduce(g, op=op, group=group)
        if divisor is not None:
            g.div_(divisor)
        t.copy_(g)
    else:
        dist.all_reduce(t, op=op, group=group)
        if divisor is not None:
            t.div_(divisor)
    if t is not grad:
        grad.copy_(t)


class SumGradAcrossGroup(torch.autograd.Function):
    """Identity forward; SUM all-reduce of the gradient over ``group`` in backward.

    The autograd-time form of :func:`reduce_grad`, applied to an activation whose consumers are
    sharded over ``group``, so each rank's backward produces only a partial gradient for it. Used by
    two axes: expert-TP inserts it on the MoE layer input (else FSDP2 divides by ``world_size`` while
    only ``dp_size`` distinct full gradients exist), and TP inserts it on the MLA rope rows, which
    are expanded to this rank's local heads and never cross a DTensor boundary. Reducing at the
    activation runs the collective once per backward, unlike a ``register_full_backward_hook`` on
    the weight, which re-reduces ``.grad`` on every micro-step.
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


def reduce_grads_bucketed(grads, *, op=dist.ReduceOp.SUM, divisor=None, group=None, fp32=False):
    """Reduce many gradients sharing one (``op``, ``group``, ``divisor``) in few collectives per dtype.

    Numerically identical to :func:`reduce_grad` per tensor, but replaces N latency-bound all-reduces
    with a handful. Every rank in ``group`` must pass grads of matching shapes in the same order so
    chunk boundaries line up; dtype buckets are reduced in a sorted, rank-stable order.
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

    # Keep several collectives in flight so each one's latency covers the next bucket's cat/upcast
    # and the previous one's scatter-back. Every rank walks `chunks` in the same order, so the
    # launch order stays matched.
    with torch.profiler.record_function("grad_sync.reduce_bucketed"):
        inflight: list[tuple] = []
        for chunk in chunks:
            inflight.append(_launch_flat_chunk(chunk, op=op, group=group, fp32=fp32))
            if len(inflight) >= BUCKET_MAX_INFLIGHT:
                _finish_flat_chunk(*inflight.pop(0), divisor=divisor)
        for pending in inflight:
            _finish_flat_chunk(*pending, divisor=divisor)


def _launch_flat_chunk(bucket, *, op, group, fp32):
    """Flatten one same-dtype chunk and start its all-reduce. Returns the state ``_finish_flat_chunk`` needs.

    A chunk holding a single contiguous gradient (any fused expert tensor above the bucket cap on a
    100B+ MoE) is reduced in its own storage: ``torch.cat`` always allocates, so flattening it would
    cost two full copies and a transient the size of the gradient.
    """
    if len(bucket) == 1 and bucket[0].is_contiguous():
        flat, scatter = bucket[0].view(-1), False
    else:
        flat, scatter = torch.cat([g.reshape(-1) for g in bucket]), True
    buf = flat.float() if fp32 and flat.dtype != torch.float32 else flat
    work = dist.all_reduce(buf, op=op, group=group, async_op=True)
    return bucket, flat, buf, work, scatter


def _finish_flat_chunk(bucket, flat, buf, work, scatter, *, divisor):
    """Wait for one chunk's all-reduce, then scale and scatter it back in place."""
    work.wait()
    if divisor is not None:
        buf.div_(divisor)
    if buf is not flat:
        flat.copy_(buf)
    if not scatter:
        return
    offset = 0
    for g in bucket:
        n = g.numel()
        g.copy_(flat[offset : offset + n].view_as(g))
        offset += n
