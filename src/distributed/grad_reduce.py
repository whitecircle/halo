"""Gradient all-reduce primitives shared by every post-backward sync path.

:func:`reduce_grad` reduces one gradient in place; :func:`reduce_grads_bucketed` coalesces many
gradients sharing a reduction into a few flat collectives; :func:`agree_grad_presence` agrees which
grads enter one. Callers are the deferred cross-replica EP sweep, the TP replicated / per-head-norm
sweep, the QLoRA sweep and the EP router grad hook.

A leaf by construction — torch, :mod:`src.env` and the :mod:`src.distributed.runtime` leaf only — so
any part of the toolkit can reduce gradients without importing a parallelism implementation.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.distributed as dist

from src.distributed.runtime import collective_device
from src.env import env_int

# Flat-buffer cap for bucketed reduction: bounds the transient cat allocation while keeping collectives few.
GRAD_BUCKET_MB = env_int("HALO_GRAD_BUCKET_MB", 256)
BUCKET_MAX_BYTES = GRAD_BUCKET_MB * 1024 * 1024
# Buckets reduced concurrently. The deferred cross-replica sweep runs after backward, so nothing else is
# left to overlap it with — the only latency to hide is its own framing (cat, fp32 upcast, scatter-back),
# which one bucket's collective can cover for the next. Peak transient is this many flat buffers under a
# same-dtype reduce, and 3x that under ``fp32``: each in-flight chunk holds the bf16 flat buffer (needed
# for the scatter-back) alongside its fp32 upcast.
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

    The autograd-time face of :func:`reduce_grad`, applied to an ACTIVATION whose consumers are
    sharded over ``group``, so each rank's backward produces only a partial gradient for it. Two
    axes need it and must not spell it twice: expert-TP inserts it on the MoE layer input (else FSDP2
    divides by ``world_size`` while only ``dp_size`` distinct full gradients exist), and TP inserts it
    on the MLA rope rows, which are expanded to this rank's local heads and never cross a DTensor
    boundary. Reducing at the activation runs the collective once per backward, unlike a
    ``register_full_backward_hook`` on the weight, which re-reduces ``.grad`` on every micro-step.
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
    """The ``params`` some rank of ``group`` produced a gradient for, each now holding one.

    COLLECTIVE over ``group``: one MAX all-reduce of a presence mask. Grad presence is rank-local (a
    router tie-break, a text-only batch beside vision adapters), so a sweep reducing only its own
    grads would skip a collective its peers enter. A param some rank touched joins the reduce on
    every rank, zero-filled where absent (its true contribution); one no rank touched keeps
    ``grad is None``, so the optimizer leaves it alone instead of applying weight decay and momentum
    to a zero gradient. ``params`` must be structural — the same list on every rank of ``group`` —
    which makes the empty-list skip rank-uniform.

    A rank whose own grads are all present fills its mask on device and never reads it back (the MAX
    is full whatever the peers hold). A rank missing some trainable grad reads the agreed mask back:
    one host sync per bucket per optimizer step, on that rank only.
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


def reduce_grads_bucketed(grads, *, op=dist.ReduceOp.SUM, divisor=None, group=None, fp32=False):
    """Reduce many gradients sharing one (``op``, ``group``, ``divisor``) in few collectives per dtype.

    Numerically identical to :func:`reduce_grad` per tensor but replaces N latency-bound all-reduces with
    a handful. Every rank in ``group`` must pass grads of matching shapes in the same order so chunk
    boundaries line up; dtype buckets are reduced in a sorted, rank-stable order.
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

    # Keep several collectives in flight so each one's latency covers the next bucket's cat/upcast and
    # the previous one's scatter-back, instead of the three running strictly one after another. Every
    # rank walks `chunks` in the same order, so the launch order stays matched.
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

    A chunk holding a single contiguous gradient — every fused expert tensor above the bucket cap on a
    100B+ MoE — is reduced in its own storage: ``torch.cat`` always allocates, so flattening it would
    cost two full copies (in and back) and a transient the size of the gradient for nothing.
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
