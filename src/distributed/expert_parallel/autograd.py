"""Autograd functions for the DeepEP dispatch and combine, and their GC-recompute replay variants."""

from __future__ import annotations

import torch

from src.distributed.expert_parallel.extension import deep_ep


def _to_topk_weights(topk_weights: torch.Tensor) -> torch.Tensor:
    """DeepEP requires top-k weights to be contiguous float32 (csrc asserts)."""
    if topk_weights.dtype != torch.float32:
        topk_weights = topk_weights.to(dtype=torch.float32)
    if not topk_weights.is_contiguous():
        topk_weights = topk_weights.contiguous()
    return topk_weights


def _dispatch_backward(ctx, grad_recv_x: torch.Tensor, grad_recv_topk_weights) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradient of a dispatch: a combine over the layout cached on ``ctx.handle``.

    Shared by the live and the GC-replay dispatch functions, which differ only in forward arity.
    """
    grad_x, grad_topk_weights = ctx.dispatcher.backend.combine_grad(
        grad_recv_x.contiguous(), grad_recv_topk_weights, ctx.handle
    )
    if grad_x.dtype != ctx.input_dtype:
        grad_x = grad_x.to(ctx.input_dtype)
    return grad_x, grad_topk_weights


def _combine_backward(ctx, grad_combined_x: torch.Tensor) -> torch.Tensor:
    """Gradient of a combine: a dispatch over the layout cached on ``ctx.handle``. Shared with the
    GC-replay combine function."""
    grad_x = ctx.dispatcher.backend.dispatch_grad(grad_combined_x.contiguous(), ctx.handle)
    if grad_x.dtype != ctx.input_dtype:
        grad_x = grad_x.to(ctx.input_dtype)
    return grad_x


class DeepEPDispatchFunction(torch.autograd.Function):
    """DeepEP dispatch (forward) / combine (backward). The dispatch backward is a combine."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        dispatcher,
    ):
        x = x.contiguous()
        topk_idx = topk_idx.contiguous().to(deep_ep().topk_idx_t)
        topk_weights = _to_topk_weights(topk_weights)

        recv_x, recv_topk_idx, recv_topk_weights, handle = dispatcher.backend.dispatch_fwd(x, topk_idx, topk_weights)

        ctx.dispatcher = dispatcher
        ctx.handle = handle
        ctx.input_dtype = x.dtype
        return recv_x, recv_topk_idx, recv_topk_weights, handle

    @staticmethod
    def backward(ctx, grad_recv_x, grad_recv_topk_idx, grad_recv_topk_weights, grad_handle):
        grad_x, grad_topk_weights = _dispatch_backward(ctx, grad_recv_x, grad_recv_topk_weights)
        return grad_x, None, grad_topk_weights, None


class DeepEPCombineFunction(torch.autograd.Function):
    """DeepEP combine (forward) / dispatch (backward). The combine backward is a dispatch reusing the
    cached layout via ``handle``."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, topk_weights: torch.Tensor, handle: object, dispatcher):
        x = x.contiguous()
        topk_weights = _to_topk_weights(topk_weights)
        combined_x = dispatcher.backend.combine_fwd(x, topk_weights, handle)
        ctx.dispatcher = dispatcher
        ctx.handle = handle
        ctx.input_dtype = x.dtype
        return combined_x

    @staticmethod
    def backward(ctx, grad_combined_x):
        return _combine_backward(ctx, grad_combined_x), None, None, None


class ReplayDispatchFunction(torch.autograd.Function):
    """Replay cached dispatch results during GC recompute; backward is the shared combine
    (:func:`_dispatch_backward`)."""

    @staticmethod
    def forward(
        ctx,
        x,
        topk_idx,
        topk_weights,
        dispatcher,
        cached_recv_x,
        cached_recv_topk_idx,
        cached_recv_topk_weights,
        cached_handle,
    ):
        ctx.dispatcher = dispatcher
        ctx.handle = cached_handle
        ctx.input_dtype = x.dtype
        return cached_recv_x, cached_recv_topk_idx, cached_recv_topk_weights, cached_handle

    @staticmethod
    def backward(ctx, grad_recv_x, grad_recv_topk_idx, grad_recv_topk_weights, grad_handle):
        grad_x, grad_topk_weights = _dispatch_backward(ctx, grad_recv_x, grad_recv_topk_weights)
        return grad_x, None, grad_topk_weights, None, None, None, None, None


class ReplayCombineFunction(torch.autograd.Function):
    """Replay cached combine results during GC recompute; backward is the shared dispatch
    (:func:`_combine_backward`)."""

    @staticmethod
    def forward(ctx, x, topk_weights, handle, dispatcher, cached_combined_x):
        ctx.dispatcher = dispatcher
        ctx.handle = handle
        ctx.input_dtype = x.dtype
        return cached_combined_x

    @staticmethod
    def backward(ctx, grad_combined_x):
        return _combine_backward(ctx, grad_combined_x), None, None, None, None
