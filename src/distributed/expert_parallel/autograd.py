"""Autograd primitives for EP MoE layers: DeepEP dispatch/combine, GC-recompute replay variants,
and Megatron scatter/reduce around expert-TP compute."""

from __future__ import annotations

import torch
import torch.distributed as dist

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

    Shared by the live and the GC-replay dispatch functions, which differ only in forward arity —
    the replay one returns cached tensors, so its backward is the same collective.
    """
    grad_x, grad_topk_weights = ctx.dispatcher.backend.combine_grad(
        grad_recv_x.contiguous(), grad_recv_topk_weights, ctx.handle
    )
    if grad_x.dtype != ctx.input_dtype:
        grad_x = grad_x.to(ctx.input_dtype)
    return grad_x, grad_topk_weights


def _combine_backward(ctx, grad_combined_x: torch.Tensor) -> torch.Tensor:
    """Gradient of a combine: a dispatch over the layout cached on ``ctx.handle`` (see
    :func:`_dispatch_backward` for why the replay twin shares it)."""
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
        num_experts: int,
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
        return grad_x, None, grad_topk_weights, None, None


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
    """Replay cached dispatch results during GC recompute; backward is the shared clean combine
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
    """Replay cached combine results during GC recompute; backward is the shared clean dispatch
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


# Expert-TP scatter-gather (Megatron-LM): input → SumGradAcrossGroup → expert_compute(W_shard) →
# ReduceFromExpertTP. The scatter half is the toolkit-wide :class:`SumGradAcrossGroup`, shared with TP.


class ReduceFromExpertTP(torch.autograd.Function):
    """All-reduce SUM forward, identity backward, on the combined MoE output (token space, after combine).

    Sums each expert-TP shard's partial per-token output; exact because combine is linear. Identity
    backward is correct: :class:`SumGradAcrossGroup` does the input-grad all-reduce on the way in, and
    each shard's weight grad is independently correct.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group):
        ctx.group = group
        out = x.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


# Atomic-free MoE token permute / unpermute for the grouped-GEMM expert path: index_add_ lowers to a bf16
# atomic with no native add, so a CAS loop serialises under high top_k. A precomputed ``inv_map`` (sorted
# positions per recv token, sentinel-padded) makes scatter-back a gather+sum. See ``_build_inv_map``.


class MoEGatherPermute(torch.autograd.Function):
    """``sorted = tokens[sorted_token_idx]`` (index_select gather) with an atomic-free gather-reduce
    backward over ``inv_map``."""

    @staticmethod
    def forward(ctx, tokens: torch.Tensor, sorted_token_idx: torch.Tensor, inv_map: torch.Tensor):
        ctx.save_for_backward(inv_map)
        return tokens.index_select(0, sorted_token_idx)

    @staticmethod
    def backward(ctx, grad_sorted):
        (inv_map,) = ctx.saved_tensors
        pad = grad_sorted.new_zeros(1, grad_sorted.shape[1])
        grad_tokens = torch.cat([grad_sorted, pad], 0)[inv_map].sum(dim=1)
        return grad_tokens, None, None


class MoEScatterUnpermute(torch.autograd.Function):
    """Atomic-free expert-output unpermute. Forward ``out[r] = sum_k expert_out[inv_map[r, k]]``
    (equivalent to index_add_, no atomics); backward is a plain gather."""

    @staticmethod
    def forward(ctx, expert_out: torch.Tensor, sorted_token_idx: torch.Tensor, inv_map: torch.Tensor):
        ctx.save_for_backward(sorted_token_idx)
        pad = expert_out.new_zeros(1, expert_out.shape[1])
        return torch.cat([expert_out, pad], 0)[inv_map].sum(dim=1)

    @staticmethod
    def backward(ctx, grad_out):
        (sorted_token_idx,) = ctx.saved_tensors
        return grad_out.index_select(0, sorted_token_idx), None, None


class MoEExpertBiasGather(torch.autograd.Function):
    """Per-expert bias broadcast ``bias[eids]`` to expert-sorted tokens, with an atomic-free backward.

    The default index_add_ backward is a bf16 atomic scatter that serialises (all of an expert's tokens
    hit the same bias row). Instead ``grad_bias`` = per-expert grad sum via one GEMM
    ``onehot(eids)ᵀ @ grad_out``: tensor-core fp32 accum, no atomics, order-independent."""

    @staticmethod
    def forward(ctx, bias: torch.Tensor, eids: torch.Tensor):
        ctx.save_for_backward(eids)
        ctx.num_experts = bias.shape[0]
        return bias.index_select(0, eids)

    @staticmethod
    def backward(ctx, grad_out):
        (eids,) = ctx.saved_tensors
        # onehot[e, t] = 1 where token t routed to expert e; ([E, N] @ [N, I]) sums grad per expert.
        onehot = (torch.arange(ctx.num_experts, device=eids.device).unsqueeze(1) == eids).to(grad_out.dtype)
        return onehot @ grad_out, None
