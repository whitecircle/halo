"""Atomic-free token permute/unpermute for grouped-GEMM MoE expert compute.

The permute's gather has ``index_add_`` as its default backward, and the unpermute is one: a bf16 kernel
that emulates an atomic add with a CAS loop, which serializes under high ``top_k``. Both directions here
reduce over a token's ``top_k`` expert rows through :func:`build_inv_map`'s ``inv_map`` (``[recv_N,
top_k]`` sorted positions, padded with the sentinel ``N_sorted``) instead. The eager form pads the source
with a zero row, gathers a ``[recv_N, top_k, H]`` transient and sums it; the kernels walk ``inv_map`` per
output row and accumulate in fp32, so neither the pad copy nor the transient exists.

The routing-weight multiply is folded into the unpermute, and its backward emits the expert-output
gradient and the routing-weight gradient from one read of each operand. The expert outputs feed only
the routing-weight gradient, so they are saved, and that gradient computed, only when the weights
need one.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from src.kernels.histogram import sync_free_bincount

_BLOCK_H = 1024


# ``n_src`` (the dispatch's sorted-row count) changes every step; see ``fused_glu._glu_fwd_kernel``.
@triton.jit(do_not_specialize=["n_src"])
def _gather_reduce_kernel(
    src_ptr,
    weight_ptr,
    inv_map_ptr,
    out_ptr,
    n_src,
    hidden,
    TOP_K: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    col_mask = cols < hidden
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        src_row = tl.load(inv_map_ptr + row * TOP_K + k)
        valid = src_row < n_src
        vals = tl.load(src_ptr + src_row * hidden + cols, mask=col_mask & valid, other=0.0).to(tl.float32)
        if HAS_WEIGHT:
            vals = vals * tl.load(weight_ptr + src_row, mask=valid, other=0.0).to(tl.float32)
        acc += vals
    tl.store(out_ptr + row * hidden + cols, acc, mask=col_mask)


# ``weight_grad`` is a runtime branch rather than a constexpr, so the one binary the EP warm-up compiles
# serves both cases.
@triton.jit(do_not_specialize=["grad_out_stride", "weight_grad"])
def _weighted_unpermute_bwd_kernel(
    grad_out_ptr,
    expert_out_ptr,
    weight_ptr,
    token_idx_ptr,
    grad_expert_ptr,
    grad_weight_ptr,
    hidden,
    grad_out_stride,
    weight_grad,
    BLOCK_H: tl.constexpr,
):
    # One program per sorted row j: grad_expert[j] = w[j] * grad_out[tok[j]], grad_w[j] = <grad_out[tok[j]], y[j]>.
    row = tl.program_id(0).to(tl.int64)
    token = tl.load(token_idx_ptr + row).to(tl.int64)
    weight = tl.load(weight_ptr + row).to(tl.float32)
    dot = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for start in range(0, hidden, BLOCK_H):
        cols = start + tl.arange(0, BLOCK_H)
        mask = cols < hidden
        grad = tl.load(grad_out_ptr + token * grad_out_stride + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(grad_expert_ptr + row * hidden + cols, grad * weight, mask=mask)
        if weight_grad:
            y = tl.load(expert_out_ptr + row * hidden + cols, mask=mask, other=0.0).to(tl.float32)
            dot += grad * y
    if weight_grad:
        tl.store(grad_weight_ptr + row, tl.sum(dot, axis=0))


def build_inv_map(sorted_token_idx: torch.Tensor, recv_N: int, width: int) -> torch.Tensor:
    """The atomic-free permute/unpermute map.

    ``inv_map[r, j]`` = the j-th sorted position whose recv token is ``r``, padded to ``width`` cols
    with sentinel ``N_sorted``. Sync-free (stable argsort + cumulative counts, no host round-trip).
    """
    device = sorted_token_idx.device
    n_sorted = sorted_token_idx.shape[0]
    inv_map = torch.full((recv_N, width), n_sorted, dtype=torch.long, device=device)
    if n_sorted == 0:
        return inv_map
    order = torch.argsort(sorted_token_idx, stable=True)
    rp = sorted_token_idx.index_select(0, order)  # recv positions, grouped & contiguous
    counts = sync_free_bincount(sorted_token_idx, recv_N, dtype=torch.long)
    starts = torch.cumsum(counts, 0) - counts
    slot = torch.arange(n_sorted, device=device) - starts.index_select(0, rp)
    inv_map[rp, slot] = order
    return inv_map


def _gather_reduce(src: torch.Tensor, inv_map: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    """``out[r] = sum_k w[j] * src[j]`` over ``j = inv_map[r, k] < len(src)``; ``w = 1`` without a weight."""
    n_out, top_k = inv_map.shape
    hidden = src.shape[-1]
    out = torch.empty((n_out, hidden), device=src.device, dtype=src.dtype)
    if n_out == 0:
        return out
    src = src.contiguous()
    grid = (n_out, triton.cdiv(hidden, _BLOCK_H))
    _gather_reduce_kernel[grid](
        src,
        weight.contiguous() if weight is not None else src,
        inv_map.contiguous(),
        out,
        src.shape[0],
        hidden,
        TOP_K=top_k,
        HAS_WEIGHT=weight is not None,
        BLOCK_H=_BLOCK_H,
    )
    return out


def padded_gather_reduce(src: torch.Tensor, inv_map: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    """:func:`gather_reduce_rows` as the padded reference: weight the rows, pad with a zero row, gather
    ``[N, top_k, H]`` and sum. The CPU path, and the eager baseline the kernel replaces."""
    if weight is not None:
        src = src * weight.unsqueeze(-1).to(src.dtype)
    pad = src.new_zeros(1, src.shape[1])
    return torch.cat([src, pad], 0)[inv_map].sum(dim=1)


def gather_reduce_rows(src: torch.Tensor, inv_map: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    """Sum each output row's ``top_k`` source rows (optionally weighted) through ``inv_map``.

    Triton on CUDA, the padded-gather reference elsewhere.
    """
    if src.is_cuda:
        return _gather_reduce(src, inv_map, weight)
    return padded_gather_reduce(src, inv_map, weight)


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
        return gather_reduce_rows(grad_sorted, inv_map), None, None


class MoEWeightedUnpermute(torch.autograd.Function):
    """``out[r] = sum_k w[j] * expert_out[j]`` for ``j = inv_map[r, k]``: the routing-weight multiply and
    the atomic-free unpermute in one pass.

    Numerically the eager ``expert_out * w`` then padded gather-sum, without the intermediate bf16
    rounding of each weighted row (the product and the sum accumulate in fp32). ``expert_out`` is saved
    only when ``weights`` needs a gradient, the one gradient that reads it.
    """

    @staticmethod
    def forward(ctx, expert_out, weights, sorted_token_idx, inv_map):
        # The backward kernel indexes both by row with unit strides.
        expert_out, weights = expert_out.contiguous(), weights.contiguous()
        ctx.weight_grad = ctx.needs_input_grad[1]
        ctx.save_for_backward(expert_out if ctx.weight_grad else None, weights, sorted_token_idx.contiguous())
        return gather_reduce_rows(expert_out, inv_map, weights)

    @staticmethod
    def backward(ctx, grad_out):
        expert_out, weights, sorted_token_idx = ctx.saved_tensors
        if not grad_out.is_cuda:
            gathered = grad_out.index_select(0, sorted_token_idx)
            grad_expert = gathered * weights.unsqueeze(-1).to(gathered.dtype)
            if not ctx.weight_grad:
                return grad_expert, None, None, None
            accumulate = torch.promote_types(expert_out.dtype, torch.float32)  # fp32 at least, fp64 kept
            grad_weights = (gathered.to(accumulate) * expert_out.to(accumulate)).sum(-1)
            return grad_expert, grad_weights.to(weights.dtype), None, None
        # Autograd casts the incoming gradient to the forward output's dtype, which is expert_out's.
        n_sorted, hidden = sorted_token_idx.shape[0], grad_out.shape[-1]
        # Read in place at its row stride: under EP the gradient arrives as a row-strided view of the
        # padded transport buffer.
        if grad_out.stride(-1) != 1:
            grad_out = grad_out.contiguous()
        grad_expert = torch.empty((n_sorted, hidden), device=grad_out.device, dtype=grad_out.dtype)
        grad_weights = torch.empty(n_sorted if ctx.weight_grad else 0, device=grad_out.device, dtype=torch.float32)
        if n_sorted:
            _weighted_unpermute_bwd_kernel[(n_sorted,)](
                grad_out,
                # Unread without the weight gradient; grad_expert carries expert_out's dtype, so the binary is the same.
                expert_out if ctx.weight_grad else grad_expert,
                weights,
                sorted_token_idx,
                grad_expert,
                grad_weights,
                hidden,
                grad_out.stride(0),
                int(ctx.weight_grad),
                BLOCK_H=_BLOCK_H,
            )
        return grad_expert, grad_weights.to(weights.dtype) if ctx.weight_grad else None, None, None
