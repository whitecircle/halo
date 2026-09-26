"""Fused token permute/unpermute kernels for grouped-GEMM MoE expert compute.

Both directions of the expert permutation reduce over a token's ``top_k`` expert rows through the
``inv_map`` built by ``EPMoELayerBase._build_inv_map`` (``[recv_N, top_k]`` sorted positions, padded
with the sentinel ``N_sorted``). The eager form pads the source with a zero row, gathers a
``[recv_N, top_k, H]`` transient and sums it; these kernels walk ``inv_map`` per output row and
accumulate in fp32, so neither the pad copy nor the transient exists.

The routing-weight multiply is folded into the unpermute, and its backward emits the expert-output
gradient and the routing-weight gradient from one read of each operand.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK_H = 1024


@triton.jit
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


@triton.jit
def _weighted_unpermute_bwd_kernel(
    grad_out_ptr,
    expert_out_ptr,
    weight_ptr,
    token_idx_ptr,
    grad_expert_ptr,
    grad_weight_ptr,
    hidden,
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
        grad = tl.load(grad_out_ptr + token * hidden + cols, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(expert_out_ptr + row * hidden + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(grad_expert_ptr + row * hidden + cols, grad * weight, mask=mask)
        dot += grad * y
    tl.store(grad_weight_ptr + row, tl.sum(dot, axis=0))


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
        weight if weight is not None else src,
        inv_map.contiguous(),
        out,
        src.shape[0],
        hidden,
        TOP_K=top_k,
        HAS_WEIGHT=weight is not None,
        BLOCK_H=_BLOCK_H,
    )
    return out


def _gather_reduce_eager(src: torch.Tensor, inv_map: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
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
    return _gather_reduce_eager(src, inv_map, weight)


class MoEWeightedUnpermute(torch.autograd.Function):
    """``out[r] = sum_k w[j] * expert_out[j]`` for ``j = inv_map[r, k]``: the routing-weight multiply and
    the atomic-free unpermute in one pass.

    Numerically the eager ``expert_out * w`` then padded gather-sum, without the intermediate bf16
    rounding of each weighted row (the product and the sum accumulate in fp32).
    """

    @staticmethod
    def forward(ctx, expert_out, weights, sorted_token_idx, inv_map):
        ctx.save_for_backward(expert_out, weights, sorted_token_idx)
        ctx.weight_dtype = weights.dtype
        return gather_reduce_rows(expert_out, inv_map, weights.contiguous())

    @staticmethod
    def backward(ctx, grad_out):
        expert_out, weights, sorted_token_idx = ctx.saved_tensors
        if not grad_out.is_cuda:
            gathered = grad_out.index_select(0, sorted_token_idx)
            grad_expert = gathered * weights.unsqueeze(-1).to(gathered.dtype)
            grad_weights = (gathered.float() * expert_out.float()).sum(-1)
            return grad_expert, grad_weights.to(ctx.weight_dtype), None, None
        n_sorted, hidden = expert_out.shape
        grad_expert = torch.empty_like(expert_out)
        grad_weights = torch.empty(n_sorted, device=expert_out.device, dtype=torch.float32)
        if n_sorted:
            _weighted_unpermute_bwd_kernel[(n_sorted,)](
                grad_out.contiguous(),
                expert_out,
                weights,
                sorted_token_idx,
                grad_expert,
                grad_weights,
                hidden,
                BLOCK_H=_BLOCK_H,
            )
        return grad_expert, grad_weights.to(ctx.weight_dtype), None, None
