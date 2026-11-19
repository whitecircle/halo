"""Triton kernels for Flash-Next's gated residual: the per-stream RMS norm and the gated stream mix, each an autograd
function. Every row is one program's work with no atomic, so a run is repeatable, and the hidden size is a runtime
argument: one compile per stream dtype and count serves every row count."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from halo.kernels.hyper_connection import inject_streams, require_shape

# One program's columns, a power of two as ``tl.arange`` takes.
_MAX_BLOCK = 512
# Rows whose fp32 products the norm weight's gradient holds at once.
_WEIGHT_GRAD_ROWS = 4096


def _block(hidden: int) -> int:
    return min(_MAX_BLOCK, triton.next_power_of_2(hidden))


def _require_streams(features: int, streams: int) -> None:
    if streams < 1 or features < 1 or features % streams:
        raise ValueError(f"{features} features do not split into {streams} streams")


@triton.jit
def _norm_fwd_kernel(x_ptr, weight_ptr, out_ptr, rstd_ptr, streams, hidden, eps, BLOCK: tl.constexpr):
    group = tl.program_id(0).to(tl.int64)
    base = group * hidden
    weight_base = (group % streams) * hidden
    squares = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, hidden, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + base + cols, mask=cols < hidden, other=0.0).to(tl.float32)
        squares += x * x
    rstd = tl.rsqrt(tl.sum(squares, axis=0) / hidden + eps)
    tl.store(rstd_ptr + group, rstd)
    for start in range(0, hidden, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        in_col = cols < hidden
        x = tl.load(x_ptr + base + cols, mask=in_col, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + weight_base + cols, mask=in_col, other=0.0).to(tl.float32)
        tl.store(out_ptr + base + cols, ((x * rstd) * (1.0 + weight)).to(out_ptr.dtype.element_ty), mask=in_col)


@triton.jit
def _norm_bwd_kernel(grad_ptr, x_ptr, weight_ptr, rstd_ptr, dx_ptr, streams, hidden, BLOCK: tl.constexpr):
    # With y = r * x * (1 + w) and r = rsqrt(mean(x^2) + eps): dx = r * dy * (1 + w) - x * r^3 * mean(dy * (1 + w) * x).
    group = tl.program_id(0).to(tl.int64)
    base = group * hidden
    weight_base = (group % streams) * hidden
    rstd = tl.load(rstd_ptr + group)
    dot = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, hidden, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        in_col = cols < hidden
        grad = tl.load(grad_ptr + base + cols, mask=in_col, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + base + cols, mask=in_col, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + weight_base + cols, mask=in_col, other=0.0).to(tl.float32)
        dot += grad * (1.0 + weight) * x
    coefficient = rstd * rstd * rstd * tl.sum(dot, axis=0) / hidden
    for start in range(0, hidden, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        in_col = cols < hidden
        grad = tl.load(grad_ptr + base + cols, mask=in_col, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + base + cols, mask=in_col, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + weight_base + cols, mask=in_col, other=0.0).to(tl.float32)
        dx = grad * (1.0 + weight) * rstd - x * coefficient
        tl.store(dx_ptr + base + cols, dx.to(dx_ptr.dtype.element_ty), mask=in_col)


@triton.jit
def _mix_cells(streams, hidden, S_PAD: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    lane = tl.arange(0, S_PAD)
    in_col = cols < hidden
    cells = row * streams * hidden + lane[:, None] * hidden + cols[None, :]
    return row, cols, in_col, cells, (lane[:, None] < streams) & in_col[None, :]


@triton.jit
def _mix_fwd_kernel(up_ptr, normed_ptr, mixed_ptr, streams, hidden, S_PAD: tl.constexpr, BLOCK: tl.constexpr):
    row, cols, in_col, cells, in_cell = _mix_cells(streams, hidden, S_PAD, BLOCK)
    dtype = mixed_ptr.dtype.element_ty
    # The eager form rounds the gate, the product and the mean to the streams' dtype.
    gate = tl.sigmoid(tl.load(up_ptr + cells, mask=in_cell, other=0.0).to(tl.float32)).to(dtype).to(tl.float32)
    normed = tl.load(normed_ptr + cells, mask=in_cell, other=0.0).to(tl.float32)
    product = (gate * normed).to(dtype).to(tl.float32)
    tl.store(mixed_ptr + row * hidden + cols, (tl.sum(product, axis=0) / streams).to(dtype), mask=in_col)


@triton.jit
def _mix_bwd_kernel(
    up_ptr, normed_ptr, grad_ptr, dup_ptr, dnormed_ptr, streams, hidden, S_PAD: tl.constexpr, BLOCK: tl.constexpr
):
    row, cols, in_col, cells, in_cell = _mix_cells(streams, hidden, S_PAD, BLOCK)
    dtype = dup_ptr.dtype.element_ty
    gate = tl.sigmoid(tl.load(up_ptr + cells, mask=in_cell, other=0.0).to(tl.float32)).to(dtype).to(tl.float32)
    normed = tl.load(normed_ptr + cells, mask=in_cell, other=0.0).to(tl.float32)
    product_grad = (tl.load(grad_ptr + row * hidden + cols, mask=in_col, other=0.0).to(tl.float32) / streams).to(dtype)
    product_grad = product_grad.to(tl.float32)[None, :]
    gate_grad = (product_grad * normed).to(dtype).to(tl.float32)
    tl.store(dup_ptr + cells, (gate_grad * (1.0 - gate) * gate).to(dtype), mask=in_cell)
    tl.store(dnormed_ptr + cells, (product_grad * gate).to(dtype), mask=in_cell)


class _GroupedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float, streams: int) -> torch.Tensor:
        _require_streams(weight.numel(), streams)
        require_shape(x, (x.numel() // weight.numel(), weight.numel()), "x")
        x, weight = x.contiguous(), weight.contiguous()
        hidden = weight.numel() // streams
        out = torch.empty_like(x)
        rstd = torch.empty(x.shape[0] * streams, device=x.device, dtype=torch.float32)
        _norm_fwd_kernel[(rstd.numel(),)](x, weight, out, rstd, streams, hidden, eps, BLOCK=_block(hidden))
        ctx.save_for_backward(x, weight, rstd)
        ctx.streams = streams
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out: torch.Tensor):
        x, weight, rstd = ctx.saved_tensors
        streams = ctx.streams
        hidden = weight.numel() // streams
        grad_out = grad_out.contiguous()
        dx = torch.empty_like(x)
        _norm_bwd_kernel[(rstd.numel(),)](grad_out, x, weight, rstd, dx, streams, hidden, BLOCK=_block(hidden))
        dweight = None
        if ctx.needs_input_grad[1]:
            dweight = torch.zeros(streams, hidden, device=x.device, dtype=torch.float32)
            for start in range(0, x.shape[0], _WEIGHT_GRAD_ROWS):
                stop = start + _WEIGHT_GRAD_ROWS
                normalized = x[start:stop].float().view(-1, streams, hidden)
                normalized = normalized * rstd[start * streams : stop * streams].view(-1, streams, 1)
                dweight += (grad_out[start:stop].float().view(-1, streams, hidden) * normalized).sum(0)
            dweight = dweight.reshape(weight.shape).to(weight.dtype)
        return dx, dweight, None, None


class _GateMix(torch.autograd.Function):
    """``mean_streams(sigmoid(up) * normed)`` for ``up = Linear(silu(down / N))``: the activation and ``up`` are
    recomputed backward, so the graph holds ``normed`` and ``down`` only."""

    @staticmethod
    def forward(ctx, normed: torch.Tensor, down: torch.Tensor, w_up: torch.Tensor, streams: int) -> torch.Tensor:
        _require_streams(w_up.shape[0], streams)
        require_shape(normed, (normed.numel() // w_up.shape[0], w_up.shape[0]), "normed")
        normed, down, w_up = normed.contiguous(), down.contiguous(), w_up.contiguous()
        hidden = w_up.shape[0] // streams
        up = F.linear(F.silu(down / streams), w_up)
        mixed = torch.empty(normed.shape[0], hidden, device=normed.device, dtype=normed.dtype)
        grid = (normed.shape[0], triton.cdiv(hidden, _block(hidden)))
        _mix_fwd_kernel[grid](
            up, normed, mixed, streams, hidden, S_PAD=triton.next_power_of_2(streams), BLOCK=_block(hidden)
        )
        ctx.save_for_backward(normed, down, w_up)
        ctx.streams = streams
        return mixed

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out: torch.Tensor):
        normed, down, w_up = ctx.saved_tensors
        streams = ctx.streams
        hidden = w_up.shape[0] // streams
        scaled = down / streams
        activation = F.silu(scaled)
        up = F.linear(activation, w_up)
        dup, dnormed = torch.empty_like(up), torch.empty_like(normed)
        grid = (normed.shape[0], triton.cdiv(hidden, _block(hidden)))
        _mix_bwd_kernel[grid](
            up, normed, grad_out.contiguous(), dup, dnormed, streams, hidden,
            S_PAD=triton.next_power_of_2(streams), BLOCK=_block(hidden),
        )  # fmt: skip
        ddown = dw_up = None
        if ctx.needs_input_grad[1]:
            ddown = torch.ops.aten.silu_backward(dup @ w_up, scaled) / streams
        if ctx.needs_input_grad[2]:
            dw_up = dup.t() @ activation
        return dnormed, ddown, dw_up, None


def grouped_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, streams: int) -> torch.Tensor:
    """``x * rsqrt(mean(x^2) + eps) * (1 + weight)`` per group of ``weight.numel() / streams`` features, for ``[rows,
    streams * hidden]`` ``x`` and a ``[streams * hidden]`` ``weight``, computed in fp32 and stored in ``x``'s dtype."""
    return _GroupedRMSNorm.apply(x, weight, eps, streams)


def gate_mix(normed: torch.Tensor, down: torch.Tensor, w_up: torch.Tensor, streams: int) -> torch.Tensor:
    """``(sigmoid(F.linear(silu(down / streams), w_up)) * normed).view(rows, streams, hidden).mean(-2)`` for ``[rows,
    streams * hidden]`` ``normed``, ``[rows, rank]`` ``down`` and a ``[streams * hidden, rank]`` ``w_up``; the roundings
    to the streams' dtype are the eager form's."""
    return _GateMix.apply(normed, down, w_up, streams)


def warm_gated_residual_kernels(
    device: torch.device,
    dtype: torch.dtype,
    norm_weight_dtype: torch.dtype,
    streams: int,
    hidden: int,
    eps: float,
    rank: int,
) -> None:
    """Compile every kernel of the gated residual and the injection for one row, forward and backward, for a norm
    ``eps`` and a mix projection of ``rank`` features. Rank-local."""
    options = {"device": device, "requires_grad": True}
    width = streams * hidden
    x = torch.zeros(1, width, dtype=dtype, **options)
    norm_weight = torch.zeros(width, dtype=norm_weight_dtype, **options)
    down = torch.zeros(1, rank, dtype=dtype, **options)
    w_up = torch.zeros(width, rank, dtype=dtype, **options)
    gate = torch.zeros(1, streams, dtype=dtype, **options)
    mixed = gate_mix(grouped_rms_norm(x, norm_weight, eps, streams), down, w_up, streams)
    out = inject_streams(mixed, x.view(1, streams, hidden), gate)
    torch.autograd.grad(out, (x, norm_weight, down, w_up, gate), torch.ones_like(out))
