"""Manifold-constrained hyper-connection kernels, CUDA only. The projection, collapse and mix of the ``[..., N, H]``
residual streams compute in fp32 and round once at the store, where the eager forms cycle the streams through several
fp32 temporaries, and the Sinkhorn projection runs inside its kernel, where the eager loop launches a hundred-odd times.
Each takes one launch per direction; a trained ``fn`` takes a second one backward."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

# One projection program's tile, measured on B300: on bf16 streams a narrower one, or fewer warps, runs the forward five
# to ten times slower through shared memory.
_PROJECTION_TOKENS = 64
_PROJECTION_COLUMNS = 64
_PROJECTION_WARPS = 8
# ``tl.dot`` takes no inner dimension under this.
_MIN_DOT_LANES = 16
# Elements of one Sinkhorn program's matrices.
_SINKHORN_TILE = 512
# Elements of a stream program's ``[N, N, columns]`` products. Each token's lanes and columns sit in one program, so the
# reductions over them need no atomic, on one warp as the products cross lanes on every pass: a mix backward on four
# takes four times as long, through shared memory.
_STREAM_TILE = 4096
_STREAM_WARPS = 1
# A batch of ``[N, N]`` matrices: axis 1 sums a column's rows, axis 2 a row's columns.
_COLUMN_SUM = tl.constexpr(1)
_ROW_SUM = tl.constexpr(2)


@triton.jit
def _projection_tokens(n_tokens, K: tl.constexpr, K_PAD: tl.constexpr, TOKENS: tl.constexpr):
    tokens = tl.program_id(0).to(tl.int64) * TOKENS + tl.arange(0, TOKENS)
    in_token = tokens < n_tokens
    lane = tl.arange(0, K_PAD)
    in_lane = lane < K
    return tokens, in_token, lane, in_lane, tokens[:, None] * K + lane[None, :], in_token[:, None] & in_lane[None, :]


@triton.jit
def _projection_columns(tokens, in_token, start, width, lane, in_lane, COLUMNS: tl.constexpr):
    cols = start + tl.arange(0, COLUMNS)
    in_col = cols < width
    return (
        tokens[:, None] * width + cols[None, :],
        in_token[:, None] & in_col[None, :],
        lane[:, None] * width + cols[None, :],
        in_lane[:, None] & in_col[None, :],
    )


def _require_shape(tensor: torch.Tensor, expected: tuple[int, ...], name: str) -> None:
    """The kernels address every input off the streams' shape, so a mismatch would read and write out of bounds."""
    if tensor.shape != expected:
        raise ValueError(f"{name} of shape {tuple(tensor.shape)} where the streams take {tuple(expected)}")


# The token count changes every batch; specialized, each class of it would compile its own binary.
@triton.jit(do_not_specialize=["n_tokens"])
def _projection_fwd_kernel(
    stream_ptr,
    fn_ptr,
    out_ptr,
    inv_rms_ptr,
    n_tokens,
    width,
    norm_eps,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    TOKENS: tl.constexpr,
    COLUMNS: tl.constexpr,
):
    tokens, in_token, lane, in_lane, out_cells, in_out = _projection_tokens(n_tokens, K, K_PAD, TOKENS)
    projected = tl.zeros((TOKENS, K_PAD), dtype=tl.float32)
    squares = tl.zeros((TOKENS, COLUMNS), dtype=tl.float32)
    for start in range(0, width, COLUMNS):
        stream_cells, in_stream, fn_cells, in_fn = _projection_columns(
            tokens, in_token, start, width, lane, in_lane, COLUMNS
        )
        stream = tl.load(stream_ptr + stream_cells, mask=in_stream, other=0.0).to(tl.float32)
        fn = tl.load(fn_ptr + fn_cells, mask=in_fn, other=0.0).to(tl.float32)
        projected = tl.dot(stream, tl.trans(fn), projected, input_precision="ieee")
        squares += stream * stream
    inv_rms = tl.rsqrt(tl.sum(squares, axis=1) / width + norm_eps)
    tl.store(inv_rms_ptr + tokens, inv_rms, mask=in_token)
    tl.store(out_ptr + out_cells, projected * inv_rms[:, None], mask=in_out)


@triton.jit(do_not_specialize=["n_tokens"])
def _projection_stream_bwd_kernel(
    grad_ptr,
    stream_ptr,
    fn_ptr,
    out_ptr,
    inv_rms_ptr,
    dstream_ptr,
    n_tokens,
    width,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    TOKENS: tl.constexpr,
    COLUMNS: tl.constexpr,
):
    # With r = 1 / rms(x) and w = r * fn @ x: dx = r * dw @ fn - x * r^2 * (dw . w) / width.
    tokens, in_token, lane, in_lane, out_cells, in_out = _projection_tokens(n_tokens, K, K_PAD, TOKENS)
    grad = tl.load(grad_ptr + out_cells, mask=in_out, other=0.0)
    inv_rms = tl.load(inv_rms_ptr + tokens, mask=in_token, other=0.0)
    norm_term = inv_rms * inv_rms * tl.sum(grad * tl.load(out_ptr + out_cells, mask=in_out, other=0.0), axis=1) / width
    for start in range(0, width, COLUMNS):
        stream_cells, in_stream, fn_cells, in_fn = _projection_columns(
            tokens, in_token, start, width, lane, in_lane, COLUMNS
        )
        stream = tl.load(stream_ptr + stream_cells, mask=in_stream, other=0.0).to(tl.float32)
        fn = tl.load(fn_ptr + fn_cells, mask=in_fn, other=0.0).to(tl.float32)
        dstream = inv_rms[:, None] * tl.dot(grad, fn, input_precision="ieee") - norm_term[:, None] * stream
        tl.store(dstream_ptr + stream_cells, dstream, mask=in_stream)


@triton.jit(do_not_specialize=["n_tokens"])
def _projection_fn_bwd_kernel(
    grad_ptr,
    stream_ptr,
    inv_rms_ptr,
    dfn_ptr,
    n_tokens,
    width,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    TOKENS: tl.constexpr,
    COLUMNS: tl.constexpr,
):
    # dfn = (r * dw)^T @ x, one program per block of columns over every token, as these sum across the tokens.
    cols = tl.program_id(0) * COLUMNS + tl.arange(0, COLUMNS)
    in_col = cols < width
    lane = tl.arange(0, K_PAD)
    in_lane = lane < K
    dfn = tl.zeros((K_PAD, COLUMNS), dtype=tl.float32)
    for start in range(0, n_tokens, TOKENS):
        tokens = start.to(tl.int64) + tl.arange(0, TOKENS)
        in_token = tokens < n_tokens
        grad = tl.load(
            grad_ptr + lane[:, None] + tokens[None, :] * K, mask=in_lane[:, None] & in_token[None, :], other=0.0
        )
        inv_rms = tl.load(inv_rms_ptr + tokens, mask=in_token, other=0.0)
        stream = tl.load(
            stream_ptr + tokens[:, None] * width + cols[None, :],
            mask=in_token[:, None] & in_col[None, :],
            other=0.0,
        ).to(tl.float32)
        dfn = tl.dot(grad * inv_rms[None, :], stream, dfn, input_precision="ieee")
    tl.store(dfn_ptr + lane[:, None] * width + cols[None, :], dfn, mask=in_lane[:, None] & in_col[None, :])


def _projection_launch(fn: torch.Tensor) -> dict:
    n_lanes = fn.shape[0]
    return {
        "K": n_lanes,
        "K_PAD": max(triton.next_power_of_2(n_lanes), _MIN_DOT_LANES),
        "TOKENS": _PROJECTION_TOKENS,
        "COLUMNS": _PROJECTION_COLUMNS,
        "num_warps": _PROJECTION_WARPS,
    }


class _StreamProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, stream: torch.Tensor, fn: torch.Tensor, norm_eps: float) -> torch.Tensor:
        _require_shape(fn, (fn.shape[0], stream.shape[-2] * stream.shape[-1]), "fn")
        stream, fn = stream.contiguous(), fn.contiguous()
        width = fn.shape[1]
        n_tokens = stream.numel() // width
        out = torch.empty((*stream.shape[:-2], fn.shape[0]), device=stream.device, dtype=torch.float32)
        inv_rms = torch.empty(n_tokens, device=stream.device, dtype=torch.float32)
        _projection_fwd_kernel[(triton.cdiv(n_tokens, _PROJECTION_TOKENS),)](
            stream, fn, out, inv_rms, n_tokens, width, norm_eps, **_projection_launch(fn)
        )
        ctx.save_for_backward(stream, fn, out, inv_rms)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out: torch.Tensor):
        stream, fn, out, inv_rms = ctx.saved_tensors
        need_stream, need_fn, _ = ctx.needs_input_grad
        width = fn.shape[1]
        n_tokens = inv_rms.numel()
        grad_out = grad_out.contiguous()
        tile = _projection_launch(fn)
        dstream, dfn = None, None
        if need_stream:
            dstream = torch.empty_like(stream, memory_format=torch.contiguous_format)
            _projection_stream_bwd_kernel[(triton.cdiv(n_tokens, _PROJECTION_TOKENS),)](
                grad_out, stream, fn, out, inv_rms, dstream, n_tokens, width, **tile
            )
        if need_fn:
            dfn = torch.empty_like(fn, memory_format=torch.contiguous_format)
            _projection_fn_bwd_kernel[(triton.cdiv(width, _PROJECTION_COLUMNS),)](
                grad_out, stream, inv_rms, dfn, n_tokens, width, **tile
            )
        return dstream, dfn, None


# Eager under torch.compile, as these kernels are already fused and, traced, the token count would become a graph input.
@torch.compiler.disable
def project_streams(stream: torch.Tensor, fn: torch.Tensor, norm_eps: float) -> torch.Tensor:
    """``F.linear(x / rms(x), fn)`` in fp32 with ``x`` the streams of ``[..., N, H]`` flattened per token, where
    ``rms(x) = sqrt(mean(x^2) + norm_eps)`` and ``fn`` is ``[K, N * H]``. Returns ``[..., K]``."""
    return _StreamProjection.apply(stream, fn, norm_eps)


@triton.jit
def _sinkhorn_cells(n_tokens, N: tl.constexpr, N_PAD: tl.constexpr, BLOCK: tl.constexpr, STEPS: tl.constexpr):
    tokens = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    lane = tl.arange(0, N_PAD)
    cells = tokens[:, None, None] * (N * N) + lane[None, :, None] * N + lane[None, None, :]
    exists = (tokens[:, None, None] < n_tokens) & (lane[None, :, None] < N) & (lane[None, None, :] < N)
    scale_cells = tokens[:, None] * (STEPS * N) + lane[None, :]
    scale_exists = (tokens[:, None] < n_tokens) & (lane[None, :] < N)
    return cells, exists, scale_cells, scale_exists


@triton.jit
def _normalize_step(matrix, scale_ptr, scale_exists, eps, AXIS: tl.constexpr, SAVE: tl.constexpr):
    scale = tl.sum(matrix, axis=AXIS) + eps
    if SAVE:
        tl.store(scale_ptr, scale, mask=scale_exists)
    # A lane past the token or the matrix divides by one, as at a zero ``eps`` its NaN would reach every later sum.
    return matrix / tl.expand_dims(tl.where(scale_exists, scale, 1.0), AXIS)


@triton.jit(do_not_specialize=["n_tokens"])
def _sinkhorn_fwd_kernel(
    comb_ptr,
    out_ptr,
    scales_ptr,
    n_tokens,
    eps,
    N: tl.constexpr,
    N_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
    LOOPS: tl.constexpr,
    STEPS: tl.constexpr,
    SAVE: tl.constexpr,
):
    cells, exists, scale_cells, scale_exists = _sinkhorn_cells(n_tokens, N, N_PAD, BLOCK, STEPS)
    scale = scales_ptr + scale_cells
    matrix = tl.load(comb_ptr + cells, mask=exists, other=0.0)
    matrix = _normalize_step(matrix, scale, scale_exists, eps, _COLUMN_SUM, SAVE)
    for loop in range(LOOPS):
        step = 2 * loop + 1
        matrix = _normalize_step(matrix, scale + step * N, scale_exists, eps, _ROW_SUM, SAVE)
        matrix = _normalize_step(matrix, scale + (step + 1) * N, scale_exists, eps, _COLUMN_SUM, SAVE)
    tl.store(out_ptr + cells, matrix, mask=exists)


@triton.jit
def _reverse_step(grad, matrix, scale_ptr, scale_exists, AXIS: tl.constexpr):
    """``y = x / s`` with ``s = sum(x, axis) + eps``: ``dx = (dy - sum(dy * y, axis)) / s`` and ``x = y * s``. A lane
    past the token or the matrix reads a unit scale, so its zeros stay zeros."""
    scale = tl.expand_dims(tl.load(scale_ptr, mask=scale_exists, other=1.0), AXIS)
    return (grad - tl.expand_dims(tl.sum(grad * matrix, axis=AXIS), AXIS)) / scale, matrix * scale


@triton.jit(do_not_specialize=["n_tokens"])
def _sinkhorn_bwd_kernel(
    grad_ptr,
    out_ptr,
    scales_ptr,
    dcomb_ptr,
    n_tokens,
    N: tl.constexpr,
    N_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
    LOOPS: tl.constexpr,
    STEPS: tl.constexpr,
):
    cells, exists, scale_cells, scale_exists = _sinkhorn_cells(n_tokens, N, N_PAD, BLOCK, STEPS)
    scale = scales_ptr + scale_cells
    matrix = tl.load(out_ptr + cells, mask=exists, other=0.0)
    grad = tl.load(grad_ptr + cells, mask=exists, other=0.0)
    for loop in range(LOOPS):
        step = STEPS - 1 - 2 * loop
        grad, matrix = _reverse_step(grad, matrix, scale + step * N, scale_exists, _COLUMN_SUM)
        grad, matrix = _reverse_step(grad, matrix, scale + (step - 1) * N, scale_exists, _ROW_SUM)
    grad, _ = _reverse_step(grad, matrix, scale, scale_exists, _COLUMN_SUM)
    tl.store(dcomb_ptr + cells, grad, mask=exists)


def _sinkhorn_launch(n_tokens: int, n: int, iters: int) -> tuple[tuple[int], dict]:
    n_pad = triton.next_power_of_2(n)
    block = _SINKHORN_TILE // (n_pad * n_pad)
    loops = max(iters - 1, 0)
    tile = {"N": n, "N_PAD": n_pad, "BLOCK": block, "LOOPS": loops, "STEPS": 2 * loops + 1}
    return (triton.cdiv(n_tokens, block),), tile


def _sinkhorn_forward(comb: torch.Tensor, iters: int, eps: float, save: bool) -> tuple[torch.Tensor, torch.Tensor]:
    n = comb.shape[-1]
    n_tokens = comb.numel() // (n * n)
    grid, tile = _sinkhorn_launch(n_tokens, n, iters)
    out = torch.empty_like(comb, memory_format=torch.contiguous_format)
    # Unwritten without ``save``.
    scales = torch.empty((n_tokens, tile["STEPS"], n), device=comb.device, dtype=comb.dtype) if save else out
    _sinkhorn_fwd_kernel[grid](comb.contiguous(), out, scales, n_tokens, eps, SAVE=save, **tile)
    return out, scales


class _SinkhornKnopp(torch.autograd.Function):
    """Saves the final matrix and each step's scale, ``[T, 2 * iters - 1, N]``: every earlier matrix follows from them
    in reverse."""

    @staticmethod
    def forward(ctx, comb: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
        out, scales = _sinkhorn_forward(comb, iters, eps, save=True)
        ctx.save_for_backward(out, scales)
        ctx.iters = iters
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out: torch.Tensor):
        out, scales = ctx.saved_tensors
        n_tokens = scales.shape[0]
        grid, tile = _sinkhorn_launch(n_tokens, out.shape[-1], ctx.iters)
        dcomb = torch.empty_like(out, memory_format=torch.contiguous_format)
        _sinkhorn_bwd_kernel[grid](grad_out.contiguous(), out, scales, dcomb, n_tokens, **tile)
        return dcomb, None, None


@torch.compiler.disable
def sinkhorn_knopp(comb: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    """``comb / (comb.sum(-2, keepdim=True) + eps)``, then ``iters - 1`` passes over rows and columns of the same, over
    ``[..., N, N]``. Saves nothing with no graph to record."""
    if torch.is_grad_enabled() and comb.requires_grad:
        return _SinkhornKnopp.apply(comb, iters, eps)
    return _sinkhorn_forward(comb, iters, eps, save=False)[0]


@triton.jit
def _lane_cells(token, N: tl.constexpr, N_PAD: tl.constexpr):
    lane = tl.arange(0, N_PAD)
    in_lane = lane < N
    return lane, in_lane, token * N + lane, token * (N * N) + lane[:, None] * N + lane[None, :]


@triton.jit
def _stream_cells(token, start, width, lane, in_lane, N: tl.constexpr, COLUMNS: tl.constexpr):
    cols = start + tl.arange(0, COLUMNS)
    in_col = cols < width
    block_cells = token * width + cols
    stream_cells = token * (N * width) + lane[:, None] * width + cols[None, :]
    return block_cells, in_col, stream_cells, in_lane[:, None] & in_col[None, :]


@triton.jit
def _lane_product(lanes, columns):
    """``lanes @ columns`` with fewer lanes than ``tl.dot`` takes."""
    return tl.sum(lanes[:, :, None] * columns[None, :, :], axis=1)


def _stream_launch(stream: torch.Tensor) -> tuple[tuple[int], dict]:
    n, width = stream.shape[-2:]
    n_pad = triton.next_power_of_2(n)
    grid = (stream.numel() // (n * width),)
    return grid, {
        "N": n,
        "N_PAD": n_pad,
        "COLUMNS": _STREAM_TILE // (n_pad * n_pad),
        "num_warps": _STREAM_WARPS,
    }


def _empty_grads(ctx, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """An input stands in for a gradient nothing asks for, which the kernel does not write."""
    return tuple(
        torch.empty_like(t, memory_format=torch.contiguous_format) if needed else t
        for t, needed in zip(inputs, ctx.needs_input_grad, strict=True)
    )


def _requested(ctx, grads: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor | None, ...]:
    return tuple(grad if needed else None for grad, needed in zip(grads, ctx.needs_input_grad, strict=True))


@triton.jit
def _collapse_fwd_kernel(
    stream_ptr, weight_ptr, out_ptr, width, N: tl.constexpr, N_PAD: tl.constexpr, COLUMNS: tl.constexpr
):
    token = tl.program_id(0).to(tl.int64)
    lane, in_lane, weight_cells, _ = _lane_cells(token, N, N_PAD)
    weight = tl.load(weight_ptr + weight_cells, mask=in_lane, other=0.0).to(tl.float32)
    for start in range(0, width, COLUMNS):
        block_cells, in_col, stream_cells, in_stream = _stream_cells(token, start, width, lane, in_lane, N, COLUMNS)
        stream = tl.load(stream_ptr + stream_cells, mask=in_stream, other=0.0).to(tl.float32)
        tl.store(out_ptr + block_cells, tl.sum(weight[:, None] * stream, axis=0), mask=in_col)


@triton.jit
def _collapse_bwd_kernel(
    grad_ptr,
    stream_ptr,
    weight_ptr,
    dstream_ptr,
    dweight_ptr,
    width,
    N: tl.constexpr,
    N_PAD: tl.constexpr,
    COLUMNS: tl.constexpr,
    NEED_STREAM: tl.constexpr,
    NEED_WEIGHT: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    lane, in_lane, weight_cells, _ = _lane_cells(token, N, N_PAD)
    weight = tl.load(weight_ptr + weight_cells, mask=in_lane, other=0.0).to(tl.float32)
    # Per column across the passes and reduced once at the token's end, as one reduction per pass costs several times
    # everything else here.
    dweight = tl.zeros((N_PAD, COLUMNS), dtype=tl.float32)
    for start in range(0, width, COLUMNS):
        block_cells, in_col, stream_cells, in_stream = _stream_cells(token, start, width, lane, in_lane, N, COLUMNS)
        grad = tl.load(grad_ptr + block_cells, mask=in_col, other=0.0).to(tl.float32)
        if NEED_STREAM:
            tl.store(dstream_ptr + stream_cells, weight[:, None] * grad[None, :], mask=in_stream)
        if NEED_WEIGHT:
            dweight += tl.load(stream_ptr + stream_cells, mask=in_stream, other=0.0).to(tl.float32) * grad[None, :]
    if NEED_WEIGHT:
        tl.store(dweight_ptr + weight_cells, tl.sum(dweight, axis=1), mask=in_lane)


class _StreamCollapse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, stream: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        _require_shape(weight, stream.shape[:-1], "weight")
        stream, weight = stream.contiguous(), weight.contiguous()
        grid, tile = _stream_launch(stream)
        out = torch.empty((*stream.shape[:-2], stream.shape[-1]), device=stream.device, dtype=stream.dtype)
        _collapse_fwd_kernel[grid](stream, weight, out, stream.shape[-1], **tile)
        ctx.save_for_backward(stream, weight)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out: torch.Tensor):
        stream, weight = ctx.saved_tensors
        grid, tile = _stream_launch(stream)
        grads = _empty_grads(ctx, stream, weight)
        need_stream, need_weight = ctx.needs_input_grad
        _collapse_bwd_kernel[grid](
            grad_out.contiguous(),
            stream,
            weight,
            *grads,
            stream.shape[-1],
            NEED_STREAM=need_stream,
            NEED_WEIGHT=need_weight,
            **tile,
        )
        return _requested(ctx, grads)


@torch.compiler.disable
def collapse_streams(stream: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``(weight.unsqueeze(-1) * stream).sum(-2)``, for ``[..., N, H]`` ``stream`` and ``[..., N]`` ``weight``, in
    ``stream``'s dtype."""
    return _StreamCollapse.apply(stream, weight)


@triton.jit
def _mix_fwd_kernel(
    block_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    width,
    N: tl.constexpr,
    N_PAD: tl.constexpr,
    COLUMNS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    lane, in_lane, post_cells, comb_cells = _lane_cells(token, N, N_PAD)
    post = tl.load(post_ptr + post_cells, mask=in_lane, other=0.0).to(tl.float32)
    comb = tl.load(comb_ptr + comb_cells, mask=in_lane[:, None] & in_lane[None, :], other=0.0).to(tl.float32)
    comb_t = tl.trans(comb)
    for start in range(0, width, COLUMNS):
        block_cells, in_col, stream_cells, in_stream = _stream_cells(token, start, width, lane, in_lane, N, COLUMNS)
        block = tl.load(block_ptr + block_cells, mask=in_col, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + stream_cells, mask=in_stream, other=0.0).to(tl.float32)
        out = post[:, None] * block[None, :] + _lane_product(comb_t, residual)
        tl.store(out_ptr + stream_cells, out, mask=in_stream)


@triton.jit
def _mix_bwd_kernel(
    grad_ptr,
    block_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    dblock_ptr,
    dresidual_ptr,
    dpost_ptr,
    dcomb_ptr,
    width,
    N: tl.constexpr,
    N_PAD: tl.constexpr,
    COLUMNS: tl.constexpr,
    NEED_BLOCK: tl.constexpr,
    NEED_RESIDUAL: tl.constexpr,
    NEED_POST: tl.constexpr,
    NEED_COMB: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    lane, in_lane, post_cells, comb_cells = _lane_cells(token, N, N_PAD)
    in_comb = in_lane[:, None] & in_lane[None, :]
    post = tl.load(post_ptr + post_cells, mask=in_lane, other=0.0).to(tl.float32)
    comb = tl.load(comb_ptr + comb_cells, mask=in_comb, other=0.0).to(tl.float32)
    # Per column across the passes, as in ``_collapse_bwd_kernel``.
    dpost = tl.zeros((N_PAD, COLUMNS), dtype=tl.float32)
    dcomb = tl.zeros((N_PAD, N_PAD, COLUMNS), dtype=tl.float32)
    for start in range(0, width, COLUMNS):
        block_cells, in_col, stream_cells, in_stream = _stream_cells(token, start, width, lane, in_lane, N, COLUMNS)
        grad = tl.load(grad_ptr + stream_cells, mask=in_stream, other=0.0).to(tl.float32)
        if NEED_BLOCK:
            tl.store(dblock_ptr + block_cells, tl.sum(post[:, None] * grad, axis=0), mask=in_col)
        if NEED_RESIDUAL:
            tl.store(dresidual_ptr + stream_cells, _lane_product(comb, grad), mask=in_stream)
        if NEED_POST:
            dpost += grad * tl.load(block_ptr + block_cells, mask=in_col, other=0.0).to(tl.float32)[None, :]
        if NEED_COMB:
            residual = tl.load(residual_ptr + stream_cells, mask=in_stream, other=0.0).to(tl.float32)
            dcomb += residual[:, None, :] * grad[None, :, :]
    if NEED_POST:
        tl.store(dpost_ptr + post_cells, tl.sum(dpost, axis=1), mask=in_lane)
    if NEED_COMB:
        tl.store(dcomb_ptr + comb_cells, tl.sum(dcomb, axis=2), mask=in_comb)


class _StreamMix(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, block: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
    ) -> torch.Tensor:
        _require_shape(block, (*residual.shape[:-2], residual.shape[-1]), "block")
        _require_shape(post, residual.shape[:-1], "post")
        _require_shape(comb, (*residual.shape[:-1], residual.shape[-2]), "comb")
        block, residual, post, comb = (t.contiguous() for t in (block, residual, post, comb))
        grid, tile = _stream_launch(residual)
        out = torch.empty_like(
            residual,
            dtype=torch.promote_types(block.dtype, residual.dtype),
            memory_format=torch.contiguous_format,
        )
        _mix_fwd_kernel[grid](block, residual, post, comb, out, residual.shape[-1], **tile)
        ctx.save_for_backward(block, residual, post, comb)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out: torch.Tensor):
        block, residual, post, comb = ctx.saved_tensors
        grid, tile = _stream_launch(residual)
        grads = _empty_grads(ctx, block, residual, post, comb)
        need_block, need_residual, need_post, need_comb = ctx.needs_input_grad
        _mix_bwd_kernel[grid](
            grad_out.contiguous(),
            block,
            residual,
            post,
            comb,
            *grads,
            residual.shape[-1],
            NEED_BLOCK=need_block,
            NEED_RESIDUAL=need_residual,
            NEED_POST=need_post,
            NEED_COMB=need_comb,
            **tile,
        )
        return _requested(ctx, grads)


@torch.compiler.disable
def mix_streams(block: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """``post.unsqueeze(-1) * block.unsqueeze(-2) + comb.transpose(-1, -2) @ residual``, for ``[..., H]`` ``block``,
    ``[..., N, H]`` ``residual``, ``[..., N]`` ``post`` and ``[..., N, N]`` ``comb``."""
    return _StreamMix.apply(block, residual, post, comb)
