"""Native grouped MoE GEMM on DeepGEMM's ``m_indices`` kernels (``lowp_backend: native``): FP8 activations (1x128)
times FP8 (128x128) or MXFP4 (1x32) weights, each expert's tokens padded to a 128-row segment. The backward is bf16
``F.grouped_mm``, or FP8 under ``lowp_backward_precision: fp8``: dgrad on the 128x128 FP8 weight and wgrad on 128x1
token tiles of the gradient and the activation. The projections of one expert FFN share their row layout, and gate and
up their activation casts, so each is built once per pass."""

from __future__ import annotations

import functools
from collections.abc import Iterable
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from halo.kernels.grouped_mm import grouped_mm_grads
from halo.kernels.lowp.deepgemm_cast_triton import (
    GRADIENT_AMAX_FLOOR,
    TILE,
    cast_rows_to_fp8,
    cast_tiles_to_fp8,
    cast_weight_blocks_to_fp8,
    cast_weight_rows_to_mxfp4,
)
from halo.kernels.lowp.quantization import WeightVersionCache
from halo.kernels.lowp.spec import LOWP_RECIPES, LowpSpec, QuantFormat

DEEPGEMM_FORMATS: tuple[QuantFormat, ...] = tuple(r.quant_format for r in LOWP_RECIPES.values() if r.native)

_ALIGN = TILE  # the contiguous layout pads each expert's row (M) segment to whole scale tiles
_FP4_K_ALIGN = 256  # the fp4 kernel's contraction (K) multiple, two e2m1 per byte
_WGRAD_MAX_GROUPS = 128  # the k-grouped kernel's scale-layout transform takes at most this many experts per call
# DeepGEMM JIT-compiles a second k-grouped kernel (one TMA store stage) once a call's experts average this many
# 128-row blocks, by its output dtype (``get_num_tma_store_stages``, csrc/jit_kernels/heuristics/sm100.hpp).
_WGRAD_LONG_GROUP_BLOCKS = {torch.float32: 32}
_WGRAD_LONG_GROUP_BLOCKS_DEFAULT = 16
_CAST_CHUNK_BYTES = 256 << 20  # the fp32 temporaries of one weight-cast call, which are the size of its input

# w -> (data, scales) per format, re-quantized once per optimizer step; the MXFP4 forward and its FP8 dgrad each keep one.
_weight_caches = {fmt: WeightVersionCache() for fmt in DEEPGEMM_FORMATS}


@functools.lru_cache(maxsize=1)
def _deep_gemm():
    """None when the package (Blackwell image only) is not installed; an install that fails to import raises."""
    try:
        import deep_gemm  # noqa: PLC0415
        from deep_gemm import utils  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        if exc.name != "deep_gemm":
            raise
        return None
    return deep_gemm, utils


def deepgemm_available() -> bool:
    return _deep_gemm() is not None


def ensure_contiguous_alignment() -> None:
    """Set DeepGEMM's process-wide M/K alignment of the contiguous layout, which its grouped kernels assert."""
    _deep_gemm()[1].set_mk_alignment_for_contiguous_layout(_ALIGN)


def _align_up(n, multiple: int):
    return (n + multiple - 1) // multiple * multiple


def _contraction_width(k: int, fmt: QuantFormat) -> int:
    """fp4 packs two e2m1 per byte, so its K is zero-padded to a multiple of 256 (result unchanged)."""
    return _align_up(k, _FP4_K_ALIGN) if fmt == "mxfp4" else k


class _RowLayout(NamedTuple):
    """Each expert's tokens at the start of its own 128-aligned segment of the ``rows``-row operand."""

    dst: torch.Tensor  # [T] int64, the operand row of each token
    src: torch.Tensor  # [rows] int32, the token of each row, -1 on padding
    m_indices: torch.Tensor  # [rows] int32, the expert of each row, -1 on padding
    segments: list[int]  # each expert's segment length
    segments_dev: torch.Tensor  # the same, int32 on the device
    rows: int


class _Memo:
    """``derive(tensor, *extra)`` for the last ``tensor`` and ``extra`` asked about, an ``extra`` object matched by
    identity. It keeps the tensor's storage (through a detached alias, which pins no graph) and the extras alive, as
    freed storage could come back to a later tensor of other contents at the same address, shape and version."""

    def __init__(self, derive):
        self._derive = derive
        self.clear()

    def __call__(self, tensor: torch.Tensor, *extra):
        if tensor.is_inference():  # no version counter to key on
            return self._derive(tensor, *extra)
        key = (tensor.device, tensor.dtype, tensor.untyped_storage().data_ptr(), tensor.storage_offset(),
               tuple(tensor.shape), tensor.stride(), tensor._version,
               *(e if isinstance(e, int) else id(e) for e in extra))  # fmt: skip
        if self._held is None or key != self._key:
            self._value = self._derive(tensor, *extra)
            self._held, self._key = (tensor.detach(), extra), key
        return self._value

    def clear(self) -> None:
        self._held = self._key = self._value = None


def _row_layout(offs: torch.Tensor, num_tokens: int, num_experts: int) -> _RowLayout:
    ends = offs.tolist()
    # Rows outside ``[starts[g], ends[g])`` of the uninitialized output stay unwritten, so ``ends[-1]`` must be ``T``.
    covers_all_tokens = len(ends) == num_experts and ends[0] >= 0 and ends[-1] == num_tokens
    if not covers_all_tokens or any(b < a for a, b in zip(ends, ends[1:], strict=False)):
        raise ValueError(
            f"offs must be [{num_experts}] non-decreasing cumulative token counts starting at >= 0 and ending "
            f"at T={num_tokens}; got {ends}. Rows outside the covered span are never written and would be "
            f"returned uninitialized."
        )
    segments = [_align_up(b - a, _ALIGN) for a, b in zip([0, *ends[:-1]], ends, strict=True)]
    counts = torch.diff(offs.long(), prepend=offs.new_zeros(1, dtype=torch.long))
    segments_dev = _align_up(counts, _ALIGN)
    expert = torch.repeat_interleave(torch.arange(num_experts, device=offs.device), counts, output_size=num_tokens)
    shift = (segments_dev.cumsum(0) - segments_dev) - (counts.cumsum(0) - counts)
    dst = torch.arange(num_tokens, device=offs.device) + shift[expert]
    rows = sum(segments)
    src = torch.full((rows,), -1, device=offs.device, dtype=torch.int32)
    src[dst] = torch.arange(num_tokens, device=offs.device, dtype=torch.int32)
    m_indices = torch.full((rows,), -1, device=offs.device, dtype=torch.int32)
    m_indices[dst] = expert.to(torch.int32)
    return _RowLayout(dst, src, m_indices, segments, segments_dev.to(torch.int32), rows)


class _SharedTiles:
    """One activation's 128x1 FP8 cast for the fp8 wgrad, shared by the projections that read it (gate and up) and
    dropped after the last one's backward."""

    def __init__(self, *_activation):
        self.tiles = None
        self.readers = self.left = 0

    def take(self, x: torch.Tensor, layout: _RowLayout, wanted: bool):
        """A reader's backward: the tiles when it runs a wgrad, cast by the first that does."""
        if wanted and self.tiles is None:
            self.tiles = cast_tiles_to_fp8(x, layout.src)
        tiles = self.tiles if wanted else None
        # Counted per pass: a second backward over a retained graph casts and drops them again.
        self.left = (self.left or self.readers) - 1
        if not self.left:
            self.tiles = None
        return tiles


# One host read of ``offs`` per expert FFN, not per projection; the gate and up GEMMs cast the same activation. The
# memo alone holds the forward's 1x128 rows, so no graph keeps an FP8 copy of an activation.
_layouts = _Memo(_row_layout)
_row_casts = _Memo(lambda x, layout, width: cast_rows_to_fp8(x, layout.src, width))
_shared_tiles = _Memo(_SharedTiles)


def _launch(kernel, *args, what: str, **kwargs) -> None:
    try:
        kernel(*args, **kwargs)
    except torch.cuda.OutOfMemoryError:
        raise
    except RuntimeError as exc:
        raise RuntimeError(
            f"DeepGEMM rejected the {what}: {exc}. A native run has no bf16 fallback, which would change this "
            f"expert's numerics mid-run; set lowp_backend: simulated to run simulated."
        ) from exc


def _cast_experts(b: torch.Tensor, cast, block_scaled: bool):
    """``(data, scales)`` of ``b`` ``[E, N, K]``, cast over chunks of experts that stay under ``_CAST_CHUNK_BYTES``.

    The MXFP4 cast scales row by row, so experts flatten as they are. The FP8 weight cast scales ``TILE`` x ``TILE``
    blocks, so each expert's N is zero-padded to a block boundary first: a block then never spans two experts, and the
    padding is the zeros the per-expert cast would add itself. Outputs are allocated dense and filled by copy, as
    DeepGEMM asserts the weight's expert stride is ``N * K``, which a slice of the padded layout does not keep at E=1."""
    E, N, K = b.shape
    n_cast = _align_up(N, TILE) if block_scaled else N
    step = max(1, _CAST_CHUNK_BYTES // (4 * n_cast * K))

    def cast_chunk(start: int):
        chunk = b[start : start + step]
        count = chunk.shape[0]
        if n_cast != N:
            chunk = F.pad(chunk, (0, 0, 0, n_cast - N))
        data_c, sf_c = cast(chunk.reshape(count * n_cast, K))
        return data_c.reshape(count, n_cast, *data_c.shape[1:])[:, :N], sf_c.reshape(count, -1, *sf_c.shape[1:])

    data_c, sf_c = cast_chunk(0)
    data = data_c.new_empty((E, N, *data_c.shape[2:]))
    sf = sf_c.new_empty((E, *sf_c.shape[1:]))
    for start in range(0, E, step):
        if start:
            data_c, sf_c = cast_chunk(start)
        data[start : start + len(data_c)] = data_c
        sf[start : start + len(sf_c)] = sf_c
        del data_c, sf_c  # the next cast starts with no chunk of the quantized weight held beside the full outputs
    return data, sf


_NON_FINITE_WEIGHT = (
    "Non-finite expert weight{}: e2m1 has no NaN code, so the MXFP4 cast gives a NaN or inf element a finite one and "
    "the run would train on it. The checkpoint or an optimizer step wrote it: check the loaded weights, or resume "
    "from an earlier save at a lower learning rate."
)
# Set on the device by an MXFP4 cast of a NaN or inf weight. Read once per model forward: a read per cast would sync
# the host with the GPU at every expert GEMM of an uncached weight.
_nonfinite_weight: dict[torch.device, torch.Tensor] = {}


def verify_finite_mxfp4_weights(named_weights: Iterable[tuple[str, torch.Tensor]] = ()) -> None:
    """One host read for every MXFP4 weight cast since the last call, which ``apply_lowp_compute`` hooks after the
    last converted layer. The raise names the non-finite ones of ``named_weights`` and drops the cached casts, so a
    caller that catches it cannot run on them. It is local to this rank; its EP peers wait in their next collective
    until it times out."""
    for flag in _nonfinite_weight.values():
        if flag.item():
            flag.zero_()
            _weight_caches["mxfp4"] = WeightVersionCache()
            names = [name for name, weight in named_weights if not weight.isfinite().all()]
            raise ValueError(_NON_FINITE_WEIGHT.format(f" {', '.join(names)}" if names else ""))


def _quant_weight(w: torch.Tensor, fmt: QuantFormat, cacheable: bool = True):
    """Cached per optimizer step unless ``cacheable`` is False, as FSDP2's pinned version counter would serve the
    step-0 quantization forever."""
    _, dg_utils = _deep_gemm()

    # A non-leaf `.to(bf16)` copy churns its id at _version 0.
    cacheable = cacheable and w.is_leaf
    fingerprint = w._version
    if cacheable:
        cached = _weight_caches[fmt].get(w, fingerprint)
        if cached is not None:
            return cached
    if w.is_cuda:
        if fmt == "mxfp8":
            operand = cast_weight_blocks_to_fp8(w)
        else:
            flag = _nonfinite_weight.get(w.device)
            if flag is None:
                flag = _nonfinite_weight[w.device] = torch.zeros(1, device=w.device, dtype=torch.int32)
            operand = cast_weight_rows_to_mxfp4(w, _contraction_width(w.shape[1], fmt), flag)
    else:  # DeepGEMM's reference casts: the FP8 operand keeps the weight's [E, K, N], the MXFP4 one is [E, N, K]
        # e2m1 has no NaN code and the cast turns a NaN amax into a 2^127 scale: the block would decode as zeros.
        if fmt == "mxfp4" and not w.isfinite().all():
            raise ValueError(_NON_FINITE_WEIGHT.format(f" of shape {tuple(w.shape)}"))
        b = w.contiguous() if fmt == "mxfp8" else w.transpose(-2, -1).contiguous()
        b = F.pad(b, (0, _contraction_width(b.shape[2], fmt) - b.shape[2]))
        cast = (
            (lambda t: dg_utils.per_token_cast_to_fp4(t, use_ue8m0=True, gran_k=32))
            if fmt == "mxfp4"
            else (lambda t: dg_utils.per_block_cast_to_fp8(t, use_ue8m0=True, gran_k=TILE))
        )
        operand = _cast_experts(b, cast, block_scaled=fmt == "mxfp8")
    if cacheable:
        _weight_caches[fmt].put(w, fingerprint, operand)
    return operand


def _deepgemm_forward(x, w, layout: _RowLayout, a_op, fmt: QuantFormat, weight_cacheable: bool):
    """bf16 ``[T, N]``; both kernels take an fp8 activation, ``a_op``."""
    deep_gemm, _ = _deep_gemm()
    E, K, N = w.shape
    b_op = _quant_weight(w, fmt, cacheable=weight_cacheable)
    d = x.new_empty(layout.rows, N, dtype=torch.bfloat16)
    kernel, recipe_a, recipe_b = (
        (deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous, (1, 128), (1, 32))
        if fmt == "mxfp4"
        else (deep_gemm.m_grouped_fp8_gemm_nn_contiguous, None, None)  # the FP8 weight is cast [E, K, N]
    )
    _launch(
        kernel,
        a_op,
        b_op,
        d,
        layout.m_indices,
        recipe=None,
        recipe_a=recipe_a,
        recipe_b=recipe_b,
        what=f"{fmt} grouped GEMM on M={layout.rows} N={N} K={K} E={E}",
    )
    return d.index_select(0, layout.dst)


def _fp8_dgrad(grad: torch.Tensor, w_fp8, layout: _RowLayout, k: int) -> torch.Tensor:
    """``grad @ W^T`` per expert, ``[T, K]``: the ``[E, K, N]`` 128x128-scaled FP8 weight, read N-major."""
    deep_gemm, _ = _deep_gemm()
    d = grad.new_empty(layout.rows, k, dtype=torch.bfloat16)
    _launch(
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous,
        cast_rows_to_fp8(grad, layout.src, grad.shape[1], GRADIENT_AMAX_FLOOR),
        w_fp8,
        d,
        layout.m_indices,
        what=f"fp8 dgrad on M={layout.rows} N={k} K={grad.shape[1]}",
    )
    return d.index_select(0, layout.dst)


def _launch_wgrad(x_op, g_op, segments: list[int], segments_dev: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The k-grouped kernel over at most ``_WGRAD_MAX_GROUPS`` experts per call. It writes every expert's block, a zero
    one for an expert with no rows, so the output needs no zero fill."""
    deep_gemm, _ = _deep_gemm()
    E, K, N = len(segments), x_op[0].shape[1], g_op[0].shape[1]
    d = g_op[0].new_empty(E, K, N, dtype=dtype)
    row = 0
    for start in range(0, E, _WGRAD_MAX_GROUPS):
        stop = min(start + _WGRAD_MAX_GROUPS, E)
        ks = segments[start:stop]
        end = row + sum(ks)
        rows, tiles = slice(row, end), slice(row // TILE, end // TILE)
        _launch(
            deep_gemm.k_grouped_fp8_gemm_tn_contiguous,
            (x_op[0][rows], x_op[1][tiles]),
            (g_op[0][rows], g_op[1][tiles]),
            d[start:stop],
            ks,
            segments_dev[start:stop],
            None,
            recipe=(1, 1, TILE),
            what=f"fp8 wgrad on M={K} N={N} E={stop - start}",
        )
        row = end
    return d


def _fp8_wgrad(x_op, grad: torch.Tensor, layout: _RowLayout, dtype: torch.dtype) -> torch.Tensor:
    """``x_e^T @ grad_e`` per expert, ``[E, K, N]`` in ``dtype``, both operands cast over 128-token tiles of one
    expert; the kernel accumulates in fp32 and rounds once into its output."""
    g_op = cast_tiles_to_fp8(grad, layout.src, GRADIENT_AMAX_FLOOR)
    return _launch_wgrad(x_op, g_op, layout.segments, layout.segments_dev, dtype)


def warm_fp8_wgrad(num_experts: int, k: int, n: int, dtype: torch.dtype, device: torch.device) -> None:
    """Compile the weight-gradient kernel's long-group variant for an ``[num_experts, k, n]`` bank in ``dtype``, on
    zero operands of the fewest rows that select it. A backward over a few rows per expert compiles the other one,
    and the first long receive would otherwise compile this one mid-step."""
    ensure_contiguous_alignment()
    rows_per_expert = TILE * _WGRAD_LONG_GROUP_BLOCKS.get(dtype, _WGRAD_LONG_GROUP_BLOCKS_DEFAULT)

    def zeros(width: int) -> tuple[torch.Tensor, torch.Tensor]:
        rows = num_experts * rows_per_expert
        data = torch.zeros(rows, width, dtype=torch.uint8, device=device).view(torch.float8_e4m3fn)
        return data, torch.ones(rows // TILE, width, dtype=torch.float32, device=device)

    segments = [rows_per_expert] * num_experts
    _launch_wgrad(zeros(k), zeros(n), segments, torch.tensor(segments, dtype=torch.int32, device=device), dtype)


class _DeepGEMMGroupedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, offs, fmt, fp8_backward, weight_cacheable):
        layout = _layouts(offs, x.shape[0], w.shape[0])
        rows = _row_casts(x, layout, _contraction_width(w.shape[1], fmt))
        out = _deepgemm_forward(x, w, layout, rows, fmt, weight_cacheable)
        ctx.layout, ctx.tiles, ctx.w_fp8 = layout, None, None
        if fp8_backward:
            ctx.tiles = _shared_tiles(x, layout)
            ctx.tiles.readers += 1
            # The dgrad's weight is the forward's cast under fp8, a cache hit here, and an FP8 cast of its own under
            # mxfp4, once per step. The backward cannot ask the cache: under saved-tensor hooks (a non-reentrant
            # checkpoint) its saved weight is another object. An uncached weight (FSDP2's, an fp32 master's bf16
            # copy) is cast again there, which costs less than holding an FP8 copy of every bank in the graph.
            if weight_cacheable and w.is_leaf:
                ctx.w_fp8 = _quant_weight(w, "mxfp8")
        ctx.save_for_backward(x, w, offs)
        return out.to(x.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        _row_casts.clear()
        _shared_tiles.clear()
        x, w, offs = ctx.saved_tensors
        needs_x, needs_w = ctx.needs_input_grad[:2]
        if ctx.tiles is not None:
            grad_x = grad_w = None
            x_op = ctx.tiles.take(x, ctx.layout, needs_w)
            if needs_x:
                w_fp8 = ctx.w_fp8 or _quant_weight(w, "mxfp8", cacheable=False)
                grad_x = _fp8_dgrad(grad_output, w_fp8, ctx.layout, w.shape[1])
            if needs_w:
                grad_w = _fp8_wgrad(x_op, grad_output, ctx.layout, w.dtype)
        else:
            grad_x, grad_w = grouped_mm_grads(
                x.to(torch.bfloat16), w.to(torch.bfloat16), grad_output.to(torch.bfloat16), offs, needs_x, needs_w
            )
        if grad_x is not None:
            grad_x = grad_x.to(x.dtype)
        if grad_w is not None:
            grad_w = grad_w.to(w.dtype)
        return grad_x, grad_w, None, None, None, None


def deepgemm_grouped_gemm(x, w, *, offs, lowp: LowpSpec, weight_cacheable: bool = True):
    """Native forward at ``lowp.precision``, with the bf16 or FP8 backward ``lowp.backward_precision`` names. On
    CUDA a non-finite MXFP4 weight raises in :func:`verify_finite_mxfp4_weights`, not here."""
    if not deepgemm_available():
        raise ImportError("deepgemm_grouped_gemm needs the deep_gemm package (Blackwell image only)")
    if offs is None:
        raise ValueError("deepgemm_grouped_gemm needs offs: the kernel's m_indices are built from the group ends")
    if x.shape[1] != w.shape[1]:
        # The row cast writes the weight's K columns, so a wider activation would be truncated, not rejected.
        raise ValueError(f"x is [T, {x.shape[1]}] but w contracts over K={w.shape[1]}; the two must match.")
    ensure_contiguous_alignment()
    fmt, fp8_backward = LOWP_RECIPES[lowp.precision].quant_format, lowp.backward_precision == "fp8"
    # Any other format would run as FP8 on the wrong weight layout; a direct caller has not been through
    # ``LowpSpec.validate``.
    if fmt not in DEEPGEMM_FORMATS:
        raise ValueError(f"deepgemm_grouped_gemm takes {DEEPGEMM_FORMATS}, got {fmt!r}.")
    return _DeepGEMMGroupedFn.apply(x, w, offs.to(torch.int32), fmt, fp8_backward, weight_cacheable)
