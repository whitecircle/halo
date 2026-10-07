"""Fused GLU activations for grouped MoE expert compute.

One row-strided kernel pair serves every combine: SiLU and tanh-GELU (SwiGLU / GeGLU), and the clamped
SwiGLU family (GptOss, DeepSeek-V4, GLM-5 Next, Step-3.7). The activation, the clamp placement and
GptOss's ``up + 1`` are ``tl.constexpr``, so each combine compiles only its own arithmetic; ``alpha`` and
the bound are runtime arguments, so every layer and family with one combine shares one compilation.

Row-strided means the kernels read the ``gate``/``up`` halves of a fused ``[gate | up]`` projection in
place. The packed entry points (:data:`PACKED_GLU_MULS`) take that projection output whole and return
its gradient as one ``[..., 2M]`` buffer, so the expert path runs no copy on either side of the activation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import tanh

from src.env import env_flag

logger = logging.getLogger(__name__)

# ``HALO_FUSED_GLU=0`` runs every combine below through its eager form on CUDA as well, the switch for a
# GPU where a GLU kernel fails to compile or launch. Read once at import.
_FUSED_GLU_ENABLED = env_flag("HALO_FUSED_GLU", True)

# Activation codes (``tl.constexpr`` in the kernels).
_SILU = 0
_GELU_TANH = 1
_CLAMPED_SILU = 2  # alpha-scaled SiLU with the clamps of the GptOss/DeepSeek-V4/Step-3.7 family

# tanh-GELU (`approximate="tanh"`): 0.5*x*(1 + tanh(sqrt(2/pi)*(x + k*x^3))). The constants are fixed by
# the reference rather than tuned: the kernel is gated on reproducing torch's own tanh-GELU. Wrapped in
# ``tl.constexpr`` because a ``@triton.jit`` body may only read module globals that are constexpr.
_GELU_SQRT_2_OVER_PI = tl.constexpr(0.7978845608028654)
_GELU_TANH_COEFF = tl.constexpr(0.044715)

# Both saturating tails, both signs, zero and the near-zero region. Sufficiency is checked by the
# registry-wide sweep in ``tests/cpu/kernels/test_silu_activation_probe.py``, which requires every other
# ``ACT2FN`` entry to be rejected for each probed activation.
_ACTIVATION_PROBE_INPUT = (-64.0, -8.0, -1.0, -0.5, -1e-3, 0.0, 1e-3, 0.5, 1.0, 8.0, 64.0)


def _computes_exactly(act_fn: object, reference_fn: Callable[[torch.Tensor], torch.Tensor], name: str) -> bool:
    """Whether ``act_fn`` reproduces ``reference_fn`` bit-for-bit over the probe vector.

    The test is behavioural rather than by type: ``ACT2FN["silu"]`` is a bare ``nn.Module`` that is
    neither ``nn.SiLU`` nor ``F.silu``, so an ``isinstance`` check would disable the kernels. A
    non-callable, a raised exception, or any result not exactly equal in shape, dtype and value returns
    False, falling back to the eager combine. Callers latch the result at construction, so an activation
    whose parameters train away from the reference would keep the kernel enabled.
    """
    if not callable(act_fn):
        return False
    try:
        probe = torch.tensor(_ACTIVATION_PROBE_INPUT, dtype=torch.float32, device="cpu")
        expected = reference_fn(probe)
        with torch.no_grad():
            out = act_fn(probe)
        return (
            isinstance(out, torch.Tensor)
            and out.shape == expected.shape
            and out.dtype == expected.dtype
            and torch.equal(out, expected)
        )
    except Exception:
        logger.warning(f"Activation {act_fn!r} could not be probed against {name}; treating it as non-{name}.")
        return False


def is_silu_activation(act_fn: object) -> bool:
    """Whether ``act_fn`` computes exactly ``F.silu``, so a SiLU-hardcoding kernel may stand in for it."""
    return _computes_exactly(act_fn, torch.nn.functional.silu, "F.silu")


def is_gelu_tanh_activation(act_fn: object) -> bool:
    """Whether ``act_fn`` computes exactly the tanh-GELU approximation the fused GeGLU kernel hardcodes.

    Exact (erf) GELU is a different function and returns False; using the kernel for it would change
    the activation on every expert.
    """
    return _computes_exactly(act_fn, lambda x: torch.nn.functional.gelu(x, approximate="tanh"), "tanh-GELU")


def silu_mul_eager(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Reference SwiGLU combine used by the fused kernel."""
    return torch.nn.functional.silu(gate) * up


def gelu_tanh_mul_eager(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Reference tanh-GeGLU combine used by Gemma 4."""
    return torch.nn.functional.gelu(gate, approximate="tanh") * up


def gptoss_glu_eager(gate: torch.Tensor, up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """Reference (unfused) GptOss clamped-SwiGLU — equivalence target for the kernel."""
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return (up + 1) * (gate * torch.sigmoid(gate * alpha))


def clamped_silu_mul_eager(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """Clamped SwiGLU shared by DeepSeek-V4 and GLM-5 Next experts (their ``_apply_gate``):
    clamp the gate from above and the up half symmetrically, then SiLU-gate."""
    return torch.nn.functional.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)


def silu_then_clamp_mul_eager(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """Step-3.7's clamped SwiGLU (``Step3p7Experts._apply_gate``): SiLU first, then clamp the activated
    gate from above and the up half symmetrically. The post-activation counterpart of
    :func:`clamped_silu_mul_eager` and not interchangeable with it: it saturates at ``limit`` where the
    pre-activation clamp saturates at ``silu(limit)``, and the gate gradient vanishes elsewhere."""
    return torch.nn.functional.silu(gate).clamp(max=limit) * up.clamp(min=-limit, max=limit)


class _GluVariant(NamedTuple):
    """One combine: its activation code, runtime ``alpha``/``limit``, and the clamped family's switches."""

    activation: int
    alpha: float = 1.0
    limit: float = 0.0
    clamp_activated: bool = False
    up_plus_one: bool = False


_SILU_VARIANT = _GluVariant(_SILU)
_GELU_TANH_VARIANT = _GluVariant(_GELU_TANH)


def _clamped_variant(limit: float, clamp_activated: bool) -> _GluVariant:
    """The clamped SwiGLU at ``alpha = 1``: the gate clamped before the SiLU, or (``clamp_activated``) the
    activated gate after it."""
    return _GluVariant(_CLAMPED_SILU, 1.0, float(limit), clamp_activated, False)


def _glu_launch(n_rows: int, width: int) -> tuple[tuple[int, int], dict]:
    """The grid and the ``BLOCK_M``/``ROWS`` tile of a GLU kernel launch over ``[n_rows, width]``.

    ``BLOCK_M`` is the column tile, ``ROWS`` the rows each program walks. The tile is keyed on the width
    alone, so the EP layer's small-token warmup compiles the exact variant a full dispatch runs. Tuned on
    the Gemma 4 and Qwen/GLM expert widths.
    """
    block_m, rows = (1024, 4) if width <= 1024 else (512, 8)
    return (triton.cdiv(n_rows, rows), triton.cdiv(width, block_m)), {"BLOCK_M": block_m, "ROWS": rows}


@triton.jit
def _gelu_tanh_terms(gate):
    """tanh-GELU of ``gate``, and the tanh its derivative reuses."""
    tanh_inner = tanh(_GELU_SQRT_2_OVER_PI * (gate + _GELU_TANH_COEFF * gate * gate * gate))
    return 0.5 * gate * (1.0 + tanh_inner), tanh_inner


@triton.jit
def _clamped_glu_terms(gate, up, alpha, limit, CLAMP_ACTIVATED: tl.constexpr, UP_PLUS_ONE: tl.constexpr):
    """The clamped family's activated gate and clamped up half, plus what the backward differentiates
    through: the gate operand, its scaled sigmoid and the GLU before any post-activation clamp."""
    up_g = tl.minimum(tl.maximum(up, -limit), limit)
    if UP_PLUS_ONE:
        up_g = up_g + 1.0
    gate_a = gate if CLAMP_ACTIVATED else tl.minimum(gate, limit)
    s = tl.sigmoid(gate_a * alpha)
    glu = gate_a * s
    activated = tl.minimum(glu, limit) if CLAMP_ACTIVATED else glu
    return activated, up_g, gate_a, s, glu


# ``n_rows`` is the dispatch's row count, which changes every step. Left specializable, Triton compiles a
# binary per class of it (1, a multiple of 16, neither), and an unseen class compiles between an EP
# dispatch and its combine.
@triton.jit(do_not_specialize=["n_rows"])
def _glu_fwd_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    n_rows,
    width,
    stride_gate,
    stride_up,
    alpha,
    limit,
    ACTIVATION: tl.constexpr,
    CLAMP_ACTIVATED: tl.constexpr,
    UP_PLUS_ONE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ROWS: tl.constexpr,
):
    # int64 row offsets: rows * stride can exceed 2**31.
    row0 = tl.program_id(0).to(tl.int64) * ROWS
    cols = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    col_mask = cols < width
    for r in tl.static_range(ROWS):
        row = row0 + r
        mask = col_mask & (row < n_rows)
        gate = tl.load(gate_ptr + row * stride_gate + cols, mask=mask).to(tl.float32)
        up = tl.load(up_ptr + row * stride_up + cols, mask=mask).to(tl.float32)
        if ACTIVATION == 0:
            out = gate * tl.sigmoid(gate) * up
        elif ACTIVATION == 1:
            activated, _ = _gelu_tanh_terms(gate)
            out = activated * up
        else:
            activated, up_g, _, _, _ = _clamped_glu_terms(gate, up, alpha, limit, CLAMP_ACTIVATED, UP_PLUS_ONE)
            out = activated * up_g
        tl.store(out_ptr + row * width + cols, out, mask=mask)


@triton.jit(do_not_specialize=["n_rows"])
def _glu_bwd_kernel(
    gate_ptr,
    up_ptr,
    dout_ptr,
    dgate_ptr,
    dup_ptr,
    n_rows,
    width,
    stride_gate,
    stride_up,
    stride_dgate,
    stride_dup,
    alpha,
    limit,
    ACTIVATION: tl.constexpr,
    CLAMP_ACTIVATED: tl.constexpr,
    UP_PLUS_ONE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ROWS: tl.constexpr,
):
    row0 = tl.program_id(0).to(tl.int64) * ROWS
    cols = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    col_mask = cols < width
    for r in tl.static_range(ROWS):
        row = row0 + r
        mask = col_mask & (row < n_rows)
        gate = tl.load(gate_ptr + row * stride_gate + cols, mask=mask).to(tl.float32)
        up = tl.load(up_ptr + row * stride_up + cols, mask=mask).to(tl.float32)
        dout = tl.load(dout_ptr + row * width + cols, mask=mask).to(tl.float32)
        if ACTIVATION == 0:
            sigmoid = tl.sigmoid(gate)
            activated = gate * sigmoid
            dgate = dout * up * (sigmoid * (1.0 + gate * (1.0 - sigmoid)))
            dup = dout * activated
        elif ACTIVATION == 1:
            activated, tanh_inner = _gelu_tanh_terms(gate)
            inner_grad = _GELU_SQRT_2_OVER_PI * (1.0 + 3.0 * _GELU_TANH_COEFF * (gate * gate))
            activation_grad = 0.5 * (1.0 + tanh_inner) + 0.5 * gate * (1.0 - tanh_inner * tanh_inner) * inner_grad
            dgate = dout * up * activation_grad
            dup = dout * activated
        else:
            activated, up_g, gate_a, s, glu = _clamped_glu_terms(gate, up, alpha, limit, CLAMP_ACTIVATED, UP_PLUS_ONE)
            dglu = s + gate_a * alpha * s * (1.0 - s)
            # Clamp subgradients follow torch: the bound itself is inside the pass-through interval.
            up_in = tl.where((up >= -limit) & (up <= limit), 1.0, 0.0)
            gate_in = tl.where(glu <= limit, 1.0, 0.0) if CLAMP_ACTIVATED else tl.where(gate <= limit, 1.0, 0.0)
            dgate = dout * up_g * dglu * gate_in
            dup = dout * activated * up_in
        tl.store(dgate_ptr + row * stride_dgate + cols, dgate, mask=mask)
        tl.store(dup_ptr + row * stride_dup + cols, dup, mask=mask)


def _as_rows(t: torch.Tensor) -> torch.Tensor:
    """``t`` as a 2D ``[rows, width]`` tensor with a unit column stride, copying only when no such view exists."""
    if t.stride(-1) != 1:
        t = t.contiguous()
    return t.reshape(-1, t.shape[-1])


def _variant_args(variant: _GluVariant) -> dict:
    return {
        "alpha": variant.alpha,
        "limit": variant.limit,
        "ACTIVATION": variant.activation,
        "CLAMP_ACTIVATED": variant.clamp_activated,
        "UP_PLUS_ONE": variant.up_plus_one,
    }


def _glu_forward(gate: torch.Tensor, up: torch.Tensor, variant: _GluVariant) -> torch.Tensor:
    n_rows, width = gate.shape
    out = torch.empty((n_rows, width), device=gate.device, dtype=gate.dtype)
    grid, tile = _glu_launch(n_rows, width)
    _glu_fwd_kernel[grid](
        gate,
        up,
        out,
        n_rows,
        width,
        gate.stride(0),
        up.stride(0),
        **_variant_args(variant),
        **tile,
    )
    return out


def _glu_backward(
    gate: torch.Tensor,
    up: torch.Tensor,
    grad_out: torch.Tensor,
    dgate: torch.Tensor,
    dup: torch.Tensor,
    variant: _GluVariant,
) -> None:
    n_rows, width = gate.shape
    grid, tile = _glu_launch(n_rows, width)
    _glu_bwd_kernel[grid](
        gate,
        up,
        grad_out.contiguous().view(n_rows, width),
        dgate,
        dup,
        n_rows,
        width,
        gate.stride(0),
        up.stride(0),
        dgate.stride(0),
        dup.stride(0),
        **_variant_args(variant),
        **tile,
    )


class _FusedGLU(torch.autograd.Function):
    """``combine(gate, up)`` over separate (possibly row-strided) halves."""

    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor, variant: _GluVariant) -> torch.Tensor:
        shape = gate.shape
        gate2d, up2d = _as_rows(gate), _as_rows(up)
        ctx.save_for_backward(gate2d, up2d)
        ctx.variant, ctx.shape = variant, shape
        return _glu_forward(gate2d, up2d, variant).view(shape)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        gate, up = ctx.saved_tensors
        dgate, dup = (
            torch.empty_like(gate, memory_format=torch.contiguous_format),
            torch.empty_like(up, memory_format=torch.contiguous_format),
        )
        _glu_backward(gate, up, grad_out, dgate, dup, ctx.variant)
        return dgate.view(ctx.shape), dup.view(ctx.shape), None


class _FusedPackedGLU(torch.autograd.Function):
    """``combine(gate, up)`` over a fused ``[..., 2M]`` projection output laid out ``[gate | up]``.

    Reads both halves in place and writes the input gradient as one ``[..., 2M]`` buffer, so neither the
    ``.contiguous()`` copies of the chunked halves nor autograd's ``cat`` behind ``chunk`` run.
    """

    @staticmethod
    def forward(ctx, gate_up: torch.Tensor, variant: _GluVariant) -> torch.Tensor:
        shape = gate_up.shape
        rows2d = _as_rows(gate_up)
        width = shape[-1] // 2
        ctx.save_for_backward(rows2d)
        ctx.variant, ctx.shape = variant, shape
        out = _glu_forward(rows2d[:, :width], rows2d[:, width:], variant)
        return out.view(*shape[:-1], width)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (rows2d,) = ctx.saved_tensors
        width = rows2d.shape[-1] // 2
        grad = torch.empty(rows2d.shape, device=rows2d.device, dtype=rows2d.dtype)
        _glu_backward(rows2d[:, :width], rows2d[:, width:], grad_out, grad[:, :width], grad[:, width:], ctx.variant)
        return grad.view(ctx.shape), None


def _split_packed(gate_up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return gate_up.chunk(2, dim=-1)


def fused_glu_enabled() -> bool:
    """Whether the fused kernels may run (``HALO_FUSED_GLU``); off, every combine here runs its eager form."""
    return _FUSED_GLU_ENABLED


def _runs_fused(t: torch.Tensor) -> bool:
    return t.is_cuda and fused_glu_enabled()


def fused_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU combine, fused on CUDA and eager elsewhere."""
    if _runs_fused(gate):
        return _FusedGLU.apply(gate, up, _SILU_VARIANT)
    return silu_mul_eager(gate, up)


def fused_gelu_tanh_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Gemma-style tanh-GeGLU combine, fused on CUDA and eager elsewhere."""
    if _runs_fused(gate):
        return _FusedGLU.apply(gate, up, _GELU_TANH_VARIANT)
    return gelu_tanh_mul_eager(gate, up)


def fused_gptoss_glu(gate: torch.Tensor, up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """:func:`gptoss_glu_eager` — clamp the gate, scale it by ``alpha`` inside the sigmoid, multiply by
    the clamped ``up + 1``. Triton-fused on CUDA, eager fallback otherwise."""
    if _runs_fused(gate):
        return _FusedGLU.apply(gate, up, _GluVariant(_CLAMPED_SILU, float(alpha), float(limit), False, True))
    return gptoss_glu_eager(gate, up, alpha, limit)


def fused_clamped_silu_mul(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """:func:`clamped_silu_mul_eager` — clamp the gate, then SiLU (``alpha = 1``), times the clamped
    ``up``. Fused on CUDA and eager elsewhere."""
    if _runs_fused(gate):
        return _FusedGLU.apply(gate, up, _clamped_variant(limit, clamp_activated=False))
    return clamped_silu_mul_eager(gate, up, limit)


def fused_silu_then_clamp_mul(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """:func:`silu_then_clamp_mul_eager` — SiLU (``alpha = 1``) then clamp the activated gate, times
    the clamped ``up``. Fused on CUDA and eager elsewhere."""
    if _runs_fused(gate):
        return _FusedGLU.apply(gate, up, _clamped_variant(limit, clamp_activated=True))
    return silu_then_clamp_mul_eager(gate, up, limit)


def fused_silu_mul_packed(gate_up: torch.Tensor) -> torch.Tensor:
    """:func:`fused_silu_mul` over a fused ``[gate | up]`` projection output."""
    if _runs_fused(gate_up):
        return _FusedPackedGLU.apply(gate_up, _SILU_VARIANT)
    return silu_mul_eager(*_split_packed(gate_up))


def fused_gelu_tanh_mul_packed(gate_up: torch.Tensor) -> torch.Tensor:
    """:func:`fused_gelu_tanh_mul` over a fused ``[gate | up]`` projection output."""
    if _runs_fused(gate_up):
        return _FusedPackedGLU.apply(gate_up, _GELU_TANH_VARIANT)
    return gelu_tanh_mul_eager(*_split_packed(gate_up))


def fused_clamped_silu_mul_packed(gate_up: torch.Tensor, limit: float) -> torch.Tensor:
    """:func:`fused_clamped_silu_mul` over a fused ``[gate | up]`` projection output."""
    if _runs_fused(gate_up):
        return _FusedPackedGLU.apply(gate_up, _clamped_variant(limit, clamp_activated=False))
    return clamped_silu_mul_eager(*_split_packed(gate_up), limit)


def fused_silu_then_clamp_mul_packed(gate_up: torch.Tensor, limit: float) -> torch.Tensor:
    """:func:`fused_silu_then_clamp_mul` over a fused ``[gate | up]`` projection output."""
    if _runs_fused(gate_up):
        return _FusedPackedGLU.apply(gate_up, _clamped_variant(limit, clamp_activated=True))
    return silu_then_clamp_mul_eager(*_split_packed(gate_up), limit)


# The packed-input form of each combine, keyed on the combine a layer latches (or, for a
# ``functools.partial`` that binds a clamp, on the function it wraps; the bound keywords carry over).
# A latch that is none of these (a bound method, the eager fallback) keeps the chunked path.
PACKED_GLU_MULS: dict[Callable, Callable[..., torch.Tensor]] = {
    fused_silu_mul: fused_silu_mul_packed,
    fused_gelu_tanh_mul: fused_gelu_tanh_mul_packed,
    fused_clamped_silu_mul: fused_clamped_silu_mul_packed,
    fused_silu_then_clamp_mul: fused_silu_then_clamp_mul_packed,
}


def packed_glu_mul(combine: Callable | None) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """The packed form of a latched combine, with any keywords a ``functools.partial`` binds, or ``None``."""
    if isinstance(combine, partial):
        packed = PACKED_GLU_MULS.get(combine.func)
        if packed is None or combine.args:
            return None
        return partial(packed, **combine.keywords)
    return PACKED_GLU_MULS.get(combine) if combine is not None else None


# Probe → the kernel that computes it. A family gets a fused combine by having one of these
# activations rather than by setting a flag.
_FUSED_GLU_MULS = (
    (is_silu_activation, fused_silu_mul),
    (is_gelu_tanh_activation, fused_gelu_tanh_mul),
)

FusedGluMul = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def resolve_fused_glu_mul(act_fn: object) -> FusedGluMul | None:
    """The fused ``act_fn(gate) * up`` kernel for this activation, or ``None`` for the eager combine."""
    for matches, fused in _FUSED_GLU_MULS:
        if matches(act_fn):
            return fused
    return None
