"""Fused GLU combines for MoE expert compute: one row-strided Triton kernel pair serves SiLU and tanh-GELU
(SwiGLU / GeGLU) and the clamped SwiGLU family (GptOss, DeepSeek-V4, GLM-5 Next, Step-3.7), each a
:class:`GluCombine`. :meth:`GluCombine.packed` reads a fused ``[gate | up]`` projection in place and
returns its gradient as one buffer, so the expert path copies on neither side of the activation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import tanh

from src.env import env_flag

logger = logging.getLogger(__name__)

# ``HALO_FUSED_GLU=0`` runs every combine eagerly on CUDA too, for a GPU where a GLU kernel fails to build.
_FUSED_GLU_ENABLED = env_flag("HALO_FUSED_GLU", True)

# Activation codes (``tl.constexpr`` in the kernels).
_SILU = 0
_GELU_TANH = 1
_CLAMPED_SILU = 2  # alpha-scaled SiLU with the clamps of the GptOss/DeepSeek-V4/Step-3.7 family

# torch's tanh-GELU, 0.5*x*(1 + tanh(sqrt(2/pi)*(x + k*x^3))); ``tl.constexpr`` because a ``@triton.jit``
# body reads only constexpr module globals.
_GELU_SQRT_2_OVER_PI = tl.constexpr(0.7978845608028654)
_GELU_TANH_COEFF = tl.constexpr(0.044715)

# Both saturating tails, both signs, zero and the near-zero region; the ``ACT2FN`` sweep in
# ``test_silu_activation_probe.py`` checks it separates every registered activation.
_ACTIVATION_PROBE_INPUT = (-64.0, -8.0, -1.0, -0.5, -1e-3, 0.0, 1e-3, 0.5, 1.0, 8.0, 64.0)


def _computes_exactly(act_fn: object, reference_fn: Callable[[torch.Tensor], torch.Tensor], name: str) -> bool:
    """Whether ``act_fn`` reproduces ``reference_fn`` bit for bit over the probe vector; False on anything
    else, a raise included.

    Behavioural rather than by type: ``ACT2FN["silu"]`` is neither ``nn.SiLU`` nor ``F.silu``. Callers
    latch the result at construction, so an activation whose parameters train away keeps the kernel.
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
    """Whether ``act_fn`` computes exactly the tanh-GELU approximation the fused GeGLU kernel hardcodes;
    exact (erf) GELU is a different function."""
    return _computes_exactly(act_fn, lambda x: torch.nn.functional.gelu(x, approximate="tanh"), "tanh-GELU")


def silu_mul_eager(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU reference."""
    return torch.nn.functional.silu(gate) * up


def gelu_tanh_mul_eager(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """tanh-GeGLU reference (Gemma 4)."""
    return torch.nn.functional.gelu(gate, approximate="tanh") * up


def gptoss_glu_eager(gate: torch.Tensor, up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """GptOss clamped-SwiGLU reference."""
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return (up + 1) * (gate * torch.sigmoid(gate * alpha))


def clamped_silu_mul_eager(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """DeepSeek-V4 and GLM-5 Next clamped SwiGLU reference: the gate clamped from above and the up half
    symmetrically, then SiLU-gated."""
    return torch.nn.functional.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)


def silu_then_clamp_mul_eager(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """Step-3.7 clamped SwiGLU reference: SiLU first, then the activated gate clamped from above and the up
    half symmetrically. Not interchangeable with :func:`clamped_silu_mul_eager`: it saturates at
    ``limit`` rather than ``silu(limit)``, and its gate gradient vanishes elsewhere."""
    return torch.nn.functional.silu(gate).clamp(max=limit) * up.clamp(min=-limit, max=limit)


def _glu_launch(n_rows: int, width: int) -> tuple[tuple[int, int], dict]:
    """The grid and ``BLOCK_M`` (column) / ``ROWS`` (rows per program) tile of a launch over
    ``[n_rows, width]``, keyed on the width alone so the EP layer's small-token warmup compiles the
    binary a full dispatch runs. Tuned on the Gemma 4 and Qwen/GLM expert widths."""
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


# ``n_rows`` changes every step; specialized, each class of it (1, a multiple of 16, neither) would compile
# its own binary, an unseen one between an EP dispatch and its combine.
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


def _kernel_args(combine: GluCombine) -> dict:
    return {
        "alpha": combine.alpha,
        "limit": combine.limit,
        "ACTIVATION": combine.activation,
        "CLAMP_ACTIVATED": combine.clamp_activated,
        "UP_PLUS_ONE": combine.up_plus_one,
    }


def _glu_forward(gate: torch.Tensor, up: torch.Tensor, combine: GluCombine) -> torch.Tensor:
    n_rows, width = gate.shape
    out = torch.empty((n_rows, width), device=gate.device, dtype=gate.dtype)
    grid, tile = _glu_launch(n_rows, width)
    _glu_fwd_kernel[grid](gate, up, out, n_rows, width, gate.stride(0), up.stride(0), **_kernel_args(combine), **tile)
    return out


def _glu_backward(
    gate: torch.Tensor,
    up: torch.Tensor,
    grad_out: torch.Tensor,
    dgate: torch.Tensor,
    dup: torch.Tensor,
    combine: GluCombine,
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
        **_kernel_args(combine),
        **tile,
    )


class _FusedGLU(torch.autograd.Function):
    """``combine(gate, up)`` over separate (possibly row-strided) halves."""

    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor, combine: GluCombine) -> torch.Tensor:
        shape = gate.shape
        gate2d, up2d = _as_rows(gate), _as_rows(up)
        ctx.save_for_backward(gate2d, up2d)
        ctx.combine, ctx.shape = combine, shape
        return _glu_forward(gate2d, up2d, combine).view(shape)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        gate, up = ctx.saved_tensors
        dgate, dup = (
            torch.empty_like(gate, memory_format=torch.contiguous_format),
            torch.empty_like(up, memory_format=torch.contiguous_format),
        )
        _glu_backward(gate, up, grad_out, dgate, dup, ctx.combine)
        return dgate.view(ctx.shape), dup.view(ctx.shape), None


class _FusedPackedGLU(torch.autograd.Function):
    """``combine(gate, up)`` over a ``[..., 2M]`` ``[gate | up]`` projection output, read in place with the
    gradient written as one buffer: no ``.contiguous()`` copies of the halves, no ``cat`` behind ``chunk``."""

    @staticmethod
    def forward(ctx, gate_up: torch.Tensor, combine: GluCombine) -> torch.Tensor:
        shape = gate_up.shape
        rows2d = _as_rows(gate_up)
        width = shape[-1] // 2
        ctx.save_for_backward(rows2d)
        ctx.combine, ctx.shape = combine, shape
        out = _glu_forward(rows2d[:, :width], rows2d[:, width:], combine)
        return out.view(*shape[:-1], width)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (rows2d,) = ctx.saved_tensors
        width = rows2d.shape[-1] // 2
        grad = torch.empty(rows2d.shape, device=rows2d.device, dtype=rows2d.dtype)
        _glu_backward(rows2d[:, :width], rows2d[:, width:], grad_out, grad[:, :width], grad[:, width:], ctx.combine)
        return grad.view(ctx.shape), None


def fused_glu_enabled() -> bool:
    """Whether the fused kernels may run (``HALO_FUSED_GLU``); off, every combine runs its eager form."""
    return _FUSED_GLU_ENABLED


@dataclass(frozen=True)
class GluCombine:
    """One ``act(gate) * up`` combine: the fused kernel on CUDA, its eager reference elsewhere and under
    ``HALO_FUSED_GLU=0``. ``alpha`` and ``limit`` are runtime kernel arguments and the rest
    ``tl.constexpr``, so every layer with one combine shares one compilation."""

    activation: int
    alpha: float = 1.0
    limit: float = 0.0
    clamp_activated: bool = False
    up_plus_one: bool = False

    @classmethod
    def silu(cls) -> GluCombine:
        return cls(_SILU)

    @classmethod
    def gelu_tanh(cls) -> GluCombine:
        return cls(_GELU_TANH)

    @classmethod
    def clamped_silu(cls, limit: float, *, clamp_activated: bool = False) -> GluCombine:
        """SiLU-gated with ``up`` clamped to ``±limit`` and the gate clamped from above before the SiLU
        (DeepSeek-V4, GLM-5 Next) or, with ``clamp_activated``, the activated gate after it (Step-3.7)."""
        return cls(_CLAMPED_SILU, limit=float(limit), clamp_activated=clamp_activated)

    @classmethod
    def gptoss(cls, alpha: float, limit: float) -> GluCombine:
        """:func:`gptoss_glu_eager`: the clamped gate scaled by ``alpha`` inside the sigmoid, times the
        clamped ``up + 1``."""
        return cls(_CLAMPED_SILU, alpha=float(alpha), limit=float(limit), up_plus_one=True)

    @property
    def name(self) -> str:
        if self.activation != _CLAMPED_SILU:
            return "silu" if self.activation == _SILU else "gelu_tanh"
        if self.up_plus_one:
            return "gptoss"
        return "silu_then_clamp" if self.clamp_activated else "clamped_silu"

    def __call__(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if gate.is_cuda and fused_glu_enabled():
            return _FusedGLU.apply(gate, up, self)
        return self._eager(gate, up)

    def packed(self, gate_up: torch.Tensor) -> torch.Tensor:
        """The combine over a fused ``[..., 2M]`` projection output laid out ``[gate | up]``."""
        if gate_up.is_cuda and fused_glu_enabled():
            return _FusedPackedGLU.apply(gate_up, self)
        return self._eager(*gate_up.chunk(2, dim=-1))

    def _eager(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if self.activation == _SILU:
            return silu_mul_eager(gate, up)
        if self.activation == _GELU_TANH:
            return gelu_tanh_mul_eager(gate, up)
        if self.up_plus_one:
            return gptoss_glu_eager(gate, up, self.alpha, self.limit)
        if self.clamp_activated:
            return silu_then_clamp_mul_eager(gate, up, self.limit)
        return clamped_silu_mul_eager(gate, up, self.limit)


def resolve_fused_glu_mul(act_fn: object) -> GluCombine | None:
    """The fused combine computing ``act_fn(gate) * up``, or None for the eager ``act_fn`` combine. A
    family gets a fused combine by having one of these activations rather than by setting a flag."""
    if is_silu_activation(act_fn):
        return GluCombine.silu()
    if is_gelu_tanh_activation(act_fn):
        return GluCombine.gelu_tanh()
    return None


def resolve_clamped_glu_mul(
    act_fn: object, limit: float, *, clamp_activated: bool
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """The clamped-SwiGLU combine: the gate clamped to ``limit`` before ``act_fn`` (DeepSeek-V4, GLM-5 Next) or,
    under ``clamp_activated``, after it (Step-3.7), and the up half to ``±limit``. Fused for a SiLU gate; any
    other activation runs the same clamps eagerly."""
    if is_silu_activation(act_fn):
        return GluCombine.clamped_silu(limit, clamp_activated=clamp_activated)

    def clamped_glu_mul_eager(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        activated = act_fn(gate).clamp(max=limit) if clamp_activated else act_fn(gate.clamp(max=limit))
        return activated * up.clamp(min=-limit, max=limit)

    return clamped_glu_mul_eager
