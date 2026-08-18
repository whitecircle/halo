"""Fused GLU activations for grouped MoE expert compute.

Two kernel pairs: an unclamped one (SiLU or tanh-GELU, plus the multiply) and one serving the whole
bounded family — GptOss, DeepSeek-V4, GLM-5 Next, Step-3.7 — off a single compilation, with ``alpha``
and the bound as runtime arguments. Folding them would cost the unclamped path an ``alpha`` multiply.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
import triton
import triton.language as tl

try:
    from triton.language.extra.libdevice import tanh
except ImportError:  # Triton < 3.4 packages CUDA libdevice under a vendor namespace.
    from triton.language.extra.cuda.libdevice import tanh

logger = logging.getLogger(__name__)

_BLOCK = 1024
_SILU = 0
_GELU_TANH = 1

# tanh-GELU (`approximate="tanh"`): 0.5*x*(1 + tanh(sqrt(2/pi)*(x + k*x^3))). The values are a CONTRACT,
# not a tuning choice: the kernel is gated on reproducing torch's own tanh-GELU. ``tl.constexpr``
# instances, because a ``@triton.jit`` body may only read module globals that are constexpr.
_GELU_SQRT_2_OVER_PI = tl.constexpr(0.7978845608028654)
_GELU_TANH_COEFF = tl.constexpr(0.044715)

# Both saturating tails, both signs, zero and the near-zero region; sufficiency is pinned by the
# registry-wide sweep in ``tests/cpu/kernels/test_silu_activation_probe.py``, which requires every other
# ``ACT2FN`` entry to be rejected for each probed activation.
_ACTIVATION_PROBE_INPUT = (-64.0, -8.0, -1.0, -0.5, -1e-3, 0.0, 1e-3, 0.5, 1.0, 8.0, 64.0)


def _computes_exactly(act_fn: object, reference_fn: Callable[[torch.Tensor], torch.Tensor], name: str) -> bool:
    """Whether ``act_fn`` reproduces ``reference_fn`` bit-for-bit over the probe vector.

    Behavioural, never nominal: ``ACT2FN["silu"]`` is a bare ``nn.Module`` that is neither ``nn.SiLU`` nor
    ``F.silu``, so an ``isinstance`` test would disarm the kernels with no numerical trace. Fails closed —
    a non-callable, a raise, or any result not exactly equal in shape, dtype and value answers False, at
    the cost of the eager combine. Callers latch the verdict at construction, so an activation trainable
    away from the reference would keep the kernel armed; nothing on the roster ships one.
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

    Exact (erf) GELU is a different function and must answer False — the kernel would silently change
    the activation on every expert.
    """
    return _computes_exactly(act_fn, lambda x: torch.nn.functional.gelu(x, approximate="tanh"), "tanh-GELU")


def silu_mul_eager(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Reference SwiGLU combine used by the fused kernel."""
    return torch.nn.functional.silu(gate) * up


def gelu_tanh_mul_eager(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Reference tanh-GeGLU combine used by Gemma 4."""
    return torch.nn.functional.gelu(gate, approximate="tanh") * up


@triton.jit
def _standard_glu_fwd_kernel(gate_ptr, up_ptr, out_ptr, n, ACTIVATION: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    gate = tl.load(gate_ptr + offs, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + offs, mask=mask).to(tl.float32)

    if ACTIVATION == 0:
        activated = gate * tl.sigmoid(gate)
    else:
        inner = _GELU_SQRT_2_OVER_PI * (gate + _GELU_TANH_COEFF * gate * gate * gate)
        activated = 0.5 * gate * (1.0 + tanh(inner))

    tl.store(out_ptr + offs, activated * up, mask=mask)


@triton.jit
def _standard_glu_bwd_kernel(
    gate_ptr,
    up_ptr,
    dout_ptr,
    dgate_ptr,
    dup_ptr,
    n,
    ACTIVATION: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    gate = tl.load(gate_ptr + offs, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + offs, mask=mask).to(tl.float32)
    dout = tl.load(dout_ptr + offs, mask=mask).to(tl.float32)

    if ACTIVATION == 0:
        sigmoid = tl.sigmoid(gate)
        activated = gate * sigmoid
        activation_grad = sigmoid * (1.0 + gate * (1.0 - sigmoid))
    else:
        gate_sq = gate * gate
        inner = _GELU_SQRT_2_OVER_PI * (gate + _GELU_TANH_COEFF * gate * gate_sq)
        tanh_inner = tanh(inner)
        activated = 0.5 * gate * (1.0 + tanh_inner)
        inner_grad = _GELU_SQRT_2_OVER_PI * (1.0 + 3.0 * _GELU_TANH_COEFF * gate_sq)
        activation_grad = 0.5 * (1.0 + tanh_inner) + 0.5 * gate * (1.0 - tanh_inner * tanh_inner) * inner_grad

    tl.store(dgate_ptr + offs, dout * up * activation_grad, mask=mask)
    tl.store(dup_ptr + offs, dout * activated, mask=mask)


class _FusedStandardGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor, activation: int) -> torch.Tensor:
        if activation not in (_SILU, _GELU_TANH):
            raise ValueError(f"Unknown fused-GLU activation code {activation!r}")
        gate, up = gate.contiguous(), up.contiguous()
        out = torch.empty_like(gate)
        n = out.numel()
        _standard_glu_fwd_kernel[(triton.cdiv(n, _BLOCK),)](gate, up, out, n, ACTIVATION=activation, BLOCK=_BLOCK)
        ctx.save_for_backward(gate, up)
        ctx.activation = activation
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        gate, up = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        dgate, dup = torch.empty_like(gate), torch.empty_like(up)
        n = gate.numel()
        _standard_glu_bwd_kernel[(triton.cdiv(n, _BLOCK),)](
            gate,
            up,
            grad_out,
            dgate,
            dup,
            n,
            ACTIVATION=ctx.activation,
            BLOCK=_BLOCK,
        )
        return dgate, dup, None


def fused_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU combine, fused on CUDA and eager elsewhere."""
    if gate.is_cuda:
        return _FusedStandardGLU.apply(gate, up, _SILU)
    return silu_mul_eager(gate, up)


def fused_gelu_tanh_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Gemma-style tanh-GeGLU combine, fused on CUDA and eager elsewhere."""
    if gate.is_cuda:
        return _FusedStandardGLU.apply(gate, up, _GELU_TANH)
    return gelu_tanh_mul_eager(gate, up)


# Probe → the kernel that computes it. A family opts into a fused combine by *having* one of these
# activations, not by setting a flag, so every gate on the roster is served by the same two entries.
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
    """Step-3.7's clamped SwiGLU (``Step3p7Experts._apply_gate``): SiLU first, then clamp the ACTIVATED
    gate from above and the up half symmetrically. The post-activation mirror of
    :func:`clamped_silu_mul_eager`, not interchangeable with it — it saturates at exactly ``limit`` where
    the pre-activation clamp saturates at ``silu(limit)``, and the gate gradient dies elsewhere."""
    return torch.nn.functional.silu(gate).clamp(max=limit) * up.clamp(min=-limit, max=limit)


@triton.jit
def _clamped_glu_fwd_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    alpha,
    limit,
    n,
    CLAMP_ACTIVATED: tl.constexpr,
    UP_PLUS_ONE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # int64 program offset: grouped-path numel can exceed 2**31, where int32 `pid * BLOCK` wraps negative.
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    gate = tl.load(gate_ptr + offs, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + offs, mask=mask).to(tl.float32)
    up_g = tl.minimum(tl.maximum(up, -limit), limit)
    if UP_PLUS_ONE:
        up_g = up_g + 1.0
    gate_a = gate if CLAMP_ACTIVATED else tl.minimum(gate, limit)
    glu = gate_a * tl.sigmoid(gate_a * alpha)
    activated = tl.minimum(glu, limit) if CLAMP_ACTIVATED else glu
    tl.store(out_ptr + offs, activated * up_g, mask=mask)


@triton.jit
def _clamped_glu_bwd_kernel(
    gate_ptr,
    up_ptr,
    dout_ptr,
    dgate_ptr,
    dup_ptr,
    alpha,
    limit,
    n,
    CLAMP_ACTIVATED: tl.constexpr,
    UP_PLUS_ONE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    gate = tl.load(gate_ptr + offs, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + offs, mask=mask).to(tl.float32)
    dout = tl.load(dout_ptr + offs, mask=mask).to(tl.float32)
    up_g = tl.minimum(tl.maximum(up, -limit), limit)
    if UP_PLUS_ONE:
        up_g = up_g + 1.0
    # Clamp subgradients follow torch: the bound itself is inside the pass-through interval.
    up_in = tl.where((up >= -limit) & (up <= limit), 1.0, 0.0)
    gate_a = gate if CLAMP_ACTIVATED else tl.minimum(gate, limit)
    s = tl.sigmoid(gate_a * alpha)
    glu = gate_a * s
    dglu = s + gate_a * alpha * s * (1.0 - s)
    if CLAMP_ACTIVATED:
        activated = tl.minimum(glu, limit)
        gate_in = tl.where(glu <= limit, 1.0, 0.0)
    else:
        activated = glu
        gate_in = tl.where(gate <= limit, 1.0, 0.0)
    tl.store(dgate_ptr + offs, dout * up_g * dglu * gate_in, mask=mask)
    tl.store(dup_ptr + offs, dout * activated * up_in, mask=mask)


class _FusedClampedGLU(torch.autograd.Function):
    """One kernel pair for the whole clamped family: ``alpha`` and the bound are runtime arguments
    (so every layer and family in the process shares one compilation), while the clamp placement and
    GptOss's ``up + 1`` are ``tl.constexpr`` and cost nothing in the emitted code."""

    @staticmethod
    def forward(
        ctx,
        gate: torch.Tensor,
        up: torch.Tensor,
        alpha: float,
        limit: float,
        clamp_activated: bool,
        up_plus_one: bool,
    ) -> torch.Tensor:
        gate, up = gate.contiguous(), up.contiguous()
        out = torch.empty_like(gate)
        n = out.numel()
        _clamped_glu_fwd_kernel[(triton.cdiv(n, _BLOCK),)](
            gate,
            up,
            out,
            alpha,
            limit,
            n,
            CLAMP_ACTIVATED=clamp_activated,
            UP_PLUS_ONE=up_plus_one,
            BLOCK=_BLOCK,
        )
        ctx.save_for_backward(gate, up)
        ctx.alpha, ctx.limit = alpha, limit
        ctx.clamp_activated, ctx.up_plus_one = clamp_activated, up_plus_one
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        gate, up = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        dgate, dup = torch.empty_like(gate), torch.empty_like(up)
        n = gate.numel()
        _clamped_glu_bwd_kernel[(triton.cdiv(n, _BLOCK),)](
            gate,
            up,
            grad_out,
            dgate,
            dup,
            ctx.alpha,
            ctx.limit,
            n,
            CLAMP_ACTIVATED=ctx.clamp_activated,
            UP_PLUS_ONE=ctx.up_plus_one,
            BLOCK=_BLOCK,
        )
        return dgate, dup, None, None, None, None


def fused_gptoss_glu(gate: torch.Tensor, up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """:func:`gptoss_glu_eager` — clamp the gate, scale it by ``alpha`` inside the sigmoid, multiply by
    the clamped ``up + 1``. Triton-fused on CUDA, eager fallback otherwise."""
    if gate.is_cuda:
        return _FusedClampedGLU.apply(gate, up, float(alpha), float(limit), False, True)
    return gptoss_glu_eager(gate, up, alpha, limit)


def fused_clamped_silu_mul(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """:func:`clamped_silu_mul_eager` — clamp the gate, then SiLU (``alpha = 1``), times the clamped
    ``up``. Fused on CUDA and eager elsewhere."""
    if gate.is_cuda:
        return _FusedClampedGLU.apply(gate, up, 1.0, float(limit), False, False)
    return clamped_silu_mul_eager(gate, up, limit)


def fused_silu_then_clamp_mul(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """:func:`silu_then_clamp_mul_eager` — SiLU (``alpha = 1``) then clamp the ACTIVATED gate, times
    the clamped ``up``. Fused on CUDA and eager elsewhere."""
    if gate.is_cuda:
        return _FusedClampedGLU.apply(gate, up, 1.0, float(limit), True, False)
    return silu_then_clamp_mul_eager(gate, up, limit)
