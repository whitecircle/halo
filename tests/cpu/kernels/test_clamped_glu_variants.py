#!/usr/bin/env python
"""Three clamped-SwiGLU variants, one Triton kernel pair — this pins what still separates them.

`fused_gptoss_glu`, `fused_clamped_silu_mul` and `fused_silu_then_clamp_mul` (and the packed forms of the
last two) are thin wrappers over the shared `_FusedGLU` / `_FusedPackedGLU`, and a variant's whole identity
is the `_GluVariant` its wrapper hands that Function: `alpha` and the bound at runtime, the clamp placement
(`CLAMP_ACTIVATED`) and GptOss's `up + 1` (`UP_PLUS_ONE`) as `tl.constexpr`. Nothing downstream reads those back, so a flipped
`UP_PLUS_ONE` or a clamp dropped from one variant is a silent numerical change on every expert of one
family and invisible everywhere else. Two pins, both CPU-only (the Triton body itself is
`tests/gpu/kernels/test_fused_glu.py`):

* the three eager references — the contract the kernel reproduces — against each other on the identity
  the fold rests on, at the bound and on both sides of it;
* the variant each wrapper hands the kernel, observed at the seam, plus a torch transcription
  of the kernel body under that tuple reproducing the variant's own eager reference and its clamp
  subgradients.

Run: ``python tests/cpu/kernels/test_clamped_glu_variants.py`` (or ``pytest -m cpu``).
"""

from collections.abc import Callable
from typing import NamedTuple

import pytest
import torch
import torch.nn.functional as F

from src.kernels import fused_glu
from src.kernels.fused_glu import (
    clamped_silu_mul_eager,
    fused_clamped_silu_mul,
    fused_clamped_silu_mul_packed,
    fused_gptoss_glu,
    fused_silu_then_clamp_mul,
    fused_silu_then_clamp_mul_packed,
    gptoss_glu_eager,
    silu_then_clamp_mul_eager,
)

ALPHA = 1.702  # GptOss's; the only variant whose sigmoid is scaled

# Two bounds. 2.0 puts the grid's ±limit entries exactly on the gate and up clamps; silu(1.0) puts the
# grid's gate=1.0 exactly on the POST-activation clamp instead — the one boundary that is otherwise
# unreachable, and where `F.silu` and the kernel's `x * sigmoid(x)` agree to the last bit (they differ
# by up to 4e-15 elsewhere, which at a bound would flip a mask and test that ULP instead of the
# subgradient convention).
LIMITS = (2.0, F.silu(torch.tensor(1.0, dtype=torch.float64)).item())


def _grid(limit: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Every (gate, up) pair over saturating tails, both signs, zero, and each bound exactly."""
    values = torch.tensor(
        [-9.0, -limit - 1e-3, -limit, -1.0, -1e-4, 0.0, 1e-4, 1.0, limit, limit + 1e-3, 9.0],
        dtype=torch.float64,
    )
    return values.repeat_interleave(values.numel()), values.repeat(values.numel())


@pytest.mark.parametrize("limit", LIMITS)
def test_gptoss_at_alpha_one_is_the_clamp_then_silu_variant_plus_its_activated_gate(limit):
    """The identity the fold rests on: GptOss differs from the clamp-then-SiLU variant only by `alpha`
    and the `+ 1` on the clamped up half, so at alpha=1 the two references differ by exactly the
    activated gate — one expression family, hence one kernel."""
    gate, up = _grid(limit)
    difference = gptoss_glu_eager(gate, up, 1.0, limit) - clamped_silu_mul_eager(gate, up, limit)
    torch.testing.assert_close(difference, F.silu(gate.clamp(max=limit)), rtol=0, atol=1e-13)


@pytest.mark.parametrize("limit", LIMITS)
def test_alpha_is_a_live_argument_of_the_gptoss_variant(limit):
    """`alpha` scales the sigmoid's input, so a fold that hardcoded SiLU would land on alpha=1: the
    two must differ everywhere the clamped gate and the `up + 1` factor are both non-zero."""
    gate, up = _grid(limit)
    moved = (gptoss_glu_eager(gate, up, ALPHA, limit) - gptoss_glu_eager(gate, up, 1.0, limit)).abs()
    live = (gate.clamp(max=limit) != 0) & (up.clamp(min=-limit, max=limit) + 1 != 0)
    assert live.any()
    assert (moved[live] > 0).all(), "alpha left the GptOss reference unchanged somewhere it must not"


@pytest.mark.parametrize("limit", LIMITS)
def test_the_two_clamp_placements_agree_inside_the_bound_and_diverge_outside(limit):
    """Clamping the gate and clamping the ACTIVATED gate are the same function while the gate is inside
    the bound (silu(g) < g there, so the post-activation clamp never fires) and different once it is
    outside — which is why the clamp placement stays per-variant instead of one of them serving both."""
    gate, up = _grid(limit)
    pre = clamped_silu_mul_eager(gate, up, limit)
    post = silu_then_clamp_mul_eager(gate, up, limit)
    inside = gate <= limit
    torch.testing.assert_close(pre[inside], post[inside], rtol=0, atol=0)
    outside = (gate > limit) & (up.abs() > 1e-3)
    assert outside.any()
    assert (pre[outside] != post[outside]).all()


def _kernel_math(
    gate: torch.Tensor,
    up: torch.Tensor,
    alpha: float,
    limit: float,
    clamp_activated: bool,
    up_plus_one: bool,
    dout: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The clamped branch of `_glu_fwd_kernel` / `_glu_bwd_kernel` transcribed to torch, fp64: same constexpr
    split, same closed-form activation gradient, same clamp masks. Deliberately not written in terms of
    the eager references — those are the contract it is checked against."""
    gate, up, dout = gate.double(), up.double(), dout.double()
    up_g = up.clamp(min=-limit, max=limit)
    if up_plus_one:
        up_g = up_g + 1.0
    up_in = ((up >= -limit) & (up <= limit)).double()
    gate_a = gate if clamp_activated else gate.clamp(max=limit)
    s = torch.sigmoid(gate_a * alpha)
    glu = gate_a * s
    dglu = s + gate_a * alpha * s * (1.0 - s)
    if clamp_activated:
        activated = glu.clamp(max=limit)
        gate_in = (glu <= limit).double()
    else:
        activated = glu
        gate_in = (gate <= limit).double()
    return activated * up_g, dout * up_g * dglu * gate_in, dout * activated * up_in


class _Variant(NamedTuple):
    wrapper: Callable[..., torch.Tensor]
    eager: Callable[..., torch.Tensor]
    numeric_args: Callable[[float], tuple]  # what a caller passes after (gate, up), at this bound
    kernel_args: Callable[[float], tuple]  # (alpha, limit, CLAMP_ACTIVATED, UP_PLUS_ONE) it owes the kernel


_VARIANTS = [
    pytest.param(
        _Variant(fused_gptoss_glu, gptoss_glu_eager, lambda lim: (ALPHA, lim), lambda lim: (ALPHA, lim, False, True)),
        id="gptoss",
    ),
    pytest.param(
        _Variant(
            fused_clamped_silu_mul, clamped_silu_mul_eager, lambda lim: (lim,), lambda lim: (1.0, lim, False, False)
        ),
        id="clamp_then_silu",
    ),
    pytest.param(
        _Variant(
            fused_silu_then_clamp_mul,
            silu_then_clamp_mul_eager,
            lambda lim: (lim,),
            lambda lim: (1.0, lim, True, False),
        ),
        id="silu_then_clamp",
    ),
]


class _CudaLike(torch.Tensor):
    """The wrappers branch on ``is_cuda`` alone, so shadowing it routes a CPU tensor down the fused
    branch — the only way to observe the kernel arguments on a machine with no free GPU."""

    is_cuda = True


def _expected_variant(kernel_args: tuple) -> fused_glu._GluVariant:
    return fused_glu._GluVariant(fused_glu._CLAMPED_SILU, *kernel_args)


def _variant_math(gate: torch.Tensor, up: torch.Tensor, variant: fused_glu._GluVariant) -> torch.Tensor:
    assert variant.activation == fused_glu._CLAMPED_SILU
    gate, up = gate.as_subclass(torch.Tensor), up.as_subclass(torch.Tensor)
    args = (variant.alpha, variant.limit, variant.clamp_activated, variant.up_plus_one)
    return _kernel_math(gate, up, *args, dout=torch.ones_like(gate))[0]


class _RecordedKernel:
    """Stands in for ``_FusedGLU``: records the variant and computes the kernel body it selects, so one
    call proves both the plumbing and what it resolves to."""

    def __init__(self):
        self.calls: list[fused_glu._GluVariant] = []

    def apply(self, gate: torch.Tensor, up: torch.Tensor, variant) -> torch.Tensor:
        self.calls.append(variant)
        return _variant_math(gate, up, variant)


class _RecordedPackedKernel(_RecordedKernel):
    """Stands in for ``_FusedPackedGLU``: the same record, over a ``[gate | up]`` input."""

    def apply(self, gate_up: torch.Tensor, variant) -> torch.Tensor:
        self.calls.append(variant)
        return _variant_math(*gate_up.chunk(2, dim=-1), variant)


@pytest.mark.parametrize("limit", LIMITS)
@pytest.mark.parametrize("variant", _VARIANTS)
def test_wrapper_hands_the_kernel_the_argument_tuple_of_its_variant(variant, limit, monkeypatch):
    """The seam itself: a wrapper that dropped its clamp, flipped `UP_PLUS_ONE` or lost `alpha` still
    returns a plausible tensor on every eager path, and only this variant says which kernel it launched."""
    recorder = _RecordedKernel()
    monkeypatch.setattr(fused_glu, "_FusedGLU", recorder)
    gate, up = _grid(limit)
    fused = variant.wrapper(gate.as_subclass(_CudaLike), up.as_subclass(_CudaLike), *variant.numeric_args(limit))
    assert recorder.calls == [_expected_variant(variant.kernel_args(limit))]
    torch.testing.assert_close(fused, variant.eager(gate, up, *variant.numeric_args(limit)), rtol=0, atol=1e-13)


_PACKED = [
    pytest.param(fused_clamped_silu_mul_packed, fused_clamped_silu_mul, id="clamp_then_silu"),
    pytest.param(fused_silu_then_clamp_mul_packed, fused_silu_then_clamp_mul, id="silu_then_clamp"),
]


@pytest.mark.parametrize("limit", LIMITS)
@pytest.mark.parametrize(("packed", "unpacked"), _PACKED)
def test_packed_wrapper_hands_the_kernel_the_same_variant_as_its_unpacked_twin(packed, unpacked, limit, monkeypatch):
    """The packed form reads ``[gate | up]`` in place; a packed wrapper that launched another variant than
    its twin would change the activation only on layers storing a fused ``gate_up``."""
    separate, fused_rec = _RecordedKernel(), _RecordedPackedKernel()
    monkeypatch.setattr(fused_glu, "_FusedGLU", separate)
    monkeypatch.setattr(fused_glu, "_FusedPackedGLU", fused_rec)
    gate, up = _grid(limit)
    expected = unpacked(gate.as_subclass(_CudaLike), up.as_subclass(_CudaLike), limit)
    got = packed(torch.cat([gate, up], dim=-1).as_subclass(_CudaLike), limit)
    assert fused_rec.calls == separate.calls and len(separate.calls) == 1
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.parametrize("limit", LIMITS)
@pytest.mark.parametrize("variant", _VARIANTS)
def test_kernel_body_under_that_tuple_reproduces_the_variants_reference_and_subgradients(variant, limit):
    """Forward and both clamp subgradients on a grid sitting exactly on every bound: torch passes the
    gradient through at the bound itself, so a mask that is open there — or a tuple selecting another
    variant's clamp — is an O(1) error on the boundary rows."""
    gate, up = _grid(limit)
    dout = torch.linspace(-1.0, 1.0, gate.numel(), dtype=torch.float64)
    out, dgate, dup = _kernel_math(gate, up, *variant.kernel_args(limit), dout=dout)
    eager_gate, eager_up = gate.clone().requires_grad_(True), up.clone().requires_grad_(True)
    eager_out = variant.eager(eager_gate, eager_up, *variant.numeric_args(limit))
    (eager_out * dout).sum().backward()
    torch.testing.assert_close(out, eager_out.detach(), rtol=0, atol=1e-13)
    torch.testing.assert_close(dgate, eager_gate.grad, rtol=0, atol=1e-13)
    torch.testing.assert_close(dup, eager_up.grad, rtol=0, atol=1e-13)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
