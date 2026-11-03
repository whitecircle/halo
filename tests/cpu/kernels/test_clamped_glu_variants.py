#!/usr/bin/env python
"""Three clamped-SwiGLU combines over one Triton kernel pair, identified only by the `GluCombine` tuple, so a flip
is silent. The Triton body is `tests/gpu/kernels/test_fused_glu.py`'s."""

from collections.abc import Callable
from typing import NamedTuple

import pytest
import torch
import torch.nn.functional as F

from halo.kernels import fused_glu
from halo.kernels.fused_glu import (
    GluCombine,
    clamped_silu_mul_eager,
    gptoss_glu_eager,
    resolve_clamped_glu_mul,
    silu_then_clamp_mul_eager,
)
from tests.common.fake_cuda import CudaLike

ALPHA = 1.702  # GptOss's; the only variant whose sigmoid is scaled.

# 2.0 lands on the gate and up clamps; silu(1.0) on the post-activation clamp, where `F.silu` and `x * sigmoid(x)`
# agree to the bit (elsewhere up to 4e-15 apart, which would flip a mask).
LIMITS = (2.0, F.silu(torch.tensor(1.0, dtype=torch.float64)).item())


def _grid(limit: float) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.tensor(
        [-9.0, -limit - 1e-3, -limit, -1.0, -1e-4, 0.0, 1e-4, 1.0, limit, limit + 1e-3, 9.0],
        dtype=torch.float64,
    )
    return values.repeat_interleave(values.numel()), values.repeat(values.numel())


@pytest.mark.parametrize("limit", LIMITS)
def test_gptoss_at_alpha_one_is_the_clamp_then_silu_variant_plus_its_activated_gate(limit):
    """At alpha=1 the references differ by exactly the activated gate: one expression family, hence one kernel."""
    gate, up = _grid(limit)
    difference = gptoss_glu_eager(gate, up, 1.0, limit) - clamped_silu_mul_eager(gate, up, limit)
    torch.testing.assert_close(difference, F.silu(gate.clamp(max=limit)), rtol=0, atol=1e-13)


@pytest.mark.parametrize("limit", LIMITS)
def test_alpha_is_a_live_argument_of_the_gptoss_variant(limit):
    """A fold that hardcoded SiLU would land on alpha=1."""
    gate, up = _grid(limit)
    moved = (gptoss_glu_eager(gate, up, ALPHA, limit) - gptoss_glu_eager(gate, up, 1.0, limit)).abs()
    live = (gate.clamp(max=limit) != 0) & (up.clamp(min=-limit, max=limit) + 1 != 0)
    assert live.any()
    assert (moved[live] > 0).all(), "alpha left the GptOss reference unchanged somewhere it must not"


@pytest.mark.parametrize("limit", LIMITS)
def test_the_two_clamp_placements_agree_inside_the_bound_and_diverge_outside(limit):
    """silu(g) < g inside the bound, so the post-activation clamp never fires there; why the placement stays
    per-variant."""
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
    """Deliberately not written in terms of the eager references, which are the contract it is checked against."""
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
    combine: Callable[[float], GluCombine]  # The combine a family latches, at this bound.
    eager: Callable[..., torch.Tensor]
    numeric_args: Callable[[float], tuple]
    kernel_args: Callable[[float], tuple]


_VARIANTS = [
    pytest.param(
        _Variant(
            lambda lim: GluCombine.gptoss(ALPHA, lim),
            gptoss_glu_eager,
            lambda lim: (ALPHA, lim),
            lambda lim: (ALPHA, lim, False, True),
        ),
        id="gptoss",
    ),
    pytest.param(
        _Variant(
            GluCombine.clamped_silu, clamped_silu_mul_eager, lambda lim: (lim,), lambda lim: (1.0, lim, False, False)
        ),
        id="clamp_then_silu",
    ),
    pytest.param(
        _Variant(
            lambda lim: GluCombine.clamped_silu(lim, clamp_activated=True),
            silu_then_clamp_mul_eager,
            lambda lim: (lim,),
            lambda lim: (1.0, lim, True, False),
        ),
        id="silu_then_clamp",
    ),
]


def _expected_combine(kernel_args: tuple) -> GluCombine:
    return GluCombine(fused_glu._CLAMPED_SILU, *kernel_args)


def _combine_math(gate: torch.Tensor, up: torch.Tensor, combine: GluCombine) -> torch.Tensor:
    assert combine.activation == fused_glu._CLAMPED_SILU
    gate, up = gate.as_subclass(torch.Tensor), up.as_subclass(torch.Tensor)
    args = (combine.alpha, combine.limit, combine.clamp_activated, combine.up_plus_one)
    return _kernel_math(gate, up, *args, dout=torch.ones_like(gate))[0]


class _RecordedKernel:
    """One call proves both the plumbing and what it resolves to."""

    def __init__(self):
        self.calls: list[GluCombine] = []

    def apply(self, gate: torch.Tensor, up: torch.Tensor, combine) -> torch.Tensor:
        self.calls.append(combine)
        return _combine_math(gate, up, combine)


class _RecordedPackedKernel(_RecordedKernel):
    """The same record, over a ``[gate | up]`` input."""

    def apply(self, gate_up: torch.Tensor, combine) -> torch.Tensor:
        self.calls.append(combine)
        return _combine_math(*gate_up.chunk(2, dim=-1), combine)


@pytest.mark.parametrize("limit", LIMITS)
@pytest.mark.parametrize("variant", _VARIANTS)
def test_combine_hands_the_kernel_the_argument_tuple_of_its_variant(variant, limit, monkeypatch):
    """A combine that dropped its clamp still returns a plausible tensor; only this tuple says which kernel it
    launched."""
    recorder = _RecordedKernel()
    monkeypatch.setattr(fused_glu, "_FusedGLU", recorder)
    gate, up = _grid(limit)
    fused = variant.combine(limit)(gate.as_subclass(CudaLike), up.as_subclass(CudaLike))
    assert recorder.calls == [_expected_combine(variant.kernel_args(limit))]
    torch.testing.assert_close(fused, variant.eager(gate, up, *variant.numeric_args(limit)), rtol=0, atol=1e-13)


@pytest.mark.parametrize("limit", LIMITS)
@pytest.mark.parametrize("variant", _VARIANTS)
def test_packed_form_hands_the_kernel_the_same_combine_as_the_separate_halves(variant, limit, monkeypatch):
    """Another combine would change the activation only on layers storing a fused ``gate_up``."""
    separate, fused_rec = _RecordedKernel(), _RecordedPackedKernel()
    monkeypatch.setattr(fused_glu, "_FusedGLU", separate)
    monkeypatch.setattr(fused_glu, "_FusedPackedGLU", fused_rec)
    gate, up = _grid(limit)
    combine = variant.combine(limit)
    expected = combine(gate.as_subclass(CudaLike), up.as_subclass(CudaLike))
    got = combine.packed(torch.cat([gate, up], dim=-1).as_subclass(CudaLike))
    assert fused_rec.calls == separate.calls and len(separate.calls) == 1
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.parametrize("limit", LIMITS)
@pytest.mark.parametrize("variant", _VARIANTS)
def test_kernel_body_under_that_tuple_reproduces_the_variants_reference_and_subgradients(variant, limit):
    """torch passes the gradient through at the bound, so a mask open there is an O(1) error on the boundary rows."""
    gate, up = _grid(limit)
    dout = torch.linspace(-1.0, 1.0, gate.numel(), dtype=torch.float64)
    out, dgate, dup = _kernel_math(gate, up, *variant.kernel_args(limit), dout=dout)
    eager_gate, eager_up = gate.clone().requires_grad_(True), up.clone().requires_grad_(True)
    eager_out = variant.eager(eager_gate, eager_up, *variant.numeric_args(limit))
    (eager_out * dout).sum().backward()
    torch.testing.assert_close(out, eager_out.detach(), rtol=0, atol=1e-13)
    torch.testing.assert_close(dgate, eager_gate.grad, rtol=0, atol=1e-13)
    torch.testing.assert_close(dup, eager_up.grad, rtol=0, atol=1e-13)


@pytest.mark.parametrize("clamp_activated", [False, True], ids=["clamp-then-act", "act-then-clamp"])
@pytest.mark.parametrize("limit", LIMITS)
def test_the_family_resolver_fuses_a_silu_gate_and_keeps_the_clamps_on_any_other(clamp_activated, limit):
    gate, up = _grid(limit)
    fused = resolve_clamped_glu_mul(F.silu, limit, clamp_activated=clamp_activated)
    reference = silu_then_clamp_mul_eager if clamp_activated else clamped_silu_mul_eager
    assert fused == GluCombine.clamped_silu(limit, clamp_activated=clamp_activated)
    torch.testing.assert_close(fused(gate, up), reference(gate, up, limit), rtol=0, atol=0)

    eager = resolve_clamped_glu_mul(F.gelu, limit, clamp_activated=clamp_activated)
    activated = F.gelu(gate).clamp(max=limit) if clamp_activated else F.gelu(gate.clamp(max=limit))
    torch.testing.assert_close(eager(gate, up), activated * up.clamp(-limit, limit), rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
