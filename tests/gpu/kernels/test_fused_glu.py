#!/usr/bin/env python
"""Fused MoE GLU kernels match their eager references, forward and backward.

Exercises the standard SwiGLU and tanh-GeGLU pair used by Mistral 4 and Gemma 4, and the clamped
family — GptOss, DeepSeek-V4 / GLM-5 Next's clamp-then-SiLU, Step-3.7's SiLU-then-clamp — which shares
one kernel pair whose bound and ``alpha`` are runtime arguments and whose clamp placement and
``up + 1`` are ``tl.constexpr``. Token counts vary because grouped expert routing produces dynamic
shapes; the variants are run in one process because that is where a constexpr keyed to the wrong
wrapper, or a compilation reused across two of them, would show.
"""

from collections.abc import Callable
from typing import NamedTuple

import pytest
import torch

from src.kernels.fused_glu import (
    clamped_silu_mul_eager,
    fused_clamped_silu_mul,
    fused_gelu_tanh_mul,
    fused_gptoss_glu,
    fused_silu_mul,
    fused_silu_then_clamp_mul,
    gelu_tanh_mul_eager,
    gptoss_glu_eager,
    silu_mul_eager,
    silu_then_clamp_mul_eager,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DIM, ALPHA, LIMIT = 2880, 1.702, 7.0


def _fwd_bwd(fn, gate, up, *args):
    out = fn(gate, up, *args)
    out.float().pow(2).sum().backward()
    return out.detach(), gate.grad, up.grad


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp(min=1e-9)).item()


def _check_standard(fused_fn, eager_fn, n, dtype, tol):
    generator = torch.Generator(device="cuda").manual_seed(n)
    base_gate = torch.randn(n, DIM, generator=generator, device="cuda", dtype=dtype) * 3
    base_up = torch.randn(n, DIM, generator=generator, device="cuda", dtype=dtype) * 3
    fused_gate, fused_up = base_gate.clone().requires_grad_(True), base_up.clone().requires_grad_(True)
    eager_gate, eager_up = base_gate.clone().requires_grad_(True), base_up.clone().requires_grad_(True)
    fused_out, fused_dgate, fused_dup = _fwd_bwd(fused_fn, fused_gate, fused_up)
    eager_out, eager_dgate, eager_dup = _fwd_bwd(eager_fn, eager_gate, eager_up)
    rels = [
        _rel(fused_out, eager_out),
        _rel(fused_dgate, eager_dgate),
        _rel(fused_dup, eager_dup),
    ]
    assert max(rels) < tol, f"{fused_fn.__name__} n={n} rel fwd/dgate/dup={rels}"


@pytest.mark.parametrize(
    ("fused_fn", "eager_fn"),
    [
        (fused_silu_mul, silu_mul_eager),
        (fused_gelu_tanh_mul, gelu_tanh_mul_eager),
    ],
)
@pytest.mark.parametrize(("dtype", "tol"), [(torch.float32, 1e-4), (torch.bfloat16, 5e-2)])
def test_standard_glu_matches_eager(fused_fn, eager_fn, dtype, tol):
    for n in (1, 333, 4096):
        _check_standard(fused_fn, eager_fn, n, dtype, tol)


class _Variant(NamedTuple):
    """One clamped variant: its wrapper, its eager reference, the numeric args it takes before the
    bound, and the tolerances its output scale earns."""

    name: str
    fused: Callable[..., torch.Tensor]
    eager: Callable[..., torch.Tensor]
    extra: tuple[float, ...]
    rel_tol: float  # fp32, on random inputs
    bound_atol: float  # absolute, on the hand-built bound grid


# GptOss's `up + 1` and its alpha put its values ~8x the other two's, so it carries its own tolerances
# and its own dtype/size sweep (which is also where the int64 program offset is exercised).
GPTOSS = _Variant("gptoss", fused_gptoss_glu, gptoss_glu_eager, (ALPHA,), 1e-3, 1e-4)
CLAMP_THEN_SILU = _Variant("clamp_then_silu", fused_clamped_silu_mul, clamped_silu_mul_eager, (), 1e-5, 1e-5)
SILU_THEN_CLAMP = _Variant("silu_then_clamp", fused_silu_then_clamp_mul, silu_then_clamp_mul_eager, (), 1e-5, 1e-5)

_EVERY_VARIANT = [pytest.param(v, id=v.name) for v in (GPTOSS, CLAMP_THEN_SILU, SILU_THEN_CLAMP)]
_SILU_VARIANTS = [pytest.param(v, id=v.name) for v in (CLAMP_THEN_SILU, SILU_THEN_CLAMP)]


def _check_clamped(variant, n, dtype, limit, tol):
    generator = torch.Generator(device="cuda").manual_seed(n)
    # ×3 so the ±limit clamps are exercised on both gate and up
    base_gate = torch.randn(n, DIM, generator=generator, device="cuda", dtype=dtype) * 3
    base_up = torch.randn(n, DIM, generator=generator, device="cuda", dtype=dtype) * 3
    fused_gate, fused_up = base_gate.clone().requires_grad_(True), base_up.clone().requires_grad_(True)
    eager_gate, eager_up = base_gate.clone().requires_grad_(True), base_up.clone().requires_grad_(True)
    fused_out, fused_dgate, fused_dup = _fwd_bwd(variant.fused, fused_gate, fused_up, *variant.extra, limit)
    eager_out, eager_dgate, eager_dup = _fwd_bwd(variant.eager, eager_gate, eager_up, *variant.extra, limit)
    rels = [_rel(fused_out, eager_out), _rel(fused_dgate, eager_dgate), _rel(fused_dup, eager_dup)]
    print(f"  {variant.name} n={n:6d} [{dtype}] limit={limit} rel fwd/dgate/dup = {[f'{r:.1e}' for r in rels]}")
    assert max(rels) < tol


def test_gptoss_fp32_matches_eager():
    for n in (1, 333, 4096, 65536):  # varying token counts: shape-agnostic launch
        _check_clamped(GPTOSS, n, torch.float32, LIMIT, 1e-3)


def test_gptoss_bf16_matches_eager():
    for n in (333, 4096, 65537):
        _check_clamped(GPTOSS, n, torch.bfloat16, LIMIT, 5e-2)


@pytest.mark.parametrize("variant", _SILU_VARIANTS)
@pytest.mark.parametrize(("dtype", "tol"), [(torch.float32, 1e-5), (torch.bfloat16, 2e-2)])
def test_clamped_glu_matches_eager(variant, dtype, tol):
    for n in (1, 17, 4096):
        _check_clamped(variant, n, dtype, LIMIT, tol)


@pytest.mark.parametrize("variant", _EVERY_VARIANT)
def test_clamped_glu_honours_every_bound_in_one_process(variant):
    """Two families (or Step-3.7's two per-layer limits) share the kernel: after warming at one bound
    over several token counts, a different bound must be computed at that bound, not the first."""
    for n in (5, 7, 3, 4, 9, 10):
        _check_clamped(variant, n, torch.float32, 10.0, variant.rel_tol)
    _check_clamped(variant, 6, torch.float32, 0.5, variant.rel_tol)
    _check_clamped(variant, 6, torch.float32, 7.0, variant.rel_tol)


@pytest.mark.parametrize("variant", _EVERY_VARIANT)
def test_clamped_glu_subgradient_at_the_bound(variant):
    """Inputs sitting exactly on the clamp bound: torch's clamp passes the gradient through at the
    bound itself, so the kernel's pass-through interval must be closed on both ends (a masked-out
    gradient is O(1) off; the tolerances only absorb the sigmoid ulp)."""
    limit = 2.0
    values = torch.tensor([-limit, limit, 0.0, limit + 1e-3, -limit - 1e-3], device="cuda")
    gate = values.repeat(8, 1)
    up = values.flip(0).repeat(8, 1)
    if variant is SILU_THEN_CLAMP:
        # silu(x) == limit exactly: solve by bisection so the activated gate sits on its own bound.
        lo, hi = torch.tensor(0.0, device="cuda"), torch.tensor(10.0, device="cuda")
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if torch.nn.functional.silu(mid) < limit else (lo, mid)
        gate = torch.cat([gate, lo.expand(8, 1)], dim=1)
        up = torch.cat([up, torch.full((8, 1), 0.5, device="cuda")], dim=1)
    fused_gate, fused_up = gate.clone().requires_grad_(True), up.clone().requires_grad_(True)
    eager_gate, eager_up = gate.clone().requires_grad_(True), up.clone().requires_grad_(True)
    fused = _fwd_bwd(variant.fused, fused_gate, fused_up, *variant.extra, limit)
    eager = _fwd_bwd(variant.eager, eager_gate, eager_up, *variant.extra, limit)
    for got, want in zip(fused, eager, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=variant.bound_atol)


def test_up_plus_one_is_all_that_separates_the_two_pre_activation_variants():
    """GptOss and the clamp-then-SiLU variant reach one kernel pair and differ only in the
    ``UP_PLUS_ONE`` constexpr, so at ``alpha=1`` their outputs must differ by exactly the activated
    gate. Measured through the kernels, in one process: a constexpr keyed to the wrong wrapper — or a
    compilation reused across the two — is invisible to every per-variant check above. The tolerance
    absorbs the fp32 cancellation of two ~8x larger products (7e-6 in eager fp64-vs-fp32); either
    constexpr flipped is off by the activated gate itself, up to `limit`."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    gate = torch.randn(257, DIM, generator=generator, device="cuda") * 3
    up = torch.randn(257, DIM, generator=generator, device="cuda") * 3
    difference = fused_gptoss_glu(gate, up, 1.0, LIMIT) - fused_clamped_silu_mul(gate, up, LIMIT)
    torch.testing.assert_close(difference, torch.nn.functional.silu(gate.clamp(max=LIMIT)), rtol=0, atol=2e-4)


def test_large_numel_int64_offset():
    """gate.numel() past 2**31 must stay correct (int64 program offset).

    On the grouped expert path at long context / high ep (e.g. ep8 past the DeepEP token ceiling,
    where skewed routing piles >745k tokens onto a rank), ``[N, 2880]`` crosses 2**31 elements. An
    int32 ``pid * BLOCK`` offset wraps negative there and the kernel illegal-accesses. Skipped on GPUs
    too small to hold the >2**31-element tensors (the bug only manifests where the activation fits)."""
    _, total = torch.cuda.mem_get_info()
    if total / 2**30 < 110:
        pytest.skip(f"needs ~110 GiB to hold >2**31-element tensors (have {total / 2**30:.0f} GiB)")
    n = 746_000  # 746000 * 2880 = 2,148,480,000 > 2**31 = 2,147,483,648
    assert n * DIM > 2**31
    _check_clamped(GPTOSS, n, torch.bfloat16, LIMIT, 5e-2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
