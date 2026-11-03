#!/usr/bin/env python
"""Fused MoE GLU kernels match their eager references, forward and backward.

The variants run in one process because that is where a constexpr keyed to the wrong combine, or a compilation reused
across two of them, shows.
"""

from collections.abc import Callable
from typing import NamedTuple
from unittest.mock import patch

import torch

from halo.kernels import fused_glu
from halo.kernels.fused_glu import (
    GluCombine,
    clamped_silu_mul_eager,
    gelu_tanh_mul_eager,
    gptoss_glu_eager,
    silu_mul_eager,
    silu_then_clamp_mul_eager,
)
from tests.common.harness import gpu_test_main, record_check
from tests.common.numerics import max_abs_rel_err
from tests.common.rank_log import log

DIM, ALPHA, LIMIT = 2880, 1.702, 7.0
# Device memory the int64-offset case needs to hold its >2**31-element tensors.
LARGE_NUMEL_MIN_GIB = 110
STANDARD_PAIRS = ((GluCombine.silu(), silu_mul_eager), (GluCombine.gelu_tanh(), gelu_tanh_mul_eager))
STANDARD_DTYPE_TOLS = ((torch.float32, 1e-4), (torch.bfloat16, 5e-2))


def _fwd_bwd(fn, gate, up, *args):
    out = fn(gate, up, *args)
    out.float().pow(2).sum().backward()
    return out.detach(), gate.grad, up.grad


def _check_standard(combine, eager_fn, n, dtype, tol):
    generator = torch.Generator(device="cuda").manual_seed(n)
    base_gate = torch.randn(n, DIM, generator=generator, device="cuda", dtype=dtype) * 3
    base_up = torch.randn(n, DIM, generator=generator, device="cuda", dtype=dtype) * 3
    fused_gate, fused_up = base_gate.clone().requires_grad_(True), base_up.clone().requires_grad_(True)
    eager_gate, eager_up = base_gate.clone().requires_grad_(True), base_up.clone().requires_grad_(True)
    fused_out, fused_dgate, fused_dup = _fwd_bwd(combine, fused_gate, fused_up)
    eager_out, eager_dgate, eager_dup = _fwd_bwd(eager_fn, eager_gate, eager_up)
    rels = [
        max_abs_rel_err(fused_out, eager_out),
        max_abs_rel_err(fused_dgate, eager_dgate),
        max_abs_rel_err(fused_dup, eager_dup),
    ]
    assert all(rel < tol for rel in rels), f"{combine.name} n={n} rel fwd/dgate/dup={rels}"


def test_standard_glu_matches_eager(combine, eager_fn, dtype, tol):
    for n in (1, 333, 4096):
        _check_standard(combine, eager_fn, n, dtype, tol)


PACKED_DTYPE_TOLS = ((torch.float32, 1e-5), (torch.bfloat16, 2e-2))
PACKED_WIDTHS = (704, 2112, 5)
# Bound 2.0 against inputs of scale 3 fires both clamps on every row.
PACKED_CLAMPED = (GluCombine.clamped_silu(2.0), GluCombine.clamped_silu(2.0, clamp_activated=True))
PACKED_CLAMPED_WIDTHS = (1536, 5)


def test_fused_gate_up_layouts_match_eager(combine, eager_fn, dtype, tol, width):
    """The packed form and the strided halves (no ``.contiguous()`` copy) must read and write the right columns: an
    off-by-``M`` stride swaps gate and up, which only a non-symmetric activation shows."""
    generator = torch.Generator(device="cuda").manual_seed(width)
    for n in (1, 333, 4099):
        base = torch.randn(n, 2 * width, generator=generator, device="cuda", dtype=dtype) * 3
        grad = torch.randn(n, width, generator=generator, device="cuda", dtype=dtype)
        reference = base.double().requires_grad_(True)
        expected = eager_fn(*reference.chunk(2, dim=-1))
        expected.backward(grad.double())
        for call in (combine.packed, lambda gu: combine(*gu.chunk(2, dim=-1))):
            gate_up = base.clone().requires_grad_(True)
            out = call(gate_up)
            out.backward(grad)
            assert out.shape == (n, width)
            assert max_abs_rel_err(out, expected) < tol
            assert max_abs_rel_err(gate_up.grad, reference.grad) < tol


def test_packed_clamped_glu_is_bit_identical_to_the_separate_halves(combine, dtype, width):
    """The packed clamped form runs the same kernel over ``[gate | up]`` read in place, so forward and gradient match
    bit for bit; a stride off by ``M`` swaps gate and up, which the asymmetric clamps show."""
    generator = torch.Generator(device="cuda").manual_seed(width)
    for n in (1, 333, 4099):
        base = torch.randn(n, 2 * width, generator=generator, device="cuda", dtype=dtype) * 3
        grad = torch.randn(n, width, generator=generator, device="cuda", dtype=dtype)
        packed_in = base.clone().requires_grad_(True)
        packed_out = combine.packed(packed_in)
        packed_out.backward(grad)
        gate, up = (half.contiguous().requires_grad_(True) for half in base.chunk(2, dim=-1))
        separate_out = combine(gate, up)
        separate_out.backward(grad)
        assert torch.equal(packed_out, separate_out)
        assert torch.equal(packed_in.grad, torch.cat([gate.grad, up.grad], dim=-1))


def test_packed_glu_keeps_leading_dims():
    gate_up = torch.randn(2, 5, 2 * 64, device="cuda", requires_grad=True)
    out = GluCombine.gelu_tanh().packed(gate_up)
    out.sum().backward()
    assert out.shape == (2, 5, 64) and gate_up.grad.shape == gate_up.shape
    torch.testing.assert_close(out, gelu_tanh_mul_eager(*gate_up.detach().chunk(2, dim=-1)))


class _Variant(NamedTuple):
    name: str
    fused: Callable[..., torch.Tensor]
    eager: Callable[..., torch.Tensor]
    extra: tuple[float, ...]
    rel_tol: float  # fp32, on random inputs.
    bound_atol: float  # Absolute, on the hand-built bound grid.


# GptOss's `up + 1` and its alpha put its values ~8x the other two's, so it carries its own tolerances and dtype/size
# sweep.
GPTOSS = _Variant(
    "gptoss",
    lambda gate, up, alpha, limit: GluCombine.gptoss(alpha, limit)(gate, up),
    gptoss_glu_eager,
    (ALPHA,),
    1e-3,
    1e-4,
)
CLAMP_THEN_SILU = _Variant(
    "clamp_then_silu",
    lambda gate, up, limit: GluCombine.clamped_silu(limit)(gate, up),
    clamped_silu_mul_eager,
    (),
    1e-5,
    1e-5,
)
SILU_THEN_CLAMP = _Variant(
    "silu_then_clamp",
    lambda gate, up, limit: GluCombine.clamped_silu(limit, clamp_activated=True)(gate, up),
    silu_then_clamp_mul_eager,
    (),
    1e-5,
    1e-5,
)

_EVERY_VARIANT = (GPTOSS, CLAMP_THEN_SILU, SILU_THEN_CLAMP)
_SILU_VARIANTS = (CLAMP_THEN_SILU, SILU_THEN_CLAMP)
CLAMPED_DTYPE_TOLS = ((torch.float32, 1e-5), (torch.bfloat16, 2e-2))


def _check_clamped(variant, n, dtype, limit, tol):
    generator = torch.Generator(device="cuda").manual_seed(n)
    # x3 so the limit clamps fire on both gate and up.
    base_gate = torch.randn(n, DIM, generator=generator, device="cuda", dtype=dtype) * 3
    base_up = torch.randn(n, DIM, generator=generator, device="cuda", dtype=dtype) * 3
    fused_gate, fused_up = base_gate.clone().requires_grad_(True), base_up.clone().requires_grad_(True)
    eager_gate, eager_up = base_gate.clone().requires_grad_(True), base_up.clone().requires_grad_(True)
    fused_out, fused_dgate, fused_dup = _fwd_bwd(variant.fused, fused_gate, fused_up, *variant.extra, limit)
    eager_out, eager_dgate, eager_dup = _fwd_bwd(variant.eager, eager_gate, eager_up, *variant.extra, limit)
    rels = [
        max_abs_rel_err(fused_out, eager_out),
        max_abs_rel_err(fused_dgate, eager_dgate),
        max_abs_rel_err(fused_dup, eager_dup),
    ]
    print(f"  {variant.name} n={n:6d} [{dtype}] limit={limit} rel fwd/dgate/dup = {[f'{r:.1e}' for r in rels]}")
    assert all(rel < tol for rel in rels), f"{variant.name} n={n} [{dtype}] limit={limit} rel fwd/dgate/dup={rels}"


def test_gptoss_fp32_matches_eager():
    for n in (1, 333, 4096, 65536):  # Varying token counts: shape-agnostic launch.
        _check_clamped(GPTOSS, n, torch.float32, LIMIT, 1e-3)


def test_gptoss_bf16_matches_eager():
    for n in (333, 4096, 65537):
        _check_clamped(GPTOSS, n, torch.bfloat16, LIMIT, 5e-2)


def test_clamped_glu_matches_eager(variant, dtype, tol):
    for n in (1, 17, 4096):
        _check_clamped(variant, n, dtype, LIMIT, tol)


def test_clamped_glu_honours_every_bound_in_one_process(variant):
    """Two families share the kernel: after warming at one bound, a different bound must be computed at that bound."""
    for n in (5, 7, 3, 4, 9, 10):
        _check_clamped(variant, n, torch.float32, 10.0, variant.rel_tol)
    _check_clamped(variant, 6, torch.float32, 0.5, variant.rel_tol)
    _check_clamped(variant, 6, torch.float32, 7.0, variant.rel_tol)


def test_clamped_glu_subgradient_at_the_bound(variant):
    """torch's clamp passes the gradient through at the bound, so the kernel's pass-through interval must be closed on
    both ends (a masked-out gradient is O(1) off; the tolerances only absorb the sigmoid ulp)."""
    limit = 2.0
    values = torch.tensor([-limit, limit, 0.0, limit + 1e-3, -limit - 1e-3], device="cuda")
    gate = values.repeat(8, 1)
    up = values.flip(0).repeat(8, 1)
    if variant is SILU_THEN_CLAMP:
        # silu(x) == limit exactly, by bisection, so the activated gate sits on its own bound.
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
    """The two variants differ only in the ``UP_PLUS_ONE`` constexpr, so at ``alpha=1`` their outputs differ by exactly
    the activated gate. The tolerance absorbs the fp32 cancellation of two ~8x larger products (7e-6 in eager
    fp64-vs-fp32); either constexpr flipped is off by the activated gate itself."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    gate = torch.randn(257, DIM, generator=generator, device="cuda") * 3
    up = torch.randn(257, DIM, generator=generator, device="cuda") * 3
    difference = GluCombine.gptoss(1.0, LIMIT)(gate, up) - GluCombine.clamped_silu(LIMIT)(gate, up)
    torch.testing.assert_close(difference, torch.nn.functional.silu(gate.clamp(max=LIMIT)), rtol=0, atol=2e-4)


def test_the_switch_runs_every_combine_eagerly_on_cuda():
    """``HALO_FUSED_GLU=0`` is the documented escape: no combine may reach a Triton kernel, and each returns its eager
    reference."""
    gate, up = (torch.randn(8, 64, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    gate_up = torch.cat([gate, up], dim=-1)
    eager_forms = {
        GluCombine.silu(): silu_mul_eager(gate, up),
        GluCombine.gelu_tanh(): gelu_tanh_mul_eager(gate, up),
        GluCombine.clamped_silu(LIMIT): clamped_silu_mul_eager(gate, up, LIMIT),
        GluCombine.clamped_silu(LIMIT, clamp_activated=True): silu_then_clamp_mul_eager(gate, up, LIMIT),
        GluCombine.gptoss(ALPHA, LIMIT): gptoss_glu_eager(gate, up, ALPHA, LIMIT),
    }
    assert {combine.name for combine in eager_forms} == {
        "silu",
        "gelu_tanh",
        "clamped_silu",
        "silu_then_clamp",
        "gptoss",
    }

    def no_kernel(*_args):
        raise AssertionError("a combine reached the fused kernel with HALO_FUSED_GLU=0")

    with (
        patch.object(fused_glu, "_FUSED_GLU_ENABLED", False),
        patch.object(fused_glu._FusedGLU, "apply", no_kernel),
        patch.object(fused_glu._FusedPackedGLU, "apply", no_kernel),
    ):
        for combine, eager in eager_forms.items():
            assert torch.equal(combine(gate, up), eager), combine.name
            assert torch.equal(combine.packed(gate_up), eager), combine.name


def _compiled_binaries(kernel) -> int:
    return len(kernel.device_caches[torch.cuda.current_device()][0])


def test_one_compile_serves_every_row_count():
    """The EP dispatch's row count changes every step, and its class (1, a multiple of 16, neither) must not select a
    new binary."""
    width = 704
    kernels = (fused_glu._glu_fwd_kernel, fused_glu._glu_bwd_kernel)

    def run(rows: int) -> None:
        gate = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        up = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        GluCombine.silu()(gate, up).sum().backward()

    # Start from an empty cache so a row count selecting its own binary shows as a new entry.
    for kernel in kernels:
        kernel.device_caches.clear()
    run(16)
    warm = [_compiled_binaries(kernel) for kernel in kernels]
    for rows in (1, 17, 33, 48, 1025):
        run(rows)
    assert [_compiled_binaries(kernel) for kernel in kernels] == warm, "a row count compiled a new binary"


def test_large_numel_int64_offset():
    """Rows address as ``row * stride`` in int64; in int32 that wraps negative past 2**31 elements (ep8 past the DeepEP
    token ceiling, where skewed routing piles >745k tokens onto a rank) and the kernel illegal-accesses."""
    n = 746_000  # 746000 * 2880 = 2,148,480,000 > 2**31 = 2,147,483,648.
    assert n * DIM > 2**31
    _check_clamped(GPTOSS, n, torch.bfloat16, LIMIT, 5e-2)


@gpu_test_main(exact_world_size=1, prefix="fused_glu", partial_state=False)
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    for combine, eager_fn in STANDARD_PAIRS:
        for dtype, tol in STANDARD_DTYPE_TOLS:
            record_check(
                checks,
                f"standard_glu_matches_eager[{combine.name}-{dtype}]",
                lambda: test_standard_glu_matches_eager(combine, eager_fn, dtype, tol),
            )
        for dtype, tol in PACKED_DTYPE_TOLS:
            for width in PACKED_WIDTHS:
                record_check(
                    checks,
                    f"fused_gate_up_layouts_match_eager[{combine.name}-{dtype}-{width}]",
                    lambda: test_fused_gate_up_layouts_match_eager(combine, eager_fn, dtype, tol, width),
                )
    for combine in PACKED_CLAMPED:
        for dtype in (torch.float32, torch.bfloat16):
            for width in PACKED_CLAMPED_WIDTHS:
                record_check(
                    checks,
                    f"packed_clamped_glu_is_bit_identical_to_the_separate_halves[{combine.name}-{dtype}-{width}]",
                    lambda: test_packed_clamped_glu_is_bit_identical_to_the_separate_halves(combine, dtype, width),
                )
    record_check(checks, "packed_glu_keeps_leading_dims", test_packed_glu_keeps_leading_dims)
    record_check(checks, "gptoss_fp32_matches_eager", test_gptoss_fp32_matches_eager)
    record_check(checks, "gptoss_bf16_matches_eager", test_gptoss_bf16_matches_eager)
    for variant in _SILU_VARIANTS:
        for dtype, tol in CLAMPED_DTYPE_TOLS:
            record_check(
                checks,
                f"clamped_glu_matches_eager[{variant.name}-{dtype}]",
                lambda: test_clamped_glu_matches_eager(variant, dtype, tol),
            )
    for variant in _EVERY_VARIANT:
        record_check(
            checks,
            f"clamped_glu_honours_every_bound_in_one_process[{variant.name}]",
            lambda: test_clamped_glu_honours_every_bound_in_one_process(variant),
        )
    for variant in _EVERY_VARIANT:
        record_check(
            checks,
            f"clamped_glu_subgradient_at_the_bound[{variant.name}]",
            lambda: test_clamped_glu_subgradient_at_the_bound(variant),
        )
    record_check(
        checks,
        "up_plus_one_is_all_that_separates_the_two_pre_activation_variants",
        test_up_plus_one_is_all_that_separates_the_two_pre_activation_variants,
    )
    record_check(checks, "one_compile_serves_every_row_count", test_one_compile_serves_every_row_count)
    record_check(
        checks, "the_switch_runs_every_combine_eagerly_on_cuda", test_the_switch_runs_every_combine_eagerly_on_cuda
    )
    total_gib = torch.cuda.mem_get_info()[1] / 2**30
    if total_gib >= LARGE_NUMEL_MIN_GIB:
        record_check(checks, "large_numel_int64_offset", test_large_numel_int64_offset)
    else:
        log(f"large_numel_int64_offset not run: needs ~{LARGE_NUMEL_MIN_GIB} GiB, have {total_gib:.0f} GiB")
    return {"checks": checks}


if __name__ == "__main__":
    run()
