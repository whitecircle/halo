#!/usr/bin/env python
"""Fused MoE GLU kernels match their eager references, forward and backward.

Exercises every combine of the one row-strided kernel pair: the standard SwiGLU and tanh-GeGLU used by
Mistral 4 and Gemma 4, and the clamped family — GptOss, DeepSeek-V4 / GLM-5 Next's clamp-then-SiLU,
Step-3.7's SiLU-then-clamp — whose bound and ``alpha`` are runtime arguments and whose clamp placement
and ``up + 1`` are ``tl.constexpr``. Token counts vary because grouped expert routing produces dynamic
shapes; the variants are run in one process because that is where a constexpr keyed to the wrong
wrapper, or a compilation reused across two of them, would show.

Run: torchrun --nproc_per_node=1 tests/gpu/kernels/test_fused_glu.py
"""

from collections.abc import Callable
from functools import partial
from typing import NamedTuple
from unittest.mock import patch

import torch

from src.kernels import fused_glu
from src.kernels.fused_glu import (
    PACKED_GLU_MULS,
    clamped_silu_mul_eager,
    fused_clamped_silu_mul,
    fused_clamped_silu_mul_packed,
    fused_gelu_tanh_mul,
    fused_gelu_tanh_mul_packed,
    fused_gptoss_glu,
    fused_silu_mul,
    fused_silu_mul_packed,
    fused_silu_then_clamp_mul,
    fused_silu_then_clamp_mul_packed,
    gelu_tanh_mul_eager,
    gptoss_glu_eager,
    silu_mul_eager,
    silu_then_clamp_mul_eager,
)
from tests.common.harness import gpu_test_main, record_check
from tests.common.utils import log, max_abs_rel_err

DIM, ALPHA, LIMIT = 2880, 1.702, 7.0
# Device memory the int64-offset case needs to hold its >2**31-element tensors.
LARGE_NUMEL_MIN_GIB = 110
STANDARD_PAIRS = ((fused_silu_mul, silu_mul_eager), (fused_gelu_tanh_mul, gelu_tanh_mul_eager))
STANDARD_DTYPE_TOLS = ((torch.float32, 1e-4), (torch.bfloat16, 5e-2))


def _fwd_bwd(fn, gate, up, *args):
    out = fn(gate, up, *args)
    out.float().pow(2).sum().backward()
    return out.detach(), gate.grad, up.grad


def _check_standard(fused_fn, eager_fn, n, dtype, tol):
    generator = torch.Generator(device="cuda").manual_seed(n)
    base_gate = torch.randn(n, DIM, generator=generator, device="cuda", dtype=dtype) * 3
    base_up = torch.randn(n, DIM, generator=generator, device="cuda", dtype=dtype) * 3
    fused_gate, fused_up = base_gate.clone().requires_grad_(True), base_up.clone().requires_grad_(True)
    eager_gate, eager_up = base_gate.clone().requires_grad_(True), base_up.clone().requires_grad_(True)
    fused_out, fused_dgate, fused_dup = _fwd_bwd(fused_fn, fused_gate, fused_up)
    eager_out, eager_dgate, eager_dup = _fwd_bwd(eager_fn, eager_gate, eager_up)
    rels = [
        max_abs_rel_err(fused_out, eager_out),
        max_abs_rel_err(fused_dgate, eager_dgate),
        max_abs_rel_err(fused_dup, eager_dup),
    ]
    assert all(rel < tol for rel in rels), f"{fused_fn.__name__} n={n} rel fwd/dgate/dup={rels}"


def test_standard_glu_matches_eager(fused_fn, eager_fn, dtype, tol):
    for n in (1, 333, 4096):
        _check_standard(fused_fn, eager_fn, n, dtype, tol)


# Packed entry, separate-halves entry and eager reference of each standard combine.
PACKED_STANDARD = (
    (fused_silu_mul_packed, fused_silu_mul, silu_mul_eager),
    (fused_gelu_tanh_mul_packed, fused_gelu_tanh_mul, gelu_tanh_mul_eager),
)
PACKED_DTYPE_TOLS = ((torch.float32, 1e-5), (torch.bfloat16, 2e-2))
PACKED_WIDTHS = (704, 2112, 5)
# Packed and separate-halves forms of the clamped combines.
PACKED_CLAMPED = (
    ("clamp_then_silu", fused_clamped_silu_mul_packed, fused_clamped_silu_mul),
    ("silu_then_clamp", fused_silu_then_clamp_mul_packed, fused_silu_then_clamp_mul),
)
PACKED_CLAMPED_WIDTHS = (1536, 5)


def test_fused_gate_up_layouts_match_eager(packed_fn, chunked_fn, eager_fn, dtype, tol, width):
    """The expert path hands the kernel the two halves of one ``[N, 2M]`` projection. The packed entry and
    the strided halves (no ``.contiguous()`` copy) must both read the right columns and write the right
    gradient columns: an off-by-``M`` stride swaps gate and up, which only a non-symmetric activation shows."""
    generator = torch.Generator(device="cuda").manual_seed(width)
    for n in (1, 333, 4099):
        base = torch.randn(n, 2 * width, generator=generator, device="cuda", dtype=dtype) * 3
        grad = torch.randn(n, width, generator=generator, device="cuda", dtype=dtype)
        reference = base.double().requires_grad_(True)
        expected = eager_fn(*reference.chunk(2, dim=-1))
        expected.backward(grad.double())
        for call in (packed_fn, lambda gu: chunked_fn(*gu.chunk(2, dim=-1))):
            gate_up = base.clone().requires_grad_(True)
            out = call(gate_up)
            out.backward(grad)
            assert out.shape == (n, width)
            assert max_abs_rel_err(out, expected) < tol
            assert max_abs_rel_err(gate_up.grad, reference.grad) < tol


def test_packed_clamped_glu_is_bit_identical_to_the_separate_halves(packed_fn, separate_fn, dtype, width):
    """The packed clamped forms run the same kernel over ``[gate | up]`` read in place, so forward and
    gradient must equal the separate-halves call bit for bit; a stride off by ``M`` swaps gate and up,
    which the asymmetric clamps show. Bound 2.0 against inputs of scale 3 fires both clamps on every row."""
    generator = torch.Generator(device="cuda").manual_seed(width)
    for n in (1, 333, 4099):
        base = torch.randn(n, 2 * width, generator=generator, device="cuda", dtype=dtype) * 3
        grad = torch.randn(n, width, generator=generator, device="cuda", dtype=dtype)
        packed_in = base.clone().requires_grad_(True)
        packed_out = packed_fn(packed_in, 2.0)
        packed_out.backward(grad)
        gate, up = (half.contiguous().requires_grad_(True) for half in base.chunk(2, dim=-1))
        separate_out = separate_fn(gate, up, 2.0)
        separate_out.backward(grad)
        assert torch.equal(packed_out, separate_out)
        assert torch.equal(packed_in.grad, torch.cat([gate.grad, up.grad], dim=-1))


def test_packed_glu_keeps_leading_dims():
    """A ``[B, S, 2M]`` input (the dense MLP path) returns ``[B, S, M]`` and a ``[B, S, 2M]`` gradient."""
    gate_up = torch.randn(2, 5, 2 * 64, device="cuda", requires_grad=True)
    out = fused_gelu_tanh_mul_packed(gate_up)
    out.sum().backward()
    assert out.shape == (2, 5, 64) and gate_up.grad.shape == gate_up.shape
    torch.testing.assert_close(out, gelu_tanh_mul_eager(*gate_up.detach().chunk(2, dim=-1)))


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

_EVERY_VARIANT = (GPTOSS, CLAMP_THEN_SILU, SILU_THEN_CLAMP)
_SILU_VARIANTS = (CLAMP_THEN_SILU, SILU_THEN_CLAMP)
CLAMPED_DTYPE_TOLS = ((torch.float32, 1e-5), (torch.bfloat16, 2e-2))


def _check_clamped(variant, n, dtype, limit, tol):
    generator = torch.Generator(device="cuda").manual_seed(n)
    # ×3 so the ±limit clamps are exercised on both gate and up
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
    for n in (1, 333, 4096, 65536):  # varying token counts: shape-agnostic launch
        _check_clamped(GPTOSS, n, torch.float32, LIMIT, 1e-3)


def test_gptoss_bf16_matches_eager():
    for n in (333, 4096, 65537):
        _check_clamped(GPTOSS, n, torch.bfloat16, LIMIT, 5e-2)


def test_clamped_glu_matches_eager(variant, dtype, tol):
    for n in (1, 17, 4096):
        _check_clamped(variant, n, dtype, LIMIT, tol)


def test_clamped_glu_honours_every_bound_in_one_process(variant):
    """Two families (or Step-3.7's two per-layer limits) share the kernel: after warming at one bound
    over several token counts, a different bound must be computed at that bound, not the first."""
    for n in (5, 7, 3, 4, 9, 10):
        _check_clamped(variant, n, torch.float32, 10.0, variant.rel_tol)
    _check_clamped(variant, 6, torch.float32, 0.5, variant.rel_tol)
    _check_clamped(variant, 6, torch.float32, 7.0, variant.rel_tol)


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


def test_the_switch_runs_every_combine_eagerly_on_cuda():
    """``HALO_FUSED_GLU=0`` is the documented escape from a GLU kernel that fails on a GPU: with it, no
    combine entry point (each separate-halves combine in ``PACKED_GLU_MULS`` and its packed form, plus
    ``fused_gptoss_glu``) may reach a Triton kernel, and each must return exactly its eager form. An
    entry point added there without an eager form here fails the coverage check."""
    gate, up = (torch.randn(8, 64, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    gate_up = torch.cat([gate, up], dim=-1)
    eager_forms = {
        fused_silu_mul: (silu_mul_eager, ()),
        fused_gelu_tanh_mul: (gelu_tanh_mul_eager, ()),
        fused_clamped_silu_mul: (clamped_silu_mul_eager, (LIMIT,)),
        fused_silu_then_clamp_mul: (silu_then_clamp_mul_eager, (LIMIT,)),
        fused_gptoss_glu: (gptoss_glu_eager, (ALPHA, LIMIT)),
    }
    assert set(PACKED_GLU_MULS) <= set(eager_forms), "a combine in PACKED_GLU_MULS has no eager form here"
    combines = []
    for separate, (eager, extra) in eager_forms.items():
        combines.append((partial(separate, gate, up, *extra), partial(eager, gate, up, *extra)))
        if separate in PACKED_GLU_MULS:
            combines.append((partial(PACKED_GLU_MULS[separate], gate_up, *extra), partial(eager, gate, up, *extra)))

    def no_kernel(*_args):
        raise AssertionError("a combine reached the fused kernel with HALO_FUSED_GLU=0")

    with (
        patch.object(fused_glu, "_FUSED_GLU_ENABLED", False),
        patch.object(fused_glu._FusedGLU, "apply", no_kernel),
        patch.object(fused_glu._FusedPackedGLU, "apply", no_kernel),
    ):
        for fused, eager in combines:
            assert torch.equal(fused(), eager())


def _compiled_binaries(kernel) -> int:
    return len(kernel.device_caches[torch.cuda.current_device()][0])


def test_one_compile_serves_every_row_count():
    """The EP dispatch's row count changes every step, and its class (1, a multiple of 16, neither) must
    not select a new binary: one warm call compiles the forward and backward kernels every later row
    count runs."""
    width = 704
    kernels = (fused_glu._glu_fwd_kernel, fused_glu._glu_bwd_kernel)

    def run(rows: int) -> None:
        gate = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        up = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        fused_silu_mul(gate, up).sum().backward()

    # Earlier checks in this process compiled these kernels at other row counts; start from an empty
    # cache so a row count that selects its own binary shows up as a new entry.
    for kernel in kernels:
        kernel.device_caches.clear()
    run(16)
    warm = [_compiled_binaries(kernel) for kernel in kernels]
    for rows in (1, 17, 33, 48, 1025):
        run(rows)
    assert [_compiled_binaries(kernel) for kernel in kernels] == warm, "a row count compiled a new binary"


def test_large_numel_int64_offset():
    """gate.numel() past 2**31 must stay correct (int64 row offsets).

    On the grouped expert path at long context / high ep (e.g. ep8 past the DeepEP token ceiling,
    where skewed routing piles >745k tokens onto a rank), ``[N, 2880]`` crosses 2**31 elements. The
    kernels address a row as ``row * stride`` in int64; in int32 that product wraps negative there and
    the kernel illegal-accesses. Not run on GPUs too small to hold the >2**31-element tensors (the bug
    only manifests where the activation fits)."""
    n = 746_000  # 746000 * 2880 = 2,148,480,000 > 2**31 = 2,147,483,648
    assert n * DIM > 2**31
    _check_clamped(GPTOSS, n, torch.bfloat16, LIMIT, 5e-2)


@gpu_test_main(exact_world_size=1, prefix="fused_glu", partial_state=False)
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    for fused_fn, eager_fn in STANDARD_PAIRS:
        for dtype, tol in STANDARD_DTYPE_TOLS:
            record_check(
                checks,
                f"standard_glu_matches_eager[{fused_fn.__name__}-{dtype}]",
                lambda: test_standard_glu_matches_eager(fused_fn, eager_fn, dtype, tol),
            )
    for packed_fn, chunked_fn, eager_fn in PACKED_STANDARD:
        for dtype, tol in PACKED_DTYPE_TOLS:
            for width in PACKED_WIDTHS:
                record_check(
                    checks,
                    f"fused_gate_up_layouts_match_eager[{packed_fn.__name__}-{dtype}-{width}]",
                    lambda: test_fused_gate_up_layouts_match_eager(packed_fn, chunked_fn, eager_fn, dtype, tol, width),
                )
    for name, packed_fn, separate_fn in PACKED_CLAMPED:
        for dtype in (torch.float32, torch.bfloat16):
            for width in PACKED_CLAMPED_WIDTHS:
                record_check(
                    checks,
                    f"packed_clamped_glu_is_bit_identical_to_the_separate_halves[{name}-{dtype}-{width}]",
                    lambda: test_packed_clamped_glu_is_bit_identical_to_the_separate_halves(
                        packed_fn, separate_fn, dtype, width
                    ),
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
