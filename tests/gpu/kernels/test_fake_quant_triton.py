#!/usr/bin/env python
"""Fused Triton MX fake-quant fast paths (mxfp8, mxfp4), verifying that
- each is bit-identical to the eager quantizer (``_block_round_trip``) on adversarial inputs: exact
  powers of two, the format-max scale boundary, zeros, -0.0, subnormals, overflow, NaN, inf, rounding
  midpoints, an fp32 one-ulp scale edge, a near-subnormal scale, and for mxfp4 every e2m1 level and
  midpoint of both signs at forty scales,
- ``fake_quant`` routes CUDA / contiguous / last-axis / MX through them and everything else to eager,
  and mxfp8 to eager on GPUs older than SM 8.9,
- a new token count never compiles a new kernel variant, which under EP would stall the other ranks,
- the straight-through backward is unchanged.

Run: torchrun --nproc_per_node=1 tests/gpu/kernels/test_fake_quant_triton.py
"""

import torch
import triton

import halo.kernels.lowp.quantization as q
from halo.kernels.lowp.fake_quant_triton import (
    _launch,
    _mxfp4_fake_quant_kernel,
    _mxfp8_fake_quant_kernel,
    fake_quant_mxfp4_triton,
    fake_quant_mxfp8_triton,
)
from halo.kernels.lowp.quantization import (
    _AMAX_FLOOR,
    E2M1_MAX,
    E4M3_MAX,
    FORMAT_BLOCK_SIZE,
    _block_round_trip,
    fake_quant,
)
from tests.common.harness import gpu_test_main, record_check

DEV = "cuda"
K = 2880
# Format → (fused kernel, the element maximum its scale targets).
FUSED = {"mxfp8": (fake_quant_mxfp8_triton, E4M3_MAX), "mxfp4": (fake_quant_mxfp4_triton, E2M1_MAX)}


def _e2m1_grid() -> torch.Tensor:
    """Rows of 32-element blocks, each led by the format max so the block scale is exactly ``2^k``, then every
    multiple of 1/8 in [-6, 6]: all e2m1 levels, all midpoints (the tie cases) and the values between them."""
    grid = torch.arange(-48, 49, device=DEV, dtype=torch.float32) / 8
    body = grid.repeat(2)[: 4 * 31].reshape(4, 31)
    lead = torch.full((4, 1), E2M1_MAX, device=DEV)
    row = torch.cat([lead, body], dim=1).reshape(-1)
    scales = 2.0 ** torch.arange(-20, 20, device=DEV, dtype=torch.float32)
    return (row[None, :] * scales[:, None]).to(torch.bfloat16)


def _cases(fmt: str):
    format_max = FUSED[fmt][1]
    g = torch.Generator(device=DEV).manual_seed(0)
    bf = torch.bfloat16
    for shape in [(4096, K), (16384, K), (32, K, K), (7, 32), (1, 64)]:
        yield f"rand{shape}", torch.randn(shape, device=DEV, generator=g, dtype=bf)
    yield "pow2", (2.0 ** torch.randint(-20, 20, (256, K), device=DEV)).to(bf)
    yield "amax_edge", torch.full((64, K), format_max, device=DEV, dtype=bf)
    yield "zeros", torch.zeros(64, K, device=DEV, dtype=bf)
    yield "negzero", torch.full((64, K), -0.0, device=DEV, dtype=bf)
    yield "tiny", torch.full((64, K), 1e-40, device=DEV, dtype=bf)
    # A format max leads every block, so the scale is 1 and each -1e-40 stays a negative subnormal: its sign must
    # survive the scaling as -0.0, which a flush to zero would lose.
    t = torch.full((64, K), -1e-40, device=DEV)
    t[:, ::32] = format_max
    yield "max_and_neg_subnormal", t.to(bf)
    yield "huge", torch.full((64, K), 3e38, device=DEV, dtype=bf)
    t = torch.randn(64, K, device=DEV, dtype=bf)
    t[3, 5] = float("nan")
    yield "nan", t
    t = torch.randn(64, K, device=DEV, dtype=bf)
    t[0, 0] = float("inf")
    t[1, 0] = float("-inf")
    yield "inf", t
    yield "midpoints", torch.randn(64, K, device=DEV, dtype=bf) * 1.0625
    t = torch.full((64, K), format_max * 2.0**5, device=DEV)
    yield "fp32_ulp_edge", torch.nextafter(t, torch.full_like(t, 1e30))
    yield "near_denorm", torch.full((64, K), 3e-36, device=DEV, dtype=bf)
    yield "empty", torch.empty(0, K, device=DEV, dtype=bf)
    yield "fp16", torch.randn(64, K, device=DEV, dtype=torch.float16)
    yield "fp32", torch.randn(64, K, device=DEV, dtype=torch.float32)
    if fmt == "mxfp4":
        yield "e2m1_grid", _e2m1_grid()


def _assert_bits_equal(out, ref, name):
    assert out.dtype == ref.dtype and out.shape == ref.shape, name
    torch.testing.assert_close(out, ref, rtol=0, atol=0, equal_nan=True, msg=name)
    # assert_close treats -0.0 == 0.0, so compare sign bits outside NaN positions as well. Masks only, as boolean
    # indexing would build int64 indices several GiB large for the 32x2880x2880 case.
    flipped = (torch.signbit(out) != torch.signbit(ref)) & ~torch.isnan(ref)
    assert not flipped.any(), f"{name}: sign bit differs"


def _kernel_supported() -> bool:
    """Read from the device, not from ``_fp8_triton_supported``, so a wrong guard on SM 8.9+ fails the tests."""
    return torch.cuda.get_device_capability() >= (8, 9)


class _Spy:
    """Counts fast-path calls per format by swapping the kernels in ``_FUSED_FAKE_QUANT``, which
    ``_FakeQuant`` reads at call time."""

    def __init__(self):
        self.calls = {fmt: 0 for fmt in q._FUSED_FAKE_QUANT}
        self._orig = dict(q._FUSED_FAKE_QUANT)

    def __enter__(self):
        for fmt, (kernel, format_max) in self._orig.items():

            def spy(t, *a, _fmt=fmt, _kernel=kernel, **k):
                self.calls[_fmt] += 1
                return _kernel(t, *a, **k)

            q._FUSED_FAKE_QUANT[fmt] = (spy, format_max)
        return self

    def __exit__(self, *_):
        q._FUSED_FAKE_QUANT.update(self._orig)


def test_bit_exact_against_eager():
    """Each kernel reproduces the eager quantizer bit-for-bit, NaN blocks and signed zeros included."""
    for fmt, (kernel, format_max) in FUSED.items():
        if fmt == "mxfp8" and not _kernel_supported():
            print("  mxfp8: below SM 8.9, the fp8 kernel cannot compile — skipped")
            continue
        n = 0
        for name, t in _cases(fmt):
            ref = _block_round_trip(t, fmt, -1).to(t.dtype)
            _assert_bits_equal(kernel(t, FORMAT_BLOCK_SIZE[fmt], format_max, _AMAX_FLOOR), ref, f"{fmt}/{name}")
            n += 1
        print(f"  {fmt}: {n} cases bit-exact — OK")


def test_routing():
    """fake_quant takes the fast path only for CUDA / contiguous / real / last-axis / MX / K % 32 == 0."""
    t = torch.randn(64, K, device=DEV, dtype=torch.bfloat16)
    expected = {"mxfp8": int(_kernel_supported()), "mxfp4": 1}  # below SM 8.9 mxfp8 runs eager
    with _Spy() as s:
        for fmt in FUSED:
            _assert_bits_equal(fake_quant(t, fmt), _block_round_trip(t, fmt, -1), f"{fmt} wired")
        assert s.calls == expected, f"fast-path calls {s.calls}, expected {expected}"
        for fmt in FUSED:
            fake_quant(t, fmt, axis=0)  # non-last axis → eager
            fake_quant(t.t().contiguous().t(), fmt)  # non-contiguous → eager
            fake_quant(t.cpu(), fmt)  # CPU → eager
            c = t.to(torch.complex64)  # complex → eager, which keeps the real part
            assert torch.equal(fake_quant(c, fmt), _block_round_trip(c, fmt, -1).to(c.dtype)), f"{fmt}: complex"
        fake_quant(t, "nvfp4")  # tensor-wide scale → eager
        assert s.calls == expected, f"fallback cases took the fast path: {s.calls}"
    for fmt in FUSED:
        try:
            fake_quant(torch.randn(8, 33, device=DEV, dtype=torch.bfloat16), fmt)
            raise AssertionError(f"{fmt}: non-divisible K did not raise")
        except ValueError:
            pass
    print("  routing and fallbacks — OK")


def test_pre_sm89_falls_back():
    """A GPU below SM 8.9 cannot compile the fp8 kernel, so mxfp8 runs eager there. mxfp4 uses no fp8 type
    and keeps its kernel."""
    t = torch.randn(64, K, device=DEV, dtype=torch.bfloat16)
    orig = torch.cuda.get_device_capability
    q._fp8_triton_supported.cache_clear()
    torch.cuda.get_device_capability = lambda *_: (8, 0)
    try:
        with _Spy() as s:
            _assert_bits_equal(fake_quant(t, "mxfp8"), _block_round_trip(t, "mxfp8", -1), "sm80")
            assert s.calls["mxfp8"] == 0, "SM 8.0 took the fp8 Triton path"
            _assert_bits_equal(fake_quant(t, "mxfp4"), _block_round_trip(t, "mxfp4", -1), "sm80 mxfp4")
            assert s.calls["mxfp4"] == 1, "SM 8.0 dropped the mxfp4 kernel"
    finally:
        torch.cuda.get_device_capability = orig
        q._fp8_triton_supported.cache_clear()
    print("  pre-SM 8.9 eager fallback — OK")


def test_no_recompile_across_token_counts():
    """Under EP the token count changes every step. ``n_blocks`` is not specialized, so a new count never compiles
    a new variant mid-run, where one rank compiling would stall the others at the next collective. Each check runs
    on a fresh copy of the kernel with the same ``do_not_specialize``, so variants that earlier tests compiled
    cannot hide a new one."""
    kernels = {"mxfp8": _mxfp8_fake_quant_kernel, "mxfp4": _mxfp4_fake_quant_kernel}
    for fmt, kernel in kernels.items():
        if fmt == "mxfp8" and not _kernel_supported():
            print("  mxfp8: below SM 8.9, the fp8 kernel cannot compile — skipped")
            continue
        fresh = triton.jit(kernel.fn, do_not_specialize=kernel.do_not_specialize)
        compiled = []

        def hook(*, fn, _fresh=fresh, _compiled=compiled, **_):
            if fn.jit_function is _fresh:
                _compiled.append(fn.name)
            return False  # a truthy return would skip the compile

        prev = triton.knobs.runtime.jit_cache_hook
        triton.knobs.runtime.jit_cache_hook = hook
        try:
            for shape in [(64, K), (7, K), (1, 32)]:  # n_blocks 5760 (divisible by 16), 630, 1
                t = torch.randn(*shape, device=DEV, dtype=torch.bfloat16)
                _launch(fresh, t, FORMAT_BLOCK_SIZE[fmt], FUSED[fmt][1], _AMAX_FLOOR)
        finally:
            triton.knobs.runtime.jit_cache_hook = prev
        assert len(compiled) == 1, f"{fmt}: 3 token counts compiled {len(compiled)} kernel variants, expected 1"
        print(f"  {fmt}: no recompile across token counts — OK")


def test_ste_backward():
    """Straight-through: the gradient passes unchanged through each fast path."""
    for fmt in FUSED:
        t = torch.randn(64, K, device=DEV, dtype=torch.bfloat16, requires_grad=True)
        g = torch.randn_like(t)
        fake_quant(t, fmt).backward(g)
        assert torch.equal(t.grad, g), f"{fmt}: backward is not identity"
    print("  STE backward — OK")


@gpu_test_main(exact_world_size=1, prefix="fake_quant_triton", partial_state=False)
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    record_check(checks, "bit_exact_against_eager", test_bit_exact_against_eager)
    record_check(checks, "routing", test_routing)
    record_check(checks, "pre_sm89_falls_back", test_pre_sm89_falls_back)
    record_check(checks, "no_recompile_across_token_counts", test_no_recompile_across_token_counts)
    record_check(checks, "ste_backward", test_ste_backward)
    return {"checks": checks}


if __name__ == "__main__":
    run()
