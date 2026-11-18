#!/usr/bin/env python
"""Fused Triton fake-quant fast paths (mxfp8, mxfp4, nvfp4), verifying that
- nvfp4's global scale matches eager's on every positive fp32 tensor amax and NaN,
- each is bit-identical to the eager quantizer (``_block_round_trip``) on CUDA on adversarial inputs: exact
  powers of two, the format-max scale boundary, zeros, -0.0, subnormals, overflow, NaN, inf, rounding
  midpoints, an fp32 one-ulp scale edge, a near-subnormal scale, for mxfp4 every e2m1 level and midpoint of
  both signs at forty scales, and for nvfp4 the same grid over 16-wide blocks, the e4m3 block-scale edges,
  tensor maxima at and past a global-scale boundary, a lone maximum in the last block, fp32 maxima where
  the reciprocal multiply and true division disagree, and a maximum whose global scale is CUDA's inexact
  ``exp2(-127)``,
- ``fake_quant`` routes CUDA / contiguous / last-axis through them and everything else to eager, and to
  eager on GPUs too old for a kernel's types (mxfp8 and nvfp4 below SM 8.9, mxfp4 below SM 8.0),
- the cached nvfp4 expert-weight round trip on axis 1 goes through its kernel, bit-exact,
- nvfp4's two launches never sync with the host,
- a new token count never compiles a new kernel variant, which under EP would stall the other ranks,
- the straight-through backward is unchanged.

Run: torchrun --nproc_per_node=1 tests/gpu/kernels/test_fake_quant_triton.py
"""

import torch
import triton
import triton.language as tl

import halo.kernels.lowp.fake_quant_triton as fq
import halo.kernels.lowp.quantization as q
from halo.kernels.lowp.fake_quant_triton import fake_quant_triton
from halo.kernels.lowp.quantization import (
    _AMAX_FLOOR,
    _ELEMENT_MAX,
    E2M1_MAX,
    FORMAT_BLOCK_SIZE,
    _block_round_trip,
    _round_trip,
    cached_fake_quant,
    fake_quant,
)
from tests.common.harness import gpu_test_main, record_check

DEV = "cuda"
K = 2880
# The table ``fake_quant`` dispatches on: format → kernel and the oldest capability that compiles it.
FUSED = q._FUSED_FAKE_QUANT


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
    format_max = _ELEMENT_MAX[fmt]
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
    # An inf puts its block on exponent 127, where CUDA's exp2(-127) is one ulp below 2^-127 again. Finite values large
    # enough to land near a rounding boundary once scaled by 2^-127 show that (eager mxfp8 multiplied by exp2(-127) and
    # differed from its kernel on about 1% of such elements).
    t = torch.exp2(torch.rand(512, K, device=DEV, generator=g) * 19 + 108)
    t = torch.where(torch.rand(512, K, device=DEV, generator=g) < 0.5, -t, t)
    t[:, ::32] = float("inf")
    t[::2, ::64] = float("-inf")
    yield "inf_beside_large_values", t
    yield "inf_beside_large_values_bf16", t.to(bf)
    yield "midpoints", torch.randn(64, K, device=DEV, dtype=bf) * 1.0625
    t = torch.full((64, K), format_max * 2.0**5, device=DEV)
    yield "fp32_ulp_edge", torch.nextafter(t, torch.full_like(t, 1e30))
    yield "near_denorm", torch.full((64, K), 3e-36, device=DEV, dtype=bf)
    # Random blocks small enough that the MX exponent clamps to -127, where CUDA's exp2(-127) is one ulp below 2^-127.
    # A constant block hides a scale one ulp off; varied values do not (eager differed from the kernel on about 4% of
    # such elements for mxfp8 before #182, and 1.5% for mxfp4 before #185). mxfp4 reaches -127 only below about 3.5e-38.
    for amax in (2e-36, 3e-38):
        tiny = (torch.rand(512, K, device=DEV, generator=g) * 2 - 1) * amax
        yield f"tiny_random_{amax:g}", tiny
        yield f"tiny_random_{amax:g}_bf16", tiny.to(bf)
    # fp32 block maxima where amax / format max lands one ulp apart under true division and under torch's reciprocal
    # multiply, moving the exponent (mxfp8: 0x04600028, mxfp4: 0x20c00011), plus maxima whose quotient is subnormal.
    # The rest of each block steps down to 2^-20 of the max, into the element formats' subnormal range, where an
    # exponent one off changes the output.
    amax = torch.tensor([0x04600028, 0x20C00011], dtype=torch.int32).view(torch.float32).tolist()
    amax += [5e-38, 3.6e-38, 3e-36, 5e-36]
    lead = torch.tensor(amax, device=DEV)[:, None]
    steps = 2.0 ** (-torch.arange(1, 32, device=DEV) / 1.5) * (-1.0) ** torch.arange(1, 32, device=DEV)
    t = torch.cat([lead, lead * steps], dim=1)
    t[0, 1] = 1.2627380721723792e-40  # in the mxfp8 row, an element whose bf16 output moves with the exponent
    yield "e8m0_edges", t
    yield "empty", torch.empty(0, K, device=DEV, dtype=bf)
    yield "fp16", torch.randn(64, K, device=DEV, dtype=torch.float16)
    yield "fp32", torch.randn(64, K, device=DEV, dtype=torch.float32)
    if fmt == "mxfp4":
        yield "e2m1_grid", _e2m1_grid()
    if fmt == "nvfp4":
        yield from _nvfp4_cases()


def _nvfp4_cases():
    """NVFP4's own edges: the e2m1 grid over 16-wide blocks, the e4m3 block-scale edges, tensor maxima on and one ulp
    past a power-of-two boundary of 2688 (E4M3_MAX x E2M1_MAX), a lone maximum in the last block, and fp32 maxima
    where eager's reciprocal multiply and a true division give different global scales (the sweep over all positive
    fp32 in the PR finds them)."""
    bf = torch.bfloat16
    # 16-wide blocks led by 6 x 2^j, so each block scale is exactly 2^j for j in [-8, 8] (the tensor amax gives a
    # global scale of 1), then every multiple of 1/8 in [-6, 6] at that scale.
    grid = torch.arange(-48, 49, device=DEV, dtype=torch.float32) / 8
    body = grid.repeat(2)[: 7 * 15].reshape(7, 15)
    row = torch.cat([torch.full((7, 1), E2M1_MAX, device=DEV), body], dim=1).reshape(-1)
    scales = 2.0 ** torch.arange(-8, 9, device=DEV, dtype=torch.float32)
    yield "e2m1_grid16", (row[None, :] * scales[:, None]).to(bf)
    # Block maxima 6 x v under a global scale near 1: v on e4m3 subnormals, ties between neighbours, values that round
    # to 0, and the top of the range, where the scale is clamped at 448. fp32 input, so each v is exact.
    edges = [
        448.0,
        2.0**-9,
        3 * 2.0**-9,
        2.0**-10,
        1.5 * 2.0**-10,
        2.0**-11,
        1.0625,
        1.1875,
        248.0,
        440.0,
        447.0,
        448.0,
    ]
    lead = torch.tensor(edges, device=DEV)[:, None] * E2M1_MAX
    steps = 2.0 ** (-torch.arange(1, 16, device=DEV) / 1.5) * (-1.0) ** torch.arange(1, 16, device=DEV)
    yield "e4m3_scale_edges", torch.cat([lead, lead * steps], dim=1)
    for k in (-10, 0, 12):
        boundary = torch.tensor(2688.0 * 2.0**k, device=DEV)
        for name, top in (("on", boundary), ("ulp_past", torch.nextafter(boundary, torch.tensor(1e30, device=DEV)))):
            t = torch.randn(8, 64, device=DEV) * 2.0**k
            t[3, 17] = top
            yield f"global_{name}_2688x2^{k}", t
    t = torch.full((8, 64), 2688.0, device=DEV, dtype=bf)
    t[5, 5] = 2704.0  # one bf16 ulp past
    yield "global_ulp_past_bf16", t
    t = torch.randn(1024, K, device=DEV, dtype=bf) * 0.01
    t[-1, -1] = 50.0
    yield "max_in_last_block", t
    for bits in (0x00A80540, 0x0328002A, 0x05A8001E, 0x2528000F, 0x2928000F):
        amax = torch.tensor([bits], dtype=torch.int32).view(torch.float32).to(DEV)
        t = amax * (2.0 ** (-torch.arange(64, device=DEV, dtype=torch.float32) / 4)).reshape(4, 16)
        t[1::2] *= -1.0
        yield f"global_recip_edge_{bits:#010x}", t
    yield _EXP2_MINUS_127_CASE, _exp2_minus_127_case()


# A tensor amax in (2688 x 2^-128, 2688 x 2^-127], about (7.9e-36, 1.58e-35], puts the global scale on exp2(-127), the
# one exponent where CUDA's exp2 is not an exact power of two and the kernel copies it. One ulp of global scale rarely
# moves a dequantized element, so this case covers the output path there, and the sweep below pins the scale's bits.
_EXP2_MINUS_127_CASE = "global_exp2_minus_127"


def _exp2_minus_127_case() -> torch.Tensor:
    amax = torch.tensor(2688.0 * 2.0**-127.5, device=DEV)
    t = amax * (2.0 ** (-torch.arange(64, device=DEV, dtype=torch.float32) / 4)).reshape(4, 16)
    t[1::2] *= -1.0
    return t


def _assert_bits_equal(out, ref, name):
    assert out.dtype == ref.dtype and out.shape == ref.shape, name
    torch.testing.assert_close(out, ref, rtol=0, atol=0, equal_nan=True, msg=name)
    # assert_close treats -0.0 == 0.0, so compare sign bits outside NaN positions as well. Masks only, as boolean
    # indexing would build int64 indices several GiB large for the 32x2880x2880 case.
    flipped = (torch.signbit(out) != torch.signbit(ref)) & ~torch.isnan(ref)
    assert not flipped.any(), f"{name}: sign bit differs"


@triton.jit
def _global_scale_probe(
    a_ptr, out_ptr, n, FMAX: tl.constexpr, SMAX: tl.constexpr, TINY: tl.constexpr, B: tl.constexpr
):
    offs = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    a = tl.load(a_ptr + offs, mask=offs < n, other=0.0)
    tl.store(out_ptr + offs, fq._nvfp4_global_scale(a, FMAX, SMAX, TINY), mask=offs < n)


def _kernel_global_scale(amax: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(amax)
    grid = (triton.cdiv(amax.numel(), 1024),)
    _global_scale_probe[grid](amax, out, amax.numel(), FMAX=E2M1_MAX, SMAX=q.E4M3_MAX, TINY=_AMAX_FLOOR, B=1024)
    return out


def test_nvfp4_global_scale_matches_eager_on_every_fp32_amax():
    """The kernel's global scale against eager's ``nvfp4_global_scale`` (the one ``_nvfp4_core`` and the export call)
    on CUDA, bit for bit, for every positive fp32 tensor amax (inf included) and a NaN. Whole tensors cannot pin it:
    one ulp of global scale, as at CUDA's inexact ``exp2(-127)`` (``0x3FFFFF``), rarely moves a dequantized element, so
    only the scale's own bits show a kernel that builds the exact power of two there."""
    chunk = 1 << 26
    for lo in range(0, 0x7F800001, chunk):
        hi = min(lo + chunk, 0x7F800001)
        amax = torch.arange(lo, hi, device=DEV, dtype=torch.int64).to(torch.int32).view(torch.float32)
        differ = q.nvfp4_global_scale(amax).view(torch.int32) != _kernel_global_scale(amax).view(torch.int32)
        if differ.any():
            first = int(amax[differ.nonzero()[0, 0]].view(torch.int32))
            raise AssertionError(
                f"{int(differ.sum())} amax values in [{lo:#010x}, {hi:#010x}) differ, first {first:#010x}"
            )
    nan = torch.tensor([float("nan")], device=DEV)
    assert q.nvfp4_global_scale(nan).isnan().item() and _kernel_global_scale(nan).isnan().item()
    minus_127 = q.nvfp4_global_scale(_exp2_minus_127_case().abs().amax().reshape(1))
    assert minus_127.view(torch.int32).item() == 0x003FFFFF, "the exp2(-127) case no longer lands on CUDA's 0x3FFFFF"
    print("  nvfp4 global scale: 2,139,095,041 fp32 maxima and NaN match eager — OK")


def _kernel_supported() -> bool:
    """Whether this GPU runs the fp8 kernels (mxfp8 and nvfp4 round through Triton's fp8 type). Stated here and read
    from the device, not from the table's ``min_capability``, so a wrong entry or a wrong guard on SM 8.9+ fails the
    tests."""
    return torch.cuda.get_device_capability() >= (8, 9)


_FP8_FORMATS = ("mxfp8", "nvfp4")


class _Spy:
    """Counts fast-path calls per format by wrapping ``fake_quant_triton`` in the quantization module, which
    ``_FakeQuant`` and ``_round_trip`` look up at call time; the kernel it is handed names the format."""

    def __init__(self):
        self.calls = {fmt: 0 for fmt in FUSED}
        self._format_of = {id(fused.kernel): fmt for fmt, fused in FUSED.items()}

    def __enter__(self):
        def spy(kernel, *a, **k):
            self.calls[self._format_of[id(kernel)]] += 1
            return fake_quant_triton(kernel, *a, **k)

        q.fake_quant_triton = spy
        return self

    def __exit__(self, *_):
        q.fake_quant_triton = fake_quant_triton


def _launch(fused, fmt: str, t: torch.Tensor, kernel=None) -> torch.Tensor:
    return fake_quant_triton(
        kernel or fused.kernel, t, FORMAT_BLOCK_SIZE[fmt], _ELEMENT_MAX[fmt], _AMAX_FLOOR, scale_max=fused.scale_max
    )


def test_bit_exact_against_eager():
    """Each kernel reproduces the CUDA eager quantizer bit-for-bit, NaN blocks and signed zeros included."""
    for fmt, fused in FUSED.items():
        if fmt in _FP8_FORMATS and not _kernel_supported():
            print(f"  {fmt}: below SM 8.9, the fp8 kernel cannot compile — skipped")
            continue
        n = 0
        for name, t in _cases(fmt):
            if t.numel() == 0 and fused.scale_max is not None:
                # Eager nvfp4 cannot take an empty tensor (its tensor amax reduces over no dim), so there is nothing to
                # match: the kernel returns an empty tensor of the input's shape, as the MX kernels do.
                out = _launch(fused, fmt, t)
                assert out.shape == t.shape and out.dtype == t.dtype, f"{fmt}/{name}"
                continue
            ref = _block_round_trip(t, fmt, -1).to(t.dtype)
            _assert_bits_equal(_launch(fused, fmt, t), ref, f"{fmt}/{name}")
            n += 1
        print(f"  {fmt}: {n} cases bit-exact — OK")


def test_routing():
    """fake_quant takes the fast path only for CUDA / contiguous / real / last-axis / K % block size == 0."""
    t = torch.randn(64, K, device=DEV, dtype=torch.bfloat16)
    fp8 = int(_kernel_supported())  # below SM 8.9 mxfp8 and nvfp4 run eager
    expected = {"mxfp8": fp8, "mxfp4": 1, "nvfp4": fp8}
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
        assert s.calls == expected, f"fallback cases took the fast path: {s.calls}"
        # K = 48 divides into nvfp4's 16-wide blocks but not MX's 32-wide ones.
        t48 = torch.randn(8, 48, device=DEV, dtype=torch.bfloat16)
        _assert_bits_equal(fake_quant(t48, "nvfp4"), _block_round_trip(t48, "nvfp4", -1), "nvfp4 K=48")
        assert s.calls["nvfp4"] == 2 * fp8, f"nvfp4 K=48 did not take the fast path: {s.calls}"
    for fmt in FUSED:
        bad = [torch.randn(8, 33, device=DEV, dtype=torch.bfloat16)]
        if fmt != "nvfp4":
            bad.append(t48)
        for x in bad:
            try:
                fake_quant(x, fmt)
                raise AssertionError(f"{fmt}: K={x.shape[-1]} did not raise")
            except ValueError:
                pass
    print("  routing and fallbacks — OK")


def test_pre_sm89_falls_back():
    """A GPU below SM 8.9 cannot compile Triton's fp8 type, so mxfp8 and nvfp4 run eager there. mxfp4 uses no fp8
    type and keeps its kernel down to SM 8.0, the oldest with a bf16 conversion. The capability is faked, so this
    checks the routing, not what Triton does on those GPUs."""
    t = torch.randn(64, K, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(4, 64, 32, device=DEV, dtype=torch.bfloat16)
    orig = torch.cuda.get_device_capability
    try:
        for capability, mxfp4_calls in [((8, 0), 1), ((7, 5), 0)]:
            q._device_capability.cache_clear()
            torch.cuda.get_device_capability = lambda *_, _c=capability: _c
            sm = f"SM {capability[0]}.{capability[1]}"
            with _Spy() as s:
                for fmt in _FP8_FORMATS:
                    _assert_bits_equal(fake_quant(t, fmt), _block_round_trip(t, fmt, -1), f"{sm} {fmt}")
                    assert s.calls[fmt] == 0, f"{sm} took the fp8 Triton path for {fmt}"
                _assert_bits_equal(_round_trip(w, "nvfp4", 1), _block_round_trip(w, "nvfp4", 1), f"{sm} nvfp4 weight")
                assert s.calls["nvfp4"] == 0, f"{sm} took the nvfp4 kernel for a weight"
                _assert_bits_equal(fake_quant(t, "mxfp4"), _block_round_trip(t, "mxfp4", -1), f"{sm} mxfp4")
                assert s.calls["mxfp4"] == mxfp4_calls, f"{sm}: {s.calls['mxfp4']} mxfp4 kernel calls"
    finally:
        torch.cuda.get_device_capability = orig
        q._device_capability.cache_clear()
    print("  pre-SM 8.9 eager fallback — OK")


def test_nvfp4_weight_round_trip_through_the_kernel():
    """EP expert weights quantize on axis 1 of ``[E, K, N]`` through ``_round_trip``, which nvfp4 cannot take
    compiled: it moves the axis last, takes the kernel and moves it back, bit-exact with eager, for a bf16 weight and
    an fp32 master, directly and through ``cached_fake_quant``."""
    if not _kernel_supported():
        print("  nvfp4 weights: below SM 8.9 — skipped")
        return
    torch.manual_seed(0)
    for dtype in (torch.bfloat16, torch.float32):
        w = torch.nn.Parameter(torch.randn(4, 2880, 1024, device=DEV, dtype=dtype) * 0.02)
        ref = _block_round_trip(w.detach(), "nvfp4", 1)
        with _Spy() as s:
            out = _round_trip(w.detach(), "nvfp4", 1)
            cached = cached_fake_quant(w, "nvfp4", 1)
        _assert_bits_equal(out, ref, f"nvfp4 weight {dtype}")
        _assert_bits_equal(cached.detach().to(torch.bfloat16), ref, f"nvfp4 cached weight {dtype}")
        assert out.is_contiguous() and out.dtype == torch.bfloat16
        assert s.calls["nvfp4"] == 2, f"nvfp4 weight kernel calls {s.calls['nvfp4']}, expected 2"
    # An fp16 weight at fp16's max: nvfp4 rounds 65504 up to 67584, past fp16's range. The round trip returns bf16 as
    # eager does, so it stays finite instead of passing through an fp16 output and becoming inf.
    w = torch.randn(2, 64, 32, device=DEV, dtype=torch.float16)
    w[1, 5, 7] = 65504.0
    ref = _block_round_trip(w, "nvfp4", 1)
    out = _round_trip(w, "nvfp4", 1)
    assert ref.isfinite().all() and ref.abs().max().item() > 65504.0, "fixture: eager no longer rounds past fp16 max"
    _assert_bits_equal(out, ref, "nvfp4 fp16 weight past fp16 max")
    print("  nvfp4 weight round trip on axis 1, direct and cached, through the kernel, fp16 past its max — OK")


def test_nvfp4_launches_do_not_sync():
    """The tensor amax stays on the device: neither launch, nor the weight path's axis moves, syncs with the host."""
    if not _kernel_supported():
        print("  nvfp4 sync: below SM 8.9 — skipped")
        return
    t = torch.randn(256, K, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(4, 64, 32, device=DEV, dtype=torch.bfloat16)
    fake_quant(t, "nvfp4")  # compile both kernels first
    _round_trip(w, "nvfp4", 1)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        fake_quant(t, "nvfp4")
        _round_trip(w, "nvfp4", 1)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    print("  nvfp4: no host sync — OK")


def test_no_recompile_across_token_counts():
    """Under EP the token count changes every step. ``n_blocks`` (and the amax pass's ``n``) is not specialized, so
    a new count never compiles a new variant mid-run, where one rank compiling would stall the others at the next
    collective. Each check runs on fresh copies of the kernels with the same ``do_not_specialize``, so variants that
    earlier tests compiled cannot hide a new one."""
    for fmt, fused in FUSED.items():
        if fmt in _FP8_FORMATS and not _kernel_supported():
            print(f"  {fmt}: below SM 8.9, the fp8 kernel cannot compile — skipped")
            continue
        fresh = triton.jit(fused.kernel.fn, do_not_specialize=fused.kernel.do_not_specialize)
        amax_kernel = fq.tensor_abs_max_bits_kernel
        fresh_amax = triton.jit(amax_kernel.fn, do_not_specialize=amax_kernel.do_not_specialize)
        compiled = []

        def hook(*, fn, _watched=(fresh, fresh_amax), _compiled=compiled, **_):
            if fn.jit_function in _watched:
                _compiled.append(fn.name)
            return False  # a truthy return would skip the compile

        prev = triton.knobs.runtime.jit_cache_hook
        triton.knobs.runtime.jit_cache_hook = hook
        fq.tensor_abs_max_bits_kernel = fresh_amax
        try:
            for shape in [(64, K), (7, K), (1, 32)]:  # n_blocks 5760 (divisible by 16), 630, 1 at 32 wide
                t = torch.randn(*shape, device=DEV, dtype=torch.bfloat16)
                _launch(fused, fmt, t, kernel=fresh)
        finally:
            triton.knobs.runtime.jit_cache_hook = prev
            fq.tensor_abs_max_bits_kernel = amax_kernel
        expected = 2 if fused.scale_max is not None else 1
        assert len(compiled) == expected, f"{fmt}: 3 token counts compiled {compiled}, expected {expected} variant(s)"
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
    record_check(checks, "nvfp4_global_scale_sweep", test_nvfp4_global_scale_matches_eager_on_every_fp32_amax)
    record_check(checks, "routing", test_routing)
    record_check(checks, "pre_sm89_falls_back", test_pre_sm89_falls_back)
    record_check(checks, "nvfp4_weight_round_trip", test_nvfp4_weight_round_trip_through_the_kernel)
    record_check(checks, "nvfp4_launches_do_not_sync", test_nvfp4_launches_do_not_sync)
    record_check(checks, "no_recompile_across_token_counts", test_no_recompile_across_token_counts)
    record_check(checks, "ste_backward", test_ste_backward)
    return {"checks": checks}


if __name__ == "__main__":
    run()
