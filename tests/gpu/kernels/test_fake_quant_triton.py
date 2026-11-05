#!/usr/bin/env python
"""Fused Triton MXFP8 fake-quant fast path, verifying that
- it is bit-identical to the eager quantizer (``_block_round_trip``) on adversarial inputs: exact
  powers of two, the 448 scale boundary, zeros, -0.0, subnormals, overflow, NaN, inf, e4m3 rounding
  midpoints, an fp32 one-ulp scale edge, and a near-subnormal scale,
- ``fake_quant`` routes CUDA / contiguous / last-axis / mxfp8 through it and everything else to eager,
  including GPUs older than SM 8.9,
- the straight-through backward is unchanged.

Run: torchrun --nproc_per_node=1 tests/gpu/kernels/test_fake_quant_triton.py
"""

import torch

import halo.kernels.lowp.quantization as q
from halo.kernels.lowp.fake_quant_triton import fake_quant_mxfp8_triton
from halo.kernels.lowp.quantization import _AMAX_FLOOR, E4M3_MAX, FORMAT_BLOCK_SIZE, _block_round_trip, fake_quant
from tests.common.harness import gpu_test_main, record_check

DEV = "cuda"
K = 2880


def _cases():
    g = torch.Generator(device=DEV).manual_seed(0)
    bf = torch.bfloat16
    for shape in [(4096, K), (16384, K), (32, K, K), (7, 32), (1, 64)]:
        yield f"rand{shape}", torch.randn(shape, device=DEV, generator=g, dtype=bf)
    yield "pow2", (2.0 ** torch.randint(-20, 20, (256, K), device=DEV)).to(bf)
    yield "amax_edge", torch.full((64, K), 448.0, device=DEV, dtype=bf)
    yield "zeros", torch.zeros(64, K, device=DEV, dtype=bf)
    yield "negzero", torch.full((64, K), -0.0, device=DEV, dtype=bf)
    yield "tiny", torch.full((64, K), 1e-40, device=DEV, dtype=bf)
    yield "huge", torch.full((64, K), 3e38, device=DEV, dtype=bf)
    t = torch.randn(64, K, device=DEV, dtype=bf)
    t[3, 5] = float("nan")
    yield "nan", t
    t = torch.randn(64, K, device=DEV, dtype=bf)
    t[0, 0] = float("inf")
    yield "inf", t
    yield "midpoints", torch.randn(64, K, device=DEV, dtype=bf) * 1.0625
    t = torch.full((64, K), 448.0 * 2.0**5, device=DEV)
    yield "fp32_ulp_edge", torch.nextafter(t, torch.full_like(t, 1e30))
    yield "near_denorm", torch.full((64, K), 3e-36, device=DEV, dtype=bf)
    yield "empty", torch.empty(0, K, device=DEV, dtype=bf)
    yield "fp16", torch.randn(64, K, device=DEV, dtype=torch.float16)
    yield "fp32", torch.randn(64, K, device=DEV, dtype=torch.float32)


def _assert_bits_equal(out, ref, name):
    assert out.dtype == ref.dtype and out.shape == ref.shape, name
    torch.testing.assert_close(out, ref, rtol=0, atol=0, equal_nan=True, msg=name)
    # assert_close treats -0.0 == 0.0, so compare sign bits outside NaN positions as well. Masks only, as boolean
    # indexing would build int64 indices several GiB large for the 32x2880x2880 case.
    flipped = (torch.signbit(out) != torch.signbit(ref)) & ~torch.isnan(ref)
    assert not flipped.any(), f"{name}: sign bit differs"


def _kernel_supported() -> bool:
    return q._fp8_triton_supported(torch.cuda.current_device())


class _Spy:
    """Counts fast-path calls. ``_FakeQuant`` resolves the kernel as a ``quantization`` module global at
    call time, so patching the module attribute is enough."""

    def __init__(self):
        self.calls = 0
        self._orig = q.fake_quant_mxfp8_triton

    def __enter__(self):
        def spy(t, *a, **k):
            self.calls += 1
            return self._orig(t, *a, **k)

        q.fake_quant_mxfp8_triton = spy
        return self

    def __exit__(self, *_):
        q.fake_quant_mxfp8_triton = self._orig


def test_bit_exact_against_eager():
    """The kernel reproduces the eager quantizer bit-for-bit, NaN blocks included."""
    if not _kernel_supported():
        print("  below SM 8.9, the fp8 kernel cannot compile — skipped")
        return
    n = 0
    for name, t in _cases():
        ref = _block_round_trip(t, "mxfp8", -1).to(t.dtype)
        _assert_bits_equal(fake_quant_mxfp8_triton(t, FORMAT_BLOCK_SIZE["mxfp8"], E4M3_MAX, _AMAX_FLOOR), ref, name)
        n += 1
    print(f"  {n} cases bit-exact — OK")


def test_routing():
    """fake_quant takes the fast path only for CUDA / contiguous / last-axis / mxfp8 / K % 32 == 0."""
    t = torch.randn(64, K, device=DEV, dtype=torch.bfloat16)
    fast = int(_kernel_supported())  # below SM 8.9 even the canonical case runs eager
    with _Spy() as s:
        _assert_bits_equal(fake_quant(t, "mxfp8"), _block_round_trip(t, "mxfp8", -1), "wired")
        assert s.calls == fast, f"canonical case: {s.calls} fast-path calls, expected {fast}"
        fake_quant(t, "mxfp8", axis=0)  # non-last axis → eager
        fake_quant(t, "mxfp4")  # other format → eager
        fake_quant(t.t().contiguous().t(), "mxfp8")  # non-contiguous → eager
        fake_quant(t.cpu(), "mxfp8")  # CPU → eager
        assert s.calls == fast, f"fallback cases took the fast path ({s.calls - fast} extra calls)"
    try:
        fake_quant(torch.randn(8, 33, device=DEV, dtype=torch.bfloat16), "mxfp8")
        raise AssertionError("non-divisible K did not raise")
    except ValueError:
        pass
    print("  routing and fallbacks — OK")


def test_pre_sm89_falls_back():
    """A GPU below SM 8.9 cannot compile the fp8 kernel, so mxfp8 runs eager there."""
    t = torch.randn(64, K, device=DEV, dtype=torch.bfloat16)
    orig = torch.cuda.get_device_capability
    q._fp8_triton_supported.cache_clear()
    torch.cuda.get_device_capability = lambda *_: (8, 0)
    try:
        with _Spy() as s:
            _assert_bits_equal(fake_quant(t, "mxfp8"), _block_round_trip(t, "mxfp8", -1), "sm80")
            assert s.calls == 0, "SM 8.0 took the fp8 Triton path"
    finally:
        torch.cuda.get_device_capability = orig
        q._fp8_triton_supported.cache_clear()
    print("  pre-SM 8.9 eager fallback — OK")


def test_ste_backward():
    """Straight-through: the gradient passes unchanged through the fast path."""
    t = torch.randn(64, K, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn_like(t)
    fake_quant(t, "mxfp8").backward(g)
    assert torch.equal(t.grad, g), "backward is not identity"
    print("  STE backward — OK")


@gpu_test_main(exact_world_size=1, prefix="fake_quant_triton", partial_state=False)
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    record_check(checks, "bit_exact_against_eager", test_bit_exact_against_eager)
    record_check(checks, "routing", test_routing)
    record_check(checks, "pre_sm89_falls_back", test_pre_sm89_falls_back)
    record_check(checks, "ste_backward", test_ste_backward)
    return {"checks": checks}


if __name__ == "__main__":
    run()
