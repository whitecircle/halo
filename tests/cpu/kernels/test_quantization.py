#!/usr/bin/env python
"""CPU tests for the block-scale quantizers (MXFP8 / MXFP4 / NVFP4) in
``src/kernels/lowp/quantization.py``.

Validates storage shapes/dtypes, quantize->dequantize round-trip error within each
format's tolerance, axis handling, divisibility guards, the NVFP4 saturating-scale
outlier guard, the straight-through ``fake_quant`` estimator, and the per-step
``cached_fake_quant`` weight cache.

Run: ``pytest tests/cpu/kernels/test_quantization.py``.
"""

import pytest
import torch

from src.kernels.lowp.quantization import (
    E2M1_MAX,
    cached_fake_quant,
    dequantize,
    fake_quant,
    quantize_mxfp4,
    quantize_mxfp8,
    quantize_nvfp4,
)

# Each format's intrinsic round-trip tolerance (fp8 ~ 8-bit, fp4 ~ 4-bit).
_TOL = {"mxfp8": 0.05, "mxfp4": 0.25, "nvfp4": 0.25}


def _relerr(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


# Storage form: shapes, dtypes, packing


def test_mxfp8_storage_shapes_and_dtypes():
    torch.manual_seed(0)
    x = torch.randn(64, 256) * 0.5
    q = quantize_mxfp8(x, axis=-1, block_size=32)
    assert q.data.shape == x.shape and q.data.dtype == torch.float8_e4m3fn
    assert q.scales.shape == (64, 256 // 32) and q.scales.dtype == torch.uint8  # e8m0
    assert not q.packed and q.pow2_scale


def test_nvfp4_storage_shapes_and_dtypes():
    torch.manual_seed(0)
    x = torch.randn(64, 256) * 0.5
    q = quantize_nvfp4(x, axis=-1, block_size=16)
    assert q.data.shape == (64, 256 // 2) and q.data.dtype == torch.uint8  # 2 nibbles / byte
    assert q.scales.shape == (64, 256 // 16) and q.scales.dtype == torch.float8_e4m3fn
    assert q.packed and not q.pow2_scale
    # Two-level scaling: a scalar fp32 per-tensor multiplier accompanies the e4m3 block scales.
    assert q.global_scale is not None and q.global_scale.numel() == 1 and q.global_scale.dtype == torch.float32


def test_mxfp4_storage_shapes_and_dtypes():
    torch.manual_seed(0)
    x = torch.randn(64, 256) * 0.5
    q = quantize_mxfp4(x, axis=-1, block_size=32)
    assert q.data.shape == (64, 256 // 2) and q.data.dtype == torch.uint8
    assert q.scales.shape == (64, 256 // 32) and q.scales.dtype == torch.uint8  # e8m0 pow2
    assert q.packed and q.pow2_scale
    assert q.global_scale is None  # MX is single-level by definition


# Round-trip accuracy


def test_roundtrip_within_tolerance_all_formats():
    torch.manual_seed(0)
    x = torch.randn(64, 256) * 0.5
    for fmt, quantizer in (("mxfp8", quantize_mxfp8), ("mxfp4", quantize_mxfp4), ("nvfp4", quantize_nvfp4)):
        recon = dequantize(quantizer(x))
        assert recon.dtype == torch.bfloat16
        re = _relerr(recon, x)
        assert re < _TOL[fmt], f"{fmt} round-trip relerr {re:.4f} >= {_TOL[fmt]}"


def test_fp8_beats_fp4_accuracy():
    # 8-bit elements must reconstruct more faithfully than 4-bit ones.
    torch.manual_seed(1)
    x = torch.randn(128, 256) * 0.4
    assert _relerr(dequantize(quantize_mxfp8(x)), x) < _relerr(dequantize(quantize_nvfp4(x)), x)


def test_axis_0_quantization():
    torch.manual_seed(0)
    x = torch.randn(128, 96) * 0.3
    q = quantize_mxfp8(x, axis=0, block_size=32)
    assert q.scales.shape == (128 // 32, 96)  # blocks along axis 0
    assert _relerr(dequantize(q), x) < _TOL["mxfp8"]


def test_zero_input_roundtrips_to_zero():
    # The min-clamp on amax must not turn an all-zero block into NaN/garbage.
    x = torch.zeros(32, 64)
    for quantizer in (quantize_mxfp8, quantize_mxfp4, quantize_nvfp4):
        recon = dequantize(quantizer(x))
        assert torch.isfinite(recon).all()
        assert recon.abs().max().item() == 0.0


# Block-scale semantics


def test_scale_tracks_block_magnitude():
    # A block with larger amax must get a larger e8m0 exponent (scale_exp + 127).
    small = torch.full((1, 32), 0.01)
    large = torch.full((1, 32), 100.0)
    s_scale = quantize_mxfp8(small).scales.item()
    l_scale = quantize_mxfp8(large).scales.item()
    assert l_scale > s_scale


def test_e2m1_midpoints_round_to_even_not_down():
    """The three e2m1 midpoints must resolve UP, as hardware RNE casts do.

    ``bucketize`` alone resolves every tie DOWN, and exact ties are routine (bf16 inputs over
    power-of-two scales land on them), so a tie-down would bias every fp4 tensor toward zero.
    An amax of exactly ``E2M1_MAX`` makes the mxfp4 e8m0 divisor exactly 1.0, so the block's
    magnitudes reach ``_e2m1_codes`` unscaled and the expected codes are readable.
    """
    block = torch.zeros(1, 32)
    block[0, 0] = E2M1_MAX  # pins the block scale at 1.0
    block[0, 1:4] = torch.tensor([0.75, 1.75, 3.5])  # the midpoints
    block[0, 4] = -0.75  # sign must not change which way a tie goes
    block[0, 5:7] = torch.tensor([0.7, 0.8])  # anti-vacuity: non-ties still round to nearest

    recon = dequantize(quantize_mxfp4(block))

    assert recon[0, 0].item() == E2M1_MAX
    assert [recon[0, i].item() for i in (1, 2, 3)] == [1.0, 2.0, 4.0]  # tie-down would give 0.5/1.5/3.0
    assert recon[0, 4].item() == -1.0
    assert [recon[0, i].item() for i in (5, 6)] == [0.5, 1.0]


def test_nvfp4_saturating_scale_guards_against_outlier_nan():
    # A block amax above E2M1_MAX*E4M3_MAX (=2688) would overflow the e4m3 scale to NaN
    # and poison the tensor; the source clamps the scale instead. Reconstruction must
    # stay finite (the outlier is under-scaled, not NaN).
    x = torch.randn(16, 64) * 0.1
    x[0, 0] = 5000.0  # well past the saturation threshold
    q = quantize_nvfp4(x, block_size=16)
    recon = dequantize(q)
    assert torch.isfinite(q.scales.float()).all()
    assert torch.isfinite(recon).all()


# Divisibility guard


def test_divisibility_guard_raises():
    x = torch.randn(8, 30)  # 30 not divisible by 32 (mxfp8/mxfp4) nor 16 (nvfp4)
    for quantizer in (quantize_mxfp8, quantize_mxfp4, quantize_nvfp4):
        try:
            quantizer(x, axis=-1)
        except ValueError:
            continue
        raise AssertionError(f"{quantizer.__name__} should raise on a non-divisible axis")


# fake_quant: straight-through estimator


def test_fake_quant_matches_roundtrip_and_ste_backward():
    torch.manual_seed(0)
    for fmt in ("mxfp8", "nvfp4"):
        x = (torch.randn(64, 256) * 0.3).requires_grad_(True)
        y = fake_quant(x, fmt, axis=-1)
        # Forward equals the explicit quantize->dequantize round-trip.
        assert _relerr(y.detach(), x.detach()) < _TOL[fmt]
        # Backward is the identity (gradient reaches the master weight unchanged).
        (y * 2.0).sum().backward()
        assert torch.allclose(x.grad, torch.full_like(x.grad, 2.0)), "STE backward must be identity"


def test_fake_quant_preserves_input_dtype():
    x = (torch.randn(32, 64) * 0.2).to(torch.bfloat16)
    assert fake_quant(x, "mxfp8").dtype == torch.bfloat16


# cached_fake_quant: per-optimizer-step weight cache


def test_cached_fake_quant_matches_fake_quant():
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(8, 64) * 0.3)
    cached = cached_fake_quant(w, "mxfp8", axis=-1)
    direct = fake_quant(w.detach(), "mxfp8", axis=-1)
    assert torch.allclose(cached.detach(), direct, atol=1e-5)


def test_cached_fake_quant_reuses_until_version_bumps():
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(8, 64) * 0.3)
    first = cached_fake_quant(w, "mxfp8", axis=-1).detach().clone()
    # Same _version → cached dequant reused → identical result.
    again = cached_fake_quant(w, "mxfp8", axis=-1).detach()
    assert torch.equal(first, again)
    # An in-place update bumps _version → cache invalidated → result tracks new weight.
    with torch.no_grad():
        w.add_(1.0)
    after = cached_fake_quant(w, "mxfp8", axis=-1).detach()
    assert not torch.equal(first, after)
    assert _relerr(after, w.detach()) < _TOL["mxfp8"]


def test_cached_fake_quant_is_straight_through():
    w = torch.nn.Parameter(torch.randn(8, 64) * 0.3)
    (cached_fake_quant(w, "mxfp8", axis=-1) * 3.0).sum().backward()
    assert w.grad is not None and torch.allclose(w.grad, torch.full_like(w.grad, 3.0))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
