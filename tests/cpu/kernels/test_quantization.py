#!/usr/bin/env python
"""The block-scale quantizers (MXFP8 / MXFP4 / NVFP4) in ``halo/kernels/lowp/quantization.py``."""

import warnings

import pytest
import torch
from torch._dynamo.exc import TorchDynamoException
from torch.overrides import TorchFunctionMode

from halo.kernels.lowp import quantization
from halo.kernels.lowp.quantization import (
    E2M1_MAX,
    cached_fake_quant,
    dequantize,
    fake_quant,
    quantize_mxfp4,
    quantize_mxfp8,
    quantize_nvfp4,
)
from tests.common.numerics import fro_rel_err

# Each format's intrinsic round-trip tolerance (fp8 ~ 8-bit, fp4 ~ 4-bit).
_TOL = {"mxfp8": 0.05, "mxfp4": 0.25, "nvfp4": 0.25}


def test_mxfp8_storage_shapes_and_dtypes():
    torch.manual_seed(0)
    x = torch.randn(64, 256) * 0.5
    q = quantize_mxfp8(x, axis=-1)
    assert q.data.shape == x.shape and q.data.dtype == torch.float8_e4m3fn
    assert q.scales.shape == (64, 256 // 32) and q.scales.dtype == torch.uint8
    assert not q.packed and q.pow2_scale


def test_nvfp4_storage_shapes_and_dtypes():
    torch.manual_seed(0)
    x = torch.randn(64, 256) * 0.5
    q = quantize_nvfp4(x, axis=-1)
    assert q.data.shape == (64, 256 // 2) and q.data.dtype == torch.uint8  # 2 nibbles / byte.
    assert q.scales.shape == (64, 256 // 16) and q.scales.dtype == torch.float8_e4m3fn
    assert q.packed and not q.pow2_scale
    # Two-level scaling: a scalar fp32 multiplier accompanies the e4m3 block scales.
    assert q.global_scale is not None and q.global_scale.numel() == 1 and q.global_scale.dtype == torch.float32


def test_mxfp4_storage_shapes_and_dtypes():
    torch.manual_seed(0)
    x = torch.randn(64, 256) * 0.5
    q = quantize_mxfp4(x, axis=-1)
    assert q.data.shape == (64, 256 // 2) and q.data.dtype == torch.uint8
    assert q.scales.shape == (64, 256 // 32) and q.scales.dtype == torch.uint8
    assert q.packed and q.pow2_scale
    assert q.global_scale is None  # MX is single-level by definition.


def test_roundtrip_within_tolerance_all_formats():
    torch.manual_seed(0)
    x = torch.randn(64, 256) * 0.5
    for fmt, quantizer in (("mxfp8", quantize_mxfp8), ("mxfp4", quantize_mxfp4), ("nvfp4", quantize_nvfp4)):
        recon = dequantize(quantizer(x))
        assert recon.dtype == torch.bfloat16
        re = fro_rel_err(recon, x)
        assert re < _TOL[fmt], f"{fmt} round-trip relerr {re:.4f} >= {_TOL[fmt]}"


def test_fp8_beats_fp4_accuracy():
    torch.manual_seed(1)
    x = torch.randn(128, 256) * 0.4
    assert fro_rel_err(dequantize(quantize_mxfp8(x)), x) < fro_rel_err(dequantize(quantize_nvfp4(x)), x)


def test_axis_0_quantization():
    torch.manual_seed(0)
    x = torch.randn(128, 96) * 0.3
    q = quantize_mxfp8(x, axis=0)
    assert q.scales.shape == (128 // 32, 96)
    assert fro_rel_err(dequantize(q), x) < _TOL["mxfp8"]


def test_zero_input_roundtrips_to_zero():
    # The min-clamp on amax must not turn an all-zero block into NaN.
    x = torch.zeros(32, 64)
    for quantizer in (quantize_mxfp8, quantize_mxfp4, quantize_nvfp4):
        recon = dequantize(quantizer(x))
        assert torch.isfinite(recon).all()
        assert recon.abs().max().item() == 0.0


def test_scale_tracks_block_magnitude():
    small = torch.full((1, 32), 0.01)
    large = torch.full((1, 32), 100.0)
    s_scale = quantize_mxfp8(small).scales.item()
    l_scale = quantize_mxfp8(large).scales.item()
    assert l_scale > s_scale


def test_e2m1_midpoints_round_to_even_not_down():
    """``bucketize`` resolves every tie down, and exact ties are routine, which would bias every fp4 tensor toward
    zero."""
    block = torch.zeros(1, 32)
    block[0, 0] = E2M1_MAX  # Pins the block scale at 1.0.
    block[0, 1:4] = torch.tensor([0.75, 1.75, 3.5])
    block[0, 4] = -0.75  # Sign must not change which way a tie goes.
    block[0, 5:7] = torch.tensor([0.7, 0.8])  # Anti-vacuity: non-ties still round to nearest.

    recon = dequantize(quantize_mxfp4(block))

    assert recon[0, 0].item() == E2M1_MAX
    assert [recon[0, i].item() for i in (1, 2, 3)] == [1.0, 2.0, 4.0]  # Tie-down would give 0.5/1.5/3.0.
    assert recon[0, 4].item() == -1.0
    assert [recon[0, i].item() for i in (5, 6)] == [0.5, 1.0]


def test_nvfp4_saturating_scale_guards_against_outlier_nan():
    # A block amax above E2M1_MAX*E4M3_MAX (=2688) overflows a lone e4m3 scale to NaN; the global scale lifts it into
    # range.
    x = torch.randn(16, 64) * 0.1
    x[0, 0] = 5000.0
    q = quantize_nvfp4(x)
    recon = dequantize(q)
    assert torch.isfinite(q.scales.float()).all()
    assert torch.isfinite(recon).all()


def test_divisibility_guard_raises():
    x = torch.randn(8, 30)  # 30 is divisible by neither 32 (mxfp8/mxfp4) nor 16 (nvfp4).
    for quantizer in (quantize_mxfp8, quantize_mxfp4, quantize_nvfp4):
        try:
            quantizer(x, axis=-1)
        except ValueError:
            continue
        raise AssertionError(f"{quantizer.__name__} should raise on a non-divisible axis")


def test_fake_quant_matches_roundtrip_and_ste_backward():
    torch.manual_seed(0)
    for fmt in ("mxfp8", "nvfp4"):
        x = (torch.randn(64, 256) * 0.3).requires_grad_(True)
        y = fake_quant(x, fmt, axis=-1)
        assert fro_rel_err(y.detach(), x.detach()) < _TOL[fmt]
        # Backward is the identity.
        (y * 2.0).sum().backward()
        assert torch.allclose(x.grad, torch.full_like(x.grad, 2.0)), "STE backward must be identity"


def test_fake_quant_preserves_input_dtype():
    x = (torch.randn(32, 64) * 0.2).to(torch.bfloat16)
    assert fake_quant(x, "mxfp8").dtype == torch.bfloat16


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
    again = cached_fake_quant(w, "mxfp8", axis=-1).detach()
    assert torch.equal(first, again)
    with torch.no_grad():
        w.add_(1.0)
    after = cached_fake_quant(w, "mxfp8", axis=-1).detach()
    assert not torch.equal(first, after)
    assert fro_rel_err(after, w.detach()) < _TOL["mxfp8"]


def test_cached_fake_quant_is_straight_through():
    w = torch.nn.Parameter(torch.randn(8, 64) * 0.3)
    (cached_fake_quant(w, "mxfp8", axis=-1) * 3.0).sum().backward()
    assert w.grad is not None and torch.allclose(w.grad, torch.full_like(w.grad, 3.0))


class _TensorsFromHostData(TorchFunctionMode):
    """Each ``torch.tensor`` is a blocking host-to-device copy on a GPU."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is torch.tensor:
            self.calls += 1
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
def test_fp4_eager_path_builds_its_tables_once(fmt):
    x = torch.randn(8, 64)
    fake_quant(x, fmt)
    with _TensorsFromHostData() as probe:
        for _ in range(3):
            fake_quant(x, fmt)
    assert probe.calls == 0


def test_compiled_mxfp4_round_trip_matches_eager_without_tracing_the_cache():
    torch._dynamo.reset()
    x = torch.randn(64, 128) * 3
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled = torch.compile(quantization._block_round_trip, dynamic=True)(x, "mxfp4", -1)
    assert torch.equal(compiled, quantization._block_round_trip(x, "mxfp4", -1))
    assert not [w for w in caught if "lru_cache" in str(w.message)]


@pytest.fixture
def fresh_compile_state(monkeypatch):
    monkeypatch.setattr(quantization, "_compiled_round_trip", None)
    monkeypatch.setattr(quantization, "_compile_failed", False)
    monkeypatch.setenv("HALO_LOWP_COMPILE", "1")


def test_a_callers_error_leaves_compile_on(fresh_compile_state):
    """It must not switch every later quantization in the process to the eager path."""
    with pytest.raises(ValueError, match="not divisible"):
        quantization._round_trip(torch.randn(4, 33), "mxfp8", -1)
    assert quantization._compile_failed is False


def test_a_compile_failure_falls_back_to_eager(fresh_compile_state, monkeypatch):
    def fail_to_compile(*args):
        raise TorchDynamoException("simulated backend failure")

    monkeypatch.setattr(quantization, "_compiled_round_trip", fail_to_compile)
    x = torch.randn(4, 64)
    out = quantization._round_trip(x, "mxfp8", -1)
    assert quantization._compile_failed is True
    assert torch.equal(out, quantization._block_round_trip(x, "mxfp8", -1))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
