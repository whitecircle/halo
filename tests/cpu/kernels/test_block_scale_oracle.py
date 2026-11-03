#!/usr/bin/env python
"""The block-scaled quantization oracle (halo/kernels/lowp/quantization.py) against an independent first-principles
reference. The registry-enumerated checks police a format added later."""

import math
import os

import pytest
import torch

import halo.kernels.lowp.quantization as q

E2M1_MAX = 6.0
E4M3_MAX = 448.0
E8M0_BIAS = 127
# Independent copy of the format definition.
_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _expected_e8m0_exp(amax: float, fmt_max: float) -> int:
    raw = math.ceil(math.log2(amax / fmt_max))
    raw = max(-E8M0_BIAS, min(E8M0_BIAS, raw))
    return raw + E8M0_BIAS


def _ref_e2m1_code(value: float, divisor: float) -> int:
    """An exact midpoint goes to the lower level, what round-half-to-even does wherever the lower level's code is
    even."""
    scaled = value / divisor
    sign = 1 if scaled < 0 else 0
    mag = min(abs(scaled), E2M1_MAX)
    boundaries = [(a + b) / 2 for a, b in zip(_LEVELS[:-1], _LEVELS[1:], strict=False)]
    level = len(_LEVELS) - 1
    for i, edge in enumerate(boundaries):
        if mag <= edge:
            level = i
            break
    return (sign << 3) | level


def test_mxfp8_scale_exponent_first_principles():
    block = torch.zeros(32, dtype=torch.float32)
    block[0] = 100.0
    block[5] = -12.5
    bst = q.quantize_mxfp8(block, axis=-1)
    assert bst.pow2_scale is True
    assert bst.scales.numel() == 1
    expected = _expected_e8m0_exp(100.0, E4M3_MAX)
    assert int(bst.scales.item()) == expected, f"mxfp8 e8m0 scale {int(bst.scales.item())} != reference {expected}"


def _unpack_codes(bst) -> list[int]:
    packed = bst.data.view(-1)
    codes = []
    for byte in packed.tolist():
        codes.append(byte & 0x0F)
        codes.append((byte >> 4) & 0x0F)
    return codes


def test_mxfp4_codes_match_first_principles():
    """amax = 6.0 makes the divisor 1.0, so codes map levels directly."""
    vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -6.0]
    block = torch.zeros(32, dtype=torch.float32)
    for i, v in enumerate(vals):
        block[i] = v

    bst = q.quantize_mxfp4(block, axis=-1)
    expected_exp = _expected_e8m0_exp(6.0, E2M1_MAX)
    assert int(bst.scales.item()) == expected_exp
    divisor = 2.0 ** (expected_exp - E8M0_BIAS)
    assert divisor == 1.0

    codes = _unpack_codes(bst)
    for i in range(32):
        v = float(block[i])
        assert codes[i] == _ref_e2m1_code(v, divisor), (
            f"element {i} value {v}: code {codes[i]} != reference {_ref_e2m1_code(v, divisor)}"
        )


def test_mxfp4_boundary_straddle_2p5_rounds_to_lower():
    """Pins the tie-break direction so a flip in rounding mode breaks the test."""
    block = torch.zeros(32, dtype=torch.float32)
    block[0] = 6.0  # Forces amax=6, so divisor 1.0.
    block[1] = 2.5
    block[2] = 2.6
    block[3] = 2.4
    bst = q.quantize_mxfp4(block, axis=-1)
    codes = _unpack_codes(bst)
    assert codes[1] == 4, f"2.5 must round to level 2.0 (code 4), got {codes[1]}"
    assert codes[2] == 5, f"2.6 must round to level 3.0 (code 5), got {codes[2]}"
    assert codes[3] == 4, f"2.4 must round to level 2.0 (code 4), got {codes[3]}"
    for i, v in [(1, 2.5), (2, 2.6), (3, 2.4)]:
        assert codes[i] == _ref_e2m1_code(v, 1.0)


def test_nvfp4_scale_is_two_level_e4m3_not_pow2():
    """``global = 2**ceil(log2(amax_tensor / (448*6)))``, ``block_scale = e4m3(amax_block / 6 / global)``; the
    power-of-two rounding makes an EP shard and the gathered export agree."""
    block = torch.zeros(16, dtype=torch.float32)
    block[0] = 30.0
    bst = q.quantize_nvfp4(block, axis=-1)
    assert bst.pow2_scale is False
    assert bst.global_scale is not None, "nvfp4 must carry a per-tensor global scale"

    expected_global = 2.0 ** math.ceil(math.log2(30.0 / (E4M3_MAX * E2M1_MAX)))
    assert bst.global_scale.item() == pytest.approx(expected_global, rel=1e-6)
    expected_scale = (
        torch.tensor(30.0 / E2M1_MAX / expected_global, dtype=torch.float32).to(torch.float8_e4m3fn).float().item()
    )
    assert expected_scale <= E4M3_MAX, "the tensor-amax block must still fit an e4m3 scale"
    assert abs(bst.scales.float().item() - expected_scale) < 1e-6, (
        f"nvfp4 e4m3 scale {bst.scales.float().item()} != reference {expected_scale}"
    )
    assert abs(float(q.dequantize(bst)[0]) - 30.0) < 1e-3


@pytest.mark.parametrize("fmt", sorted(q.QUANTIZERS))
def test_round_trip_is_scale_invariant_across_magnitudes(fmt):
    """Both breakages are silent: a single-level NVFP4 ``e4m3`` scale underflows to zero, and an amax floor above
    fp32's smallest normal rescales every block."""
    torch.manual_seed(0)
    base = torch.randn(128, 256, dtype=torch.float32)
    errors = {}
    for exponent in range(4, -32, -4):
        std = 10.0**exponent
        x = base * std
        recon = q.dequantize(q.QUANTIZERS[fmt](x)).double()
        # float64 norms: at std 1e-28 the squared magnitudes underflow fp32 and the ratio reads NaN.
        errors[std] = ((recon - x.double()).norm() / x.double().norm()).item()

    reference = errors[1.0]
    for std, err in errors.items():
        assert err == pytest.approx(reference, abs=0.02), (
            f"{fmt} round-trip relerr at weight std {std:g} is {err:.4f} vs {reference:.4f} at std 1.0 — "
            "small-magnitude blocks are being destroyed"
        )


def test_mxfp4_round_trip_compile_bit_identical():
    """The module claims the e8m0 MXFP4 round-trip compiles bit-identically."""
    torch.manual_seed(0)
    weight = torch.randn(64, 128, dtype=torch.float32) * 3.0

    def _round_trip_with_compile(flag: str) -> torch.Tensor:
        prev = os.environ.get("HALO_LOWP_COMPILE")
        prev_compiled = q._compiled_round_trip
        prev_failed = q._compile_failed
        os.environ["HALO_LOWP_COMPILE"] = flag
        # Reset the compile caches. importlib.reload would rebind the classes and fail isinstance checks in sibling
        # tests order-dependently.
        q._compiled_round_trip = None
        q._compile_failed = False
        try:
            out = q._round_trip(weight, "mxfp4", -1)
        finally:
            if prev is None:
                os.environ.pop("HALO_LOWP_COMPILE", None)
            else:
                os.environ["HALO_LOWP_COMPILE"] = prev
            q._compiled_round_trip = prev_compiled
            q._compile_failed = prev_failed
        return out

    off = _round_trip_with_compile("0")
    on = _round_trip_with_compile("1")
    assert off.dtype == on.dtype == torch.bfloat16
    assert torch.equal(off, on), "MXFP4 round-trip not bit-identical between compile off and on"


@pytest.mark.parametrize("fmt", sorted(q.QUANTIZERS))
@pytest.mark.parametrize("ep_size", [2, 4, 8])
def test_quantizing_an_expert_shard_matches_quantizing_the_gathered_bank(fmt, ep_size):
    """Training quantizes the slice an EP rank holds and the export the gathered bank; e4m3 is invariant under a
    power-of-two rescale, so a plain ``amax/2688`` global fails here."""
    torch.manual_seed(0)
    bank = torch.randn(8, 64, 256, dtype=torch.float32) * 0.02
    bank[0] *= 1e-3  # One expert far below the bank amax.

    gathered = q.dequantize(q.QUANTIZERS[fmt](bank, axis=-1))
    sharded = torch.cat([q.dequantize(q.QUANTIZERS[fmt](s, axis=-1)) for s in bank.chunk(ep_size, dim=0)], dim=0)
    assert torch.equal(sharded, gathered), (
        f"{fmt} at ep{ep_size}: the shard-quantized bank differs from the gathered one by up to "
        f"{(sharded.float() - gathered.float()).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("fmt", sorted(q.QUANTIZERS))
@pytest.mark.parametrize("poison", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_input_propagates_through_every_format(fmt, poison):
    """``e2m1`` has no NaN encoding and a NaN exponent casts to 0, so an unguarded MXFP4 block decodes as a finite
    ~3.5e-38; asserted on the dequantized value since each format stores the poison differently."""
    torch.manual_seed(0)
    x = torch.randn(4, 64, dtype=torch.float32) * 0.3
    x[1, 7] = poison
    recon = q.dequantize(q.QUANTIZERS[fmt](x))
    assert not torch.isfinite(recon).all(), (
        f"{fmt} absorbed a {poison} input: max |recon| = {recon.abs().max().item():.3e}"
    )


@pytest.mark.parametrize("fmt", sorted(q.QUANTIZERS))
def test_finite_input_never_produces_nan(fmt):
    """The 0xFF code is reserved."""
    torch.manual_seed(0)
    x = torch.randn(4, 64, dtype=torch.float32) * 0.3
    x[0, 0] = 5e4
    x[2, :] = 0.0
    recon = q.dequantize(q.QUANTIZERS[fmt](x))
    assert torch.isfinite(recon).all(), f"{fmt} produced a non-finite value from finite input"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
