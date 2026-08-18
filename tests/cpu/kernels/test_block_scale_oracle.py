#!/usr/bin/env python
"""CPU tests for the block-scaled quantization oracle (src/kernels/lowp/quantization.py).

Pins the MX / NVFP4 quant math against an INDEPENDENT first-principles reference
(not by re-calling the module's own helpers):

  * the e8m0 block scale exponent == ceil(log2(amax / format_max)), biased by 127;
  * the per-element e2m1 code == sign<<3 | round-to-nearest magnitude level,
    including values that straddle a bucket boundary (e.g. 2.5, the midpoint of
    levels 2.0 and 3.0);
  * the MXFP4 (power-of-two scale) round-trip is bit-identical with
    HALO_LOWP_COMPILE on vs off (the module documents this);
  * NVFP4's two-level scaling makes the round-trip scale-invariant;
  * every registered format propagates NaN instead of absorbing it.

The last two are enumerated over the module's own format registry, so a format
added later is policed by them without touching this file.

Run: python tests/cpu/kernels/test_block_scale_oracle.py  (or pytest)
"""

import math
import os

import pytest
import torch

import src.kernels.lowp.quantization as q

E2M1_MAX = 6.0
E4M3_MAX = 448.0
E8M0_BIAS = 127
# e2m1 magnitudes, codes 0..7 (independent copy of the format definition).
_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _expected_e8m0_exp(amax: float, fmt_max: float) -> int:
    """Reference e8m0 biased exponent: ceil(log2(amax / fmt_max)) + 127, clamped."""
    raw = math.ceil(math.log2(amax / fmt_max))
    raw = max(-E8M0_BIAS, min(E8M0_BIAS, raw))
    return raw + E8M0_BIAS


def _ref_e2m1_code(value: float, divisor: float) -> int:
    """Reference e2m1 code for one element: sign<<3 | nearest-level index.

    Round-to-nearest using the midpoints between adjacent levels, matching the
    bucketize used by the module (lower level chosen at an exact midpoint).
    """
    scaled = value / divisor
    sign = 1 if scaled < 0 else 0
    mag = min(abs(scaled), E2M1_MAX)
    # Nearest level by midpoint boundaries. torch.bucketize(v, edges) with the default
    # right=False returns the count of edges with edge < v ... in fact maps a value
    # exactly ON a boundary to the LOWER level (it returns bucket i for edges[i-1] < v <= edges[i]).
    # Mirror that with mag <= edge so an exact midpoint (e.g. 2.5) lands on the lower level.
    boundaries = [(a + b) / 2 for a, b in zip(_LEVELS[:-1], _LEVELS[1:], strict=False)]
    level = len(_LEVELS) - 1
    for i, edge in enumerate(boundaries):
        if mag <= edge:
            level = i
            break
    return (sign << 3) | level


# MXFP8 scale exponent


def test_mxfp8_scale_exponent_first_principles():
    """A single 32-element block: the stored e8m0 scale == ceil(log2(amax/448))+127."""
    block = torch.zeros(32, dtype=torch.float32)
    block[0] = 100.0  # amax = 100
    block[5] = -12.5
    bst = q.quantize_mxfp8(block, axis=-1, block_size=32)
    assert bst.pow2_scale is True
    assert bst.scales.numel() == 1
    expected = _expected_e8m0_exp(100.0, E4M3_MAX)
    assert int(bst.scales.item()) == expected, f"mxfp8 e8m0 scale {int(bst.scales.item())} != reference {expected}"


# MXFP4 e2m1 codes — incl. bucket-boundary straddle


def _unpack_codes(bst) -> list[int]:
    """Recover the per-element e2m1 codes from a packed fp4 BlockScaledTensor (single block)."""
    packed = bst.data.view(-1)
    codes = []
    for byte in packed.tolist():
        codes.append(byte & 0x0F)
        codes.append((byte >> 4) & 0x0F)
    return codes


def test_mxfp4_codes_match_first_principles():
    """Hand-built block: every e2m1 code matches the independent reference, and
    the e8m0 scale matches ceil(log2(amax/6))+127."""
    # amax = 6.0 -> ceil(log2(6/6)) = 0 -> divisor = 2^0 = 1.0, so codes map levels directly.
    vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -6.0]
    block = torch.zeros(32, dtype=torch.float32)
    for i, v in enumerate(vals):
        block[i] = v

    bst = q.quantize_mxfp4(block, axis=-1, block_size=32)
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
    """2.5 is the exact midpoint between levels 2.0 and 3.0. bucketize(right=False)
    sends a value on the boundary to the LOWER level (2.0, code 4). This pins the
    tie-break direction so a flip in rounding mode breaks the test."""
    block = torch.zeros(32, dtype=torch.float32)
    block[0] = 6.0  # forces amax=6 -> divisor 1.0
    block[1] = 2.5  # exactly on the 2.0/3.0 boundary
    block[2] = 2.6  # clearly above -> level 3.0 (code 5)
    block[3] = 2.4  # clearly below -> level 2.0 (code 4)
    bst = q.quantize_mxfp4(block, axis=-1, block_size=32)
    codes = _unpack_codes(bst)
    assert codes[1] == 4, f"2.5 must round to level 2.0 (code 4), got {codes[1]}"
    assert codes[2] == 5, f"2.6 must round to level 3.0 (code 5), got {codes[2]}"
    assert codes[3] == 4, f"2.4 must round to level 2.0 (code 4), got {codes[3]}"
    # Cross-check against the independent reference too.
    for i, v in [(1, 2.5), (2, 2.6), (3, 2.4)]:
        assert codes[i] == _ref_e2m1_code(v, 1.0)


def test_nvfp4_scale_is_two_level_e4m3_not_pow2():
    """NVFP4 stores an e4m3 (non-power-of-two) block scale RELATIVE to a per-tensor fp32 global scale.

    Reference (the NVFP4 contract): ``global = 2**ceil(log2(amax_tensor / (448*6)))`` and
    ``block_scale = e4m3(amax_block / 6 / global)``, so an element decodes as
    ``level * block_scale * global``. The power-of-two rounding is what makes an EP shard and the
    gathered export agree; see ``test_quantizing_an_expert_shard_matches_quantizing_the_gathered_bank``.
    """
    block = torch.zeros(16, dtype=torch.float32)
    block[0] = 30.0
    bst = q.quantize_nvfp4(block, axis=-1, block_size=16)
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


@pytest.mark.parametrize("fmt", sorted(q._QUANTIZERS))
def test_round_trip_is_scale_invariant_across_magnitudes(fmt):
    """A block-scaled round-trip's RELATIVE error must not depend on the tensor's magnitude.

    Every format here carries a per-block scale precisely so that only the shape of the distribution
    matters. Two ways of breaking that, both silent — the block quantizes to all zeros and nothing
    warns:

    * a single-level absolute NVFP4 ``e4m3`` block scale underflows to zero once
      ``amax_block < E2M1_MAX x`` half of e4m3's smallest subnormal (5.86e-3), which covers the
      residual-scaled ``down_proj`` init ``0.02/sqrt(2L)``;
    * an amax floor set to a round epsilon rather than fp32's smallest normal rescales every block of
      a tensor whose amax falls below it.

    Enumerated over the module's format registry and over the magnitude range a bf16 master can hold,
    so either failure mode is caught for any format, present or future.
    """
    torch.manual_seed(0)
    base = torch.randn(128, 256, dtype=torch.float32)
    errors = {}
    for exponent in range(4, -32, -4):  # 1e4 down to 1e-28
        std = 10.0**exponent
        x = base * std
        recon = q.dequantize(q._QUANTIZERS[fmt](x)).double()
        # float64 norms: at std 1e-28 the squared magnitudes underflow fp32 and the ratio reads NaN.
        errors[std] = ((recon - x.double()).norm() / x.double().norm()).item()

    reference = errors[1.0]
    for std, err in errors.items():
        assert err == pytest.approx(reference, abs=0.02), (
            f"{fmt} round-trip relerr at weight std {std:g} is {err:.4f} vs {reference:.4f} at std 1.0 — "
            "small-magnitude blocks are being destroyed"
        )


# MXFP4 round-trip: bit-identical with compile on vs off


def test_mxfp4_round_trip_compile_bit_identical():
    """The module claims the e8m0 (power-of-two) MXFP4 round-trip compiles
    bit-identically. Run the WEIGHT round-trip (which compiles) with
    HALO_LOWP_COMPILE=1 and =0 on the same input and assert exact equality."""
    torch.manual_seed(0)
    weight = torch.randn(64, 128, dtype=torch.float32) * 3.0

    def _round_trip_with_compile(flag: str) -> torch.Tensor:
        prev = os.environ.get("HALO_LOWP_COMPILE")
        prev_compiled = q._compiled_round_trip
        prev_failed = q._compile_failed
        os.environ["HALO_LOWP_COMPILE"] = flag
        # Reset the module's compile caches for this flag. Do NOT importlib.reload:
        # reloading re-executes the module in its shared dict, rebinding its classes
        # and poisoning every earlier ``from src.kernels.lowp.quantization import ...``
        # (isinstance checks in sibling tests then fail order-dependently).
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


@pytest.mark.parametrize("fmt", sorted(q._QUANTIZERS))
@pytest.mark.parametrize("ep_size", [2, 4, 8])
def test_quantizing_an_expert_shard_matches_quantizing_the_gathered_bank(fmt, ep_size):
    """Slicing a fused expert bank along the expert axis must not change any block's quantization.

    This is the QAT<->export contract: training quantizes the ``[E/ep, ...]`` slice an EP rank holds,
    while ``quantize_to_lowp`` quantizes the gathered ``[E, ...]`` checkpoint. They must produce the same
    dequantized weight, or the exported model is not the model that was trained.

    It is what constrains NVFP4's per-tensor global scale to a POWER OF TWO: e4m3 rounding is invariant
    under a power-of-two rescale, so the two tensors' differing amax cannot move a block scale. The bank
    below spans three orders of magnitude across experts, so a plain ``amax/2688`` global fails here.
    """
    torch.manual_seed(0)
    bank = torch.randn(8, 64, 256, dtype=torch.float32) * 0.02
    bank[0] *= 1e-3  # one expert far below the bank amax

    gathered = q.dequantize(q._QUANTIZERS[fmt](bank, axis=-1))
    sharded = torch.cat([q.dequantize(q._QUANTIZERS[fmt](s, axis=-1)) for s in bank.chunk(ep_size, dim=0)], dim=0)
    assert torch.equal(sharded, gathered), (
        f"{fmt} at ep{ep_size}: the shard-quantized bank differs from the gathered one by up to "
        f"{(sharded.float() - gathered.float()).abs().max().item():.3e}"
    )


# NaN must survive every format


@pytest.mark.parametrize("fmt", sorted(q._QUANTIZERS))
@pytest.mark.parametrize("poison", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_input_propagates_through_every_format(fmt, poison):
    """A non-finite element must come back non-finite, for every registered block-scaled format.

    A format that absorbs it lets a diverged run continue with plausible-looking weights: MXFP4 is the
    trap, because ``e2m1`` has no NaN encoding and casting a NaN exponent to ``uint8`` gives 0, so an
    unguarded block decodes as a finite ~3.5e-38. Enumerated over the registry so a new format cannot
    regress the invariant silently, and asserted on the DEQUANTIZED value rather than on any storage
    field, because each format carries the poison differently (e4m3 data, e4m3 scale, e8m0's 0xFF code)
    and NVFP4's per-tensor global scale spreads it across the whole tensor rather than one block.
    """
    torch.manual_seed(0)
    x = torch.randn(4, 64, dtype=torch.float32) * 0.3
    x[1, 7] = poison
    recon = q.dequantize(q._QUANTIZERS[fmt](x))
    assert not torch.isfinite(recon).all(), (
        f"{fmt} absorbed a {poison} input: max |recon| = {recon.abs().max().item():.3e}"
    )


@pytest.mark.parametrize("fmt", sorted(q._QUANTIZERS))
def test_finite_input_never_produces_nan(fmt):
    """The other direction: no finite input may manufacture a NaN (the 0xFF code is reserved)."""
    torch.manual_seed(0)
    x = torch.randn(4, 64, dtype=torch.float32) * 0.3
    x[0, 0] = 5e4  # scale-saturating outlier
    x[2, :] = 0.0  # all-zero block
    recon = q.dequantize(q._QUANTIZERS[fmt](x))
    assert torch.isfinite(recon).all(), f"{fmt} produced a non-finite value from finite input"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
