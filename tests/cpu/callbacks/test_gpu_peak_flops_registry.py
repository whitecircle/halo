"""Cross-checks on the hardware peak-FLOPS registry every MFU number divides by.

Every MFU number the toolkit reports divides by an entry of ``GPU_PEAK_FLOPS``, so a wrong entry
silently rescales the headline metric with nothing else in the system disagreeing. These tests pin the
registry against an INDEPENDENT source — the die's own SM count and clock — rather than against the
constants themselves, so they also reject the next SKU added with a marketing (or sparsity) figure.
"""

import pytest

from src.hardware import GPU_PEAK_FLOPS, GpuPeak, _classify_gpu_name, get_gpu_peak_flops

# Dense bf16/fp16 tensor-core FLOPs per SM per clock, by architecture. Blackwell doubles Hopper's
# per-SM rate; GA102 (workstation) ships a quarter of GA100's tensor throughput per SM.
_ARCH_BF16_FLOPS_PER_SM_CLOCK = {
    "ampere_ga100": 2048,
    "ampere_ga102": 1024,
    "hopper": 4096,
    "blackwell": 8192,
}

# (SM count, max boost clock in GHz, architecture) per registry key, from the vendor die specs.
# Independent of the registry: the ceiling below is computed from these, never from the table.
_DIE_SPECS = {
    "H100_SXM": (132, 1.980, "hopper"),
    "H100_PCIE": (114, 1.755, "hopper"),
    "H100_NVL": (132, 1.785, "hopper"),
    "H200": (132, 1.980, "hopper"),
    "B200": (148, 1.965, "blackwell"),
    "GB200": (148, 2.100, "blackwell"),
    "B300": (148, 2.032, "blackwell"),
    "A100": (108, 1.410, "ampere_ga100"),
    "A6000": (84, 1.860, "ampere_ga102"),
}

# Published peaks embed a nominal rather than max boost clock; allow a little headroom so the test
# rejects marketing/sparsity inflation (tens of percent) without tripping on clock rounding.
_CLOCK_TOLERANCE = 1.02

# Dense tensor-core rate of each dtype relative to bf16, fixed by the tensor-core datapath widths.
# ``fp32`` is a CUDA-core (non-tensor) rate and has no such relation, so it is excluded.
_DTYPE_RATIO_TO_BF16 = {"fp16": 1.0, "fp8": 2.0, "fp4": 4.0, "tf32": 0.5}


def _bf16_ceiling(gpu_key: str) -> float:
    sms, clock_ghz, arch = _DIE_SPECS[gpu_key]
    return sms * clock_ghz * 1e9 * _ARCH_BF16_FLOPS_PER_SM_CLOCK[arch]


def test_every_registry_key_has_a_die_spec():
    """A new SKU must arrive with the die spec that lets the ceiling test police it."""
    missing = sorted(set(GPU_PEAK_FLOPS) - set(_DIE_SPECS))
    assert not missing, (
        f"GPU_PEAK_FLOPS keys {missing} have no entry in _DIE_SPECS, so their peaks are unchecked. "
        "Add (SM count, max boost GHz, architecture) for each."
    )


def test_a_new_registry_entry_needs_no_classifier_branch(monkeypatch):
    """An SKU added to the registry is classifiable from its own declaration alone.

    This is what the per-entry aliases buy: with the classification held somewhere else, a row added
    without its matching branch is worse than an absent one — ``detect_gpu_model`` answers None and
    every MFU / S-MFU line silently drops out of the run's metrics on that hardware.
    """
    monkeypatch.setitem(GPU_PEAK_FLOPS, "B400", GpuPeak(aliases=("B400",), flops={"bf16": 4000.0e12}))

    assert _classify_gpu_name("NVIDIA B400 180GB HBM4") == "B400"
    assert get_gpu_peak_flops("B400", "bf16") == 4000.0e12


def test_every_entry_is_reachable_through_its_own_aliases():
    """Each alias must resolve to the entry that declares it, upper-cased as the names arrive.

    Aliases are ranked by specificity rather than by declaration order, so the case this catches is
    one SKU's name containing another's (``GB200`` over ``B200``): shadowed, the shadowed die's MFU
    would be divided by the wrong peak instead of going missing.
    """
    for key, entry in GPU_PEAK_FLOPS.items():
        assert entry.aliases, f"{key} declares no alias, so no reported device name can resolve to it"
        for alias in entry.aliases:
            assert alias == alias.upper(), f"{key} alias {alias!r} is matched against an upper-cased name"
            assert _classify_gpu_name(alias) == key, (
                f"{key} alias {alias!r} classifies as {_classify_gpu_name(alias)!r}: it is shadowed by "
                "another entry's alias, and this die's MFU would divide by that entry's peak."
            )


@pytest.mark.parametrize("gpu_key", sorted(_DIE_SPECS))
def test_bf16_peak_does_not_exceed_architectural_ceiling(gpu_key):
    """bf16 peak must sit under SMs x clock x FLOPs-per-SM-per-clock.

    This is the assertion that catches a dense figure replaced by a sparsity (2x) or marketing
    multiplier: such a number exceeds what the die can physically retire.
    """
    entry = GPU_PEAK_FLOPS[gpu_key].flops
    ceiling = _bf16_ceiling(gpu_key) * _CLOCK_TOLERANCE
    assert entry["bf16"] <= ceiling, (
        f"{gpu_key} bf16 peak {entry['bf16'] / 1e12:.1f} TFLOP/s exceeds its architectural ceiling "
        f"{ceiling / 1e12:.1f} TFLOP/s ({_DIE_SPECS[gpu_key][0]} SMs x "
        f"{_DIE_SPECS[gpu_key][1]} GHz x {_ARCH_BF16_FLOPS_PER_SM_CLOCK[_DIE_SPECS[gpu_key][2]]} "
        "FLOP/SM/clk). A peak above the ceiling is a sparsity or marketing figure, not a dense one, "
        "and it understates every reported MFU."
    )


@pytest.mark.parametrize("gpu_key", sorted(_DIE_SPECS))
def test_bf16_peak_is_not_implausibly_low(gpu_key):
    """Guard the other direction: a peak far below the ceiling would inflate MFU past 100%."""
    entry = GPU_PEAK_FLOPS[gpu_key].flops
    floor = _bf16_ceiling(gpu_key) * 0.5
    assert entry["bf16"] >= floor, (
        f"{gpu_key} bf16 peak {entry['bf16'] / 1e12:.1f} TFLOP/s is below half its ceiling "
        f"{_bf16_ceiling(gpu_key) / 1e12:.1f} TFLOP/s — MFU computed against it would overstate."
    )


@pytest.mark.parametrize("gpu_key", sorted(GPU_PEAK_FLOPS))
def test_tensor_core_dtype_ratios_match_the_datapath(gpu_key):
    """fp16/fp8/fp4/tf32 peaks are fixed multiples of bf16 on every supported architecture.

    Catches a single-dtype typo, and catches a row where only some dtypes were rescaled.
    """
    entry = GPU_PEAK_FLOPS[gpu_key].flops
    bf16 = entry["bf16"]
    for dtype, ratio in _DTYPE_RATIO_TO_BF16.items():
        if dtype not in entry:
            continue
        expected = bf16 * ratio
        assert entry[dtype] == pytest.approx(expected, rel=0.01), (
            f"{gpu_key} {dtype} peak {entry[dtype] / 1e12:.1f} TFLOP/s is not {ratio}x its bf16 peak "
            f"{bf16 / 1e12:.1f} TFLOP/s (expected {expected / 1e12:.1f}); the tensor-core datapath "
            "fixes this ratio, so one of the two entries is wrong."
        )


def test_b300_matches_the_b200_dense_tensor_core_rates():
    """B300 is the same 148-SM die as B200: Blackwell Ultra's uplift is FP4 and HBM, not bf16.

    Pins the specific regression this file was written for — B300 once carried a ~1.47x-inflated row,
    which understated every MFU on the toolkit's primary training hardware by ~32%.
    """
    for dtype in ("bf16", "fp16", "fp8", "tf32"):
        assert GPU_PEAK_FLOPS["B300"].flops[dtype] == GPU_PEAK_FLOPS["B200"].flops[dtype], (
            f"B300 {dtype} peak diverges from B200's; both are the 148-SM Blackwell die and a "
            "measured large dense GEMM on B300 reaches ~80% of the B200 figure."
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
