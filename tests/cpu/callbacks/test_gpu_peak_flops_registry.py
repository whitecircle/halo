"""Cross-checks on the peak-FLOPS registry every MFU divides by. A wrong entry silently rescales the headline metric,
so peaks are pinned against the die's own SM count and clock, which rejects a marketing or sparsity figure too."""

import subprocess

import pytest

import halo.hardware as hardware
from halo.hardware import GPU_PEAK_FLOPS, GpuPeak, _classify_gpu_name, detect_gpu_model, get_gpu_peak_flops

# Dense bf16/fp16 tensor-core FLOPs per SM per clock. GA102 (workstation) ships a lower per-SM rate than GA100.
_ARCH_BF16_FLOPS_PER_SM_CLOCK = {
    "ampere_ga100": 2048,
    "ampere_ga102": 1024,
    "hopper": 4096,
    "blackwell": 8192,
}

# (SM count, max boost GHz, architecture) from the vendor die specs, independent of the registry the ceiling checks.
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

# Published peaks embed a nominal clock. This allows clock rounding but not marketing inflation (tens of percent).
_CLOCK_TOLERANCE = 1.02

# Set by the tensor-core datapath widths. ``fp32`` is a CUDA-core rate with no such relation, so it is excluded.
_DTYPE_RATIO_TO_BF16 = {"fp16": 1.0, "fp8": 2.0, "fp4": 4.0, "tf32": 0.5}


def _bf16_ceiling(gpu_key: str) -> float:
    sms, clock_ghz, arch = _DIE_SPECS[gpu_key]
    return sms * clock_ghz * 1e9 * _ARCH_BF16_FLOPS_PER_SM_CLOCK[arch]


def test_every_registry_key_has_a_die_spec():
    missing = sorted(set(GPU_PEAK_FLOPS) - set(_DIE_SPECS))
    assert not missing, (
        f"GPU_PEAK_FLOPS keys {missing} have no entry in _DIE_SPECS, so their peaks are unchecked. "
        "Add (SM count, max boost GHz, architecture) for each."
    )


def test_a_new_registry_entry_needs_no_classifier_branch(monkeypatch):
    """A row without a matching classifier branch is worse than an absent one: ``detect_gpu_model`` answers None and
    every MFU line silently drops out on that hardware."""
    monkeypatch.setitem(GPU_PEAK_FLOPS, "B400", GpuPeak(aliases=("B400",), flops={"bf16": 4000.0e12}))

    assert _classify_gpu_name("NVIDIA B400 180GB HBM4") == "B400"
    assert get_gpu_peak_flops("B400", "bf16") == 4000.0e12


def test_every_entry_is_reachable_through_its_own_aliases():
    """Aliases rank by specificity, so this catches one SKU's name containing another's (``GB200`` over ``B200``)."""
    for key, entry in GPU_PEAK_FLOPS.items():
        assert entry.aliases, f"{key} declares no alias, so no reported device name can resolve to it"
        for alias in entry.aliases:
            assert alias == alias.upper(), f"{key} alias {alias!r} is matched against an upper-cased name"
            assert _classify_gpu_name(alias) == key, (
                f"{key} alias {alias!r} classifies as {_classify_gpu_name(alias)!r}: it is shadowed by "
                "another entry's alias, and this die's MFU would divide by that entry's peak."
            )


def _fake_device(monkeypatch, torch_name: str, smi):
    monkeypatch.setattr(hardware.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(hardware.torch.cuda, "get_device_name", lambda index: torch_name)

    def run(cmd, **kwargs):
        if isinstance(smi, BaseException):
            raise smi
        return subprocess.CompletedProcess(cmd, 0, stdout=smi, stderr="")

    monkeypatch.setattr(hardware.subprocess, "run", run)


def test_detection_prefers_the_torch_name_and_falls_back_to_nvidia_smi(monkeypatch):
    _fake_device(monkeypatch, "NVIDIA B300 SXM6 AC", smi=AssertionError("nvidia-smi must not run"))
    assert detect_gpu_model() == "B300"
    _fake_device(monkeypatch, "GRID vGPU", smi="NVIDIA H100 PCIe\n")
    assert detect_gpu_model() == "H100_PCIE"


@pytest.mark.parametrize(
    "smi", [FileNotFoundError("nvidia-smi"), subprocess.TimeoutExpired("nvidia-smi", 5), "Unknown GPU\n"]
)
def test_detection_answers_none_when_no_name_classifies(monkeypatch, smi):
    """MFU is then left out of the metrics rather than taking the run down."""
    _fake_device(monkeypatch, "GRID vGPU", smi=smi)
    assert detect_gpu_model() is None


@pytest.mark.parametrize("gpu_key", sorted(_DIE_SPECS))
def test_bf16_peak_does_not_exceed_architectural_ceiling(gpu_key):
    """A dense figure replaced by a sparsity (2x) or marketing multiplier exceeds what the die can retire."""
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
    entry = GPU_PEAK_FLOPS[gpu_key].flops
    floor = _bf16_ceiling(gpu_key) * 0.5
    assert entry["bf16"] >= floor, (
        f"{gpu_key} bf16 peak {entry['bf16'] / 1e12:.1f} TFLOP/s is below half its ceiling "
        f"{_bf16_ceiling(gpu_key) / 1e12:.1f} TFLOP/s — MFU computed against it would overstate."
    )


@pytest.mark.parametrize("gpu_key", sorted(GPU_PEAK_FLOPS))
def test_tensor_core_dtype_ratios_match_the_datapath(gpu_key):
    """Catches a single-dtype typo, and a row where only some dtypes were rescaled."""
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


@pytest.mark.parametrize("gpu_key", [key for key, spec in _DIE_SPECS.items() if spec[2] == "blackwell"])
def test_every_blackwell_die_has_an_fp4_peak(gpu_key):
    """The tensor cores run MXFP4 and NVFP4 on every Blackwell die; a missing entry leaves a native 4-bit run unscored."""
    assert "fp4" in GPU_PEAK_FLOPS[gpu_key].flops


def test_b300_matches_the_b200_dense_tensor_core_rates():
    """Same 148-SM die, since Blackwell Ultra's uplift is FP4 and HBM.
    A ~1.47x-inflated B300 row would understate every MFU by ~32%."""
    for dtype in ("bf16", "fp16", "fp8", "tf32"):
        assert GPU_PEAK_FLOPS["B300"].flops[dtype] == GPU_PEAK_FLOPS["B200"].flops[dtype], (
            f"B300 {dtype} peak diverges from B200's; both are the 148-SM Blackwell die and a "
            "measured large dense GEMM on B300 reaches ~80% of the B200 figure."
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
