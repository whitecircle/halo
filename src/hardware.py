"""Host and accelerator facts: architecture predicates, GPU model detection, the dense Tensor-Core
peak-FLOPS table, and available host RAM.

Two lookup mechanisms, deliberately: the kernel predicates read compute capability (what a kernel is
compiled for), while the FLOPS table is keyed by SKU name because peaks differ within one
architecture. MFU and S-MFU percentages divide by a table entry, so a wrong value rescales every
reported utilization.
"""

import subprocess
from dataclasses import dataclass

import torch
from transformers.utils import logging

logger = logging.get_logger(__name__)


def is_blackwell_gpu() -> bool:
    """Whether the local CUDA device is data-center Blackwell (SM100+), the FA4 architecture.

    Shared by the attention dispatch, the CP flash-attn resolver and the loader's CP substitution.
    ``>=`` rather than ``== 10`` so a future SM11x device is not demoted off FA4. The open upper
    bound also admits consumer Blackwell (SM 12.x), where FA4's SM100 kernels do not run; that is not
    a supported target, so bound it only if it becomes one.
    """
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


def is_hopper_gpu() -> bool:
    """Whether the local CUDA device is Hopper (SM90), the FA3 architecture."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


@dataclass(frozen=True)
class GpuPeak:
    """One GPU model's dense Tensor-Core peaks by precision, plus the name fragments identifying it.

    An SKU added to the table is classified through its own ``aliases``, with no change to
    :func:`_classify_gpu_name`: an alias matches when every whitespace-separated token of it appears
    in the reported name ("H100 PCI-E"), and the most specific alias wins.
    """

    aliases: tuple[str, ...]
    flops: dict[str, float]


# Dense Tensor Core TFLOPS (no structured sparsity). FP8: Hopper+; FP4: Blackwell only.
GPU_PEAK_FLOPS = {
    "H100_SXM": GpuPeak(
        aliases=("H100",),
        flops={
            "bf16": 989.5e12,
            "fp16": 989.5e12,
            "fp8": 1979.0e12,
            "tf32": 494.7e12,
            "fp32": 67.0e12,
        },
    ),
    "H100_PCIE": GpuPeak(
        aliases=("H100 PCIE", "H100 PCI-E"),
        flops={
            "bf16": 756.5e12,
            "fp16": 756.5e12,
            "fp8": 1513.0e12,
            "tf32": 378.0e12,
            "fp32": 51.0e12,
        },
    ),
    "H100_NVL": GpuPeak(
        aliases=("H100 NVL",),
        flops={
            "bf16": 835.5e12,
            "fp16": 835.5e12,
            "fp8": 1670.5e12,
            "tf32": 417.5e12,
            "fp32": 60.0e12,
        },
    ),
    "H200": GpuPeak(
        aliases=("H200",),
        flops={
            "bf16": 989.5e12,
            "fp16": 989.5e12,
            "fp8": 1979.0e12,
            "tf32": 494.7e12,
            "fp32": 67.0e12,
        },
    ),
    "B200": GpuPeak(
        aliases=("B200",),
        flops={
            "bf16": 2250.0e12,
            "fp16": 2250.0e12,
            "fp8": 4500.0e12,
            "tf32": 1125.0e12,
            "fp32": 75.0e12,
        },
    ),
    "GB200": GpuPeak(
        aliases=("GB200",),
        flops={
            "bf16": 2500.0e12,
            "fp16": 2500.0e12,
            "fp8": 5000.0e12,
            "tf32": 1250.0e12,
            "fp32": 80.0e12,
        },
    ),
    "B300": GpuPeak(  # same 148-SM die as B200; the uplift is FP4 and HBM, not bf16
        aliases=("B300",),
        flops={
            "bf16": 2250.0e12,
            "fp16": 2250.0e12,
            "fp8": 4500.0e12,
            "fp4": 9000.0e12,
            "tf32": 1125.0e12,
            "fp32": 75.0e12,
        },
    ),
    "A100": GpuPeak(  # same compute for SXM4 and PCIe
        aliases=("A100",),
        flops={
            "bf16": 312.0e12,
            "fp16": 312.0e12,
            "tf32": 156.0e12,
            "fp32": 19.5e12,
        },
    ),
    "A6000": GpuPeak(
        aliases=("A6000",),
        flops={
            "bf16": 154.8e12,
            "fp16": 154.8e12,
            "tf32": 77.4e12,
            "fp32": 38.7e12,
        },
    ),
}


def _classify_gpu_name(gpu_name_upper: str) -> str | None:
    """Map a GPU name string (upper-cased) to a key in GPU_PEAK_FLOPS via the entries' own aliases.

    Aliases are tried most-specific first (token count, then length), so ``H100 PCIE`` matches before
    the SXM entry's plain ``H100`` and ``GB200`` before the ``B200`` contained in its name.
    """
    candidates = sorted(
        ((key, tuple(alias.split())) for key, entry in GPU_PEAK_FLOPS.items() for alias in entry.aliases),
        key=lambda candidate: (len(candidate[1]), sum(len(token) for token in candidate[1])),
        reverse=True,
    )
    return next((key for key, tokens in candidates if all(token in gpu_name_upper for token in tokens)), None)


def detect_gpu_model() -> str | None:
    """Detect the GPU model name and return a key into GPU_PEAK_FLOPS."""
    if not torch.cuda.is_available():
        return None

    try:
        gpu_name = torch.cuda.get_device_name(0)
        result = _classify_gpu_name(gpu_name.upper())
        if result is not None:
            return result

        # nvidia-smi exposes vGPU / MIG names torch does not.
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if proc.returncode == 0:
                result = _classify_gpu_name(proc.stdout.strip().upper())
                if result is not None:
                    return result
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return None
    except Exception:
        logger.debug("GPU model detection failed", exc_info=True)
        return None


def get_gpu_peak_flops(gpu_model: str, precision: str) -> float | None:
    """Get peak FLOPS for the given GPU model and precision, or None when either is unknown."""
    entry = GPU_PEAK_FLOPS.get(gpu_model)
    return entry.flops.get(precision) if entry is not None else None


def available_host_ram_bytes() -> int | None:
    """``MemAvailable`` from ``/proc/meminfo`` in bytes, or ``None`` where it cannot be read (non-Linux)."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None
