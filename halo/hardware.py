"""Host and accelerator facts. Kernel predicates read compute capability; the FLOPS table is keyed by SKU name, because
peaks differ within one architecture."""

import logging
import math
import os
import subprocess
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

# v2's unified hierarchy at the root, v1's cpu controller under cpu/.
_CGROUP_ROOT = "/sys/fs/cgroup"
# One `<id>:<controllers>:<path>` line per hierarchy (v2's has no controllers).
_PROC_SELF_CGROUP = "/proc/self/cgroup"


def current_device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")


def is_blackwell_gpu() -> bool:
    """Data-center Blackwell (SM100+), the FA4 architecture; open-ended so a later SM keeps FA4."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


def is_hopper_gpu() -> bool:
    """Hopper (SM90), the FA3 architecture."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


def supports_grouped_mm() -> bool:
    """SM90+ compute capability. A failing read propagates: the answer picks the expert weight layout, and a rank
    answering differently from its peers would hang the expert all-gather."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major * 10 + minor) >= 90


@dataclass(frozen=True)
class GpuPeak:
    """One GPU's dense Tensor-Core peaks by precision and its name aliases; an alias matches when every token of it is
    in the reported name, and the most specific wins."""

    aliases: tuple[str, ...]
    flops: dict[str, float]


# Dense Tensor Core peaks in FLOP/s (no structured sparsity). FP8: Hopper+; FP4: Blackwell only.
GPU_PEAK_FLOPS = {
    "H100_SXM": GpuPeak(
        aliases=("H100",),
        flops={"bf16": 989.5e12, "fp16": 989.5e12, "fp8": 1979.0e12, "tf32": 494.7e12, "fp32": 67.0e12},
    ),
    "H100_PCIE": GpuPeak(
        aliases=("H100 PCIE", "H100 PCI-E"),
        flops={"bf16": 756.5e12, "fp16": 756.5e12, "fp8": 1513.0e12, "tf32": 378.0e12, "fp32": 51.0e12},
    ),
    "H100_NVL": GpuPeak(
        aliases=("H100 NVL",),
        flops={"bf16": 835.5e12, "fp16": 835.5e12, "fp8": 1670.5e12, "tf32": 417.5e12, "fp32": 60.0e12},
    ),
    "H200": GpuPeak(
        aliases=("H200",),
        flops={"bf16": 989.5e12, "fp16": 989.5e12, "fp8": 1979.0e12, "tf32": 494.7e12, "fp32": 67.0e12},
    ),
    "B200": GpuPeak(
        aliases=("B200",),
        flops={"bf16": 2250.0e12, "fp16": 2250.0e12, "fp8": 4500.0e12, "tf32": 1125.0e12, "fp32": 75.0e12},
    ),
    "GB200": GpuPeak(
        aliases=("GB200",),
        flops={"bf16": 2500.0e12, "fp16": 2500.0e12, "fp8": 5000.0e12, "tf32": 1250.0e12, "fp32": 80.0e12},
    ),
    "B300": GpuPeak(  # same 148-SM die as B200; FP4 and HBM are the uplift
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
        flops={"bf16": 312.0e12, "fp16": 312.0e12, "tf32": 156.0e12, "fp32": 19.5e12},
    ),
    "A6000": GpuPeak(
        aliases=("A6000",),
        flops={"bf16": 154.8e12, "fp16": 154.8e12, "tf32": 77.4e12, "fp32": 38.7e12},
    ),
}
PEAK_KEY_BY_DTYPE = {torch.bfloat16: "bf16", torch.float16: "fp16"}


def _classify_gpu_name(gpu_name_upper: str) -> str | None:
    """Most specific alias first (``H100 PCIE`` before ``H100``, ``GB200`` before ``B200``)."""
    candidates = sorted(
        ((key, tuple(alias.split())) for key, entry in GPU_PEAK_FLOPS.items() for alias in entry.aliases),
        key=lambda candidate: (len(candidate[1]), sum(len(token) for token in candidate[1])),
        reverse=True,
    )
    return next((key for key, tokens in candidates if all(token in gpu_name_upper for token in tokens)), None)


def detect_gpu_model() -> str | None:
    if not torch.cuda.is_available():
        return None

    try:
        result = _classify_gpu_name(torch.cuda.get_device_name(0).upper())
        if result is not None:
            return result
        # nvidia-smi exposes vGPU / MIG names torch does not.
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return _classify_gpu_name(proc.stdout.strip().upper()) if proc.returncode == 0 else None
    except Exception:
        logger.debug("GPU model detection failed", exc_info=True)
        return None


def get_gpu_peak_flops(gpu_model: str, precision: str) -> float | None:
    entry = GPU_PEAK_FLOPS.get(gpu_model)
    return entry.flops.get(precision) if entry is not None else None


def _fp32_compute_precision() -> str:
    """``float32_matmul_precision`` routes fp32 matmuls to Tensor Cores (``high`` → TF32, ``medium`` → bf16), whose
    higher peak an ``"fp32"`` key would report MFU above."""
    precision = torch.get_float32_matmul_precision()
    if precision == "medium":
        return "bf16"
    if precision == "high" or torch.backends.cuda.matmul.allow_tf32:
        return "tf32"
    return "fp32"


def peak_precision_key(dtype: torch.dtype | None) -> str | None:
    if dtype == torch.float32:
        return _fp32_compute_precision()
    return PEAK_KEY_BY_DTYPE.get(dtype)


def available_host_ram_bytes() -> int | None:
    """``MemAvailable`` in bytes, or ``None`` where it cannot be read (non-Linux)."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _read_cgroup_file(path: str) -> str | None:
    """``path`` stripped, or ``None`` where unreadable (other cgroup version, a level the mount hides, no cgroup)."""
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def _own_cgroup_dirs() -> list[str]:
    """The directories of this process's cgroup and every ancestor up to the mount, in v2's unified hierarchy and
    v1's cpu controller. A level the mount does not show reads as no quota, so the walk still reaches the mount's
    root: v1 Docker mounts the container's own cgroup there, which ``/proc/self/cgroup`` names by its host path."""
    dirs = []
    for line in (_read_cgroup_file(_PROC_SELF_CGROUP) or "").splitlines():
        _, controllers, path = line.split(":", 2)
        if controllers and "cpu" not in controllers.split(","):
            continue
        mount = os.path.join(_CGROUP_ROOT, "cpu") if controllers else _CGROUP_ROOT
        parts = [part for part in path.split("/") if part]
        dirs += [os.path.join(mount, *parts[:depth]) for depth in range(len(parts) + 1)]
    return dirs


def _cpu_quota_at(directory: str) -> int | None:
    """CPUs the cgroup at ``directory`` allows, rounded up, or ``None``: v2's ``cpu.max`` (``"<quota> <period>"``,
    ``max`` when unlimited), else v1's ``cpu.cfs_quota_us`` (``-1`` when unlimited) over ``cpu.cfs_period_us``."""
    cpu_max = _read_cgroup_file(os.path.join(directory, "cpu.max"))
    if cpu_max is not None:
        quota, period = cpu_max.split()
    else:
        quota = _read_cgroup_file(os.path.join(directory, "cpu.cfs_quota_us"))
        period = _read_cgroup_file(os.path.join(directory, "cpu.cfs_period_us"))
    if quota is None or period is None or quota == "max" or int(quota) < 0:
        return None
    return math.ceil(int(quota) / int(period))


def _cgroup_cpu_quota() -> int | None:
    """The tightest quota on this process's cgroup and its ancestors, since a parent's quota (a Kubernetes pod, a
    systemd slice) binds every cgroup below it."""
    quotas = (_cpu_quota_at(directory) for directory in _own_cgroup_dirs())
    return min((quota for quota in quotas if quota is not None), default=None)


def available_cpu_count() -> int:
    """Affinity set (a container cpuset, a NUMA binding), else the host count, capped by the cgroup quotas
    (``docker --cpus``, a Kubernetes CPU limit, a systemd ``CPUQuota``), which affinity does not show."""
    count = max(1, len(os.sched_getaffinity(0))) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    quota = _cgroup_cpu_quota()
    return count if quota is None else min(count, quota)
