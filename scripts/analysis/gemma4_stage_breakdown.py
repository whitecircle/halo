#!/usr/bin/env python
"""Per-stage GPU time of one training step from a torch.profiler Chrome trace (rank 0).

Kernels are grouped by name into stages; every kernel between a step's gradient-norm kernel and its last
Adam kernel counts as the optimizer step (that window holds the clip). Communication overlaps compute, so
the per-stage sums are scaled to the GPU-busy time, and the measured step time minus GPU-busy time is
reported as idle (CPU-bound launches and host syncs).

Usage: python scripts/analysis/gemma4_stage_breakdown.py <trace.json[.gz]> <measured_step_ms> <profiled_steps>
"""

import collections
import gzip
import json
import re
import sys

STAGES = (
    (r"nccl", "FSDP communication"),
    (r"deep_ep", "Expert dispatch/combine (DeepEP)"),
    (r"fmha|flex_attention|attention|softmax", "Attention"),
    (r"GroupProblemShape|grouped", "Expert GEMMs"),
    (r"_glu_fwd_kernel|_glu_bwd_kernel|geglu|gelu|swiglu", "GeGLU activation"),
    (r"rms_norm|layer_norm|GammaBeta|LayerNorm|RMSNorm", "RMSNorm"),
    (r"cross_entropy|CrossEntropy|log_softmax|nll_loss", "Loss"),
    (
        r"gather|scatter|index|radix|sort|CatArray|chunk_cat|split_with_sizes|_gather_reduce|"
        r"_weighted_unpermute|histogram|cumsum|DeviceScan",
        "Expert token movement",
    ),
    (r"nvjet|cublas|gemm|cutlass|xmma|sm90_|sm100_", "Dense GEMMs"),
)
OPTIMIZER = "Optimizer step (clip + AdamW)"
IDLE = "GPU idle (CPU-bound launches, host syncs)"


def stage_of(name: str) -> str:
    for pattern, stage in STAGES:
        if re.search(pattern, name, re.IGNORECASE):
            return stage
    return "Other elementwise"


def optimizer_windows(kernels: list[dict]) -> list[tuple[float, float]]:
    """[first grad-norm kernel, end of the last Adam kernel] per optimizer step."""
    windows, i = [], 0
    while i < len(kernels):
        if "LpNorm" not in kernels[i]["name"]:
            i += 1
            continue
        last, j = None, i
        while j < len(kernels):
            name = kernels[j]["name"]
            if last is not None and "LpNorm" in name and kernels[j]["ts"] > kernels[last]["ts"] + 50_000:
                break
            if "_adam" in name:
                last = j
            j += 1
        if last is None:
            i += 1
            continue
        windows.append((kernels[i]["ts"], kernels[last]["ts"] + kernels[last]["dur"]))
        i = last + 1
    return windows


def busy_ms(kernels: list[dict]) -> float:
    """Union of kernel intervals over all streams, in ms."""
    total, start, end = 0.0, None, None
    for a, b in sorted((e["ts"], e["ts"] + e["dur"]) for e in kernels if e.get("cat") == "kernel"):
        if end is None or a > end:
            if end is not None:
                total += end - start
            start, end = a, b
        else:
            end = max(end, b)
    return (total + (end - start if end is not None else 0.0)) / 1e3


def main() -> None:
    path, step_ms, steps = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
        trace = json.load(f)
    kinds = ("kernel", "gpu_memcpy", "gpu_memset")
    kernels = sorted(
        (e for e in trace["traceEvents"] if e.get("ph") == "X" and e.get("cat") in kinds), key=lambda e: e["ts"]
    )
    windows = optimizer_windows(kernels)
    stages = collections.Counter()
    for e in kernels:
        in_optimizer = any(a <= e["ts"] <= b for a, b in windows) and "nccl" not in e["name"].lower()
        stages[OPTIMIZER if in_optimizer else stage_of(e["name"])] += e["dur"] / 1e3 / steps
    busy = busy_ms(kernels) / steps
    scale = busy / sum(stages.values())
    bar = {stage: round(ms * scale, 2) for stage, ms in stages.most_common()}
    bar[IDLE] = round(max(step_ms - busy, 0.0), 2)
    result = {
        "step_ms": step_ms,
        "gpu_busy_ms": busy,
        "optimizer_windows": len(windows),
        "stages_ms": {stage: round(ms, 2) for stage, ms in stages.most_common()},
        "bar_ms": bar,
    }
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
