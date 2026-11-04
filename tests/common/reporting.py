"""Structured perf and memory reporting for GPU tests and benchmarks: a flat metrics snapshot, the launcher's result
line, benchmark lines. Standard library only: the launcher process imports this module and stays free of torch."""

import json
import os
import shlex
import statistics
import sys
from collections.abc import Iterator

RESULT_SENTINEL = "__HALO_TEST_RESULT__"
BENCH_SENTINEL = "__HALO_BENCH__"
# As the EfficiencyCallback rounds them.
BENCH_DECIMALS = 2


def snapshot_efficiency(cb) -> dict:
    """Flatten an ``EfficiencyCallback``; MFU/S-MFU/TFLOPS nest under ``"diagnostics"`` so a golden-value diff never
    gates on them."""
    headline = {
        "tokens_per_second": cb.tps.avg_tokens_per_second,  # Per GPU, the headline golden-gate metric.
        "cluster_tokens_per_second": cb.tps.avg_cluster_tokens_per_second,
        "peak_allocated_gb": cb.memory.peak_allocated_gb,
        "training_peak_allocated_gb": cb.memory.training_peak_allocated_gb,
        "avg_step_time_seconds": cb.time.avg_step_time_seconds,
    }
    flops = cb.flops_model
    if flops is not None:
        local_params, sparsity_factor = flops.local_params, flops.sparsity_factor
    else:
        # A FLOPs estimate that raised leaves the rank's parameter count to the model itself.
        local_params = sum(p.numel() for p in cb.model_ref.parameters()) if cb.model_ref is not None else 0
        sparsity_factor = 1.0
    diagnostics = {
        "gpu_model": cb.gpu_model,
        "precision": cb.precision,
        "mfu_percent": cb.mfu.avg_mfu_percent,
        "smfu_percent": cb.smfu.avg_smfu_percent,
        "achieved_tflops_dense": cb.mfu.avg_tflops_per_sec,
        "achieved_tflops_active": cb.smfu.avg_smfu_tflops_per_sec,
        "local_params_b": round(local_params / 1e9, 3),
        "sparsity_factor": sparsity_factor,
    }
    return {**headline, "diagnostics": diagnostics}


def format_table(payload: dict) -> str:
    rows = [
        ("tokens/s/GPU", payload.get("tokens_per_second")),
        ("cluster tokens/s", payload.get("cluster_tokens_per_second")),
        ("peak mem (GiB)", payload.get("peak_allocated_gb")),
        ("avg step time (s)", payload.get("avg_step_time_seconds")),
    ]
    width = max(len(label) for label, _ in rows)
    lines = ["  " + label.ljust(width) + " : " + str(value) for label, value in rows]
    return "\n".join(lines)


def format_benchmark_report(cb) -> str:
    """Render an ``EfficiencyCallback`` as a benchmark report, headline first and setup-dependent diagnostics below."""
    snap = snapshot_efficiency(cb)
    diag = snap["diagnostics"]
    lines = [
        "  --- throughput (headline) ---",
        f"  tokens/s/GPU      : {snap['tokens_per_second']}",
        f"  cluster tokens/s  : {snap['cluster_tokens_per_second']}",
        f"  peak memory (GiB) : {snap['peak_allocated_gb']}",
        f"  avg step time (s) : {snap['avg_step_time_seconds']}",
        "  --- diagnostics (setup-dependent; not gated) ---",
        f"  MFU %             : {diag['mfu_percent']}",
        f"  S-MFU %           : {diag['smfu_percent']}",
        f"  TFLOPS (dense)    : {diag['achieved_tflops_dense']}",
        f"  TFLOPS (active)   : {diag['achieved_tflops_active']}",
        f"  GPU / precision   : {diag['gpu_model']} / {diag['precision']}",
    ]
    return "\n".join(lines)


def _print_sentinel_line(sentinel: str, payload: dict) -> None:
    """Print ``payload`` after ``sentinel`` in one write: unbuffered ``print`` writes the newline apart, and ranks'
    lines would run together."""
    sys.stdout.write(f"{sentinel} {json.dumps(payload)}\n")
    sys.stdout.flush()


def _emit_bench_line(key: str, metrics: dict) -> None:
    """Print this rank's benchmark line, with rank, world size and command, so a run is compared only with its own
    row."""
    line = {
        "key": key,
        "rank": int(os.environ.get("RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "command": shlex.join(sys.argv),
        "metrics": metrics,
    }
    _print_sentinel_line(BENCH_SENTINEL, line)


def emit_benchmark(key: str, cb) -> None:
    """Print this rank's throughput line for ``key``: the snapshot plus mean and median step ms, so the band tells
    stalled steps from a slower steady step."""
    step_ms = [seconds * 1e3 for seconds in cb.measured_step_times]
    metrics = {
        **snapshot_efficiency(cb),
        "mean_step_time_ms": round(statistics.fmean(step_ms), BENCH_DECIMALS),
        "median_step_time_ms": round(statistics.median(step_ms), BENCH_DECIMALS),
    }
    _emit_bench_line(key, metrics)


def emit_save_benchmark(key: str, seconds: float, peak_rss_gib: float) -> None:
    """Print this rank's checkpoint-save line: wall time and peak resident host memory."""
    metrics = {"save_seconds": seconds, "save_peak_rss_gib": peak_rss_gib}
    _emit_bench_line(key, {name: round(value, BENCH_DECIMALS) for name, value in metrics.items()})


def emit_result(
    status: str,
    checks: dict | None = None,
    metrics: dict | None = None,
    error: str | None = None,
) -> None:
    """Print the single machine-readable result line the launcher parses; ``status`` is ``"pass" | "fail" |
    "error"``."""
    line = {
        "status": status,
        "checks": checks or {},
        "metrics": metrics or {},
    }
    if error:
        line["error"] = error
    _print_sentinel_line(RESULT_SENTINEL, line)


def _sentinel_payloads(stdout: str, sentinel: str) -> Iterator[str]:
    for raw in stdout.splitlines():
        idx = raw.find(sentinel)
        if idx != -1:
            yield raw[idx + len(sentinel) :].strip()


def parse_result(stdout: str) -> dict | None:
    found = None
    for payload in _sentinel_payloads(stdout, RESULT_SENTINEL):
        try:
            found = json.loads(payload)
        except json.JSONDecodeError:
            continue
    return found


def parse_benchmarks(stdout: str) -> list[dict]:
    """Every benchmark line in order; one that does not parse raises, since it would drop a rank's measurement."""
    lines = []
    for payload in _sentinel_payloads(stdout, BENCH_SENTINEL):
        try:
            lines.append(json.loads(payload))
        except json.JSONDecodeError as error:
            raise ValueError(f"unreadable benchmark line {payload[:200]!r}: {error}") from error
    return lines
