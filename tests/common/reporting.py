"""Structured perf/mem reporting for GPU tests and benchmarks.

Three jobs:

1. Snapshot an :class:`~halo.callbacks.efficiency.EfficiencyCallback`
   into a flat, JSON-serializable dict with a stable schema. The **headline**
   metrics are hardware-anchored and unambiguous — tokens/s/GPU, cluster
   tokens/s, peak memory (GiB), step time. MFU / S-MFU / achieved-TFLOPS are
   kept only under a ``diagnostics`` sub-key: setup-dependent, easy to
   misread, never the gated number.

2. Emit a single machine-readable result line that the pytest launcher
   (``tests/gpu/conftest.py``) parses to tell a real FAIL (``status="fail"``)
   from an infra ERROR (process died, no line emitted). The line is prefixed
   with :data:`RESULT_SENTINEL` so it survives interleaved torchrun stdout.

3. Emit and parse the :data:`BENCH_SENTINEL` lines a benchmark prints, one per
   rank and key. The key names the golden ``tests/baselines/bench/<key>.json``
   that ``tests/common/equivalence/bench_band.py`` holds a ``make bench`` run to.

Standard library only: the launcher process imports this module and stays free of torch.
"""

import json
import os
import shlex
import statistics
import sys
from collections.abc import Iterator

RESULT_SENTINEL = "__HALO_TEST_RESULT__"
BENCH_SENTINEL = "__HALO_BENCH__"
# A benchmark line's metrics carry two decimals, as the EfficiencyCallback rounds them.
BENCH_DECIMALS = 2


def snapshot_efficiency(cb) -> dict:
    """Flatten an ``EfficiencyCallback`` into a baseline-ready dict.

    Headline metrics live at the top level; MFU/S-MFU/TFLOPS diagnostics are
    nested under ``"diagnostics"`` so a golden-value diff never gates on them.
    """
    headline = {
        "tokens_per_second": cb.tps.avg_tokens_per_second,  # per-GPU — the headline golden-gate metric
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
    """Render the headline metrics as a small aligned table for the log."""
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
    """Render an ``EfficiencyCallback`` as a benchmark report.

    Headline is the hardware-anchored set (tokens/s/GPU, cluster tokens/s, peak
    memory, step time); MFU / S-MFU / achieved-TFLOPS follow under a clearly
    labelled ``diagnostics`` block (setup-dependent — not the comparison metric).
    """
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
    """Print ``payload`` after ``sentinel`` in one write. torchrun starts its workers unbuffered, where ``print``
    writes the newline on its own and lines the ranks print at once run into each other; a line written whole
    stays whole on a pipe up to ``PIPE_BUF`` bytes."""
    sys.stdout.write(f"{sentinel} {json.dumps(payload)}\n")
    sys.stdout.flush()


def _emit_bench_line(key: str, metrics: dict) -> None:
    """Print this rank's benchmark line for ``key``: the rank, the world size and the command that measured
    ``metrics``, so a baseline records how it was measured and a run is compared only with its own row."""
    line = {
        "key": key,
        "rank": int(os.environ.get("RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "command": shlex.join(sys.argv),
        "metrics": metrics,
    }
    _print_sentinel_line(BENCH_SENTINEL, line)


def emit_benchmark(key: str, cb) -> None:
    """Print this rank's training-throughput line for ``key``, the golden ``tests/baselines/bench/<key>.json``
    (e.g. ``"sft_ep_gpt-oss-20b_ep8_s4096"``): the :func:`snapshot_efficiency` of ``cb``, plus the mean and the
    median of its measured step times in milliseconds, fine enough for the band to tell a mean a few stalled
    steps moved from a slower steady step."""
    step_ms = [seconds * 1e3 for seconds in cb.measured_step_times]
    metrics = {
        **snapshot_efficiency(cb),
        "mean_step_time_ms": round(statistics.fmean(step_ms), BENCH_DECIMALS),
        "median_step_time_ms": round(statistics.median(step_ms), BENCH_DECIMALS),
    }
    _emit_bench_line(key, metrics)


def emit_save_benchmark(key: str, seconds: float, peak_rss_gib: float) -> None:
    """Print this rank's checkpoint-save line for ``key``: the save's wall time and this rank's peak resident host
    memory while it ran."""
    metrics = {"save_seconds": seconds, "save_peak_rss_gib": peak_rss_gib}
    _emit_bench_line(key, {name: round(value, BENCH_DECIMALS) for name, value in metrics.items()})


def emit_result(
    status: str,
    checks: dict | None = None,
    metrics: dict | None = None,
    error: str | None = None,
) -> None:
    """Print the single machine-readable result line the launcher parses.

    ``status`` is one of ``"pass" | "fail" | "error"``. ``checks`` is the
    name→bool map of assertions; ``metrics`` is an optional headline snapshot;
    ``error`` carries an exception summary when ``status == "error"``.
    """
    line = {
        "status": status,
        "checks": checks or {},
        "metrics": metrics or {},
    }
    if error:
        line["error"] = error
    _print_sentinel_line(RESULT_SENTINEL, line)


def _sentinel_payloads(stdout: str, sentinel: str) -> Iterator[str]:
    """The text after ``sentinel`` on each line of ``stdout`` that carries it, in order."""
    for raw in stdout.splitlines():
        idx = raw.find(sentinel)
        if idx != -1:
            yield raw[idx + len(sentinel) :].strip()


def parse_result(stdout: str) -> dict | None:
    """Extract the last emitted result line from captured ``stdout`` (launcher side)."""
    found = None
    for payload in _sentinel_payloads(stdout, RESULT_SENTINEL):
        try:
            found = json.loads(payload)
        except json.JSONDecodeError:
            continue
    return found


def parse_benchmarks(stdout: str) -> list[dict]:
    """Every benchmark line in ``stdout``, in order. A line that does not parse raises: a lost line would drop one
    rank's measurement from the run it belongs to."""
    lines = []
    for payload in _sentinel_payloads(stdout, BENCH_SENTINEL):
        try:
            lines.append(json.loads(payload))
        except json.JSONDecodeError as error:
            raise ValueError(f"unreadable benchmark line {payload[:200]!r}: {error}") from error
    return lines
