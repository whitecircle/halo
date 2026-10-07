"""Structured perf/mem reporting for GPU tests and benchmarks.

Two jobs:

1. Snapshot an :class:`~src.callbacks.efficiency.EfficiencyCallback`
   into a flat, JSON-serializable dict with a stable schema. The **headline**
   metrics are hardware-anchored and unambiguous — tokens/s/GPU, cluster
   tokens/s, peak memory (GiB), step time. MFU / S-MFU / achieved-TFLOPS are
   kept only under a ``diagnostics`` sub-key: setup-dependent, easy to
   misread, never the gated number.

2. Emit a single machine-readable result line that the pytest launcher
   (``tests/gpu/conftest.py``) parses to tell a real FAIL (``status="fail"``)
   from an infra ERROR (process died, no line emitted). The line is prefixed
   with :data:`RESULT_SENTINEL` so it survives interleaved torchrun stdout.

Golden baselines (``tests/baselines/<key>.json``) carry a :func:`emit_benchmark`
payload. Nothing reads them automatically — the comparison is by hand, on the
headline metrics only (``agent-docs/contributing/README.md``).

Standard library only: the launcher process imports this module and stays free of torch.
"""

import json

RESULT_SENTINEL = "__HALO_TEST_RESULT__"
BENCH_SENTINEL = "__HALO_BENCH__"


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
    diagnostics = {
        "gpu_model": cb.state.gpu_model,
        "precision": cb.state.precision,
        "mfu_percent": cb.mfu.avg_mfu_percent,
        "smfu_percent": cb.smfu.avg_smfu_percent,
        "achieved_tflops_dense": cb.mfu.avg_tflops_per_sec,
        "achieved_tflops_active": cb.smfu.avg_smfu_tflops_per_sec,
        "local_params_b": round(cb.mfu.local_params / 1e9, 3),
        "sparsity_factor": cb.smfu.sparsity_factor,
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


def emit_benchmark(key: str, cb) -> None:
    """Print a parseable benchmark line so a refresh run can seed a golden baseline.

    ``key`` names the golden file ``tests/baselines/<key>.json`` (e.g.
    ``"sft_ep_gpt-oss-20b_ep8_s4096"``); the snapshot's headline ``tokens_per_second`` +
    ``peak_allocated_gb`` are what a refresh is compared against, by hand.
    """
    payload = {"key": key, "metrics": snapshot_efficiency(cb)}
    print(f"{BENCH_SENTINEL} {json.dumps(payload)}", flush=True)


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
    print(f"{RESULT_SENTINEL} {json.dumps(line)}", flush=True)


def parse_result(stdout: str) -> dict | None:
    """Extract the last emitted result line from captured ``stdout`` (launcher side)."""
    found = None
    for raw in stdout.splitlines():
        idx = raw.find(RESULT_SENTINEL)
        if idx != -1:
            try:
                found = json.loads(raw[idx + len(RESULT_SENTINEL) :].strip())
            except json.JSONDecodeError:
                continue
    return found
