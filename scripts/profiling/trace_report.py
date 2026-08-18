"""Generate TraceLens analysis reports for the torch.profiler traces this toolkit writes.

Turns the per-rank Chrome traces produced by ``TorchProfilerCallback`` (``enable_torch_profiler: true``)
or ``torch_profiler_session`` — ``<label>-rankNN.trace.json[.gz]`` under the profiling output dir —
into analysis workbooks, written next to the traces:

  - ``<label>-rankNN.tracelens.xlsx`` — per-rank perf report: hierarchical GPU timeline
    (compute / communication / idle attribution), unique-op launcher tables, per-op roofline
    placement, and a collective-analysis sheet.
  - ``<label>.collective.xlsx`` — for multi-rank trace sets (``ranks="all"`` captures): NCCL
    collective latency / bus bandwidth / cross-rank skew, the straggler-triage view.

Uses `TraceLens <https://github.com/AMD-AGI/TraceLens>`_ (the ``profiling`` dependency group — in the
training images by default). For per-report knobs beyond this wrapper (short-kernel studies, CSV
output, report diffs) call the TraceLens CLIs directly: ``TraceLens_generate_perf_report_pytorch``,
``TraceLens_compare_perf_reports_pytorch``.

Usage (inside the training image):
  python scripts/profiling/trace_report.py                      # $HALO_DATA_ROOT/profiling/torch, all sets
  python scripts/profiling/trace_report.py <trace_dir> --label trace-cycle1
  python scripts/profiling/trace_report.py --no-collective      # skip the multi-rank NCCL report
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import re
import tempfile
from pathlib import Path

from src.env import torch_trace_dir
from src.log import configure_cli_logging

configure_cli_logging()
logger = logging.getLogger(__name__)

# Artifact naming contract of src/diagnostics/profiling.py::export_profiler_artifacts:
# "<label>-rank<NN>.trace.json[.gz]" with the rank zero-padded to the world-size width.
_TRACE_NAME_RE = re.compile(r"^(?P<label>.+)-rank(?P<rank>\d+)\.trace(?P<suffix>\.json(?:\.gz)?)$")

# The perf report hangs every GPU kernel off the cpu_op that launched it, so a CUDA-only capture
# carries no tree to report on. Detected by scanning for the category rather than parsing the whole
# trace, since these files reach hundreds of MB decompressed.
_CPU_OP_MARKER = b'"cpu_op"'
_SCAN_CHUNK = 1 << 20


def discover_trace_sets(trace_dir: Path) -> dict[str, dict[int, Path]]:
    """Group the trace files in ``trace_dir`` into per-label rank sets.

    Returns ``{label: {rank: path}}`` for every file matching the exporter's naming contract,
    labels and ranks sorted. Non-matching files are ignored.
    """
    sets: dict[str, dict[int, Path]] = {}
    for path in sorted(trace_dir.iterdir()):
        match = _TRACE_NAME_RE.match(path.name)
        if match is None or not path.is_file():
            continue
        sets.setdefault(match["label"], {})[int(match["rank"])] = path
    return {label: dict(sorted(ranks.items())) for label, ranks in sorted(sets.items())}


def has_cpu_ops(trace_path: Path) -> bool:
    """Whether the trace holds CPU-side op events, i.e. was not a CUDA-only capture."""
    opener = gzip.open if trace_path.suffix == ".gz" else open
    carry = b""
    with opener(trace_path, "rb") as handle:
        while chunk := handle.read(_SCAN_CHUNK):
            if _CPU_OP_MARKER in carry + chunk:
                return True
            carry = chunk[-len(_CPU_OP_MARKER) :]  # a marker split across two reads
    return False


def write_rank_report(trace_path: Path) -> Path:
    """Run the TraceLens per-trace perf report for one rank's trace; return the .xlsx path."""
    match = _TRACE_NAME_RE.match(trace_path.name)
    if match is None:
        raise ValueError(f"not a profiler trace artifact: {trace_path.name}")
    if not has_cpu_ops(trace_path):
        raise ValueError(
            f"{trace_path.name} holds no cpu_op events — a CUDA-only capture, as produced by "
            "TorchProfilerCallback(cpu_activity=False). The perf report attributes every kernel to "
            "the cpu_op that launched it, so re-capture with cpu_activity left at its default."
        )
    # Deferred import: TraceLens is an optional (profiling-group) dependency; keep discovery and
    # --help usable, with a pointed error only when a report is actually requested.
    from TraceLens.Reporting.generate_perf_report_pytorch import (  # noqa: PLC0415 — optional profiling-group dep, kept out of --help
        generate_perf_report_pytorch,
    )

    out_path = trace_path.parent / f"{match['label']}-rank{match['rank']}.tracelens.xlsx"
    generate_perf_report_pytorch(profile_json_path=str(trace_path), output_xlsx_path=str(out_path))
    return out_path


def write_collective_report(label: str, ranks: dict[int, Path]) -> Path | None:
    """Run the TraceLens multi-rank NCCL collective report for one trace set.

    TraceLens expands its ``trace_pattern`` placeholder with unpadded ranks ``0..world_size-1``,
    while our artifacts are zero-padded — bridge with consecutively-named symlinks. Skips (with a
    warning) sets whose ranks are not dense from 0, e.g. a ``ranks="0,8"`` capture.
    """
    if sorted(ranks) != list(range(len(ranks))):
        logger.warning("[%s] ranks %s not dense from 0 — skipping collective report", label, sorted(ranks))
        return None
    suffixes = {m["suffix"] for p in ranks.values() if (m := _TRACE_NAME_RE.match(p.name))}
    if len(suffixes) != 1:
        logger.warning("[%s] mixed trace suffixes %s — skipping collective report", label, sorted(suffixes))
        return None
    from TraceLens.Reporting.generate_multi_rank_collective_report_pytorch import (  # noqa: PLC0415 — optional profiling-group dep, kept out of --help
        generate_collective_report,
    )

    suffix = suffixes.pop()
    out_path = next(iter(ranks.values())).parent / f"{label}.collective.xlsx"
    with tempfile.TemporaryDirectory(prefix="tracelens-") as tmp:
        for rank, path in ranks.items():
            os.symlink(path.resolve(), Path(tmp) / f"trace_rank_{rank}{suffix}")
        generate_collective_report(
            trace_pattern=str(Path(tmp) / f"trace_rank_*{suffix}"),
            world_size=len(ranks),
            output_xlsx_path=str(out_path),
        )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "trace_dir",
        nargs="?",
        default=torch_trace_dir(),
        type=Path,
        help="Directory with <label>-rankNN.trace.json[.gz] traces (default: %(default)s).",
    )
    parser.add_argument("--label", default=None, help="Only process trace sets with this label.")
    parser.add_argument(
        "--no-collective",
        action="store_true",
        help="Skip the multi-rank NCCL collective report for multi-rank trace sets.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Exit 0 even when some trace sets produced no report (default: any failure is fatal).",
    )
    args = parser.parse_args()

    if not args.trace_dir.is_dir():
        parser.error(f"trace dir does not exist: {args.trace_dir}")
    trace_sets = discover_trace_sets(args.trace_dir)
    if args.label is not None:
        trace_sets = {label: ranks for label, ranks in trace_sets.items() if label == args.label}
    if not trace_sets:
        parser.error(f"no traces matching '<label>-rankNN.trace.json[.gz]' found in {args.trace_dir}")

    # Trace sets are independent, and one unreportable capture (a CUDA-only trace, a truncated file
    # from a killed run) must not cost the reports of every other label in the directory.
    failed = []
    for label, ranks in trace_sets.items():
        logger.info("[%s] %d rank trace(s)", label, len(ranks))
        try:
            for rank, path in ranks.items():
                logger.info("[%s] rank %d perf report → %s", label, rank, write_rank_report(path))
            if len(ranks) > 1 and not args.no_collective:
                collective = write_collective_report(label, ranks)
                if collective is not None:
                    logger.info("[%s] collective report → %s", label, collective)
        except Exception as exc:
            failed.append(label)
            logger.error("[%s] report failed: %s", label, exc)

    if failed:
        logger.error("%d of %d trace set(s) failed: %s", len(failed), len(trace_sets), ", ".join(failed))
        if not args.keep_going:
            # Exiting 0 on a partial run reports missing workbooks as generated ones: the caller sees
            # success, then analyses whichever labels happened to survive without knowing any are gone.
            raise SystemExit(
                f"{len(failed)} of {len(trace_sets)} trace set(s) produced no report "
                f"({', '.join(failed)}); the failures are logged above. Pass --keep-going to accept "
                f"a partial run."
            )


if __name__ == "__main__":
    main()
