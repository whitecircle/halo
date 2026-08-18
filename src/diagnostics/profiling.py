"""Opt-in profiling helpers: operator/kernel traces, CUDA memory snapshots, and summaries.

Artifacts are rank-tagged so per-rank files never collide. Only global rank 0 profiles by
default (``ranks="0"``); ``ranks="all"`` captures every rank for rank-skew triage.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch.profiler import ProfilerActivity, profile

from src.distributed.runtime import (
    current_device,
    get_global_rank,
    rank_tag,
)
from src.env import memory_snapshot_dir, torch_trace_dir

logger = logging.getLogger(__name__)

__all__ = [
    "should_profile_this_rank",
    "ensure_artifact_dir",
    "log_cuda_memory",
    "reset_peak_memory_stats",
    "start_memory_history",
    "stop_memory_history",
    "dump_memory_snapshot",
    "cuda_memory_history",
    "torch_profiler_session",
    "export_profiler_artifacts",
]


def _parse_rank_spec(spec: str | int | None) -> set[int] | None:
    """Parse a rank-selection spec into a set of global ranks.

    ``"all"`` → ``None`` (every rank); int/str (``"0"``); comma list (``"0,8"``);
    ``None``/empty → ``{0}`` (the safe default).
    """
    if spec is None or spec == "":
        return {0}
    if isinstance(spec, int):
        return {spec}
    spec = str(spec).strip().lower()
    if spec == "all":
        return None
    return {int(p) for p in spec.split(",") if p.strip() != ""}


def should_profile_this_rank(spec: str | int | None) -> bool:
    """Whether the current global rank is selected by ``spec``."""
    selected = _parse_rank_spec(spec)
    if selected is None:
        return True
    return get_global_rank() in selected


def ensure_artifact_dir(output_dir: str) -> None:
    """Create ``output_dir`` for this rank's artifacts.

    Not ``fs_aware_makedirs``: its barrier would deadlock against the ranks that the ``ranks``
    selection returned early on.
    """
    os.makedirs(output_dir, exist_ok=True)


def reset_peak_memory_stats() -> None:
    """Reset the CUDA peak-memory counters (call at the start of a region of interest)."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def log_cuda_memory(tag: str = "") -> dict[str, float]:
    """Log and return this rank's current/peak CUDA memory (GiB).

    Peak counters are left running; :func:`reset_peak_memory_stats` scopes them to a region.
    """
    if not torch.cuda.is_available():
        return {}
    gib = 1024.0**3
    stats = {
        "allocated_gib": torch.cuda.memory_allocated() / gib,
        "reserved_gib": torch.cuda.memory_reserved() / gib,
        "max_allocated_gib": torch.cuda.max_memory_allocated() / gib,
        "max_reserved_gib": torch.cuda.max_memory_reserved() / gib,
    }
    logger.info(
        "[mem%s] %s: allocated=%.2f reserved=%.2f peak_alloc=%.2f peak_reserved=%.2f GiB",
        f" {rank_tag()}",
        tag or "cuda",
        stats["allocated_gib"],
        stats["reserved_gib"],
        stats["max_allocated_gib"],
        stats["max_reserved_gib"],
    )
    return stats


def start_memory_history(max_entries: int = 100_000) -> bool:
    """Begin recording CUDA allocation history (call sites + stacks).

    Returns True if started, False when CUDA or the recorder API is unavailable. Recording has
    measurable overhead, so scope it to a few steps.
    """
    if not torch.cuda.is_available():
        return False
    torch.cuda.memory._record_memory_history(max_entries=max_entries)
    return True


def stop_memory_history() -> None:
    """Stop CUDA allocation history recording."""
    if not torch.cuda.is_available():
        return
    torch.cuda.memory._record_memory_history(enabled=None)


def dump_memory_snapshot(path: str) -> str | None:
    """Dump the recorded CUDA memory snapshot to ``path`` (a ``.pickle``).

    Load at https://pytorch.org/memory_viz. Returns the path written, or None on failure.
    """
    if not torch.cuda.is_available():
        return None
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        torch.cuda.memory._dump_snapshot(path)
    except Exception as exc:  # diagnostics must never crash training
        logger.warning("Failed to dump CUDA memory snapshot to %s: %s", path, exc)
        return None
    logger.info("[mem %s] dumped CUDA memory snapshot → %s", rank_tag(), path)
    return path


@contextmanager
def cuda_memory_history(
    output_dir: str | None = None,
    *,
    ranks: str | int | None = "0",
    max_entries: int = 100_000,
    label: str = "snapshot",
) -> Iterator[None]:
    """Record CUDA allocation history for the region and dump a snapshot to
    ``<output_dir>/<label>-rankNN.pickle``. Only ``ranks`` record/dump; the rest no-op.
    ``output_dir`` defaults under ``HALO_DATA_ROOT``."""
    output_dir = output_dir or memory_snapshot_dir()
    active = torch.cuda.is_available() and should_profile_this_rank(ranks)
    if not active:
        yield
        return

    ensure_artifact_dir(output_dir)
    started = start_memory_history(max_entries=max_entries)
    try:
        yield
    finally:
        if started:
            path = os.path.join(output_dir, f"{label}-{rank_tag()}.pickle")
            dump_memory_snapshot(path)
            stop_memory_history()


def export_profiler_artifacts(
    prof: torch.profiler.profile,
    output_dir: str,
    *,
    label: str,
    with_stack: bool,
    profile_memory: bool,
    export_memory_timeline: bool,
) -> None:
    """Write Chrome trace, collapsed stacks (flame graph), an optional memory
    timeline, and a top-ops table for ``prof`` to ``output_dir``."""
    tag = rank_tag()
    prefix = os.path.join(output_dir, f"{label}-{tag}")
    written: list[str] = []

    def _record(path: str) -> None:
        """Keep a written artifact, dropping one an exporter left empty."""
        if os.path.getsize(path) == 0:
            os.unlink(path)
            logger.warning("[profiler %s] %s came back empty — dropped", tag, os.path.basename(path))
            return
        written.append(os.path.basename(path))

    try:
        prof.export_chrome_trace(f"{prefix}.trace.json.gz")
        _record(f"{prefix}.trace.json.gz")
    except Exception as exc:
        logger.warning("export_chrome_trace failed: %s", exc)

    if with_stack:
        for metric, suffix in (
            ("self_cuda_time_total", "cuda"),
            ("self_cpu_time_total", "cpu"),
        ):
            try:
                prof.export_stacks(f"{prefix}.stacks_{suffix}.txt", metric)
                _record(f"{prefix}.stacks_{suffix}.txt")
            except Exception as exc:
                logger.warning("export_stacks(%s) failed: %s", metric, exc)

    if profile_memory and export_memory_timeline:
        try:
            prof.export_memory_timeline(f"{prefix}.memory_timeline.html", device=str(current_device()))
            _record(f"{prefix}.memory_timeline.html")
        except Exception as exc:
            logger.warning("export_memory_timeline failed: %s", exc)

    try:
        sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
        table = prof.key_averages().table(sort_by=sort_key, row_limit=25)
        with open(f"{prefix}.top_ops.txt", "w") as fh:
            fh.write(table)
        _record(f"{prefix}.top_ops.txt")
    except Exception as exc:
        logger.warning("key_averages().table failed: %s", exc)

    logger.info("[profiler %s] wrote %s → %s.*", tag, ", ".join(written) or "nothing", prefix)


@contextmanager
def torch_profiler_session(
    output_dir: str | None = None,
    *,
    ranks: str | int | None = "0",
    label: str = "trace",
    record_shapes: bool = True,
    profile_memory: bool = True,
    with_stack: bool = True,
    with_flops: bool = False,
    export_memory_timeline: bool = True,
) -> Iterator[torch.profiler.profile | None]:
    """One-shot :class:`torch.profiler.profile` over a bounded code region.

    For schedule-based in-loop profiling of an HF Trainer run, prefer
    :class:`src.callbacks.profiler.TorchProfilerCallback`. Yields the profiler (or
    ``None`` on non-selected ranks); on exit writes per-rank trace/stacks/table to ``output_dir``
    (defaults under ``HALO_DATA_ROOT``).
    """
    output_dir = output_dir or torch_trace_dir()
    if not should_profile_this_rank(ranks):
        yield None
        return

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    ensure_artifact_dir(output_dir)
    started = time.perf_counter()
    with profile(
        activities=activities,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        with_flops=with_flops,
    ) as prof:
        yield prof

    logger.info(
        "[profiler %s] region '%s' profiled in %.2fs; exporting…",
        rank_tag(),
        label,
        time.perf_counter() - started,
    )
    export_profiler_artifacts(
        prof,
        output_dir,
        label=label,
        with_stack=with_stack,
        profile_memory=profile_memory,
        export_memory_timeline=export_memory_timeline,
    )
