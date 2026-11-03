"""Opt-in CUDA memory and torch-profiler capture; artifacts are rank-tagged, global rank 0 profiles by default."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import NamedTuple

import torch
from torch.profiler import ProfilerAction, ProfilerActivity, profile

from halo.env import data_path, torch_trace_dir
from halo.hardware import current_device
from halo.runtime.ranks import get_global_rank, rank_tag

logger = logging.getLogger(__name__)

# Alloc/free events the recorded CUDA memory history keeps before dropping the oldest.
_MEMORY_HISTORY_MAX_ENTRIES = 100_000


class CudaMemoryGiB(NamedTuple):
    """This rank's CUDA allocator counters in GiB; the peaks run since the last peak reset."""

    allocated: float
    reserved: float
    peak_allocated: float
    peak_reserved: float


def _parse_rank_spec(spec: str | int | None) -> set[int] | None:
    """Global ranks a spec selects: ``"all"`` → ``None`` (every rank), an int or ``"0,8"`` list, and
    ``None`` / empty → ``{0}``."""
    if spec is None or spec == "":
        return {0}
    if isinstance(spec, int):
        return {spec}
    spec = str(spec).strip().lower()
    if spec == "all":
        return None
    return {int(p) for p in spec.split(",") if p.strip() != ""}


def should_profile_this_rank(spec: str | int | None) -> bool:
    selected = _parse_rank_spec(spec)
    if selected is None:
        return True
    return get_global_rank() in selected


def read_cuda_memory() -> CudaMemoryGiB | None:
    if not torch.cuda.is_available():
        return None
    return CudaMemoryGiB(
        allocated=torch.cuda.memory_allocated() / 2**30,
        reserved=torch.cuda.memory_reserved() / 2**30,
        peak_allocated=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved=torch.cuda.max_memory_reserved() / 2**30,
    )


def log_cuda_memory(tag: str = "") -> None:
    memory = read_cuda_memory()
    if memory is None:
        return
    logger.info(
        "[mem %s] %s: allocated=%.2f reserved=%.2f peak_alloc=%.2f peak_reserved=%.2f GiB",
        rank_tag(),
        tag or "cuda",
        *memory,
    )


def start_memory_history(max_entries: int = _MEMORY_HISTORY_MAX_ENTRIES) -> bool:
    """Costly, so scope it to a few steps; False without CUDA."""
    if not torch.cuda.is_available():
        return False
    torch.cuda.memory._record_memory_history(max_entries=max_entries)
    return True


def stop_memory_history() -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.memory._record_memory_history(enabled=None)


def dump_memory_snapshot(path: str) -> str | None:
    """Dump to ``path`` (a ``.pickle`` for https://pytorch.org/memory_viz); the path, or None on failure."""
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
    max_entries: int = _MEMORY_HISTORY_MAX_ENTRIES,
    label: str = "snapshot",
) -> Iterator[None]:
    """Record allocation history over the region on ``ranks`` and dump ``<output_dir>/<label>-rankNN.pickle``."""
    output_dir = output_dir or data_path("profiling", "memory")
    active = torch.cuda.is_available() and should_profile_this_rank(ranks)
    if not active:
        yield
        return

    # A plain makedirs: a barrier would hang on the ranks the selection left out.
    os.makedirs(output_dir, exist_ok=True)
    start_memory_history(max_entries=max_entries)
    try:
        yield
    finally:
        dump_memory_snapshot(os.path.join(output_dir, f"{label}-{rank_tag()}.pickle"))
        stop_memory_history()


def build_profiler(
    output_dir: str,
    *,
    cpu_activity: bool,
    record_shapes: bool,
    profile_memory: bool,
    with_stack: bool,
    with_flops: bool = False,
    schedule: Callable[[int], ProfilerAction] | None = None,
    on_trace_ready: Callable[[profile], None] | None = None,
) -> profile:
    """An unentered profiler tracing CUDA, and CPU ops with ``cpu_activity``; creates ``output_dir`` without a barrier,
    which unselected ranks would never join."""
    activities = [ProfilerActivity.CPU] if cpu_activity else []
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    os.makedirs(output_dir, exist_ok=True)
    return profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=on_trace_ready,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        with_flops=with_flops,
    )


def export_profiler_artifacts(
    prof: profile,
    output_dir: str,
    *,
    label: str,
    with_stack: bool,
    profile_memory: bool,
    export_memory_timeline: bool,
) -> None:
    """Write ``prof``'s Chrome trace, collapsed stacks, optional memory timeline and top-ops table."""
    tag = rank_tag()
    prefix = os.path.join(output_dir, f"{label}-{tag}")
    written: list[str] = []

    def _record(path: str) -> None:
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
) -> Iterator[profile | None]:
    """Profile a bounded region on ``ranks`` (yielding the profiler, else ``None``) and write artifacts on exit;
    in-loop profiling is ``TorchProfilerCallback``."""
    output_dir = output_dir or torch_trace_dir()
    if not should_profile_this_rank(ranks):
        yield None
        return

    profiler = build_profiler(
        output_dir,
        cpu_activity=True,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        with_flops=with_flops,
    )
    started = time.perf_counter()
    with profiler:
        yield profiler

    logger.info(
        "[profiler %s] region '%s' profiled in %.2fs; exporting…",
        rank_tag(),
        label,
        time.perf_counter() - started,
    )
    export_profiler_artifacts(
        profiler,
        output_dir,
        label=label,
        with_stack=with_stack,
        profile_memory=profile_memory,
        export_memory_timeline=export_memory_timeline,
    )
