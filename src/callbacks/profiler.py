"""Schedule-based :mod:`torch.profiler` integration for the HF Trainer loop.

Captures a bounded ``wait`` → ``warmup`` → ``active`` window and writes per-rank artifacts (Chrome
trace, memory-allocation timeline, top-ops table, optional CUDA memory-snapshot pickle). Opt-in and
one-shot because profiling costs throughput; only global rank 0 profiles unless ``ranks`` says more.
"""

from __future__ import annotations

import logging
import os

import torch
from torch.profiler import ProfilerActivity, profile, schedule
from transformers import TrainerCallback

from src.diagnostics.profiling import (
    dump_memory_snapshot,
    ensure_artifact_dir,
    export_profiler_artifacts,
    should_profile_this_rank,
    start_memory_history,
    stop_memory_history,
)
from src.distributed.runtime import rank_tag
from src.env import torch_trace_dir

logger = logging.getLogger(__name__)


class TorchProfilerCallback(TrainerCallback):
    """Wire :class:`torch.profiler.profile` into training via a step schedule.

    Args:
        output_dir: per-rank artifact directory (defaults under ``HALO_DATA_ROOT``).
        wait/warmup/active/repeat: the profiler schedule; the trace covers the ``active`` window.
        ranks: which global ranks profile — ``"0"`` (default), ``"all"``, or a comma list ``"0,8"``.
        record_shapes/profile_memory/with_stack: forwarded to profile (``with_stack`` is what flame
            graphs need, ``profile_memory`` the memory timeline).
        memory_snapshot: also record CUDA allocation history over the window and dump a ``.pickle``.
        label: filename prefix for artifacts.
    """

    def __init__(
        self,
        output_dir: str | None = None,
        *,
        wait: int = 5,
        warmup: int = 1,
        active: int = 3,
        repeat: int = 1,
        ranks: str | int | None = "0",
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = True,
        export_memory_timeline: bool = True,
        memory_snapshot: bool = False,
        cpu_activity: bool = True,
        label: str = "trace",
    ) -> None:
        self.output_dir = output_dir or torch_trace_dir()
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.ranks = ranks
        # CPU op tracing is single-threaded and starves the GPUs on a large model; False = CUDA-only.
        self.cpu_activity = cpu_activity
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.export_memory_timeline = export_memory_timeline
        self.memory_snapshot = memory_snapshot
        self.label = label

        self._active_rank = should_profile_this_rank(ranks)
        self._prof = None
        self._step = 0
        self._cycle = 0
        self._mem_start = wait + warmup
        self._mem_stop = wait + warmup + active
        self._mem_recording = False
        self._mem_dumped = False

    def on_train_begin(self, args, state, control, **kwargs):
        if not self._active_rank:
            return

        activities = [ProfilerActivity.CPU] if self.cpu_activity else []
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        ensure_artifact_dir(self.output_dir)
        self._prof = profile(
            activities=activities,
            schedule=schedule(wait=self.wait, warmup=self.warmup, active=self.active, repeat=self.repeat),
            on_trace_ready=self._on_trace_ready,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
        )
        self._prof.__enter__()
        logger.info(
            "[profiler %s] torch.profiler active: wait=%d warmup=%d active=%d repeat=%d → %s",
            rank_tag(),
            self.wait,
            self.warmup,
            self.active,
            self.repeat,
            self.output_dir,
        )

    def on_step_end(self, args, state, control, **kwargs):
        if self._prof is None:
            return
        if self.memory_snapshot and not self._mem_dumped:
            if self._step == self._mem_start and not self._mem_recording:
                self._mem_recording = start_memory_history()
            elif self._step == self._mem_stop and self._mem_recording:
                self._dump_memory()

        self._prof.step()
        self._step += 1

    def on_train_end(self, args, state, control, **kwargs):
        if self._prof is None:
            return
        if self._mem_recording and not self._mem_dumped:
            self._dump_memory()
        try:
            self._prof.__exit__(None, None, None)
        finally:
            self._prof = None

    def _on_trace_ready(self, prof) -> None:
        """Called by torch.profiler at the end of each active window."""
        self._cycle += 1
        export_profiler_artifacts(
            prof,
            self.output_dir,
            label=f"{self.label}-cycle{self._cycle}",
            with_stack=self.with_stack,
            profile_memory=self.profile_memory,
            export_memory_timeline=self.export_memory_timeline,
        )

    def _dump_memory(self) -> None:
        path = os.path.join(self.output_dir, f"mem-{self.label}-{rank_tag()}.pickle")
        dump_memory_snapshot(path)
        stop_memory_history()
        self._mem_recording = False
        self._mem_dumped = True
