#!/usr/bin/env python
"""CPU correctness tests for TorchProfilerCallback.

Drives the callback through the HF Trainer callback lifecycle (on_train_begin →
on_step_end x N → on_train_end) with real torch.profiler activity and asserts it
honors the wait->warmup->active schedule, writes a chrome-trace artifact, and
no-ops on non-selected ranks.

Run: python tests/cpu/callbacks/test_profiler_callback.py
"""

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

from src.callbacks.profiler import TorchProfilerCallback


def _drive(cb, n_steps):
    """Run the callback through a fake training loop with real compute per step."""
    args = SimpleNamespace(process_index=0, local_process_index=0)
    state = SimpleNamespace(global_step=0, is_world_process_zero=True)
    control = SimpleNamespace()
    cb.on_train_begin(args, state, control)
    for i in range(n_steps):
        x = torch.randn(64, 64) @ torch.randn(64, 64)  # real op for the profiler to record
        x.sum().item()
        state.global_step = i + 1
        cb.on_step_end(args, state, control)
    cb.on_train_end(args, state, control)


def test_profiler_writes_trace_artifact():
    """A full wait->warmup->active cycle writes at least one trace artifact."""
    with tempfile.TemporaryDirectory() as d:
        cb = TorchProfilerCallback(
            output_dir=d,
            wait=1,
            warmup=1,
            active=2,
            repeat=1,
            with_stack=False,
            export_memory_timeline=False,
            memory_snapshot=False,
            ranks="0",
        )
        _drive(cb, n_steps=5)
        artifacts = [f for _, _, files in os.walk(d) for f in files]
        assert artifacts, f"profiler wrote no artifacts to {d}"
        assert any(("json" in f) or ("trace" in f.lower()) for f in artifacts), f"no chrome trace among {artifacts}"
        assert cb._prof is None, "profiler not cleaned up after on_train_end"


def test_profiler_inactive_rank_is_noop():
    """rank spec excluding this rank => no profiler, no artifacts."""
    with tempfile.TemporaryDirectory() as d:
        cb = TorchProfilerCallback(output_dir=d, wait=1, warmup=1, active=1, ranks="3")
        _drive(cb, n_steps=4)
        assert cb._prof is None, "inactive rank should not hold a profiler"
        artifacts = [f for _, _, files in os.walk(d) for f in files]
        assert not artifacts, f"inactive rank should write nothing, got {artifacts}"


def test_profiler_no_active_window_no_crash():
    """Fewer steps than wait+warmup+active completes cleanly (no crash, cleaned up)."""
    with tempfile.TemporaryDirectory() as d:
        cb = TorchProfilerCallback(
            output_dir=d, wait=5, warmup=2, active=3, ranks="0", export_memory_timeline=False, memory_snapshot=False
        )
        _drive(cb, n_steps=3)
        assert cb._prof is None, "profiler should be cleaned up in on_train_end"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
