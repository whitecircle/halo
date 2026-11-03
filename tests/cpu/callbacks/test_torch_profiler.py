#!/usr/bin/env python
"""TorchProfilerCallback through the HF callback lifecycle, with real torch.profiler activity."""

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch
from torch.profiler import ProfilerAction

from halo.callbacks.torch_profiler import TorchProfilerCallback
from halo.diagnostics.profiling import build_profiler


def _drive(cb, n_steps):
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


def test_profiler_is_built_from_the_callbacks_settings(tmp_path, monkeypatch):
    built: list[tuple[str, dict[str, object]]] = []

    def recording_build_profiler(output_dir, **settings):
        built.append((output_dir, settings))
        return build_profiler(output_dir, **settings)

    monkeypatch.setattr("halo.callbacks.torch_profiler.build_profiler", recording_build_profiler)
    cb = TorchProfilerCallback(
        output_dir=str(tmp_path),
        wait=2,
        warmup=1,
        active=2,
        repeat=1,
        ranks="0",
        cpu_activity=False,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        export_memory_timeline=False,
    )
    _drive(cb, n_steps=1)

    [(output_dir, settings)] = built
    schedule = settings.pop("schedule")
    assert output_dir == str(tmp_path)
    assert settings == {
        "cpu_activity": False,
        "record_shapes": False,
        "profile_memory": False,
        "with_stack": False,
        "on_trace_ready": cb._on_trace_ready,
    }
    assert [schedule(step) for step in range(8)] == [
        ProfilerAction.NONE,
        ProfilerAction.NONE,
        ProfilerAction.WARMUP,
        ProfilerAction.RECORD,
        ProfilerAction.RECORD_AND_SAVE,
        ProfilerAction.NONE,
        ProfilerAction.NONE,
        ProfilerAction.NONE,
    ]


def test_profiler_inactive_rank_is_noop():
    with tempfile.TemporaryDirectory() as d:
        cb = TorchProfilerCallback(output_dir=d, wait=1, warmup=1, active=1, ranks="3")
        _drive(cb, n_steps=4)
        assert cb._prof is None, "inactive rank should not hold a profiler"
        artifacts = [f for _, _, files in os.walk(d) for f in files]
        assert not artifacts, f"inactive rank should write nothing, got {artifacts}"


def test_profiler_no_active_window_no_crash():
    with tempfile.TemporaryDirectory() as d:
        cb = TorchProfilerCallback(
            output_dir=d, wait=5, warmup=2, active=3, ranks="0", export_memory_timeline=False, memory_snapshot=False
        )
        _drive(cb, n_steps=3)
        assert cb._prof is None, "profiler should be cleaned up in on_train_end"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
