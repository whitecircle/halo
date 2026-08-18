#!/usr/bin/env python
"""Test the mixin's OOM-diagnosability helpers in single-process isolation.

The thin-margin warning and the rank-local OOM banner exist so a one-rank OOM inside a distributed
step does not read as a DeepEP/cuBLAS kernel fault: the hot rank OOMs by a fraction of a GiB and its
peers die on Xid 43 launch failures inside combine. No GPU or distributed init.

Both are PER-RANK verdicts emitted through accelerate's ``MultiProcessAdapter``, which drops records
on every non-main rank unless ``main_process_only=False`` — so the kwarg is pinned here alongside the
message text. Without it each one prints only when rank 0 is the affected rank, which is precisely
the case it was not written for.

Run: python tests/cpu/trainers/test_memory_margin_diagnostics.py
"""

import types

import pytest
import torch

from src.trainers.mixins.base import (
    _MEMORY_MARGIN_WARN_RATIO,
    DistributedTrainerMixin,
    emit_primary_failure,
    oom_banner,
    thin_memory_margin_message,
)

_GIB = 1024**3


def test_message_fires_above_the_ratio():
    total = 268 * _GIB
    message = thin_memory_margin_message(int(total * 0.94), total, rank=2)
    assert message is not None
    assert "RANK 2" in message
    assert "94%" in message
    assert "per_device_train_batch_size" in message


def test_message_silent_below_the_ratio():
    total = 268 * _GIB
    assert thin_memory_margin_message(int(total * (_MEMORY_MARGIN_WARN_RATIO - 0.01)), total, rank=0) is None


def test_message_boundary_is_the_ratio():
    total = 100 * _GIB
    at = int(total * _MEMORY_MARGIN_WARN_RATIO)
    assert thin_memory_margin_message(at, total, rank=0) is not None
    assert thin_memory_margin_message(at - _GIB, total, rank=0) is None


def test_message_survives_zero_total():
    assert thin_memory_margin_message(0, 0, rank=0) is None


def test_oom_banner_names_the_primary_failure_and_the_collateral():
    banner = oom_banner(5, RuntimeError("CUDA out of memory. Tried to allocate 7.84 GiB"))
    assert "RANK 5" in banner
    assert "PRIMARY FAILURE" in banner
    assert "Xid 43" in banner
    assert "7.84 GiB" in banner


class _RecordingLogger:
    """Stands in for the module's accelerate adapter, capturing the kwargs each call passes."""

    def __init__(self):
        self.calls: list[tuple[str, str, dict]] = []

    def error(self, msg, **kwargs):
        self.calls.append(("error", msg, kwargs))

    def warning(self, msg, **kwargs):
        self.calls.append(("warning", msg, kwargs))


def test_primary_failure_banner_escapes_the_main_process_only_default(monkeypatch):
    """The banner names the rank that OOMed, and this module logs through accelerate's
    ``MultiProcessAdapter``, which drops records on every non-main rank unless
    ``main_process_only=False``. On the default the banner prints only when rank 0 is the one that
    died — never in the case it exists for, which is what left a real 4-rank OOM post-mortem with
    peer watchdog timeouts and no primary cause."""
    recorder = _RecordingLogger()
    monkeypatch.setattr("src.trainers.mixins.base.logger", recorder)

    emit_primary_failure(3, RuntimeError("CUDA out of memory. Tried to allocate 53.56 GiB"))

    assert len(recorder.calls) == 1
    level, msg, kwargs = recorder.calls[0]
    assert level == "error"
    assert kwargs.get("main_process_only") is False, (
        "the banner must be emitted on every rank; the adapter's default silently drops it on the "
        "non-zero rank that actually OOMed"
    )
    assert "RANK 3" in msg and "53.56 GiB" in msg


def test_thin_margin_warning_escapes_the_main_process_only_default(monkeypatch):
    """Same adapter, same trap: the thin-margin verdict is per-rank by construction (MoE routing
    skew concentrates dispatch buffers and expert activations on hot ranks), so a main-process-only
    emit reports it for the one rank whose margin is least likely to be the problem.

    CUDA is faked rather than skipped: on the CPU tier a skip would leave the kwarg unpinned, which
    is the whole behaviour under test."""
    recorder = _RecordingLogger()
    monkeypatch.setattr("src.trainers.mixins.base.logger", recorder)
    monkeypatch.setattr("src.trainers.mixins.base.thin_memory_margin_message", lambda *a, **k: "thin margin")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 99 * 1024**3)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _idx: types.SimpleNamespace(total_memory=100 * 1024**3)
    )
    me = types.SimpleNamespace(_memory_margin_checked=False, state=types.SimpleNamespace(global_step=1))

    DistributedTrainerMixin._warn_once_on_thin_memory_margin(me)

    assert recorder.calls, "the thin-margin warning never reached the logger"
    level, msg, kwargs = recorder.calls[0]
    assert level == "warning" and msg == "thin margin"
    assert kwargs.get("main_process_only") is False, (
        "a per-rank margin verdict emitted main-process-only is invisible on the hot rank it describes"
    )


def test_margin_check_runs_once_and_only_after_the_first_step():
    calls = []
    me = types.SimpleNamespace(
        _memory_margin_checked=False,
        state=types.SimpleNamespace(global_step=0),
        _record=calls,
    )
    check = DistributedTrainerMixin._warn_once_on_thin_memory_margin
    check(me)
    assert not me._memory_margin_checked, "must not latch before the first optimizer step"
    me.state.global_step = 1
    check(me)  # latches (and on a CUDA-less host returns without reading allocator stats)
    assert me._memory_margin_checked
    marker = object()
    me._memory_margin_checked = marker
    check(me)
    assert me._memory_margin_checked is marker, "second call must be a no-op"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
