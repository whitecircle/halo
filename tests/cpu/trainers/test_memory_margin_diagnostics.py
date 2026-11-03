#!/usr/bin/env python
"""The mixin's OOM-diagnosability helpers, so a one-rank OOM does not read as a DeepEP/cuBLAS kernel fault.
Both are per-rank verdicts, so each must log on a rank that is not the main process."""

import types
from unittest import mock

import pytest
import torch

from halo.trainers.distributed_trainer import (
    MEMORY_MARGIN_WARN_RATIO,
    DistributedTrainerMixin,
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
    assert thin_memory_margin_message(int(total * (MEMORY_MARGIN_WARN_RATIO - 0.01)), total, rank=0) is None


def test_message_boundary_is_the_ratio():
    total = 100 * _GIB
    at = int(total * MEMORY_MARGIN_WARN_RATIO)
    assert thin_memory_margin_message(at, total, rank=0) is not None
    assert thin_memory_margin_message(at - _GIB, total, rank=0) is None


def test_message_survives_zero_total():
    assert thin_memory_margin_message(0, 0, rank=0) is None


class _OOMingTrainer:
    def training_step(self, model, inputs, num_items_in_batch=None):
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 53.56 GiB")


class _StepHost(DistributedTrainerMixin, _OOMingTrainer):
    def _accumulate_unmasked_output_tokens(self, local_count):
        pass

    def _accumulate_attention_flops(self, inputs):
        pass


def _oom_step():
    host = object.__new__(_StepHost)
    host._memory_margin_checked = True
    host._window_modules = []
    host.accelerator = types.SimpleNamespace(sync_gradients=True)
    DistributedTrainerMixin.training_step(host, None, {})


def _recorder(monkeypatch, main: bool) -> mock.MagicMock:
    recorder = mock.MagicMock()
    monkeypatch.setattr("halo.trainers.distributed_trainer.logger", recorder)
    monkeypatch.setattr("halo.trainers.distributed_trainer.is_global_main_process", lambda: main)
    return recorder


def test_primary_failure_banner_logs_off_the_main_process(monkeypatch):
    """Guarded to the main process it would print only when rank 0 died, leaving peer timeouts and no primary cause."""
    recorder = _recorder(monkeypatch, main=False)
    monkeypatch.setattr("halo.trainers.distributed_trainer.get_global_rank", lambda: 3)

    with pytest.raises(torch.OutOfMemoryError):
        _oom_step()

    recorder.error.assert_called_once()
    (msg,), _ = recorder.error.call_args
    assert "RANK 3" in msg and "53.56 GiB" in msg
    assert "PRIMARY FAILURE" in msg and "Xid 43" in msg, "the banner must name the primary failure and the collateral"


def test_thin_margin_warning_logs_off_the_main_process(monkeypatch):
    """MoE routing skew concentrates buffers on hot ranks, so a main-process-only emit reports the rank least at risk.
    CUDA is faked because a skip on the CPU tier would leave the emit unpinned."""
    recorder = _recorder(monkeypatch, main=False)
    monkeypatch.setattr("halo.trainers.distributed_trainer.thin_memory_margin_message", lambda *a, **k: "thin margin")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 99 * 1024**3)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _idx: types.SimpleNamespace(total_memory=100 * 1024**3)
    )
    me = types.SimpleNamespace(_memory_margin_checked=False, state=types.SimpleNamespace(global_step=1))

    DistributedTrainerMixin._warn_once_on_thin_memory_margin(me)

    recorder.warning.assert_called_once_with("thin margin")


@pytest.mark.parametrize("main", [True, False])
def test_the_released_buffers_line_logs_on_the_main_process_only(monkeypatch, main):
    """The control for the two above."""
    recorder = _recorder(monkeypatch, main)
    monkeypatch.setattr("halo.trainers.distributed_trainer.free_unclaimed_ep_buffers", lambda: 2)
    monkeypatch.setattr("halo.trainers.distributed_trainer.barrier", lambda: None)

    DistributedTrainerMixin.cleanup_ep(types.SimpleNamespace(_ep_layers=lambda: []))

    assert recorder.info.called is main


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
