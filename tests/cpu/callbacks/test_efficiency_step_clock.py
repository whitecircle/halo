#!/usr/bin/env python
"""``EfficiencyCallback`` times a step between CUDA events at its boundaries and never drains the device for it.

A host clock read at a step boundary is only right once the device has run everything queued, so a clock built on
it synchronizes twice per optimizer step: the GPU then idles while the host fetches the next batch and logs. The
callback records an event at each boundary instead and reads the elapsed time at the next step end or at the log,
waiting only for an end event the device has not reached. A scripted device stands in for CUDA: kernels the host
launches run behind the work already queued, an event completes when the device reaches it, and every wait is
counted.

    python tests/cpu/callbacks/test_efficiency_step_clock.py
"""

from types import SimpleNamespace

import pytest
import torch

from halo.callbacks import efficiency
from halo.callbacks.efficiency import EfficiencyCallback
from tests.common.parallelism import make_parallelism_config

STEP_TOKENS = 1000


class _Device:
    """A host clock and a device queue: launched work runs back to back behind what is queued, from the moment it
    is launched at the earliest."""

    def __init__(self):
        self.now = 0.0
        self.busy_until = 0.0
        self.drains = 0
        self.event_waits = 0

    def perf_counter(self) -> float:
        return self.now

    def launch(self, seconds: float) -> None:
        self.busy_until = max(self.busy_until, self.now) + seconds

    def synchronize(self) -> None:
        self.drains += 1
        self.now = max(self.now, self.busy_until)

    def event(self, *args, **kwargs) -> "_Event":
        return _Event(self)


class _Event:
    def __init__(self, device: _Device):
        self.device = device
        self.at = None

    def record(self, *args, **kwargs) -> None:
        self.at = max(self.device.busy_until, self.device.now)

    def query(self) -> bool:
        return self.at <= self.device.now

    def synchronize(self) -> None:
        if not self.query():
            self.device.event_waits += 1
            self.device.now = self.at

    def elapsed_time(self, end: "_Event") -> float:
        assert self.query() and end.query(), "elapsed_time read before the device reached both events"
        return (end.at - self.at) * 1000.0


@pytest.fixture
def device(monkeypatch) -> _Device:
    scripted = _Device()
    monkeypatch.setattr(efficiency, "time", SimpleNamespace(perf_counter=scripted.perf_counter))
    monkeypatch.setattr(efficiency, "read_cuda_memory", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", scripted.synchronize)
    monkeypatch.setattr(torch.cuda, "Event", scripted.event)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *args, **kwargs: None)
    return scripted


def _callback() -> EfficiencyCallback:
    callback = EfficiencyCallback(make_parallelism_config(world_size=1, gpus_per_node=1), n_warmup_steps=0)
    callback.training_args = SimpleNamespace(per_device_train_batch_size=1, gradient_accumulation_steps=1)
    return callback


def _step(callback: EfficiencyCallback, device: _Device, step: int, *, launch: float, kernels: float) -> None:
    """Optimizer step ``step``: ``launch`` seconds of host time to queue ``kernels`` seconds of device work."""
    callback.on_step_begin(None, SimpleNamespace(global_step=step - 1), None)
    device.now += launch
    device.launch(kernels)
    callback.on_step_end(None, SimpleNamespace(global_step=step, num_input_tokens_seen=step * STEP_TOKENS), None)


def _log(callback: EfficiencyCallback, step: int) -> dict:
    logs: dict = {}
    callback.on_log(None, SimpleNamespace(global_step=step), None, logs=logs)
    return logs


def test_a_step_is_timed_over_its_own_device_work_without_draining_the_device(device):
    """The previous step's tail is still running when this step begins and the host launches inside it: the step
    spans its kernels alone, and only the log waits, for the one end event the device has not reached."""
    callback = _callback()
    device.launch(5.0)

    _step(callback, device, 1, launch=0.25, kernels=2.0)
    assert device.drains == 0 and device.event_waits == 0, "a step boundary waited on the device"

    logs = _log(callback, 1)
    assert device.drains == 0, "the step clock drained the device"
    assert device.event_waits == 1, "the log must wait for the end event, and only for it"
    assert callback.measured_step_times == (2.0,)
    assert logs["step_time_seconds"] == 2.0
    assert logs["step_tokens_per_second"] == pytest.approx(STEP_TOKENS / 2.0)


def test_each_step_is_read_one_step_later_without_a_wait(device):
    """A host-bound run: the device finishes each step while the host prepares the next one, so the earlier end
    marks have passed when the next step end reads them. A step finds the device idle and spans its launch too, the
    time the device waited on the host inside the step."""
    callback = _callback()
    launch, kernels = 0.1, (0.5, 0.7, 0.2, 0.9)
    for step, seconds in enumerate(kernels, start=1):
        _step(callback, device, step, launch=launch, kernels=seconds)
        device.now += 1.0  # the host prepares the next batch while the queue drains
    logs = _log(callback, len(kernels))

    spans = [launch + seconds for seconds in kernels]
    assert device.drains == 0 and device.event_waits == 0
    assert callback.measured_step_times == pytest.approx(spans)
    assert logs["avg_step_time_seconds"] == round(sum(spans) / len(spans), 2)
    assert logs["avg_tokens_per_second"] == round(len(spans) * STEP_TOKENS / sum(spans), 2)


def test_a_log_waits_for_the_latest_step_only(device):
    """The device has reached step 1's end by step 2's (the host reads each step's gradient norm before stepping),
    so the log waits for step 2's end event alone."""
    callback = _callback()
    _step(callback, device, 1, launch=0.0, kernels=3.0)
    device.now = 3.0
    _step(callback, device, 2, launch=0.0, kernels=4.0)

    _log(callback, 2)

    assert device.drains == 0 and device.event_waits == 1
    assert callback.measured_step_times == (3.0, 4.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
