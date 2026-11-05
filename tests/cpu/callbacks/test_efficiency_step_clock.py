#!/usr/bin/env python
"""``EfficiencyCallback`` times each step from the previous step's end to its own, on CUDA events, without a drain.
A scripted device stands in for CUDA: an event completes when the device reaches it, and every wait is counted."""

from types import SimpleNamespace

import pytest
import torch

from halo.callbacks import efficiency
from halo.callbacks.efficiency import EfficiencyCallback
from halo.diagnostics import phase_timers
from tests.common.parallelism import make_parallelism_config

STEP_TOKENS = 1000
# A device-bound dense step and the host work around it, in seconds.
FETCH, LAUNCH, KERNELS, DRAIN, LOG, COLLECTIVE = 0.002, 0.01, 0.3, 0.0005, 0.003, 0.001
# The probed window, as the bench rows read it: wall clock from the end of WINDOW_START to the end of WINDOW_END.
WARMUP_STEPS, WINDOW_START, WINDOW_END = 3, 5, 20


class _Device:
    """Launched work runs back to back behind what is queued, from the moment it is launched at the earliest."""

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
        self.read_back()

    def read_back(self) -> None:
        """A host read of a device value (a loss ``item()``, a probe's synchronize), not the callback's wait."""
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


class _StepWorkSource:
    """Drained after the step's end mark: an all-reduce of the step's counts, read back on the host."""

    def __init__(self, device: _Device):
        self.device = device

    def drain_step_work(self) -> tuple[float, int]:
        self.device.launch(COLLECTIVE)
        self.device.read_back()
        self.device.now += DRAIN
        return 0.0, 0


@pytest.fixture
def device(monkeypatch) -> _Device:
    scripted = _Device()
    monkeypatch.setattr(phase_timers, "time", SimpleNamespace(perf_counter=scripted.perf_counter))
    monkeypatch.setattr(efficiency, "read_cuda_memory", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", scripted.synchronize)
    monkeypatch.setattr(torch.cuda, "Event", scripted.event)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *args, **kwargs: None)
    return scripted


def _callback(n_warmup_steps: int = 0) -> EfficiencyCallback:
    callback = EfficiencyCallback(
        make_parallelism_config(world_size=1, gpus_per_node=1), n_warmup_steps=n_warmup_steps
    )
    callback.training_args = SimpleNamespace(per_device_train_batch_size=1, gradient_accumulation_steps=1)
    callback.on_train_begin(None, SimpleNamespace(global_step=0), None)
    return callback


def _step(callback: EfficiencyCallback, device: _Device, step: int, *, launch: float, kernels: float) -> None:
    callback.on_step_begin(None, SimpleNamespace(global_step=step - 1), None)
    device.now += launch
    device.launch(kernels)
    callback.on_step_end(None, SimpleNamespace(global_step=step, num_input_tokens_seen=step * STEP_TOKENS), None)


def _log(callback: EfficiencyCallback, step: int) -> dict:
    logs: dict = {}
    callback.on_log(None, SimpleNamespace(global_step=step), None, logs=logs)
    return logs


def test_a_step_is_timed_over_its_own_device_work_without_draining_the_device(device):
    """Work queued before the first step is still running when it begins, and the host launches inside it. The step
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
    """On a host-bound run the device finishes each step while the host fetches the next batch, so earlier end marks
    have passed when the next step end reads them, and the fetch between steps counts."""
    callback = _callback()
    launch, kernels, fetch = 0.1, (0.5, 0.7, 0.2, 0.9), 1.0
    device.now += fetch
    first_begin = device.now
    ends = []
    for step, seconds in enumerate(kernels, start=1):
        _step(callback, device, step, launch=launch, kernels=seconds)
        ends.append(device.busy_until)
        device.now += fetch
    logs = _log(callback, len(kernels))

    spans = [end - start for start, end in zip([first_begin, *ends[:-1]], ends, strict=True)]
    assert device.drains == 0 and device.event_waits == 0
    assert callback.measured_step_times == pytest.approx(spans)
    assert sum(callback.measured_step_times) == pytest.approx(ends[-1] - first_begin)
    assert logs["avg_step_time_seconds"] == round(sum(spans) / len(spans), 2)
    assert logs["avg_tokens_per_second"] == round(len(spans) * STEP_TOKENS / sum(spans), 2)


def test_the_steps_sum_to_the_wall_clock_of_the_window_they_span(device):
    """A clock from each step's begin to its end would miss the fetch, the drain and the log, about 2% of every step
    here, and one that restarted after the drain would still miss the drain."""
    callback = _callback(WARMUP_STEPS)
    # Held for the whole run, since the callback keeps only a weak reference to its source.
    source = _StepWorkSource(device)
    callback.bind_attention_work_source(source)
    wall, counted = {}, {}
    for step in range(1, WINDOW_END + 1):
        device.now += FETCH
        _step(callback, device, step, launch=LAUNCH, kernels=KERNELS)
        if step in (WINDOW_START, WINDOW_END):
            device.read_back()
            wall[step], counted[step] = device.now, len(callback.measured_step_times)
        device.launch(COLLECTIVE)
        device.read_back()
        device.now += LOG
        _log(callback, step)

    window = wall[WINDOW_END] - wall[WINDOW_START]
    steps = WINDOW_END - WINDOW_START
    in_step_spans = steps * (LAUNCH + KERNELS)
    assert window > 1.01 * in_step_spans, "the premise: the host time between steps is over 1% of the window"
    measured = callback.measured_step_times[counted[WINDOW_START] : counted[WINDOW_END]]
    assert len(measured) == steps
    assert sum(measured) == pytest.approx(window, rel=1e-12)
    assert device.drains == 0 and device.event_waits == 0


@pytest.mark.parametrize("hook", ["on_epoch_begin", "on_evaluate", "on_save", "on_train_begin"])
def test_a_step_after_a_new_epoch_an_evaluation_a_save_or_a_new_run_starts_at_its_own_begin(device, hook):
    """These gaps are not training, so the next step starts at its own begin; the steps around it keep their fetch."""
    callback = _callback()

    def train(steps) -> None:
        for step in steps:
            device.now += FETCH
            _step(callback, device, step, launch=LAUNCH, kernels=KERNELS)
            device.read_back()

    train((1, 2))
    device.now += 30.0
    getattr(callback, hook)(None, SimpleNamespace(global_step=2), None)
    train((3, 4))
    _log(callback, 4)

    own_span, chained = LAUNCH + KERNELS, FETCH + LAUNCH + KERNELS
    assert callback.measured_step_times == pytest.approx((own_span, chained, own_span, chained))


def test_a_log_waits_for_the_latest_step_only(device):
    """The host reads each gradient norm before stepping, so the device has reached the first end by the second."""
    callback = _callback()
    _step(callback, device, 1, launch=0.0, kernels=3.0)
    device.now = 3.0
    _step(callback, device, 2, launch=0.0, kernels=4.0)

    _log(callback, 2)

    assert device.drains == 0 and device.event_waits == 1
    assert callback.measured_step_times == (3.0, 4.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
