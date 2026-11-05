#!/usr/bin/env python
"""The phase timers' accounting (exclusive spans, the unaccounted remainder), the optimizer-side callback, the trainer
mixin's report over the ranks, and the efficiency callback's use of the moved boundary marks."""

from types import SimpleNamespace

import pytest
import torch

from halo.callbacks import efficiency
from halo.callbacks.phase_timers import MICRO_STEP_END, PhaseTimerCallback
from halo.diagnostics import phase_timers
from halo.diagnostics.phase_timers import UNACCOUNTED, PhaseTimers
from halo.trainers.grpo.phase_timing import PhaseTimedMixin
from tests.common.gloo import run_gloo_ranks


@pytest.fixture
def clock(monkeypatch):
    """A host clock the test steps by hand, and no GPU, so every mark is a clock reading."""
    now = [0.0]
    monkeypatch.setattr(phase_timers.time, "perf_counter", lambda: now[0])
    monkeypatch.setattr(phase_timers.torch.cuda, "is_available", lambda: False)
    return now


def test_a_disabled_timer_takes_no_mark_and_reports_nothing(monkeypatch):
    def forbidden():
        raise AssertionError("a disabled timer took a mark")

    monkeypatch.setattr(phase_timers, "mark_boundary", forbidden)
    monkeypatch.setattr(phase_timers.time, "perf_counter", forbidden)
    timers = PhaseTimers(enabled=False)
    with timers.span("micro_step"):
        timers.checkpoint(MICRO_STEP_END)
    timers.since(MICRO_STEP_END, "reduce")
    timers.begin_run()
    timers.finish_step()
    assert timers.span("a") is timers.span("b"), "the shared no-op, not an allocation per span"
    assert timers.drain() == {}


def test_a_suspended_timer_records_no_span_but_the_wall_time_counts(clock):
    timers = PhaseTimers(enabled=True)
    timers.begin_run()
    with timers.suspended():
        with timers.span("loss_forward"):
            clock[0] = 4.0
        timers.checkpoint(MICRO_STEP_END)
    clock[0] = 6.0
    timers.finish_step()

    assert timers.drain() == {UNACCOUNTED: 6.0}
    with timers.span("loss_forward"):
        pass
    assert "loss_forward" in {span.name for span in timers._spans}, "recording resumes after the block"


def test_a_span_reports_its_time_less_the_spans_nested_in_it(clock):
    timers = PhaseTimers(enabled=True)
    timers.begin_run()
    clock[0] = 1.0
    with timers.span("micro_step"):
        clock[0] = 2.0
        with timers.span("rollout", "host"):
            clock[0] = 5.0
        clock[0] = 6.0
    clock[0] = 10.0
    timers.finish_step()

    report = timers.drain()
    assert report["rollout"] == 3.0
    assert report["micro_step"] == 2.0, "5 s inside the span, 3 of them the rollout's"
    assert report[UNACCOUNTED] == 5.0, "10 s of wall time, 5 s in spans"


@pytest.mark.parametrize(("timed", "untimed_seconds"), [(("rollout", "build"), 0.0), (("rollout",), 4.0)])
def test_unaccounted_shows_a_phase_nobody_timed(clock, timed, untimed_seconds):
    """Both steps run two 4 s phases; the second times only one of them."""
    timers = PhaseTimers(enabled=True)
    timers.begin_run()
    for name in ("rollout", "build"):
        if name in timed:
            with timers.span(name):
                clock[0] += 4.0
        else:
            clock[0] += 4.0
    timers.finish_step()
    assert timers.drain()[UNACCOUNTED] == untimed_seconds


def test_the_report_is_per_step_and_resets(clock):
    timers = PhaseTimers(enabled=True)
    timers.begin_run()
    for _ in range(2):
        with timers.span("optimizer"):
            clock[0] += 2.0
        timers.finish_step()
    assert timers.drain()["optimizer"] == 2.0
    assert timers.drain() == {}, "drained spans and steps are gone"


def test_nothing_is_reported_before_a_step_ends(clock):
    timers = PhaseTimers(enabled=True)
    with timers.span("rollout"):
        clock[0] = 1.0
    assert timers.drain() == {}


def test_the_callback_times_the_reduce_and_the_optimizer_between_the_micro_steps_and_the_step_end(clock):
    timers = PhaseTimers(enabled=True)
    callback = PhaseTimerCallback(timers)
    callback.on_train_begin(None, None, None)
    clock[0] = 3.0
    timers.checkpoint(MICRO_STEP_END)
    clock[0] = 4.5
    callback.on_pre_optimizer_step(None, None, None)
    clock[0] = 8.5
    callback.on_optimizer_step(None, None, None)
    clock[0] = 9.0
    callback.on_step_end(None, None, None)

    report = timers.drain()
    assert report["reduce"] == 1.5
    assert report["optimizer"] == 4.0
    assert report[UNACCOUNTED] == 9.0 - 5.5


def test_efficiency_reads_the_shared_boundary_marks():
    assert efficiency.mark_boundary is phase_timers.mark_boundary
    assert efficiency.seconds_between is phase_timers.seconds_between


class _Loop:
    """What the mixin wraps: records the order of the inherited calls."""

    def __init__(self):
        self.calls = []
        self.model = SimpleNamespace(training=True)
        self.callbacks = []

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def training_step(self, model, inputs, num_items_in_batch=None):
        self.calls.append("training_step")
        return torch.tensor(1.0)

    def _compute_loss(self, model, inputs):
        self.calls.append("_compute_loss")
        return torch.tensor(2.0)

    def _finish_fsdp2_backward(self, model):
        self.calls.append("_finish_fsdp2_backward")

    def log(self, logs, start_time=None):
        self.calls.append("log")


class _Timed(PhaseTimedMixin, _Loop):
    pass


def _timed(enabled: bool) -> _Timed:
    trainer = _Timed()
    trainer._log_phase_times = enabled
    trainer._configure_phase_timers()
    return trainer


def test_the_flag_off_keeps_the_loop_untouched(clock):
    trainer = _timed(False)
    logs = {}
    assert trainer.training_step(None, None).item() == 1.0
    trainer._compute_loss(None, None)
    trainer._finish_fsdp2_backward(None)
    trainer.log(logs)
    assert trainer.calls == ["training_step", "_compute_loss", "_finish_fsdp2_backward", "log"]
    assert not trainer._phase_timers.enabled and not trainer.callbacks and logs == {}


def test_the_flag_on_reports_each_wrapped_phase_at_the_next_log(clock):
    trainer = _timed(True)
    assert len(trainer.callbacks) == 1
    trainer._configure_phase_timers()
    assert len(trainer.callbacks) == 1, "idempotent"
    timers = trainer._phase_timers
    timers.begin_run()
    for name in ("training_step", "_compute_loss", "_finish_fsdp2_backward"):
        clock[0] += 1.0
        if name == "training_step":
            trainer.training_step(None, None)
        elif name == "_compute_loss":
            trainer._compute_loss(None, None)
        else:
            trainer._finish_fsdp2_backward(None)
    timers.finish_step()

    logs = {}
    trainer.log(logs)
    assert set(logs) == {
        f"phase_time/{phase}{suffix}"
        for phase in ("micro_step", "loss_forward", "post_backward_finish", UNACCOUNTED)
        for suffix in ("_s", "_max_s")
    }
    assert logs["phase_time/loss_forward_s"] == logs["phase_time/loss_forward_max_s"]
    assert trainer.calls[-1] == "log"


def test_an_eval_log_reduces_nothing(clock):
    trainer = _timed(True)
    trainer.model.training = False
    logs = {}
    trainer.log(logs)
    assert logs == {} and trainer.calls == ["log"]


def _reduce_worker(rank: int) -> None:
    now = [0.0]
    phase_timers.time.perf_counter = lambda: now[0]
    trainer = _timed(True)
    trainer._phase_timers.begin_run()
    with trainer._phase_timers.span("rollout", "host"):
        now[0] = 1.0 + 2.0 * rank
    trainer._phase_timers.finish_step()
    logs = {}
    trainer.log(logs)
    assert logs["phase_time/rollout_s"] == 2.0, "the rank mean of 1 s and 3 s"
    assert logs["phase_time/rollout_max_s"] == 3.0, "the slowest rank"


def test_the_report_is_the_rank_mean_and_the_slowest_rank():
    run_gloo_ranks(_reduce_worker, 2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
