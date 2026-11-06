"""Opt-in per-phase timers: spans read off CUDA events recorded on the current stream, so timing a phase never drains
the queue as a clock read would. Durations resolve at :meth:`PhaseTimers.drain`, at a point that already syncs."""

import time
from collections import defaultdict
from contextlib import AbstractContextManager, contextmanager, nullcontext
from enum import StrEnum
from typing import Literal, NamedTuple

import torch

# A CUDA event on the current stream, or the host clock without a GPU.
BoundaryMark = torch.cuda.Event | float
# ``"host"``: a phase that ends in a collective or a host read, so the host clock is already fenced. ``"stream"``: a
# phase whose device work outlasts its launches.
SpanKind = Literal["host", "stream"]

UNACCOUNTED = "unaccounted"


class Phase(StrEnum):
    """The spans the on-policy GRPO trainers' step loop times, which every caller of :meth:`PhaseTimers.span` names."""

    ROLLOUT = "rollout"
    SYNC = "sync"
    BUILD = "build"
    NO_GRAD = "no_grad"
    NO_GRAD_REF = "no_grad_ref"
    MICRO_STEP = "micro_step"
    LOSS_FORWARD = "loss_forward"
    POST_BACKWARD_FINISH = "post_backward_finish"
    REDUCE = "reduce"
    OPTIMIZER = "optimizer"
    LOG_EMIT = "log_emit"


_NO_SPAN = nullcontext()


class _Span(NamedTuple):
    name: str
    parent: str | None
    start: BoundaryMark
    end: BoundaryMark


def mark_boundary() -> BoundaryMark:
    """An event recorded behind the queued work, timing a phase without draining the queue as a clock read would."""
    if not torch.cuda.is_available():
        return time.perf_counter()
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def seconds_between(start: BoundaryMark, end: BoundaryMark) -> float:
    """Seconds from ``start`` to ``end``, waiting only for an end event the device has not reached yet."""
    if isinstance(end, float):
        return end - start
    if not end.query():
        end.synchronize()
    return start.elapsed_time(end) / 1000.0


class PhaseTimers:
    """Spans keyed by phase name, each reported at its exclusive time: a span's duration less its directly nested
    spans'. Disabled, every method returns at once and records nothing."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._spans: list[_Span] = []
        self._open: list[str] = []
        self._checkpoints: dict[str, BoundaryMark] = {}
        self._steps = 0
        self._wall = 0.0
        self._step_started: float | None = None
        self._suspended = False

    def span(self, name: str, kind: SpanKind = "stream") -> AbstractContextManager[None]:
        """Time the block under ``name``, nested inside the span open around it."""
        return self._timed(name, kind) if self.enabled and not self._suspended else _NO_SPAN

    @contextmanager
    def suspended(self):
        """Record no span inside the block; the step's wall time still counts."""
        self._suspended = True
        try:
            yield
        finally:
            self._suspended = False

    def checkpoint(self, name: str) -> None:
        """Remember now under ``name``, for :meth:`since`."""
        if self.enabled:
            self._checkpoints[name] = mark_boundary()

    def since(self, checkpoint: str, name: str) -> None:
        """Record a span called ``name`` from ``checkpoint`` to now."""
        if self.enabled and not self._suspended:
            self._spans.append(_Span(name, None, self._checkpoints[checkpoint], mark_boundary()))

    def begin_run(self) -> None:
        if self.enabled:
            self._step_started = time.perf_counter()

    def finish_step(self) -> None:
        """One optimizer step ended: its wall time is what the spans must add up to."""
        if not self.enabled:
            return
        now = time.perf_counter()
        if self._step_started is not None:
            self._wall += now - self._step_started
        self._step_started = now
        self._steps += 1

    def drain(self) -> dict[str, float]:
        """Seconds per optimizer step since the last drain, by phase, with :data:`UNACCOUNTED` the wall time no span
        holds (negative where spans overlap that should nest). Empty before the first step ends. Waits for events the
        device has not reached."""
        if not self.enabled or not self._steps:
            return {}
        inclusive: dict[str, float] = defaultdict(float)
        nested: dict[str, float] = defaultdict(float)
        for span in self._spans:
            seconds = seconds_between(span.start, span.end)
            inclusive[span.name] += seconds
            if span.parent is not None:
                nested[span.parent] += seconds
        # Clamped: a host span inside a stream span is read off another clock, so their difference can dip below zero.
        exclusive = {name: max(total - nested[name], 0.0) for name, total in inclusive.items()}
        steps = self._steps
        report = {name: seconds / steps for name, seconds in exclusive.items()}
        report[UNACCOUNTED] = (self._wall - sum(exclusive.values())) / steps
        self._spans.clear()
        self._steps, self._wall = 0, 0.0
        return report

    @contextmanager
    def _timed(self, name: str, kind: SpanKind):
        parent = self._open[-1] if self._open else None
        start = time.perf_counter() if kind == "host" else mark_boundary()
        self._open.append(name)
        try:
            yield
        finally:
            self._open.pop()
            end = time.perf_counter() if kind == "host" else mark_boundary()
            self._spans.append(_Span(name, parent, start, end))
