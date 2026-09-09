#!/usr/bin/env python
"""``bounded_event_sync`` is what stands between a wedged peer and a parked trainer.

Both clients drain their broadcasts through it, and it sits on the critical path once per chunk:
its deadline must fire (the group has no watchdog the trainer can rely on), its poll cadence is a
per-chunk tax, and the env override must stretch the drain deadline alone — a failure-path cleanup
under the same override would wait out the override before aborting.

    python tests/cpu/grpo/test_weight_sync_bounded_sync.py
"""

import time

import pytest

from src.distributed.nccl.transport import pynccl


class _Event:
    def __init__(self, ready_after_s: float):
        self._ready_at = time.monotonic() + ready_after_s

    def query(self) -> bool:
        return time.monotonic() >= self._ready_at


def test_a_never_ready_event_raises_at_the_deadline_instead_of_parking():
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="did not complete"):
        pynccl.bounded_event_sync(_Event(float("inf")), timeout_s=0.05, what="test broadcast")
    assert time.monotonic() - started < 1.0, "the deadline did not bound the wait"


def test_the_poll_cadence_is_kept_at_a_millisecond():
    """A 1 GiB chunk crosses EFA in ~11 ms and the wait runs once per chunk, so a coarse poll
    (50 ms) is a multiple of the transfer itself on every chunk."""
    started = time.monotonic()
    pynccl.bounded_event_sync(_Event(0.002), timeout_s=1.0, what="test broadcast")
    assert time.monotonic() - started < 0.04, "an event ready after 2 ms took a coarse poll interval to notice"


def test_the_env_override_replaces_the_drain_deadline_only(monkeypatch):
    monkeypatch.setattr(pynccl, "_SYNC_TIMEOUT_OVERRIDE", 2.0)
    assert pynccl.resolve_drain_timeout_s() == 2.0
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="did not complete"):
        pynccl.bounded_event_sync(_Event(float("inf")), timeout_s=0.02, what="cleanup")
    assert time.monotonic() - started < 1.0, "a short failure-path deadline was stretched by the drain override"

    monkeypatch.setattr(pynccl, "_SYNC_TIMEOUT_OVERRIDE", None)
    assert pynccl.resolve_drain_timeout_s() == pynccl.BROADCAST_DRAIN_TIMEOUT_S


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
