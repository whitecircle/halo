#!/usr/bin/env python
"""``DeepEPDispatcher.__del__`` reaches its shutdown check after module teardown, with ``sys.is_finalizing`` bound
as a default argument, hence this patches ``__del__.__defaults__``."""

from __future__ import annotations

import inspect
import sys

import pytest

from halo.distributed.expert_parallel.deepep.dispatcher import DeepEPDispatcher


class _Dispatcher:
    """``__del__`` touches only ``destroy`` on ``self``."""

    def __init__(self):
        self.destroy_calls: list[bool] = []

    def destroy(self, *, free_buffer: bool = True) -> None:
        self.destroy_calls.append(free_buffer)


def test_the_shutdown_probe_is_bound_at_definition_time():
    """No module-global lookup happens inside the finalizer."""
    assert list(inspect.signature(DeepEPDispatcher.__del__).parameters) == ["self", "_is_finalizing"]
    assert DeepEPDispatcher.__del__.__defaults__ == (sys.is_finalizing,)


def test_destroy_is_skipped_during_interpreter_shutdown(monkeypatch):
    monkeypatch.setattr(DeepEPDispatcher.__del__, "__defaults__", (lambda: True,))
    stub = _Dispatcher()

    DeepEPDispatcher.__del__(stub)

    assert stub.destroy_calls == [], "destroy() on a torn-down CUDA context crashes the process"


def test_destroy_drops_the_claim_without_freeing_while_the_interpreter_runs(monkeypatch):
    """Freeing runs a collective on DeepEP's barrier, and GC fires at a moment no other rank agrees on, so a live
    finalizer only drops the claim."""
    monkeypatch.setattr(DeepEPDispatcher.__del__, "__defaults__", (lambda: False,))
    stub = _Dispatcher()

    DeepEPDispatcher.__del__(stub)

    assert stub.destroy_calls == [False]


def test_patching_sys_is_finalizing_does_not_reach_the_finalizer(monkeypatch):
    """A test patching ``sys.is_finalizing`` would silently assert nothing."""
    monkeypatch.setattr(sys, "is_finalizing", lambda: True)
    stub = _Dispatcher()

    DeepEPDispatcher.__del__(stub)

    assert stub.destroy_calls == [False], "the default captured the ORIGINAL sys.is_finalizing"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
