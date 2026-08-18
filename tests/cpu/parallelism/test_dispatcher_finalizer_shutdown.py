#!/usr/bin/env python
"""``DeepEPDispatcher.__del__`` must reach its shutdown check even after the module is torn down.

The finalizer skips ``destroy()`` during interpreter shutdown, because the CUDA context may already
be corrupted. That check has to run without touching a module global: torchrun teardown can null this
module's globals before the finalizer fires, and a bare ``sys.is_finalizing()`` then raises —
printing an "Exception ignored in: <function DeepEPDispatcher.__del__>" traceback at the end of every
EP run. So ``sys.is_finalizing`` is bound as a DEFAULT ARGUMENT, captured at definition time, and
lives in the function object rather than in module state.

That binding is also why this file patches ``__del__.__defaults__`` instead of ``sys.is_finalizing``:
the latter cannot reach the finalizer, which is the property under test. Nothing here
constructs a real dispatcher — it needs CUDA and DeepEP — and nothing needs to: the finalizer only
calls back into ``self``.

Run: ``python tests/cpu/parallelism/test_dispatcher_finalizer_shutdown.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import inspect
import sys

import pytest

from src.distributed.expert_parallel.dispatcher import DeepEPDispatcher


class _Dispatcher:
    """Everything ``__del__`` touches on ``self`` — which is only ``destroy``."""

    def __init__(self):
        self.destroy_calls: list[bool] = []

    def destroy(self, *, free_buffer: bool = True) -> None:
        self.destroy_calls.append(free_buffer)


def test_the_shutdown_probe_is_bound_at_definition_time():
    """The whole point of the fix: no module-global lookup happens inside the finalizer."""
    assert list(inspect.signature(DeepEPDispatcher.__del__).parameters) == ["self", "_is_finalizing"]
    assert DeepEPDispatcher.__del__.__defaults__ == (sys.is_finalizing,)


def test_destroy_is_skipped_during_interpreter_shutdown(monkeypatch):
    monkeypatch.setattr(DeepEPDispatcher.__del__, "__defaults__", (lambda: True,))
    stub = _Dispatcher()

    DeepEPDispatcher.__del__(stub)

    assert stub.destroy_calls == [], "destroy() on a torn-down CUDA context crashes the process"


def test_destroy_drops_the_claim_without_freeing_while_the_interpreter_runs(monkeypatch):
    """Anti-vacuity for the skip, and the second half of the finalizer's contract: freeing runs a
    collective on DeepEP's device-side barrier, and the GC fires at a moment no other rank agrees
    on — so a live-interpreter finalizer must drop the claim only."""
    monkeypatch.setattr(DeepEPDispatcher.__del__, "__defaults__", (lambda: False,))
    stub = _Dispatcher()

    DeepEPDispatcher.__del__(stub)

    assert stub.destroy_calls == [False]


def test_patching_sys_is_finalizing_does_not_reach_the_finalizer(monkeypatch):
    """Documents the consequence of the definition-time binding: the finalizer is deaf to a later
    rebind of ``sys.is_finalizing``, so a test that patched it would silently assert nothing."""
    monkeypatch.setattr(sys, "is_finalizing", lambda: True)
    stub = _Dispatcher()

    DeepEPDispatcher.__del__(stub)

    assert stub.destroy_calls == [False], "the default captured the ORIGINAL sys.is_finalizing"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
