#!/usr/bin/env python
"""A checkpoint write that fails must not strand the other ranks in the next collective.

The gathered EP save and the PP stage save both STREAM: the save rank's disk writes sit between the
per-layer expert all-gathers. A write that raises there takes that rank out of the loop while its
peers block in layer k+1's gather until the NCCL watchdog fires — the failure is reported as a
timeout on a collective, minutes later, naming neither the disk nor the layer.

:class:`DeferredRankFailure` is the seam: record the failure, keep entering the collectives, then
make the verdict uniform. The assertions below pin the two halves that matter — that a failing step
does NOT propagate out of ``run`` (so the loop continues into its collectives), and that ``reject``
still raises afterwards carrying the original cause (so the failure is never swallowed).

Run: pytest tests/cpu/checkpoint/test_deferred_write_failure.py
"""

import sys

import pytest

from src.distributed.runtime import DeferredRankFailure


def test_failing_step_does_not_propagate():
    """The whole point: a raising write is recorded, so the caller reaches its next collective."""
    guard = DeferredRankFailure("test write")

    def boom():
        raise OSError(28, "No space left on device")

    assert guard.run(boom) is None
    assert guard.reason is not None
    assert "No space left on device" in guard.reason


def test_collectives_after_a_failure_still_run():
    """Models the streaming loop: layer 1's write fails, layers 2..N must still gather.

    A ``run`` that re-raised (or a caller that stopped looping) would leave ``gathers`` short — which
    at runtime is precisely the rank-skew that hangs the job.
    """
    guard = DeferredRankFailure("test write")
    gathers = []
    for layer in range(4):

        def write(layer=layer):
            if layer == 1:
                raise OSError(28, "No space left on device")

        gathers.append(layer)  # stands in for the collective every rank must enter
        guard.run(write)

    assert gathers == [0, 1, 2, 3]
    with pytest.raises(RuntimeError, match="No space left on device"):
        guard.reject()


def test_steps_after_a_failure_are_skipped():
    """Once the filesystem has failed, later writes are not attempted and the FIRST cause is kept."""
    guard = DeferredRankFailure("test write")
    attempted = []

    def first():
        attempted.append("first")
        raise OSError(28, "No space left on device")

    def second():
        attempted.append("second")
        raise OSError(5, "Input/output error")

    guard.run(first)
    guard.run(second)

    assert attempted == ["first"]
    assert "No space left on device" in guard.reason


def test_clean_run_is_transparent():
    """No failure → the step's return value passes straight through and reject() is a no-op.

    ``writer.close()`` is staged through ``run`` for its return value, so a seam that swallowed it
    would silently index an empty weight map.
    """
    guard = DeferredRankFailure("test write")
    assert guard.run(lambda: ({"a": "shard-0"}, 17)) == ({"a": "shard-0"}, 17)
    assert guard.reason is None
    guard.reject()  # must not raise


def test_reject_uses_the_callers_exception_type():
    """A caller documenting ValueError must keep raising ValueError once made rank-uniform."""
    guard = DeferredRankFailure("test config", exc_type=ValueError)
    guard.run(lambda: (_ for _ in ()).throw(ValueError("bad shape")))
    with pytest.raises(ValueError, match="bad shape"):
        guard.reject()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
