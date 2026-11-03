#!/usr/bin/env python
"""A failing checkpoint write must not strand other ranks in the next collective. :class:`DeferredRankFailure`
records it, keeps entering collectives, then makes the verdict uniform."""

import pytest

from halo.runtime.consensus import DeferredRankFailure


def test_failing_step_does_not_propagate():
    guard = DeferredRankFailure("test write")

    def boom():
        raise OSError(28, "No space left on device")

    assert guard.run(boom) is None
    assert guard.reason is not None
    assert "No space left on device" in guard.reason


def test_collectives_after_a_failure_still_run():
    """A short ``gathers`` is the rank skew that hangs the job."""
    guard = DeferredRankFailure("test write")
    gathers = []
    for layer in range(4):

        def write(layer=layer):
            if layer == 1:
                raise OSError(28, "No space left on device")

        gathers.append(layer)  # Stands in for the collective every rank must enter.
        guard.run(write)

    assert gathers == [0, 1, 2, 3]
    with pytest.raises(RuntimeError, match="No space left on device"):
        guard.reject()


def test_steps_after_a_failure_are_skipped():
    """The first cause is kept."""
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
    """``writer.close()`` goes through ``run`` for its value, so swallowing it would index an empty weight map."""
    guard = DeferredRankFailure("test write")
    assert guard.run(lambda: ({"a": "shard-0"}, 17)) == ({"a": "shard-0"}, 17)
    assert guard.reason is None
    guard.reject()


def test_reject_uses_the_callers_exception_type():
    guard = DeferredRankFailure("test config", exc_type=ValueError)
    guard.run(lambda: (_ for _ in ()).throw(ValueError("bad shape")))
    with pytest.raises(ValueError, match="bad shape"):
        guard.reject()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
