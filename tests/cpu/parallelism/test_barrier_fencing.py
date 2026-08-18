#!/usr/bin/env python3
"""A rank-gated write must release its peers even when the write raises.

Every checkpoint writer is one rank doing I/O while all ranks must reach a barrier. Unfenced, a
local I/O failure becomes a job-wide hang: ENOSPC, EIO or a stale NFS handle on one node raises
past the barrier while its peers sit in it until the NCCL watchdog fires — and the traceback then
names the barrier, not the disk. At 512 ranks the chance that *some* node hits that is what makes
this a fence rather than a convention.

``barrier_on_exit`` owes two things at once: the peers are released, AND the writer's exception
still propagates (swallowing it would leave a truncated checkpoint that reads as a success).

Usage:
    python tests/cpu/parallelism/test_barrier_fencing.py
"""

import sys
from unittest.mock import patch

import pytest

from src.distributed import runtime
from src.distributed.runtime import barrier_on_exit

_MOD = "src.distributed.runtime"


def test_barrier_runs_when_the_body_raises():
    """The whole point: a raising writer must not strand its peers in the barrier."""
    with patch(f"{_MOD}.barrier") as fenced_barrier:
        with pytest.raises(OSError, match="No space left"):
            with barrier_on_exit():
                raise OSError("No space left on device")
        fenced_barrier.assert_called_once()


def test_the_writers_exception_still_propagates():
    """Releasing the peers must not swallow the failure — a truncated checkpoint is not a success."""
    with patch(f"{_MOD}.barrier"):
        with pytest.raises(RuntimeError, match="stale file handle"):
            with barrier_on_exit():
                raise RuntimeError("stale file handle")


def test_barrier_runs_after_the_body_on_the_happy_path():
    """Ordering is load-bearing: barrier before the write would publish a half-written checkpoint."""
    order = []
    with patch(f"{_MOD}.barrier", side_effect=lambda *a, **k: order.append("barrier")):
        with barrier_on_exit():
            order.append("write")
    assert order == ["write", "barrier"]


def test_the_group_is_forwarded():
    """A save fenced on a subgroup must barrier on THAT group — the default group would deadlock
    against ranks that never enter this code path."""
    sentinel = object()
    with patch(f"{_MOD}.barrier") as fenced_barrier:
        with barrier_on_exit(sentinel):
            pass
        fenced_barrier.assert_called_once_with(sentinel)


def test_barrier_is_a_no_op_without_a_process_group():
    """Single-process runs (and these tests) call the same writers; the fence must stay inert."""
    assert not runtime.dist.is_initialized()
    with barrier_on_exit():
        pass  # reaching here without raising IS the assertion


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
