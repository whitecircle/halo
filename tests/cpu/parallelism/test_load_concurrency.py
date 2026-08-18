#!/usr/bin/env python
"""``resolve_load_concurrency`` — the per-node weight-load throttle must not disarm on a 4-GPU tray.

``sequential_load_within_node`` admits ``max_concurrent`` ranks per node at a time so N ranks do not
each materialize a 100B+ checkpoint into host RAM simultaneously. It disarms whenever the value is
at or above the node width — correct for an explicit request, ruinous as a flat default: on a
GB200/GB300 tray (LOCAL_WORLD_SIZE=4 — the very hardware NVL72 is built from) a default of 4 admits
every rank at once and the throttle protects nothing.

The unset spelling is therefore ``None``, resolved against the node; every explicit value passes
through verbatim, **including 4**. An in-band sentinel would make ``max_concurrent_loading: 4``
on a 4-GPU tray silently mean 2 — the opposite of what it says.

Run: pytest tests/cpu/parallelism/test_load_concurrency.py
"""

import sys

import pytest
import torch.distributed as dist

from src.distributed import filesystem, runtime
from src.distributed.filesystem import MAX_CONCURRENT_LOADING_CAP, resolve_load_concurrency
from tests.common.distributed import FakeStore


def test_four_gpu_node_still_throttles_when_unset():
    """The finding: LOCAL_WORLD_SIZE=4 with a flat default of 4 admitted the whole node at once."""
    resolved = resolve_load_concurrency(None, local_world_size=4)
    assert resolved == 2, f"a 4-GPU tray must keep a throttle, got {resolved}"
    assert resolved < 4, "a value at or above the node width disarms sequential_load_within_node"


def test_eight_gpu_node_keeps_the_documented_width():
    """Anti-regression: an 8-GPU node (the shape the recipes target) must still load 4 at a time."""
    assert resolve_load_concurrency(None, local_world_size=8) == MAX_CONCURRENT_LOADING_CAP == 4


def test_a_wide_node_is_capped_not_halved():
    """The cap is what keeps a 16-rank node from admitting 8 concurrent checkpoint reads."""
    assert resolve_load_concurrency(None, local_world_size=16) == 4
    assert resolve_load_concurrency(None, local_world_size=72) == 4


def test_two_gpu_node_never_resolves_below_one():
    """``local_world_size // 2`` is 1 here; 0 would mean 'no throttle', the opposite of the intent."""
    assert resolve_load_concurrency(None, local_world_size=2) == 1
    assert resolve_load_concurrency(None, local_world_size=1) == 1


def test_an_explicit_value_passes_through_untouched():
    """Explicit settings keep EXACT current semantics — including the two that disarm."""
    assert resolve_load_concurrency(0, local_world_size=4) == 0  # documented "all ranks at once"
    assert resolve_load_concurrency(1, local_world_size=8) == 1  # fully sequential
    assert resolve_load_concurrency(8, local_world_size=8) == 8  # explicitly disarmed
    assert resolve_load_concurrency(3, local_world_size=8) == 3


@pytest.mark.parametrize("local_world_size", [4, 8])
def test_an_explicit_four_is_not_read_as_the_unset_default(local_world_size):
    """The reason the sentinel is out-of-band.

    An in-band ``4`` as the "untouched" marker hands an operator who deliberately wrote
    ``max_concurrent_loading: 4`` on a 4-GPU tray a 2 — a value they never asked for, and one that
    halves load throughput on a box whose RAM is known to be fine.
    """
    assert resolve_load_concurrency(4, local_world_size=local_world_size) == 4


def test_the_throttle_actually_uses_the_resolved_width(monkeypatch):
    """End-to-end through ``sequential_load_within_node``: on a 4-rank node with nothing set, rank 2
    must WAIT for rank 0 (batch 0) instead of sailing through unthrottled.

    Asserted against the store rather than the resolver so a future refactor that computes the right
    number and then forgets to use it still fails.
    """
    store = FakeStore()
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(runtime.c10d, "_get_default_store", lambda: store)
    monkeypatch.setattr(filesystem, "get_local_world_size", lambda: 4)
    monkeypatch.setattr(filesystem, "get_node_rank", lambda: 0)
    monkeypatch.setattr(filesystem, "get_local_rank", lambda: 2)

    # FakeStore.wait raises instead of blocking, so "would have blocked" is observable.
    with pytest.raises(RuntimeError, match="seq_load/model"):
        with filesystem.sequential_load_within_node(tag="model", max_concurrent=None):
            pass


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
