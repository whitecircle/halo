#!/usr/bin/env python
"""``resolve_load_concurrency``: the per-node weight-load throttle must not disarm on a 4-GPU tray. The unset spelling
is ``None``; an in-band sentinel would make ``max_concurrent_loading: 4`` on such a tray silently mean 2."""

import pytest
import torch.distributed as dist

from halo.runtime import consensus, load_coordination
from halo.runtime.load_coordination import MAX_CONCURRENT_LOADING_CAP, resolve_load_concurrency
from tests.common.distributed import FakeStore


def test_four_gpu_node_still_throttles_when_unset():
    """A flat default of 4 would admit the whole node at once."""
    resolved = resolve_load_concurrency(None, local_world_size=4)
    assert resolved == 2, f"a 4-GPU tray must keep a throttle, got {resolved}"
    assert resolved < 4, "a value at or above the node width disarms throttled_load_within_node"


def test_eight_gpu_node_keeps_the_documented_width():
    assert resolve_load_concurrency(None, local_world_size=8) == MAX_CONCURRENT_LOADING_CAP == 4


def test_a_wide_node_is_capped_not_halved():
    """The cap keeps a 16-rank node from admitting 8 concurrent checkpoint reads."""
    assert resolve_load_concurrency(None, local_world_size=16) == 4
    assert resolve_load_concurrency(None, local_world_size=72) == 4


def test_two_gpu_node_never_resolves_below_one():
    """0 would mean no throttle."""
    assert resolve_load_concurrency(None, local_world_size=2) == 1
    assert resolve_load_concurrency(None, local_world_size=1) == 1


def test_an_explicit_value_passes_through_untouched():
    assert resolve_load_concurrency(0, local_world_size=4) == 0  # Documented as all ranks at once.
    assert resolve_load_concurrency(1, local_world_size=8) == 1
    assert resolve_load_concurrency(8, local_world_size=8) == 8  # Explicitly disarmed.
    assert resolve_load_concurrency(3, local_world_size=8) == 3


@pytest.mark.parametrize("local_world_size", [4, 8])
def test_an_explicit_four_is_not_read_as_the_unset_default(local_world_size):
    assert resolve_load_concurrency(4, local_world_size=local_world_size) == 4


def test_the_throttle_actually_uses_the_resolved_width(monkeypatch):
    """Against the store, so computing the right number and forgetting to use it still fails."""
    store = FakeStore()
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(consensus.c10d, "_get_default_store", lambda: store)
    monkeypatch.setattr(load_coordination, "get_local_world_size", lambda: 4)
    monkeypatch.setattr(load_coordination, "get_node_rank", lambda: 0)
    monkeypatch.setattr(load_coordination, "get_local_rank", lambda: 2)

    # FakeStore.wait raises instead of blocking, so a blocked wait is observable.
    with pytest.raises(RuntimeError, match="seq_load/model"):
        with load_coordination.throttled_load_within_node(max_concurrent=None):
            pass


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
