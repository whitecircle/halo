"""``sequential_load_within_node`` (``halo/runtime/load_coordination.py``): repeated uses of a tag stay isolated, or
the second runs unthrottled into the CPU-RAM OOM."""

import pytest
import torch.distributed as dist

from halo.env import store_timeout
from halo.runtime import consensus, load_coordination
from tests.common.distributed import FakeStore


@pytest.fixture()
def fake_node(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(consensus.c10d, "_get_default_store", lambda: store)
    monkeypatch.setattr(load_coordination, "get_local_world_size", lambda: 2)
    monkeypatch.setattr(load_coordination, "get_node_rank", lambda: 0)
    return store


def _run_call(monkeypatch, local_rank):
    monkeypatch.setattr(load_coordination, "get_local_rank", lambda: local_rank)
    with load_coordination.sequential_load_within_node(max_concurrent=1):
        pass


def test_repeated_use_writes_distinct_done_keys(fake_node, monkeypatch):
    _run_call(monkeypatch, local_rank=0)
    _run_call(monkeypatch, local_rank=0)
    assert len(fake_node.set_keys) == 2
    assert fake_node.set_keys[0] != fake_node.set_keys[1]


def test_second_use_throttles_again(fake_node, monkeypatch):
    """It must not sail through on the first round's done-key."""
    _run_call(monkeypatch, local_rank=0)  # Round 1: rank 0 done.
    _run_call(monkeypatch, local_rank=1)  # Round 1: rank 1 proceeds on rank 0's done-key.
    with pytest.raises(RuntimeError):  # Round 2: rank 1 waits on a round-2 key that is not set yet.
        _run_call(monkeypatch, local_rank=1)


def test_second_round_proceeds_in_order(fake_node, monkeypatch):
    _run_call(monkeypatch, local_rank=0)
    _run_call(monkeypatch, local_rank=1)
    _run_call(monkeypatch, local_rank=0)
    _run_call(monkeypatch, local_rank=1)


def test_phase_keys_cleaned_after_round(fake_node, monkeypatch):
    """Only the bounded per-participant entry counters may persist."""
    _run_call(monkeypatch, local_rank=0)
    _run_call(monkeypatch, local_rank=1)
    leftover = [k for k in fake_node.kv if "/p1/" in k]
    assert leftover == [], f"phase keys leaked: {leftover}"
    # Asserted by count so this cannot pass vacuously if the throttle no-ops.
    assert len(fake_node.kv) == 2, f"expected only the per-participant counters: {sorted(fake_node.kv)}"
    assert all("/entered" in k for k in fake_node.kv), f"unexpected persistent keys: {sorted(fake_node.kv)}"


def test_failed_wait_still_marks_this_rank_done(fake_node, monkeypatch):
    """Otherwise every later batch waits out the store timeout (hours) on a key that never appears."""
    monkeypatch.setattr(load_coordination, "get_local_world_size", lambda: 3)
    with pytest.raises(RuntimeError):
        _run_call(monkeypatch, local_rank=1)
    assert any(k.endswith("/rank1_done") for k in fake_node.set_keys), "a timed-out rank never marked itself done"

    _run_call(monkeypatch, local_rank=0)
    _run_call(monkeypatch, local_rank=2)  # Blocks forever on rank1_done unless the failure marked it.


def test_wait_failure_names_the_tag_and_the_divergence_cause(fake_node, monkeypatch):
    """A raw "socket timeout" cannot tell a peer still working from participants that disagree on entries or scope."""
    monkeypatch.setattr(load_coordination, "get_local_world_size", lambda: 3)
    with pytest.raises(RuntimeError) as excinfo:
        _run_call(monkeypatch, local_rank=2)  # Waits on ranks 0/1, which never ran.
    message = str(excinfo.value)
    assert "seq_load/model" in message, f"the tag must be named: {message}"
    assert "DIST_STORE_TIMEOUT_HOURS" in message, f"the knob to raise must be named: {message}"
    assert "entered this tag" in message or "scope" in message, f"the divergence cause must be named: {message}"


def test_under_counted_participants_raises_instead_of_hanging_the_node(fake_node, monkeypatch):
    """``get_local_world_size()`` falls back to the visible device count; under-reporting collects done-keys while
    unknown ranks still wait on them."""
    monkeypatch.setattr(load_coordination, "get_local_world_size", lambda: 2)
    with pytest.raises(RuntimeError, match="LOCAL_WORLD_SIZE"):
        consensus.StorePhase("seq_load/model", "node0", 3, 2, store_timeout())


def test_unthrottled_shapes_touch_no_store(fake_node, monkeypatch):
    monkeypatch.setattr(load_coordination, "get_local_rank", lambda: 0)
    with load_coordination.sequential_load_within_node(max_concurrent=0):
        pass
    monkeypatch.setattr(load_coordination, "get_local_world_size", lambda: 1)
    with load_coordination.sequential_load_within_node(max_concurrent=1):
        pass
    assert fake_node.set_keys == [] and fake_node.kv == {}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
