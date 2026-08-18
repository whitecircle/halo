"""CPU tests for ``sequential_load_within_node`` (``src/distributed/runtime.py``).

The throttle coordinates via done-keys in the torch.distributed store. Repeated uses of one tag
must stay isolated — otherwise the second use sees the first call's done-keys and runs completely
unthrottled (every rank loads at once → the CPU-RAM OOM the throttle exists to prevent). Isolation
comes from ``_StorePhase``: each participant's phase number derives from its own store-held entry
counter (no module-level state), phase keys are phase-namespaced, and the last participant to
leave a phase deletes its transient keys (bounded store footprint).

Each rank is simulated in this one test process against one shared fake store, which mirrors the
real store's cross-process visibility; the per-participant counters live in that store, so a
simulated rank's call count advances exactly as its own process's would.

    python tests/cpu/parallelism/test_sequential_load_throttle.py
"""

import sys

import pytest
import torch.distributed as dist

from src.distributed import filesystem, runtime
from tests.common.distributed import FakeStore


@pytest.fixture()
def fake_node(monkeypatch):
    """Pretend to be one rank of a 2-GPU node with a fake coordination store."""
    store = FakeStore()
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(runtime.c10d, "_get_default_store", lambda: store)
    monkeypatch.setattr(filesystem, "get_local_world_size", lambda: 2)
    monkeypatch.setattr(filesystem, "get_node_rank", lambda: 0)
    return store


def _run_call(monkeypatch, local_rank):
    """Simulate one use by ``local_rank`` (its phase number lives in the shared store)."""
    monkeypatch.setattr(filesystem, "get_local_rank", lambda: local_rank)
    with filesystem.sequential_load_within_node(tag="model", max_concurrent=1):
        pass


def test_repeated_use_writes_distinct_done_keys(fake_node, monkeypatch):
    """Rank 0's done-key must differ between two uses of the SAME tag — identical keys make the
    second use unthrottled for the waiting ranks."""
    _run_call(monkeypatch, local_rank=0)
    _run_call(monkeypatch, local_rank=0)
    assert len(fake_node.set_keys) == 2
    assert fake_node.set_keys[0] != fake_node.set_keys[1]


def test_second_use_throttles_again(fake_node, monkeypatch):
    """After a completed first round, a waiting rank's SECOND use must block until rank 0 finishes
    the second round — not sail through on the first round's done-key."""
    _run_call(monkeypatch, local_rank=0)  # round 1: rank 0 done
    _run_call(monkeypatch, local_rank=1)  # round 1: rank 1 proceeds on rank 0's done-key
    with pytest.raises(RuntimeError):  # round 2: rank 1 must wait on a round-2 key that isn't set yet
        _run_call(monkeypatch, local_rank=1)


def test_second_round_proceeds_in_order(fake_node, monkeypatch):
    """A second round coordinates like the first when rank 0 goes first."""
    _run_call(monkeypatch, local_rank=0)  # round 1
    _run_call(monkeypatch, local_rank=1)
    _run_call(monkeypatch, local_rank=0)  # round 2: rank 0 finishes ...
    _run_call(monkeypatch, local_rank=1)  # ... now rank 1 proceeds


def test_phase_keys_cleaned_after_round(fake_node, monkeypatch):
    """The last participant to leave a phase deletes its transient keys — only the bounded
    per-participant entry counters may persist."""
    _run_call(monkeypatch, local_rank=0)
    _run_call(monkeypatch, local_rank=1)
    leftover = [k for k in fake_node.kv if "/p1/" in k]
    assert leftover == [], f"phase keys leaked: {leftover}"
    # Exactly the two entry counters — asserted by count so this cannot pass vacuously if the
    # throttle ever no-ops and touches the store not at all.
    assert len(fake_node.kv) == 2, f"expected only the per-participant counters: {sorted(fake_node.kv)}"
    assert all("/entered" in k for k in fake_node.kv), f"unexpected persistent keys: {sorted(fake_node.kv)}"


def test_failed_wait_still_marks_this_rank_done(fake_node, monkeypatch):
    """A rank whose wait times out must STILL publish its done-key.

    Otherwise every later batch on the node waits out the full store timeout (hours) on a key that
    will never appear, and the phase's keys never get collected. Sequence on a 3-rank node: rank 1
    waits first and times out (rank 0 has not run), rank 0 then completes, and rank 2 — which waits
    on BOTH earlier ranks — must proceed.
    """
    monkeypatch.setattr(filesystem, "get_local_world_size", lambda: 3)
    with pytest.raises(RuntimeError):
        _run_call(monkeypatch, local_rank=1)
    assert any(k.endswith("/rank1_done") for k in fake_node.set_keys), "a timed-out rank never marked itself done"

    _run_call(monkeypatch, local_rank=0)
    _run_call(monkeypatch, local_rank=2)  # blocks forever on rank1_done unless the failure marked it


def test_wait_failure_names_the_tag_and_the_divergence_cause(fake_node, monkeypatch):
    """A store wait that runs out must be self-diagnosing.

    The two causes are indistinguishable from the raw store error (a bare "socket timeout"): either
    the writing peer is still working, or the participants disagree on how many times they entered
    this tag / on their coordination scope and this one is waiting on a key nobody will ever write.
    Without both named, the operator sees an unattributed multi-hour stall.
    """
    monkeypatch.setattr(filesystem, "get_local_world_size", lambda: 3)
    with pytest.raises(RuntimeError) as excinfo:
        _run_call(monkeypatch, local_rank=2)  # waits on ranks 0/1, which never ran
    message = str(excinfo.value)
    assert "seq_load/model" in message, f"the tag must be named: {message}"
    assert "DIST_STORE_TIMEOUT_HOURS" in message, f"the knob to raise must be named: {message}"
    assert "entered this tag" in message or "scope" in message, f"the divergence cause must be named: {message}"


def test_under_counted_participants_raises_instead_of_hanging_the_node(fake_node, monkeypatch):
    """A participant index outside the declared count must fail loud at construction.

    ``num`` comes from ``get_local_world_size()``, which falls back to the visible CUDA device count;
    a bare launcher with fewer visible devices than local ranks under-reports it. Over-reporting only
    leaks keys, but under-reporting makes the phase collect its done-keys as soon as that many ranks
    leave — while the ranks it does not know about are still waiting on them for the whole store
    timeout. Here LOCAL_WORLD_SIZE says 4 while only 2 devices are visible, so rank 3 is out of range
    and can prove the count wrong before anyone waits.
    """
    monkeypatch.setattr(filesystem, "get_local_world_size", lambda: 2)
    with pytest.raises(RuntimeError, match="LOCAL_WORLD_SIZE"):
        filesystem._StorePhase("seq_load/model", "node0", 3, 2, runtime.get_store_timeout())


def test_unthrottled_shapes_touch_no_store(fake_node, monkeypatch):
    monkeypatch.setattr(filesystem, "get_local_rank", lambda: 0)
    with filesystem.sequential_load_within_node(tag="model", max_concurrent=0):
        pass
    monkeypatch.setattr(filesystem, "get_local_world_size", lambda: 1)
    with filesystem.sequential_load_within_node(tag="model", max_concurrent=1):
        pass
    assert fake_node.set_keys == [] and fake_node.kv == {}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
