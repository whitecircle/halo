"""CPU tests for the SGLang weight-sync client's teardown contract.

Four facts are load-bearing and invisible in a passing run:

  * the engine-side release of the group name runs WHILE the local ``dist.destroy_process_group``
    runs, not after it returns: under NCCL's cuMem transports each side's finalize waits for the
    other, so a local-first order parks the local destroy forever;
  * the atexit invocation must NOT enter ``dist.destroy_process_group``: the group's peers are
    engine processes that never enter destroy, so with an interrupted sync in flight the destroy
    BLOCKS rather than raises — wedging interpreter exit. The engine-side name release and the
    rendezvous-store drop still run, or the server refuses every future join under the name;
  * the explicit ``close_communicator()`` (the reconnect path) must keep the full local destroy,
    or the c10d group name and rendezvous port leak and the next client to the same server cannot
    form its group — unless a drain deadline aborted the group, after which torch has dropped its
    bookkeeping and a destroy raises;
  * the quiesce this teardown has to lift is lifted on SGLang's OWN route, with the body its
    handler requires — the class attributes that carry both are read once, on a failing path, where
    a wrong value is a warning line and an engine left paused.

    python tests/cpu/grpo/test_weight_sync_teardown.py
"""

import inspect
import sys
import threading
from unittest.mock import patch

import pytest
import torch

import src.distributed.nccl.clients.sglang as sglang_module
from src.distributed.nccl.clients.sglang import SGLangWeightSyncClient
from src.distributed.nccl.transport import torch_group


def _offline_client() -> SGLangWeightSyncClient:
    with patch.object(SGLangWeightSyncClient, "check_server"):
        return SGLangWeightSyncClient(base_url="http://localhost:30000")


def _teardown_probe(client, monkeypatch) -> list[tuple[str, object]]:
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(sglang_module, "destroy_weight_update_group", lambda group: calls.append(("destroy", group)))
    monkeypatch.setattr(
        sglang_module, "drop_weight_update_group_bookkeeping", lambda group: calls.append(("drop", group))
    )
    monkeypatch.setattr(client, "_destroy_remote_group", lambda: calls.append(("remote", None)))
    return calls


def test_the_pause_is_lifted_on_sglangs_own_route_with_a_body(monkeypatch):
    """``/continue_generation`` parses a request dataclass, so an empty body is REJECTED — and the
    rejection surfaces only as a warning from ``_lift_pause``, leaving the engine quiesced and every
    later rollout queued behind it. Route and body are both class attributes; neither is otherwise
    exercised outside a live server."""
    client = _offline_client()
    client._paused = True
    posts: list[tuple[str, dict]] = []
    monkeypatch.setattr(client, "_post_once", lambda path, **kwargs: posts.append((path, kwargs)))

    client._lift_pause(timeout=1.0, context="test")

    assert [path for path, _ in posts] == ["/continue_generation"], posts
    assert posts[0][1].get("json") == {}, "the handler rejects an empty body — RESUME_PAYLOAD must ship {}"
    assert client._paused is False


def test_explicit_close_keeps_the_full_local_destroy(monkeypatch):
    client = _offline_client()
    group = object()
    client._group = group
    calls = _teardown_probe(client, monkeypatch)

    client.close_communicator()

    assert ("destroy", group) in calls, "the reconnect path must destroy the local group or its name/port leak"
    assert all(kind != "drop" for kind, _ in calls)
    assert ("remote", None) in calls
    assert client._group is None and client._store is None


def test_the_engine_side_release_runs_while_the_local_destroy_blocks(monkeypatch):
    """The local destroy blocks until the engine drops its half, so the engine must already have been
    asked by then. The fake destroy waits for the engine-side release to start; a client that only
    releases the engine after the destroy returned never satisfies it."""
    client = _offline_client()
    client._group = object()
    remote_started = threading.Event()
    monkeypatch.setattr(client, "_destroy_remote_group", remote_started.set)
    seen_during_destroy: list[bool] = []
    monkeypatch.setattr(
        sglang_module,
        "destroy_weight_update_group",
        lambda group: seen_during_destroy.append(remote_started.wait(timeout=2.0)),
    )

    client.close_communicator()

    assert seen_during_destroy == [True], (
        "the engine-side /destroy_weights_update_group did not start while the local destroy ran — "
        "under cuMem transports that order parks destroy_process_group forever"
    )


def test_an_aborted_group_is_dropped_not_destroyed(monkeypatch):
    """After a drain deadline aborted the group torch has already removed its bookkeeping; a destroy
    raises "Invalid process group", and the engine-side release must still run."""
    client = _offline_client()
    group = object()
    client._group = group
    client._aborted = True
    calls = _teardown_probe(client, monkeypatch)

    client.close_communicator()

    assert all(kind != "destroy" for kind, _ in calls), "close entered dist.destroy_process_group on an aborted group"
    assert ("drop", group) in calls
    assert ("remote", None) in calls


def test_a_settle_deadline_aborts_the_group_and_retires_the_client(monkeypatch):
    """A peer that never joins parks the broadcast; the settle's deadline must abort the group (or
    every later device sync hangs behind the spinning kernel) and every later update must be refused.
    A slot with nothing in flight settles without touching the event machinery."""
    client = _offline_client()
    group = object()
    client._group = group
    aborted: list[object] = []
    monkeypatch.setattr(sglang_module.c10d, "_abort_process_group", aborted.append)

    def expired(event, timeout_s, what):
        raise RuntimeError(f"{what} did not complete")

    monkeypatch.setattr(sglang_module, "bounded_event_sync", expired)

    client._settle(0, timeout_s=1.0)  # nothing recorded for the slot: no wait, no abort
    assert aborted == []

    client._inflight[0] = object()
    with pytest.raises(RuntimeError, match="did not complete"):
        client._settle(0, timeout_s=1.0)

    assert aborted == [group], "the deadline did not abort the weight-update group"
    assert client._aborted is True
    assert client._inflight[0] is None, "a settled slot must not be waited on twice"
    with pytest.raises(RuntimeError, match="aborted"):
        client.begin_weight_update()


def test_chunks_are_staged_in_reused_arenas_settled_before_reuse(monkeypatch):
    """Per-tensor device allocations churn the allocator on the critical path and, once freed with a
    pending NCCL event, block the next upload inside the allocator; the arenas sidestep both. The
    views must reproduce every tensor's values, dtype and shape; an arena must be reused for a chunk
    that fits and grown for one that does not; and the sends it last carried must be settled before
    it is overwritten."""
    client = _offline_client()
    client._sync_device = torch.device("cpu")
    settled: list[object] = []
    monkeypatch.setattr(sglang_module, "bounded_event_sync", lambda event, timeout_s, what: settled.append(event))
    first = [torch.arange(6, dtype=torch.bfloat16).reshape(2, 3), torch.full((5,), 2.5, dtype=torch.float32)]

    staged = list(client._stage_on_device(first, 0))
    arena = client._device_arenas[0]
    assert arena is not None and arena.dtype == torch.uint8
    for view, tensor in zip(staged, first, strict=True):
        assert view.dtype == tensor.dtype and view.shape == tensor.shape
        assert torch.equal(view, tensor)
        assert view.untyped_storage().data_ptr() == arena.untyped_storage().data_ptr(), "a view outside the arena"
    assert settled == [], "nothing was in flight, so nothing was waited on"

    pending = object()
    client._inflight[0] = pending
    list(client._stage_on_device([torch.ones(3, dtype=torch.bfloat16)], 0))
    assert settled == [pending], "the arena's previous sends were not settled before its reuse"
    assert client._device_arenas[0] is arena, "a chunk that fits must reuse the arena"

    oversized = [torch.zeros(arena.numel() + 1, dtype=torch.uint8)]
    staged = list(client._stage_on_device(oversized, 0))
    assert client._device_arenas[0] is not arena and client._device_arenas[0].numel() >= oversized[0].numel()
    assert torch.equal(staged[0], oversized[0])
    assert client._device_arenas[1] is None, "the other slot is untouched"

    client.close_communicator(_local_destroy=False)
    assert client._device_arenas == [None, None] and client._inflight == [None, None], (
        "a closed client must not pin device memory"
    )


def test_the_atexit_invocation_skips_the_local_nccl_destroy(monkeypatch):
    client = _offline_client()
    group = object()
    client._group = group
    calls = _teardown_probe(client, monkeypatch)

    client.close_communicator(_local_destroy=False)

    assert all(kind != "destroy" for kind, _ in calls), (
        "the atexit path entered dist.destroy_process_group — against a dead engine that blocks "
        "instead of raising and wedges interpreter exit"
    )
    assert ("drop", group) in calls, "the rank-map bookkeeping must still be dropped"
    assert ("remote", None) in calls, "skipping the engine-side release leaves the server refusing every future join"
    assert client._group is None and client._store is None


def test_init_registers_the_no_destroy_variant_at_atexit():
    """Read from the source: the registration site lives inside ``init_communicator``, which needs a
    live engine to execute — but the flag it registers is what decides whether interpreter exit can
    wedge, so its spelling is the contract."""
    source = inspect.getsource(SGLangWeightSyncClient.init_communicator)
    assert "atexit.register(self.close_communicator, _local_destroy=False)" in source, (
        "init_communicator no longer registers the no-local-destroy teardown at atexit"
    )


def test_bookkeeping_drop_never_calls_destroy(monkeypatch):
    entered: list[object] = []
    monkeypatch.setattr(torch_group.dist, "destroy_process_group", lambda group: entered.append(group))

    group = object()
    torch_group.c10d._world.pg_group_ranks[group] = {0: 0}
    try:
        torch_group.drop_weight_update_group_bookkeeping(group)
        assert entered == [], "drop_weight_update_group_bookkeeping entered dist.destroy_process_group"
        assert group not in torch_group.c10d._world.pg_group_ranks
    finally:
        torch_group.c10d._world.pg_group_ranks.pop(group, None)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
