"""CPU tests for the SGLang weight-sync client's teardown contract.

Two facts are load-bearing and invisible in a passing run:

  * the atexit invocation must NOT enter ``dist.destroy_process_group``: the group's peers are
    engine processes that never enter destroy, so with an interrupted sync in flight the destroy
    BLOCKS rather than raises — wedging interpreter exit. The engine-side name release and the
    rendezvous-store drop still run, or the server refuses every future join under the name;
  * the explicit ``close_communicator()`` (the reconnect path) must keep the full local destroy,
    or the c10d group name and rendezvous port leak and the next client to the same server cannot
    form its group;
  * the quiesce this teardown has to lift is lifted on SGLang's OWN route, with the body its
    handler requires — the class attributes that carry both are read once, on a failing path, where
    a wrong value is a warning line and an engine left paused.

    python tests/cpu/grpo/test_weight_sync_teardown.py
"""

import inspect
import sys
from unittest.mock import patch

import pytest

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
