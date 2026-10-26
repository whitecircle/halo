"""CPU tests for the SGLang weight-sync client's teardown contract.

Five facts are load-bearing and invisible in a passing run:

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
  * that abort never raises: it runs inside the drain-deadline handler, where an escaping error
    would replace the deadline error the operator needs;
  * the quiesce this teardown has to lift is lifted on SGLang's OWN route, with the body its
    handler requires — the class attributes that carry both are read once, on a failing path, where
    a wrong value is a warning line and an engine left paused.

    python tests/cpu/weight_sync/test_teardown.py
"""

import inspect
import logging
import threading

import pytest

import src.weight_sync.clients.sglang as sglang_module
from src.weight_sync.clients.sglang import SGLangWeightSyncClient
from src.weight_sync.transport import c10d_group
from tests.common.weight_sync import offline_client


def _teardown_probe(client, monkeypatch) -> list[tuple[str, object]]:
    """Records the local release as ``("destroy" | "drop", group)`` and the engine-side one as ``("remote", None)``."""
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        sglang_module,
        "release_weight_update_group",
        lambda group, *, destroy: calls.append(("destroy" if destroy else "drop", group)),
    )
    monkeypatch.setattr(client, "_destroy_remote_group", lambda: calls.append(("remote", None)))
    return calls


def test_the_pause_is_lifted_on_sglangs_own_route_with_a_body(monkeypatch):
    """``/continue_generation`` parses a request dataclass, so an empty body is REJECTED — and the
    rejection surfaces only as a warning from ``_lift_pause``, leaving the engine quiesced and every
    later rollout queued behind it. Route and body are both class attributes; neither is otherwise
    exercised outside a live server."""
    client = offline_client(SGLangWeightSyncClient)
    client._paused = True
    posts: list[tuple[str, dict]] = []
    monkeypatch.setattr(client, "request", lambda method, path, **kwargs: posts.append((path, kwargs)))

    client._lift_pause(timeout=1.0, context="test")

    assert [path for path, _ in posts] == ["/continue_generation"], posts
    assert posts[0][1].get("json") == {}, "the handler rejects an empty body — RESUME_REQUEST must ship {}"
    assert client._paused is False


def test_explicit_close_keeps_the_full_local_destroy(monkeypatch):
    client = offline_client(SGLangWeightSyncClient)
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
    client = offline_client(SGLangWeightSyncClient)
    client._group = object()
    remote_started = threading.Event()
    monkeypatch.setattr(client, "_destroy_remote_group", remote_started.set)
    seen_during_destroy: list[bool] = []
    monkeypatch.setattr(
        sglang_module,
        "release_weight_update_group",
        lambda group, *, destroy: seen_during_destroy.append(destroy and remote_started.wait(timeout=2.0)),
    )

    client.close_communicator()

    assert seen_during_destroy == [True], (
        "the engine-side /destroy_weights_update_group did not start while the local destroy ran — "
        "under cuMem transports that order parks destroy_process_group forever"
    )


def test_an_aborted_group_is_dropped_not_destroyed(monkeypatch):
    """After a drain deadline aborted the group torch has already removed its bookkeeping; a destroy
    raises "Invalid process group", and the engine-side release must still run."""
    client = offline_client(SGLangWeightSyncClient)
    group = object()
    client._group = group
    client._aborted = True
    calls = _teardown_probe(client, monkeypatch)

    client.close_communicator()

    assert all(kind != "destroy" for kind, _ in calls), "close entered dist.destroy_process_group on an aborted group"
    assert ("drop", group) in calls
    assert ("remote", None) in calls


def test_a_client_that_never_formed_its_group_does_not_ask_the_engine_to_release_one(monkeypatch):
    """A failed or never-attempted formation already released the engine side; a second close would
    only post a redundant release and delay the teardown by its timeout."""
    client = offline_client(SGLangWeightSyncClient)
    calls = _teardown_probe(client, monkeypatch)

    client.close_communicator()

    assert calls == [], f"close touched the group machinery with no group formed: {calls}"


def test_the_atexit_invocation_skips_the_local_nccl_destroy(monkeypatch):
    client = offline_client(SGLangWeightSyncClient)
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


def test_a_release_without_destroy_never_enters_the_nccl_destroy(monkeypatch):
    entered: list[object] = []
    monkeypatch.setattr(c10d_group.dist, "destroy_process_group", lambda group: entered.append(group))

    group = object()
    c10d_group.c10d._world.pg_group_ranks[group] = {0: 0}
    try:
        c10d_group.release_weight_update_group(group, destroy=False)
        assert entered == [], "a release without destroy entered dist.destroy_process_group"
        assert group not in c10d_group.c10d._world.pg_group_ranks, "the rank-map entry outlived the release"
    finally:
        c10d_group.c10d._world.pg_group_ranks.pop(group, None)


def test_a_failing_destroy_warns_and_still_drops_the_rank_map_entry(monkeypatch, caplog):
    """The destroy runs against an engine that may already be gone; its failure must not mask the error
    that triggered the teardown, and the group's rank map must not outlive it."""

    def failing_destroy(group):
        raise RuntimeError("Invalid process group")

    monkeypatch.setattr(c10d_group.dist, "destroy_process_group", failing_destroy)
    group = object()
    c10d_group.c10d._world.pg_group_ranks[group] = {0: 0}
    try:
        with caplog.at_level(logging.WARNING, logger=c10d_group.logger.name):
            c10d_group.release_weight_update_group(group, destroy=True)
        assert group not in c10d_group.c10d._world.pg_group_ranks, "the rank-map entry outlived a failed destroy"
        assert any("Invalid process group" in record.getMessage() for record in caplog.records)
    finally:
        c10d_group.c10d._world.pg_group_ranks.pop(group, None)


def test_a_failing_abort_warns_instead_of_raising(monkeypatch, caplog):
    def failing_abort(group):
        raise RuntimeError("abort refused by NCCL")

    monkeypatch.setattr(c10d_group.c10d, "_abort_process_group", failing_abort)

    with caplog.at_level(logging.WARNING, logger=c10d_group.logger.name):
        c10d_group.abort_weight_update_group(object())

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("abort refused by NCCL" in message for message in warnings), (
        f"a failed abort must surface as a warning naming the error, got {warnings}"
    )


def test_aborting_a_group_that_never_formed_does_not_enter_torch(monkeypatch):
    entered: list[object] = []
    monkeypatch.setattr(c10d_group.c10d, "_abort_process_group", entered.append)

    c10d_group.abort_weight_update_group(None)

    assert entered == [], "abort_weight_update_group(None) called torch's abort"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
