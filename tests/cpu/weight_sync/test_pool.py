#!/usr/bin/env python
"""The weight-sync pool keeps several rollout servers in step with one push.

What is pinned: each server's group port (its own, else the run's base port plus its index); ONE
snapshot per param on the pool's resolved sync device, shared read-only by every client and released
once every server has sent the chunk holding it; no server quiesced before a full chunk exists, and none
for a param the pool refuses; a failed server isolated from its healthy siblings and the failure raised;
a server that failed AFTER its first chunk went out NOT reconnected (the trainer kept no copy of what
landed, so a reconnect would leave that engine part old and part new while still answering /health),
one that failed before streaming reconnected and re-sent; and a reconnect that retires the old client
first and builds the replacement like the first connect (port, NIC, device and co-load scope).

Run: ``python tests/cpu/weight_sync/test_pool.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import weakref

import pytest
import torch

from src.weight_sync.chunking import declared_co_load_groups, resolve_sync_device
from src.weight_sync.clients.sglang import SGLangWeightSyncClient
from src.weight_sync.clients.vllm import VLLMWeightSyncClient
from src.weight_sync.pool import WeightSyncPool
from tests.common.weight_sync import ServerlessSGLangClient, Wire, offline_client

_BUDGET = 4096


def _live(refs: list[weakref.ref]) -> int:
    return sum(1 for ref in refs if ref() is not None)


def _bare_pool(num_clients: int, wires: list[Wire] | None = None) -> tuple[WeightSyncPool, list[VLLMWeightSyncClient]]:
    """A pool over ``num_clients`` bare clients, without the NCCL/HTTP handshake."""
    configs = [{"url": f"http://server{i}:8000", "group_port": 51216 + i} for i in range(num_clients)]
    pool = WeightSyncPool(
        server_configs=configs, connection_timeout=0.0, client_cls=VLLMWeightSyncClient, base_group_port=51216
    )
    clients = []
    for index, config in enumerate(configs):
        client = (wires[index] if wires else Wire()).attach(offline_client(VLLMWeightSyncClient))
        client.base_url = config["url"]  # the pool's failure report names the server
        clients.append(client)
    pool._clients = clients
    pool._initialized = True
    return pool, clients


def test_unconfigured_group_ports_are_seeded_from_the_run_knob():
    """``vllm_group_port`` must mean the same thing whichever client shape is built.

    An entry without its own ``group_port`` takes ``base_group_port + index``; binding the module
    default instead would listen on 51216/51217 for a run that configured 52000, while the engines
    dial the port they were launched with — every sync then times out with nothing to point at.
    """
    pool = WeightSyncPool(
        server_configs=[
            {"url": "http://s0:8000"},
            {"url": "http://s1:8000"},
            {"url": "http://s2:8000", "group_port": 9000},
        ],
        connection_timeout=0.0,
        client_cls=VLLMWeightSyncClient,
        base_group_port=52000,
    )
    assert [pool._group_port(i) for i in range(3)] == [52000, 52001, 9000]


def test_pool_stages_one_snapshot_per_param(monkeypatch):
    """The multi-server fan-out must stage ONE snapshot per param, not one per client."""
    pool, clients = _bare_pool(3)

    allocations = []
    real_empty_like = torch.empty_like

    def counting_empty_like(*args, **kwargs):
        allocations.append(kwargs.get("device"))
        return real_empty_like(*args, **kwargs)

    monkeypatch.setattr(torch, "empty_like", counting_empty_like)

    weights = torch.randn(8, 8)
    pool.update_named_param("model.layers.0.q_proj.weight", weights)

    assert len(allocations) == 1, (
        f"{len(allocations)} staged allocations for one param across {len(clients)} clients — "
        f"per-client copies multiply the staged chunk by the server count (must be exactly 1)"
    )
    buffered = [client._param_buffer[0][1] for client in clients]
    assert all(t is buffered[0] for t in buffered), "clients must share ONE snapshot by reference"
    assert all(client._param_buffer[0][0] == "model.layers.0.q_proj.weight" for client in clients)


def test_pool_stages_shared_snapshots_on_its_normalized_device(monkeypatch):
    """``init_communicator(0)`` must stage on CUDA device 0: the pool stores the resolved
    ``torch.device``, so an int index (0 included, which is falsy) or a bare ``"cuda"`` names the same
    GPU the clients join on."""
    assert resolve_sync_device(0) == torch.device("cuda", 0)
    joined: list = []

    class _StubClient:
        BACKEND_NAME = "stub"

        def __init__(self, base_url, group_port, connection_timeout, group_host=None):
            self._param_buffer: list = []

        def init_communicator(self, device):
            joined.append(device)

        def buffer_param(self, name, snapshot):
            self._param_buffer.append((name, snapshot))

    pool = WeightSyncPool(
        server_configs=[{"url": "http://server0:8000"}, {"url": "http://server1:8000"}],
        connection_timeout=0.0,
        client_cls=_StubClient,
        base_group_port=51216,
    )
    pool.init_communicator(0)
    assert pool._device == torch.device("cuda", 0), f"stored {pool._device!r} for device index 0"
    assert joined == [torch.device("cuda", 0)] * 2, "clients must join on the same normalized device"

    staged_on: list = []

    def fake_snapshot(name, weights, *, device, **cut):
        staged_on.append(device)
        return weights.detach().clone()

    monkeypatch.setattr("src.weight_sync.pool.snapshot_to_sync_device", fake_snapshot)
    pool.update_named_param("w", torch.zeros(4))
    assert staged_on == [torch.device("cuda", 0)], "the shared snapshot was not staged on the sync device"


def test_pool_shared_snapshot_is_immutable_copy():
    """The shared snapshot must still be a copy: a post-buffer in-place mutation (the next optimizer
    step) must not revert any client's buffered weight."""
    pool, clients = _bare_pool(2)
    weights = torch.randn(4, 4)  # contiguous, so an aliasing .contiguous() would return it as-is
    original = weights.clone()

    pool.update_named_param("w", weights)
    weights.add_(1.0)  # simulate an optimizer step before the flush

    for client in clients:
        _, stored = client._param_buffer[0]
        assert stored.data_ptr() != weights.data_ptr(), "buffered by reference to the source"
        assert torch.equal(stored, original), "shared snapshot mutated by a later in-place update"


def test_pool_releases_a_chunks_snapshots_once_every_server_sent_it(monkeypatch):
    """Multi-server: a shared snapshot must die once the chunk holding it went out everywhere — the
    pool keeps no store of its own, so a reference kept anywhere would grow the staged footprint with
    the model rather than the chunk."""
    monkeypatch.setattr("src.weight_sync.pool.WEIGHT_SYNC_CHUNK_BYTES", 4096)
    pool, clients = _bare_pool(2, wires=[Wire(retain=False), Wire(retain=False)])
    refs: list[weakref.ref] = []

    for index in range(10):
        pool.update_named_param(f"w{index}", torch.zeros(128, dtype=torch.float32))
        refs.append(weakref.ref(clients[0]._param_buffer[-1][1]))
    assert 0 < _live(refs) < len(refs), "no chunk went out mid-stream — the flush never ran"
    pool.finish_weight_update()

    assert _live(refs) == 0, f"{_live(refs)} snapshots still referenced after the sync"
    assert all(client._param_buffer == [] for client in clients)


def test_no_server_is_quiesced_until_a_chunk_is_ready_to_send(monkeypatch):
    """The quiesce window opens at the first FULL chunk, not at the first gathered parameter.

    Every server stops serving for as long as its update is open, so the update must not be opened
    while the gather is still assembling the first chunk's worth of weights.
    """
    monkeypatch.setattr("src.weight_sync.pool.WEIGHT_SYNC_CHUNK_BYTES", 4096)
    wires = [Wire(), Wire()]
    pool, _clients = _bare_pool(2, wires=wires)

    for index in range(8):  # 512 B each: exactly the budget, nothing to send yet
        pool.update_named_param(f"w{index}", torch.zeros(128, dtype=torch.float32))
    assert [wire.opened for wire in wires] == [0, 0], "a server was quiesced before a full chunk existed"

    pool.update_named_param("w8", torch.zeros(128, dtype=torch.float32))
    assert [wire.opened for wire in wires] == [1, 1], "the full chunk did not open the update"

    pool.finish_weight_update()
    assert [wire.closed for wire in wires] == [1, 1], "the update stayed open after the final flush"


def test_the_pool_refuses_an_unsyncable_param_before_any_server_gets_the_pending_chunk(monkeypatch):
    """The pool snapshots for its clients, so their own check never runs: with a full chunk pending, a
    0-dim param must be refused before that chunk quiesces a server for a sync that cannot complete."""
    monkeypatch.setattr("src.weight_sync.pool.WEIGHT_SYNC_CHUNK_BYTES", _BUDGET)
    wires = [Wire(), Wire()]
    pool, clients = _bare_pool(2, wires=wires)
    pool.update_named_param("w", torch.zeros(_BUDGET // 4, dtype=torch.float32))

    with pytest.raises(ValueError, match="0-dimensional"):
        pool.update_named_param("scale", torch.tensor(1.0))

    assert [wire.opened for wire in wires] == [0, 0], "a server was quiesced for a param the pool then refused"
    assert [wire.chunks for wire in wires] == [[], []], "the pending chunk went out for a refused param"
    assert all(client._chunks_sent == 0 and not client._update_open for client in clients)
    pool.update_named_param("bias", torch.zeros(1, dtype=torch.float32))
    assert [wire.opened for wire in wires] == [1, 1], "the pending chunk was not full: nothing was at stake"


def test_pool_flush_failure_isolation():
    """A failed flush on server A clears only A's buffer; B still flushes the intact shared
    snapshot afterwards, and the pool raises (fail loud, no stale-policy rollouts)."""
    wires = [Wire(fail_on_send=True), Wire(), Wire()]
    pool, clients = _bare_pool(3, wires=wires)

    weights = torch.randn(4, 4)
    expected = weights.clone()
    pool.update_named_param("w", weights)

    with pytest.raises(RuntimeError, match="1/3 vLLM server"):
        pool.finish_weight_update()

    assert clients[0]._param_buffer == [], "failed client's buffer must clear (no re-broadcast)"
    for index in (1, 2):
        assert clients[index]._param_buffer == [], "healthy client must drain despite the sibling failure"
        (name, sent) = wires[index].sent[0]
        assert name == "w"
        assert torch.equal(sent, expected), "healthy flush read a corrupted/freed shared snapshot"


def test_a_server_that_already_streamed_a_chunk_is_not_retried(monkeypatch):
    """No replay once the engine holds part of the new weights.

    The trainer keeps no copy of what already went out, so a reconnect + re-flush would hand the
    fresh engine the tail alone — a model that is part old and part new, serving happily and
    answering /health. The failure has to reach the caller with that reason instead.
    """
    monkeypatch.setattr("src.weight_sync.pool.WEIGHT_SYNC_CHUNK_BYTES", 4096)
    wire = Wire()
    pool, clients = _bare_pool(1, wires=[wire])
    reconnects = []
    monkeypatch.setattr(WeightSyncPool, "_reconnect", lambda self, index: reconnects.append(index))

    for index in range(10):  # crosses the budget, so a chunk is on the wire before the tail
        pool.update_named_param(f"w{index}", torch.zeros(128, dtype=torch.float32))
    assert clients[0]._chunks_sent > 0, "the fixture never streamed a chunk — the test would be vacuous"
    assert not clients[0].can_replay_sync, "a client that streamed a chunk must report itself unreplayable"

    wire.fail_on_send = True
    with pytest.raises(RuntimeError, match="already streaming"):
        pool.finish_weight_update()

    assert reconnects == [], "a half-updated engine was reconnected and re-sent only the tail"
    assert clients[0]._param_buffer == [], "the unsendable buffer must be dropped, not re-broadcast"


def test_a_server_that_failed_before_streaming_is_still_retried(monkeypatch):
    """Anti-over-rejection: the dominant recovery case — an engine restarted BETWEEN syncs, whose
    first chunk fails — is still reconnected and re-sent, because nothing has landed on it yet."""
    wire = Wire(fail_on_send=True)
    pool, clients = _bare_pool(1, wires=[wire])

    def reconnect(self, index):
        wire.fail_on_send = False  # the replacement engine accepts the replay

    monkeypatch.setattr(WeightSyncPool, "_reconnect", reconnect)

    pool.update_named_param("w", torch.zeros(64, dtype=torch.float32))
    assert clients[0].can_replay_sync, "nothing was streamed yet — this sync must still be replayable"

    pool.finish_weight_update()  # must NOT raise

    assert [name for name, _ in wire.sent] == ["w"], "the recovered flush did not deliver the params"


def test_reconnect_retires_the_old_client_before_probing_the_replacement():
    """The old client owns the ``/resume`` for the pause its failed sync left; a replacement probed
    first would only time out against the still-paused server."""
    order: list[str] = []

    class RecordingClient:
        def __init__(self, **kwargs):
            self.base_url = kwargs["base_url"]
            self.group_port = kwargs.get("group_port")
            self._param_buffer: list = []
            order.append("construct new client")

        def init_communicator(self, device):
            order.append("probe + init new client")

        def close_communicator(self):
            order.append("close old client")

        def drain_param_buffer(self):
            order.append("drain old buffer")
            buffered, self._param_buffer = self._param_buffer, []
            return buffered

        def buffer_param(self, name, snapshot):
            self._param_buffer.append((name, snapshot))

    pool = WeightSyncPool(
        server_configs=[{"url": "http://server0:8000", "group_port": 51216}],
        connection_timeout=0.0,
        client_cls=RecordingClient,
        base_group_port=51216,
    )
    old = RecordingClient(base_url="http://server0:8000")
    old.buffer_param("w", torch.zeros(4))
    order.clear()
    pool._clients = [old]
    pool._initialized = True
    pool._device = torch.device("cpu")

    pool._reconnect(0)
    new_client = pool._clients[0]

    assert order.index("close old client") < order.index("probe + init new client"), (
        f"the replacement was probed before the old client lifted its pause: {order}"
    )
    assert [name for name, _ in new_client._param_buffer] == ["w"], "the pending buffer was not carried over"
    assert new_client is not old, "the replacement was not swapped into the client pool"
    # The engine node dials back to the configured port, so the replacement must rebind it, not drift.
    assert new_client.group_port == 51216, f"reconnect moved off the configured group_port: {new_client.group_port}"


def test_pool_scopes_every_client_it_holds_and_builds():
    """The co-load scope is the served model's, so a client rebuilt by a reconnect must carry it too."""
    pool, clients = _bare_pool(2)

    def factory(**kwargs):
        client = Wire().attach(offline_client(VLLMWeightSyncClient))
        client.init_communicator = lambda device: None  # no NCCL group to form here
        return client

    pool._client_cls = factory
    pool._device = torch.device("cpu")
    pool.scope_co_load_groups(["model.layers.0.self_attn.kv_a_proj_with_mqa"])
    scoped = declared_co_load_groups(
        VLLMWeightSyncClient.CO_LOADED_PARAM_GROUPS, ["model.layers.0.self_attn.kv_a_proj_with_mqa"]
    )
    assert all(client._co_load_groups == scoped for client in clients)
    old = clients[0]
    old.close_communicator = lambda: None  # a bare client has no group to tear down
    pool._reconnect(0)
    rebuilt = pool._clients[0]
    assert rebuilt is not old and rebuilt._co_load_groups == scoped


def test_a_pool_keeps_the_engines_groups_until_a_push_scopes_them():
    """A client built before any push must hold the pair and refuse it incomplete, not send it apart."""
    pool = WeightSyncPool(
        [{"url": f"http://localhost:{30000 + index}"} for index in range(2)],
        connection_timeout=0.0,
        client_cls=ServerlessSGLangClient,
        base_group_port=51216,
    )
    pool.init_communicator("cpu")

    assert all(client._co_load_groups == SGLangWeightSyncClient.CO_LOADED_PARAM_GROUPS for client in pool._clients)


def test_a_reconnected_client_is_built_like_the_first_connect():
    """A reconnect re-flushes the failed sync's buffer through the replacement, so it must dial back on
    the server's configured port and NIC and keep the scope the push gave the pool: an unscoped
    replacement would hold a model without ``q_a_proj`` back for a partner it never declares."""
    pool = WeightSyncPool(
        [{"url": "http://localhost:30000", "group_port": 52000, "group_host": "10.0.0.7"}],
        connection_timeout=0.0,
        client_cls=ServerlessSGLangClient,
        base_group_port=51216,
    )
    pool.init_communicator("cpu")
    pool.scope_co_load_groups(
        f"model.layers.0.self_attn.{leaf}" for leaf in ("q_b_proj", "kv_a_proj_with_mqa", "o_proj")
    )
    first = pool._clients[0]
    scoped = (("self_attn.kv_a_proj_with_mqa.weight",),)
    assert first._co_load_groups == scoped

    pool._reconnect(0)

    replacement = pool._clients[0]
    assert replacement is not first
    for client in (first, replacement):
        assert (client.group_port, client.group_host, client.sync_device) == (52000, "10.0.0.7", torch.device("cpu"))
    assert replacement._co_load_groups == scoped, "the replacement lost the scope the push gave the pool"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
