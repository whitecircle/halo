#!/usr/bin/env python
"""The weight sync must STREAM to the engine: it stages one chunk, never the model.

Staging every gathered tensor before the push holds a full model copy on the forwarding rank — 800
GB at 400B — and with N servers a per-client copy multiplies it again (4 × ~120B bf16 ≈ 0.9 TB).
Both wire protocols take an update as a sequence of declared chunks inside one quiesce, so the client
opens the update, sends each chunk as the budget fills, and closes it at the end. Staging lives on
the sync GPU, not in pinned host memory: a chunk that transits the host is copied out and back over
PCIe before the NCCL broadcast, and those two copies cap the push well below the fabric. What is
pinned here:

* the buffered weight is a **snapshot** on the sync device, not an alias — a later in-place mutation
  (the next optimizer step) must not rewrite a weight that has not gone out yet;
* the multi-server fan-out stages ONE snapshot per param, shared read-only by every client, and
  releases it once every server has sent the chunk holding it;
* **peak** staged residency stays at one chunk (plus the tensor that crossed the budget), for the
  single-server client and for the multi-server fan-out — the property the streaming exists for,
  asserted by counting live snapshots at their peak;
* the chunk is cut BEFORE the budget is exceeded, so nothing downstream has to re-split an
  already-budgeted chunk;
* the engines are quiesced only once a chunk is ready to go out, not from the first gathered param;
* every parameter reaches the engine exactly once, in order, across the chunk boundaries;
* a server that failed AFTER its first chunk went out is NOT retried: the trainer kept no copy of
  what already landed, so a reconnect would leave that engine part old and part new while still
  answering /health.

Run: ``python tests/cpu/weight_sync/test_param_buffer.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import weakref
from collections.abc import Callable

import pytest
import torch

from src.trainers.grpo.rollout.weight_sync_clients import InferenceClientManager
from src.weight_sync.chunking import (
    declared_co_load_groups,
    resolve_sync_device,
    snapshot_to_sync_device,
)
from src.weight_sync.clients.sglang import SGLangWeightSyncClient
from src.weight_sync.clients.vllm import VLLMWeightSyncClient
from src.weight_sync.transport.packed_broadcast import PACKED_BUFFER_BYTES
from tests.common.repo import probe_findings
from tests.common.weight_sync import Wire, offline_client, push

# The chunk budget the cut-rule cases run against, small enough for a few CPU tensors to cross it.
_BUDGET = 4096


def _bare_client(wire: Wire | None = None) -> VLLMWeightSyncClient:
    """An offline client wired to ``wire``.

    Only the three engine phases are stubbed; the buffering, the chunk budget and the update
    bookkeeping under test are the client's own.
    """
    client = offline_client(VLLMWeightSyncClient)
    (wire or Wire()).attach(client)
    return client


def _bare_manager(
    num_clients: int, wires: list[Wire] | None = None
) -> tuple[InferenceClientManager, list[VLLMWeightSyncClient]]:
    """A manager over ``num_clients`` bare clients, without the NCCL/HTTP handshake."""
    configs = [{"url": f"http://server{i}:8000", "group_port": 51216 + i} for i in range(num_clients)]
    manager = InferenceClientManager(
        server_configs=configs, connection_timeout=0.0, client_cls=VLLMWeightSyncClient, base_group_port=51216
    )
    clients = []
    for index, config in enumerate(configs):
        client = _bare_client(wires[index] if wires else None)
        client.base_url = config["url"]  # the manager's failure report names the server
        clients.append(client)
    manager._clients = clients
    manager._initialized = True
    return manager, clients


def _live(refs: list[weakref.ref]) -> int:
    return sum(1 for ref in refs if ref() is not None)


def test_update_named_param_buffers_a_snapshot():
    client = _bare_client()
    weights = torch.randn(8, 8)  # contiguous, so an aliasing .contiguous() would return it as-is
    original = weights.clone()

    client.update_named_param("model.layers.0.q_proj.weight", weights)

    name, stored = client._param_buffer[0]
    assert name == "model.layers.0.q_proj.weight"
    assert stored.device == weights.device, "before init_communicator the snapshot stays on the source's device"
    assert stored.data_ptr() != weights.data_ptr(), "buffered by reference — must be a snapshot"

    weights.add_(1.0)  # simulate an optimizer step before the flush
    assert torch.equal(stored, original), "buffered weight mutated by a later in-place update"


def _snapshot(
    weights: torch.Tensor,
    device: torch.device | None = None,
    *,
    buffered_bytes: int = 0,
    flush: Callable[[], object] = lambda: None,
) -> torch.Tensor:
    """:func:`snapshot_to_sync_device` of ``weights`` against ``_BUDGET``, by default with nothing buffered."""
    return snapshot_to_sync_device(
        "w", weights, device=device, buffered_bytes=buffered_bytes, budget=_BUDGET, flush=flush
    )


def test_a_snapshot_is_contiguous_and_on_the_requested_device():
    """A strided view is snapshotted as a contiguous copy (both wire formats send flat bytes), on the
    device asked for — the sync GPU in production; here the CPU stands in for it."""
    strided = torch.arange(16, dtype=torch.float32).reshape(4, 4).t()
    snapshot = _snapshot(strided, torch.device("cpu"))
    assert snapshot.is_contiguous() and snapshot.device.type == "cpu"
    assert snapshot.dtype == strided.dtype, "the wire declares the source dtype, so the snapshot must keep it"
    assert torch.equal(snapshot, strided)
    assert _snapshot(strided).device == strided.device, "no device pins the source's own"


@pytest.mark.parametrize(("buffered", "item", "cut"), [(0, 8192, False), (2048, 2048, False), (2048, 2052, True)])
def test_a_chunk_goes_out_only_when_the_next_tensor_would_pass_the_budget(buffered, item, cut):
    """The pending chunk goes out when this tensor would take it past the budget, not once it is past:
    reaching the budget exactly sends nothing, and a tensor above the budget on an empty buffer becomes
    its own chunk without sending an empty one."""
    flushes = []
    _snapshot(torch.zeros(item // 4), buffered_bytes=buffered, flush=lambda: flushes.append(1))
    assert flushes == ([1] if cut else []), f"{buffered} B buffered + {item} B against a {_BUDGET} B budget"


def test_an_unsyncable_param_is_refused_before_the_pending_chunk_goes_out():
    """A full buffer and a 0-dim param: the refusal comes first, so no chunk is sent (and no engine
    quiesced) for a sync that cannot complete."""
    flushes = []
    with pytest.raises(ValueError, match="0-dimensional"):
        _snapshot(torch.tensor(1.0), buffered_bytes=_BUDGET, flush=lambda: flushes.append(1))
    assert flushes == [], "the pending chunk was flushed for a param the sync then refused"


def test_the_snapshot_is_taken_after_the_flush():
    """The pending chunk goes out before the copy is allocated, so the two are never held at once. A
    flush that rewrites the source shows which came first."""
    weights = torch.zeros(1024)
    snapshot = _snapshot(weights, buffered_bytes=_BUDGET, flush=lambda: weights.fill_(2.0))
    assert torch.equal(snapshot, torch.full((1024,), 2.0)), "the snapshot was taken before the flush ran"


def test_staged_residency_never_exceeds_one_chunk(monkeypatch):
    """The bound the streaming exists for, measured: peak live snapshots ≈ one chunk.

    Ten params of an eighth of the budget each: buffering the model would leave all ten alive at
    once, streaming leaves at most a chunk's worth. Counted through weak references, so a snapshot
    the client still holds anywhere is counted; the engine side copies what it receives and keeps no
    reference (``retain=False``), as a real one does.
    """
    monkeypatch.setattr("src.weight_sync.clients.base.WEIGHT_SYNC_CHUNK_BYTES", 4096)
    param_bytes, num_params, budget = 512, 10, 4096
    client = _bare_client(Wire(retain=False))
    refs: list[weakref.ref] = []
    peak = 0

    for index in range(num_params):
        client.update_named_param(f"w{index}", torch.zeros(param_bytes // 4, dtype=torch.float32))
        if client._param_buffer:
            refs.append(weakref.ref(client._param_buffer[-1][1]))
        peak = max(peak, _live(refs))

    assert peak * param_bytes <= budget + param_bytes, (
        f"{peak} snapshots alive at once ({peak * param_bytes} B) — the streamed sync must not "
        f"hold more than one chunk ({budget} B) plus the tensor that crossed the budget"
    )
    assert peak < num_params, "every param stayed resident: the chunk flush never ran"


def test_streaming_sends_every_param_once_in_order(monkeypatch):
    """Anti-vacuity for the bound above: chunking must not drop, duplicate or reorder a param."""
    monkeypatch.setattr("src.weight_sync.clients.base.WEIGHT_SYNC_CHUNK_BYTES", 4096)
    wire = Wire()
    client = _bare_client(wire)
    names = [f"w{index}" for index in range(10)]

    push(client, [(name, torch.zeros(128, dtype=torch.float32)) for name in names])

    assert [name for name, _ in wire.sent] == names, "params were dropped, duplicated or reordered"
    assert len(wire.chunks) > 1, "the payload never chunked — the budget did nothing"
    assert (wire.opened, wire.closed) == (1, 1), (
        f"one quiesced update per sync, got open={wire.opened} close={wire.closed} — a per-chunk "
        f"pause/reload would churn the engine's KV cache and re-meta the model for every chunk"
    )


def test_a_sync_that_fits_in_one_chunk_still_opens_and_closes_one_update():
    """The small-model path: no mid-gather chunk, everything rides the closing flush."""
    wire = Wire()
    client = _bare_client(wire)

    client.update_named_param("w", torch.zeros(64, dtype=torch.float32))
    assert wire.sent == [], "a sub-budget param must not open an update on its own"

    client.reset_prefix_cache()
    assert [name for name, _ in wire.sent] == ["w"]
    assert (wire.opened, wire.closed) == (1, 1)


def test_manager_stages_one_snapshot_per_param(monkeypatch):
    """The multi-server fan-out must stage ONE snapshot per param, not one per client."""
    manager, clients = _bare_manager(3)

    allocations = []
    real_empty_like = torch.empty_like

    def counting_empty_like(*args, **kwargs):
        allocations.append(kwargs.get("device"))
        return real_empty_like(*args, **kwargs)

    monkeypatch.setattr(torch, "empty_like", counting_empty_like)

    weights = torch.randn(8, 8)
    manager.update_named_param("model.layers.0.q_proj.weight", weights)

    assert len(allocations) == 1, (
        f"{len(allocations)} staged allocations for one param across {len(clients)} clients — "
        f"per-client copies multiply the staged chunk by the server count (must be exactly 1)"
    )
    buffered = [client._param_buffer[0][1] for client in clients]
    assert all(t is buffered[0] for t in buffered), "clients must share ONE snapshot by reference"
    assert all(client._param_buffer[0][0] == "model.layers.0.q_proj.weight" for client in clients)


def test_manager_stages_shared_snapshots_on_its_normalized_device(monkeypatch):
    """``init_communicators(0)`` must stage on CUDA device 0: the manager stores the resolved
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

    manager = InferenceClientManager(
        server_configs=[{"url": "http://server0:8000"}, {"url": "http://server1:8000"}],
        connection_timeout=0.0,
        client_cls=_StubClient,
        base_group_port=51216,
    )
    manager.init_communicators(0)
    assert manager._device == torch.device("cuda", 0), f"stored {manager._device!r} for device index 0"
    assert joined == [torch.device("cuda", 0)] * 2, "clients must join on the same normalized device"

    staged_on: list = []

    def fake_snapshot(name, weights, *, device, **cut):
        staged_on.append(device)
        return weights.detach().clone()

    monkeypatch.setattr("src.trainers.grpo.rollout.weight_sync_clients.snapshot_to_sync_device", fake_snapshot)
    manager.update_named_param("w", torch.zeros(4))
    assert staged_on == [torch.device("cuda", 0)], "the shared snapshot was not staged on the sync device"


def test_manager_shared_snapshot_is_immutable_copy():
    """The shared snapshot must still be a copy: a post-buffer in-place mutation (the next optimizer
    step) must not revert any client's buffered weight."""
    manager, clients = _bare_manager(2)
    weights = torch.randn(4, 4)  # contiguous, so an aliasing .contiguous() would return it as-is
    original = weights.clone()

    manager.update_named_param("w", weights)
    weights.add_(1.0)  # simulate an optimizer step before the flush

    for client in clients:
        _, stored = client._param_buffer[0]
        assert stored.data_ptr() != weights.data_ptr(), "buffered by reference to the source"
        assert torch.equal(stored, original), "shared snapshot mutated by a later in-place update"


def test_manager_releases_a_chunks_snapshots_once_every_server_sent_it(monkeypatch):
    """Multi-server: a shared snapshot must die once the chunk holding it went out everywhere — the
    manager keeps no pool, so a reference kept anywhere would grow the staged footprint with the
    model rather than the chunk."""
    monkeypatch.setattr("src.trainers.grpo.rollout.weight_sync_clients.WEIGHT_SYNC_CHUNK_BYTES", 4096)
    manager, clients = _bare_manager(2, wires=[Wire(retain=False), Wire(retain=False)])
    refs: list[weakref.ref] = []

    for index in range(10):
        manager.update_named_param(f"w{index}", torch.zeros(128, dtype=torch.float32))
        refs.append(weakref.ref(clients[0]._param_buffer[-1][1]))
    assert 0 < _live(refs) < len(refs), "no chunk went out mid-stream — the flush never ran"
    manager.reset_prefix_cache()

    assert _live(refs) == 0, f"{_live(refs)} snapshots still referenced after the sync"
    assert all(client._param_buffer == [] for client in clients)


def test_manager_flush_failure_isolation():
    """A failed flush on server A clears only A's buffer; B still flushes the intact shared
    snapshot afterwards, and the manager raises (fail loud, no stale-policy rollouts)."""
    wires = [Wire(fail_on_send=True), Wire(), Wire()]
    manager, clients = _bare_manager(3, wires=wires)

    weights = torch.randn(4, 4)
    expected = weights.clone()
    manager.update_named_param("w", weights)

    with pytest.raises(RuntimeError, match="1/3 vLLM server"):
        manager.reset_prefix_cache()

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
    monkeypatch.setattr("src.trainers.grpo.rollout.weight_sync_clients.WEIGHT_SYNC_CHUNK_BYTES", 4096)
    wire = Wire()
    manager, clients = _bare_manager(1, wires=[wire])
    reconnects = []
    monkeypatch.setattr(InferenceClientManager, "reconnect_client", lambda self, index: reconnects.append(index))

    for index in range(10):  # crosses the budget, so a chunk is on the wire before the tail
        manager.update_named_param(f"w{index}", torch.zeros(128, dtype=torch.float32))
    assert clients[0]._chunks_sent > 0, "the fixture never streamed a chunk — the test would be vacuous"
    assert not clients[0].can_replay_sync, "a client that streamed a chunk must report itself unreplayable"

    wire.fail_on_send = True
    with pytest.raises(RuntimeError, match="already streaming"):
        manager.reset_prefix_cache()

    assert reconnects == [], "a half-updated engine was reconnected and re-sent only the tail"
    assert clients[0]._param_buffer == [], "the unsendable buffer must be dropped, not re-broadcast"


def test_manager_scopes_every_client_it_holds_and_builds():
    """The co-load scope is the served model's, so a client rebuilt by a reconnect must carry it too."""
    manager, clients = _bare_manager(2)

    def factory(**kwargs):
        client = _bare_client()
        client.init_communicator = lambda device: None  # no NCCL group to form here
        return client

    manager._client_factory = factory
    manager._device = torch.device("cpu")
    manager.scope_co_load_groups(["model.layers.0.self_attn.kv_a_proj_with_mqa"])
    scoped = declared_co_load_groups(
        VLLMWeightSyncClient.CO_LOADED_PARAM_GROUPS, ["model.layers.0.self_attn.kv_a_proj_with_mqa"]
    )
    assert all(client._co_load_groups == scoped for client in clients)
    old = clients[0]
    old.close_communicator = lambda: None  # a bare client has no group to tear down
    rebuilt = manager.reconnect_client(0)
    assert rebuilt is not old and rebuilt._co_load_groups == scoped


def test_a_server_that_failed_before_streaming_is_still_retried(monkeypatch):
    """Anti-over-rejection: the dominant recovery case — an engine restarted BETWEEN syncs, whose
    first chunk fails — is still reconnected and re-sent, because nothing has landed on it yet."""
    wire = Wire(fail_on_send=True)
    manager, clients = _bare_manager(1, wires=[wire])

    def reconnect(self, index):
        wire.fail_on_send = False  # the replacement engine accepts the replay
        return self._clients[index]

    monkeypatch.setattr(InferenceClientManager, "reconnect_client", reconnect)

    manager.update_named_param("w", torch.zeros(64, dtype=torch.float32))
    assert clients[0].can_replay_sync, "nothing was streamed yet — this sync must still be replayable"

    manager.reset_prefix_cache()  # must NOT raise

    assert [name for name, _ in wire.sent] == ["w"], "the recovered flush did not deliver the params"


def test_chunks_are_cut_before_the_budget_is_exceeded(monkeypatch):
    """The budget must bound what goes OUT, not what went out plus the tensor that crossed it.

    Cutting after it is reached puts the crossing tensor in the chunk on the wire: up to a second
    budget of staged device memory here, and the same overshoot in the receive buffers the engine
    allocates for every declared name before the first byte arrives.
    """
    monkeypatch.setattr("src.weight_sync.clients.base.WEIGHT_SYNC_CHUNK_BYTES", 4096)
    budget, oversize = 4096, 4096 + 512
    wire = Wire()
    client = _bare_client(wire)
    sizes = [1536, 1536, 1536, 512, oversize, 256]  # irregular, one tensor above the budget

    push(client, [(f"w{index}", torch.zeros(size // 4, dtype=torch.float32)) for index, size in enumerate(sizes)])

    assert [name for name, _ in wire.sent] == [f"w{i}" for i in range(len(sizes))]
    for chunk in wire.chunks:
        chunk_bytes = sum(tensor.numel() * tensor.element_size() for _, tensor in chunk)
        assert chunk_bytes <= budget or len(chunk) == 1, (
            f"a chunk of {chunk_bytes} B crossed the {budget} B budget with {len(chunk)} tensors — only a "
            f"single tensor larger than the budget may exceed it (the protocols describe whole tensors)"
        )


def test_sglang_declares_one_request_per_chunk(monkeypatch):
    """The chunk budget IS the server-side bound on SGLang: the engine allocates ``torch.empty`` for
    every declared name before receiving any of them.

    A second, stricter budget inside the client would re-split every already-budgeted chunk — each
    fragment its own ``/update_weights_from_distributed`` declare, thread and round-trip.
    """
    monkeypatch.setattr("src.weight_sync.clients.base.WEIGHT_SYNC_CHUNK_BYTES", 4096)
    client = offline_client(SGLangWeightSyncClient)
    declares: list[list[str]] = []
    client.begin_weight_update = lambda: None
    client._send_chunk = lambda chunk, flush_cache: declares.append([name for name, _ in chunk])
    client._lift_pause = lambda timeout, context: None

    # 1536 B each: two per 4 KiB chunk.
    push(client, [(f"w{index}", torch.zeros(384, dtype=torch.float32)) for index in range(10)])

    assert declares == [[f"w{index}", f"w{index + 1}"] for index in range(0, 10, 2)], (
        f"{len(declares)} declares for 5 budgeted chunks — a second budget re-split them: {declares}"
    )


def test_no_server_is_quiesced_until_a_chunk_is_ready_to_send(monkeypatch):
    """The quiesce window opens at the first FULL chunk, not at the first gathered parameter.

    Every server stops serving for as long as its update is open, so the update must not be opened
    while the gather is still assembling the first chunk's worth of weights.
    """
    monkeypatch.setattr("src.trainers.grpo.rollout.weight_sync_clients.WEIGHT_SYNC_CHUNK_BYTES", 4096)
    wires = [Wire(), Wire()]
    manager, _clients = _bare_manager(2, wires=wires)

    for index in range(8):  # 512 B each: exactly the budget, nothing to send yet
        manager.update_named_param(f"w{index}", torch.zeros(128, dtype=torch.float32))
    assert [wire.opened for wire in wires] == [0, 0], "a server was quiesced before a full chunk existed"

    manager.update_named_param("w8", torch.zeros(128, dtype=torch.float32))
    assert [wire.opened for wire in wires] == [1, 1], "the full chunk did not open the update"

    manager.reset_prefix_cache()
    assert [wire.closed for wire in wires] == [1, 1], "the update stayed open after the final flush"


def test_the_pool_refuses_an_unsyncable_param_before_any_server_gets_the_pending_chunk(monkeypatch):
    """The pool snapshots for its clients, so their own check never runs: with a full chunk pending, a
    0-dim param must be refused before that chunk quiesces a server for a sync that cannot complete."""
    monkeypatch.setattr("src.trainers.grpo.rollout.weight_sync_clients.WEIGHT_SYNC_CHUNK_BYTES", _BUDGET)
    wires = [Wire(), Wire()]
    manager, clients = _bare_manager(2, wires=wires)
    manager.update_named_param("w", torch.zeros(_BUDGET // 4, dtype=torch.float32))

    with pytest.raises(ValueError, match="0-dimensional"):
        manager.update_named_param("scale", torch.tensor(1.0))

    assert [wire.opened for wire in wires] == [0, 0], "a server was quiesced for a param the pool then refused"
    assert [wire.chunks for wire in wires] == [[], []], "the pending chunk went out for a refused param"
    assert all(client._chunks_sent == 0 and not client._update_open for client in clients)
    manager.update_named_param("bias", torch.zeros(1, dtype=torch.float32))
    assert [wire.opened for wire in wires] == [1, 1], "the pending chunk was not full: nothing was at stake"


@pytest.mark.parametrize(
    ("value", "expected"), [(None, PACKED_BUFFER_BYTES), ("2048", 2 * 2**30), ("0", PACKED_BUFFER_BYTES)]
)
def test_the_chunk_budget_is_read_once_from_the_env_in_megabytes(monkeypatch, value, expected):
    """``HALO_WEIGHT_SYNC_CHUNK_MB`` sizes the chunk staged on the sync GPU (and the SGLang client's
    device arena), and the default is the transport's own packed buffer, the size a chunk is re-packed
    into on the way out; a non-positive value falls back. Read in a fresh interpreter, since the module
    latches it."""
    if value is None:
        monkeypatch.delenv("HALO_WEIGHT_SYNC_CHUNK_MB", raising=False)
    else:
        monkeypatch.setenv("HALO_WEIGHT_SYNC_CHUNK_MB", value)
    script = "import src.weight_sync.chunking as c; print('CHUNK_BYTES:' + str(c.WEIGHT_SYNC_CHUNK_BYTES))"
    [budget] = probe_findings(script, "CHUNK_BYTES:")
    assert int(budget) == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
