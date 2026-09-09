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
  (PEFT unmerge, the next optimizer step) must not rewrite a weight that has not gone out yet;
* the multi-server fan-out stages ONE snapshot per param, shared read-only by every client, and
  releases it once every server has sent the chunk holding it;
* **peak** staged residency stays at one chunk (plus the tensor that crossed the budget), for the
  single-server client and for the multi-server fan-out — the property the streaming exists for,
  asserted by counting live snapshots at their peak;
* the chunk is cut BEFORE the budget is exceeded, on one rule shared by the streamed path and the
  whole-payload one, so nothing downstream has to re-split an already-budgeted chunk;
* the engines are quiesced only once a chunk is ready to go out, not from the first gathered param;
* every parameter reaches the engine exactly once, in order, across the chunk boundaries;
* a server that failed AFTER its first chunk went out is NOT retried: the trainer kept no copy of
  what already landed, so a reconnect would leave that engine part old and part new while still
  answering /health.

Run: ``python tests/cpu/grpo/test_weight_sync_param_buffer.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import weakref

import pytest
import torch

from src.distributed.nccl.clients.base import resolve_weight_sync_chunk_bytes, snapshot_param
from src.distributed.nccl.clients.sglang import SGLangWeightSyncClient
from src.distributed.nccl.clients.vllm import VLLMWeightSyncClient
from src.distributed.nccl.transport.packed_tensor import DEFAULT_PACKED_BUFFER_SIZE_BYTES
from src.trainers.grpo.rollout.weight_sync_clients import InferenceClientManager


class _Wire:
    """The engine side of one client: records each chunk put on the wire, in order.

    Stubs the per-engine seams only (``_broadcast_chunk`` and the phase calls), so the client's own
    chunk accounting — the count ``can_replay_sync`` reads — runs as it does in production.
    """

    def __init__(self, fail_on_send: bool = False, retain: bool = True):
        self.chunks: list[list[tuple[str, torch.Tensor]]] = []
        self.opened = 0
        self.closed = 0
        self.fail_on_send = fail_on_send
        # A real engine copies what it receives into its own storage and keeps no reference to the
        # trainer's snapshot. The lifetime tests need that; the others want the values back.
        self.retain = retain
        self.client: VLLMWeightSyncClient | None = None

    def attach(self, client: VLLMWeightSyncClient) -> VLLMWeightSyncClient:
        self.client = client
        client.begin_weight_update = self._begin
        client._broadcast_chunk = self._send
        client.end_weight_update = self._end
        return client

    def _begin(self):
        self.opened += 1

    def _send(self, named_params, final: bool = False):
        if self.fail_on_send:
            raise ConnectionError("server died mid-flush")
        self.chunks.append(list(named_params) if self.retain else [(name, None) for name, _ in named_params])

    def _end(self, tail):
        try:
            self.client.send_weights(tail, final=True)  # as both real clients close: through the seam
        finally:
            self.closed += 1  # the real clients close and resume in a finally too

    @property
    def sent(self) -> list[tuple[str, torch.Tensor]]:
        return [item for chunk in self.chunks for item in chunk]


def _bare_client(wire: _Wire | None = None) -> VLLMWeightSyncClient:
    """A client without the HTTP handshake (``__init__`` probes a live server), wired to ``wire``.

    Only the three engine phases are stubbed; the buffering, the chunk budget and the update
    bookkeeping under test are the client's own.
    """
    client = VLLMWeightSyncClient.__new__(VLLMWeightSyncClient)
    client._reset_buffer_state()
    (wire or _Wire()).attach(client)
    return client


def _bare_manager(
    num_clients: int, wires: list[_Wire] | None = None
) -> tuple[InferenceClientManager, list[VLLMWeightSyncClient]]:
    """A manager over ``num_clients`` bare clients, without the NCCL/HTTP handshake."""
    configs = [{"url": f"http://server{i}:8000", "group_port": 51216 + i} for i in range(num_clients)]
    manager = InferenceClientManager(server_configs=configs)
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

    weights.add_(1.0)  # simulate PEFT unmerge / optimizer step before the flush
    assert torch.equal(stored, original), "buffered weight mutated by a later in-place update"


def test_a_snapshot_is_contiguous_and_on_the_requested_device():
    """A strided view is snapshotted as a contiguous copy (both wire formats send flat bytes), on the
    device asked for — the sync GPU in production; here the CPU stands in for it."""
    strided = torch.arange(16, dtype=torch.float32).reshape(4, 4).t()
    snapshot = snapshot_param(strided, torch.device("cpu"))
    assert snapshot.is_contiguous() and snapshot.device.type == "cpu"
    assert torch.equal(snapshot, strided)
    assert snapshot_param(strided, None).device == strided.device, "no device pins the source's own"


def test_staged_residency_never_exceeds_one_chunk(monkeypatch):
    """The bound the streaming exists for, measured: peak live snapshots ≈ one chunk.

    Ten params of an eighth of the budget each: buffering the model would leave all ten alive at
    once, streaming leaves at most a chunk's worth. Counted through weak references, so a snapshot
    the client still holds anywhere is counted; the engine side copies what it receives and keeps no
    reference (``retain=False``), as a real one does.
    """
    monkeypatch.setattr("src.distributed.nccl.clients.base.WEIGHT_SYNC_CHUNK_BYTES", 4096)
    param_bytes, num_params, budget = 512, 10, 4096
    client = _bare_client(_Wire(retain=False))
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
    monkeypatch.setattr("src.distributed.nccl.clients.base.WEIGHT_SYNC_CHUNK_BYTES", 4096)
    wire = _Wire()
    client = _bare_client(wire)
    names = [f"w{index}" for index in range(10)]

    for name in names:
        client.update_named_param(name, torch.zeros(128, dtype=torch.float32))
    client.reset_prefix_cache()

    assert [name for name, _ in wire.sent] == names, "params were dropped, duplicated or reordered"
    assert len(wire.chunks) > 1, "the payload never chunked — the budget did nothing"
    assert (wire.opened, wire.closed) == (1, 1), (
        f"one quiesced update per sync, got open={wire.opened} close={wire.closed} — a per-chunk "
        f"pause/reload would churn the engine's KV cache and re-meta the model for every chunk"
    )


def test_a_sync_that_fits_in_one_chunk_still_opens_and_closes_one_update():
    """The small-model path: no mid-gather chunk, everything rides the closing flush."""
    wire = _Wire()
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
    """``init_communicators(0)`` must stage on CUDA device 0: an int index is falsy, so a helper that
    falls back on ``device or source`` silently stages on the source's device instead. The manager
    stores a normalized ``torch.device`` and the helper tests for ``None`` explicitly."""
    joined: list = []

    class _StubClient:
        BACKEND_NAME = "stub"

        def __init__(self, base_url, group_port, connection_timeout, group_host=None):
            self._param_buffer: list = []

        def init_communicator(self, device):
            joined.append(device)

        def buffer_param(self, name, snapshot):
            self._param_buffer.append((name, snapshot))

    manager = InferenceClientManager(server_configs=[{"url": "http://server0:8000"}, {"url": "http://server1:8000"}])
    manager._client_factory = _StubClient
    manager.init_communicators(0)
    assert manager._device == torch.device("cuda", 0), f"stored {manager._device!r} for device index 0"
    assert joined == [torch.device("cuda", 0)] * 2, "clients must join on the same normalized device"

    staged_on: list = []

    def fake_snapshot(weights, device):
        staged_on.append(device)
        return weights.detach().clone()

    monkeypatch.setattr("src.trainers.grpo.rollout.weight_sync_clients.snapshot_param", fake_snapshot)
    manager.update_named_param("w", torch.zeros(4))
    assert staged_on == [torch.device("cuda", 0)], "the shared snapshot was not staged on the sync device"


def test_manager_shared_snapshot_is_immutable_copy():
    """The shared snapshot must still be a copy: a post-buffer in-place mutation (PEFT unmerge,
    optimizer step) must not revert any client's buffered weight."""
    manager, clients = _bare_manager(2)
    weights = torch.randn(4, 4)  # contiguous, so an aliasing .contiguous() would return it as-is
    original = weights.clone()

    manager.update_named_param("w", weights)
    weights.add_(1.0)  # simulate PEFT unmerge before the flush

    for client in clients:
        _, stored = client._param_buffer[0]
        assert stored.data_ptr() != weights.data_ptr(), "buffered by reference to the source"
        assert torch.equal(stored, original), "shared snapshot mutated by a later in-place update"


def test_manager_releases_a_chunks_snapshots_once_every_server_sent_it(monkeypatch):
    """Multi-server: a shared snapshot must die once the chunk holding it went out everywhere — the
    manager keeps no pool, so a reference kept anywhere would grow the staged footprint with the
    model rather than the chunk."""
    monkeypatch.setattr("src.trainers.grpo.rollout.weight_sync_clients.WEIGHT_SYNC_CHUNK_BYTES", 4096)
    manager, clients = _bare_manager(2, wires=[_Wire(retain=False), _Wire(retain=False)])
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
    wires = [_Wire(fail_on_send=True), _Wire(), _Wire()]
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
    wire = _Wire()
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


def test_a_server_that_failed_before_streaming_is_still_retried(monkeypatch):
    """Anti-over-rejection: the dominant recovery case — an engine restarted BETWEEN syncs, whose
    first chunk fails — is still reconnected and re-sent, because nothing has landed on it yet."""
    wire = _Wire(fail_on_send=True)
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
    budget of pinned host memory here, and the same overshoot in the receive buffers the engine
    allocates for every declared name before the first byte arrives.
    """
    monkeypatch.setattr("src.distributed.nccl.clients.base.WEIGHT_SYNC_CHUNK_BYTES", 4096)
    budget, oversize = 4096, 4096 + 512
    wire = _Wire()
    client = _bare_client(wire)
    sizes = [1536, 1536, 1536, 512, oversize, 256]  # irregular, one tensor above the budget

    for index, size in enumerate(sizes):
        client.update_named_param(f"w{index}", torch.zeros(size // 4, dtype=torch.float32))
    client.reset_prefix_cache()

    assert [name for name, _ in wire.sent] == [f"w{i}" for i in range(len(sizes))]
    for chunk in wire.chunks:
        chunk_bytes = sum(tensor.numel() * tensor.element_size() for _, tensor in chunk)
        assert chunk_bytes <= budget or len(chunk) == 1, (
            f"a chunk of {chunk_bytes} B crossed the {budget} B budget with {len(chunk)} tensors — only a "
            f"single tensor larger than the budget may exceed it (the protocols describe whole tensors)"
        )


def test_the_whole_payload_path_cuts_on_the_same_boundaries(monkeypatch):
    """One chunker: a payload handed over in one call is cut where streaming it would have cut it.

    Two rules meant the SGLang client re-split every already-budgeted chunk at a second, stricter
    boundary — one `/update_weights_from_distributed` declare, thread and round-trip per fragment.
    """
    monkeypatch.setattr("src.distributed.nccl.clients.base.WEIGHT_SYNC_CHUNK_BYTES", 4096)
    params = [(f"w{index}", torch.zeros(384, dtype=torch.float32)) for index in range(10)]

    streamed_wire = _Wire()
    streamed = _bare_client(streamed_wire)
    for name, tensor in params:
        streamed.update_named_param(name, tensor)
    streamed.reset_prefix_cache()

    whole_wire = _Wire()
    _bare_client(whole_wire).sync_model_weights(params)

    boundaries = [[name for name, _ in chunk] for chunk in streamed_wire.chunks]
    assert len(boundaries) > 1, "the payload never chunked — the budget did nothing"
    assert [[name for name, _ in chunk] for chunk in whole_wire.chunks] == boundaries
    assert (whole_wire.opened, whole_wire.closed) == (1, 1), "the whole-payload path re-quiesced per chunk"


def test_sglang_declares_one_request_per_chunk(monkeypatch):
    """The chunk budget IS the server-side bound on SGLang: the engine allocates ``torch.empty`` for
    every declared name before receiving any of them.

    A second, stricter budget inside the client re-split every already-budgeted chunk — each
    fragment its own ``/update_weights_from_distributed`` declare, thread and round-trip.
    """
    monkeypatch.setattr("src.distributed.nccl.clients.base.WEIGHT_SYNC_CHUNK_BYTES", 4096)
    client = SGLangWeightSyncClient.__new__(SGLangWeightSyncClient)
    client._reset_buffer_state()
    declares: list[list[str]] = []
    client.begin_weight_update = lambda: None
    client._send_chunk = lambda chunk, flush_cache: declares.append([name for name, _ in chunk])
    client._lift_pause = lambda timeout, context: None

    for index in range(10):  # 1536 B each: two per 4 KiB chunk
        client.update_named_param(f"w{index}", torch.zeros(384, dtype=torch.float32))
    client.reset_prefix_cache()

    assert declares == [[f"w{index}", f"w{index + 1}"] for index in range(0, 10, 2)], (
        f"{len(declares)} declares for 5 budgeted chunks — a second budget re-split them: {declares}"
    )


def test_no_server_is_quiesced_until_a_chunk_is_ready_to_send(monkeypatch):
    """The quiesce window opens at the first FULL chunk, not at the first gathered parameter.

    Every server stops serving for as long as its update is open, so the update must not be opened
    while the gather is still assembling the first chunk's worth of weights.
    """
    monkeypatch.setattr("src.trainers.grpo.rollout.weight_sync_clients.WEIGHT_SYNC_CHUNK_BYTES", 4096)
    wires = [_Wire(), _Wire()]
    manager, _clients = _bare_manager(2, wires=wires)

    for index in range(8):  # 512 B each: exactly the budget, nothing to send yet
        manager.update_named_param(f"w{index}", torch.zeros(128, dtype=torch.float32))
    assert [wire.opened for wire in wires] == [0, 0], "a server was quiesced before a full chunk existed"

    manager.update_named_param("w8", torch.zeros(128, dtype=torch.float32))
    assert [wire.opened for wire in wires] == [1, 1], "the full chunk did not open the update"

    manager.reset_prefix_cache()
    assert [wire.closed for wire in wires] == [1, 1], "the update stayed open after the final flush"


def test_the_default_chunk_budget_is_one_packed_buffer(monkeypatch):
    """The default budget is the transport's own staging size, not an arbitrary number — a chunk is
    re-packed into exactly that on the way out."""
    monkeypatch.delenv("HALO_WEIGHT_SYNC_CHUNK_MB", raising=False)
    assert resolve_weight_sync_chunk_bytes() == DEFAULT_PACKED_BUFFER_SIZE_BYTES


def test_the_chunk_budget_is_read_from_the_env_in_megabytes(monkeypatch):
    """``HALO_WEIGHT_SYNC_CHUNK_MB`` sizes the pinned host chunk (and the SGLang client's device
    arenas); an ignored or misparsed value silently leaves the default in place."""
    monkeypatch.setenv("HALO_WEIGHT_SYNC_CHUNK_MB", "2048")
    assert resolve_weight_sync_chunk_bytes() == 2 * 2**30
    monkeypatch.setenv("HALO_WEIGHT_SYNC_CHUNK_MB", "0")
    assert resolve_weight_sync_chunk_bytes() == DEFAULT_PACKED_BUFFER_SIZE_BYTES, "a non-positive value must fall back"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
