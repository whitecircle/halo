#!/usr/bin/env python
"""The weight sync must STREAM to the engine: it buffers one chunk, never the model.

Buffering every gathered tensor before the push holds a full model copy on the forwarding rank — 800
GB at 400B. Both wire protocols take an update as a sequence of declared chunks inside one quiesce, so
the client opens the update, sends each chunk as the budget fills, and closes it at the end. The
snapshots live on the sync GPU, not in pinned host memory: a chunk that transits the host is copied out
and back over PCIe before the NCCL broadcast, and those two copies cap the push well below the fabric.
What is pinned here (the multi-server pool in ``test_pool.py``):

* the buffered weight is a **snapshot** on the sync device, not an alias — a later in-place mutation
  (the next optimizer step) must not rewrite a weight that has not gone out yet;
* **peak** buffered residency stays at one chunk (plus the tensor that crossed the budget) — the
  property the streaming exists for, asserted by counting live snapshots at their peak;
* the chunk is cut BEFORE the budget is exceeded, so nothing downstream has to re-split an
  already-budgeted chunk;
* the engine is quiesced only once a chunk is ready to go out, not from the first gathered param;
* every parameter reaches the engine exactly once, in order, across the chunk boundaries.

Run: ``python tests/cpu/weight_sync/test_param_buffer.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import weakref
from collections.abc import Callable

import pytest
import torch

from halo.weight_sync.chunking import snapshot_to_sync_device
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient
from halo.weight_sync.transport.packed_broadcast import PACKED_BUFFER_BYTES
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


def test_snapshot_residency_never_exceeds_one_chunk(monkeypatch):
    """The bound the streaming exists for, measured: peak live snapshots ≈ one chunk.

    Ten params of an eighth of the budget each: buffering the model would leave all ten alive at
    once, streaming leaves at most a chunk's worth. Counted through weak references, so a snapshot
    the client still holds anywhere is counted; the engine side copies what it receives and keeps no
    reference (``retain=False``), as a real one does.
    """
    monkeypatch.setattr("halo.weight_sync.clients.base.WEIGHT_SYNC_CHUNK_BYTES", 4096)
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
    monkeypatch.setattr("halo.weight_sync.clients.base.WEIGHT_SYNC_CHUNK_BYTES", 4096)
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

    client.finish_weight_update()
    assert [name for name, _ in wire.sent] == ["w"]
    assert (wire.opened, wire.closed) == (1, 1)


def test_chunks_are_cut_before_the_budget_is_exceeded(monkeypatch):
    """The budget must bound what goes OUT, not what went out plus the tensor that crossed it.

    Cutting after it is reached puts the crossing tensor in the chunk on the wire: up to a second
    budget of snapshot device memory here, and the same overshoot in the receive buffers the engine
    allocates for every declared name before the first byte arrives.
    """
    monkeypatch.setattr("halo.weight_sync.clients.base.WEIGHT_SYNC_CHUNK_BYTES", 4096)
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
    monkeypatch.setattr("halo.weight_sync.clients.base.WEIGHT_SYNC_CHUNK_BYTES", 4096)
    client = offline_client(SGLangWeightSyncClient)
    declares: list[list[str]] = []
    client._start_engine_update = lambda: None
    client._broadcast_chunk = lambda chunk, final: declares.append([name for name, _ in chunk])
    client._lift_pause = lambda timeout, context: None
    client._end_session = lambda: None

    # 1536 B each: two per 4 KiB chunk.
    push(client, [(f"w{index}", torch.zeros(384, dtype=torch.float32)) for index in range(10)])

    assert declares == [[f"w{index}", f"w{index + 1}"] for index in range(0, 10, 2)], (
        f"{len(declares)} declares for 5 budgeted chunks — a second budget re-split them: {declares}"
    )


@pytest.mark.parametrize(
    ("value", "expected"), [(None, PACKED_BUFFER_BYTES), ("2048", 2 * 2**30), ("0", PACKED_BUFFER_BYTES)]
)
def test_the_chunk_budget_is_read_once_from_the_env_in_megabytes(monkeypatch, value, expected):
    """``HALO_WEIGHT_SYNC_CHUNK_MB`` sizes the chunk held on the sync GPU (and the SGLang client's
    device arena), and the default is the transport's own packed buffer, the size a chunk is re-packed
    into on the way out; a non-positive value falls back. Read in a fresh interpreter, since the module
    latches it."""
    if value is None:
        monkeypatch.delenv("HALO_WEIGHT_SYNC_CHUNK_MB", raising=False)
    else:
        monkeypatch.setenv("HALO_WEIGHT_SYNC_CHUNK_MB", value)
    script = "import halo.weight_sync.chunking as c; print('CHUNK_BYTES:' + str(c.WEIGHT_SYNC_CHUNK_BYTES))"
    [budget] = probe_findings(script, "CHUNK_BYTES:")
    assert int(budget) == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
