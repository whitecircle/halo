#!/usr/bin/env python
"""CPU tests for the SGLang weight-sync client's device staging and deferred settle.

The engine acknowledges a chunk as soon as its data arrived, while the sender's kernels retire a
little later; the client must not wait for that tail before declaring the next chunk (that would
serialize each chunk's tail with the next declaration), yet a peer that never joins must still be
bounded. What is pinned:

  * a chunk's uploads land in one device arena, sized to the chunk budget and grown for a tensor
    above it, as views carrying each tensor's values, dtype and shape;
  * the arena is reused across chunks with no host wait — a chunk's sends are settled two chunks
    later — and every chunk's broadcasts go out in declared order;
  * a chunk that fails before its event is recorded drains the send stream as a whole under the
    cleanup deadline, and the deadline aborts the group and retires the client;
  * the end of a sync settles what is outstanding and drops the arena, so it does not pin the sync
    GPU at the largest chunk's size between syncs.

    python tests/cpu/grpo/test_weight_sync_sglang_staging.py
"""

import contextlib

import pytest
import torch

import src.distributed.nccl.clients.sglang as sglang_module
from tests.common.weight_sync import offline_sglang_client

_BUDGET = 4096


class _FakeStream:
    def __init__(self):
        self.waited_on: list = []

    def wait_stream(self, other):
        self.waited_on.append(other)


class _FakeEvent:
    def __init__(self):
        self.stream = None

    def record(self, stream):
        self.stream = stream


@pytest.fixture
def client(monkeypatch):
    """A client staging on the CPU, with the CUDA stream/event calls stubbed and the engine's declare
    answered at once; the arena, the settle bookkeeping and the failure path run as in production."""
    handle = offline_sglang_client()
    handle._sync_device = torch.device("cpu")
    handle._group = object()
    handle._send_stream = _FakeStream()
    monkeypatch.setattr(sglang_module, "WEIGHT_SYNC_CHUNK_BYTES", _BUDGET)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: _FakeStream())
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(handle, "_post_once", lambda path, **kwargs: None)
    monkeypatch.setattr(handle, "_post", lambda path, **kwargs: None)
    monkeypatch.setattr(handle, "_lift_pause", lambda timeout, context: None)
    return handle


_DRAIN_DEADLINE = 1234.5


def _settle_probe(monkeypatch) -> list:
    """Records every settle as ``(event, timeout_s)``; the drain deadline resolves to a sentinel."""
    settled: list = []
    monkeypatch.setattr(sglang_module, "resolve_drain_timeout_s", lambda: _DRAIN_DEADLINE)
    monkeypatch.setattr(
        sglang_module, "bounded_event_sync", lambda event, timeout_s, what: settled.append((event, timeout_s))
    )
    return settled


def test_chunks_are_staged_in_one_arena_sized_to_the_budget(client, monkeypatch):
    """Per-tensor device allocations churn the allocator on the critical path and, once freed with a
    pending NCCL event, block the next upload inside the allocator; the arena sidesteps both."""
    settled = _settle_probe(monkeypatch)
    first = [torch.arange(6, dtype=torch.bfloat16).reshape(2, 3), torch.full((5,), 2.5, dtype=torch.float32)]

    staged = list(client._stage_on_device(first))
    arena = client._arena
    assert arena is not None and arena.dtype == torch.uint8
    assert arena.numel() == _BUDGET, "the arena is sized to the chunk budget, not to the chunk"
    for view, tensor in zip(staged, first, strict=True):
        assert view.dtype == tensor.dtype and view.shape == tensor.shape
        assert torch.equal(view, tensor)
        assert view.untyped_storage().data_ptr() == arena.untyped_storage().data_ptr(), "a view outside the arena"
    assert settled == [], "nothing was in flight, so nothing was waited on"

    pending = _FakeEvent()
    client._inflight.append(pending)
    oversized = [torch.zeros(_BUDGET + 1, dtype=torch.uint8)]
    staged = list(client._stage_on_device(oversized))
    assert client._arena is not arena and client._arena.numel() >= oversized[0].numel(), (
        "a tensor above the budget must grow the arena"
    )
    assert settled == [(pending, _DRAIN_DEADLINE)], (
        "the arena was regrown while the previous chunk's sends could still be reading it"
    )
    assert torch.equal(staged[0], oversized[0])


def test_a_chunks_sends_are_settled_two_chunks_later_not_before_the_next_declaration(client, monkeypatch):
    settled = _settle_probe(monkeypatch)
    broadcasts: list[torch.Tensor] = []
    monkeypatch.setattr(sglang_module.dist, "broadcast", lambda tensor, src, group: broadcasts.append(tensor.clone()))
    chunks = [[(f"w{index}", torch.full((3,), float(index), dtype=torch.bfloat16))] for index in range(3)]

    client._send_chunk(chunks[0], flush_cache=False)
    first = client._inflight[-1]
    assert isinstance(first, _FakeEvent) and first.stream is client._send_stream, (
        "the chunk's event must be recorded on the send stream"
    )
    assert len(client._send_stream.waited_on) == 1 and len(broadcasts) == 1, (
        "the send stream must be ordered behind the caller's stream before the chunk's first broadcast"
    )
    assert settled == [], "the first chunk's sends were waited for before anything else was declared"
    arena = client._arena

    client._send_chunk(chunks[1], flush_cache=False)
    assert settled == [], "the second chunk's declaration waited on the first chunk's sends"
    assert client._arena is arena, "a chunk that fits must reuse the arena"

    client._send_chunk(chunks[2], flush_cache=False)
    assert settled == [(first, _DRAIN_DEADLINE)], "the third chunk did not settle the first chunk's sends"
    assert [tensor.tolist() for tensor in broadcasts] == [[0.0] * 3, [1.0] * 3, [2.0] * 3], (
        "broadcasts must carry each chunk's values in declared order"
    )


def test_a_chunk_that_fails_mid_upload_drains_the_stream_and_aborts_on_deadline(client, monkeypatch):
    """A broadcast that raises before the chunk's event is recorded (a NCCL error surfaced
    synchronously, an OOM growing the arena) leaves the engine's workers parked in the broadcasts it
    declared. The failure path must bound the stream as a whole and abort the group when that
    deadline passes, or the reconnect's destroy blocks forever behind them."""

    def failing_broadcast(tensor, src, group):
        raise RuntimeError("NCCL error: invalid argument")

    monkeypatch.setattr(sglang_module.dist, "broadcast", failing_broadcast)
    aborted: list = []
    monkeypatch.setattr(sglang_module.c10d, "_abort_process_group", aborted.append)
    waited: list = []

    def expired(event, timeout_s, what):
        waited.append((event, timeout_s))
        raise RuntimeError(f"{what} did not complete")

    monkeypatch.setattr(sglang_module, "bounded_event_sync", expired)
    list(client._stage_on_device([torch.ones(3, dtype=torch.bfloat16)]))  # the arena of an earlier chunk
    previous = _FakeEvent()
    client._inflight.append(previous)  # the previous chunk, still in flight on the same peer

    with pytest.raises(RuntimeError, match="did not complete"):
        client._send_chunk([("w", torch.ones(3, dtype=torch.bfloat16))], flush_cache=False)

    assert aborted == [client._group], "the failure path did not abort the weight-update group"
    assert client._aborted and not client._inflight
    assert len(waited) == 1 and waited[0][1] == sglang_module._CLEANUP_TIMEOUT_S, (
        "the failure path must wait under the cleanup deadline, not the drain deadline"
    )
    assert waited[0][0] is not previous and waited[0][0].stream is client._send_stream, (
        "the failure path must drain the whole send stream through a fresh event, not the previous chunk's"
    )
    with pytest.raises(RuntimeError, match="aborted"):
        client.begin_weight_update()


def test_the_end_of_a_sync_settles_the_outstanding_sends_and_drops_the_arena(client, monkeypatch):
    settled = _settle_probe(monkeypatch)
    list(client._stage_on_device([torch.ones(3, dtype=torch.bfloat16)]))
    pending = [_FakeEvent(), _FakeEvent()]
    client._inflight.extend(pending)
    assert client._arena is not None, "nothing was staged — the test would be vacuous"

    client.end_weight_update([])

    assert settled == [(event, _DRAIN_DEADLINE) for event in pending], (
        "every outstanding chunk's sends must be settled before the arena is released"
    )
    assert client._arena is None and not client._inflight, "a finished sync left the arena pinned"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
