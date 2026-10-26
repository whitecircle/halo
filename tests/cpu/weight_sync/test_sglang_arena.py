#!/usr/bin/env python
"""CPU tests for the SGLang weight-sync client's device staging and deferred settle.

The engine acknowledges a chunk once it has queued the receives, before the sender's kernels
retire; the client must not wait for them before declaring the next chunk (that would serialize each
chunk's tail with the next declaration), yet a peer that never joins must still be bounded. What is
pinned:

  * a chunk's uploads land in one device arena, sized to the chunk budget and grown for a tensor
    above it, as views carrying each tensor's values, dtype and shape;
  * the arena is allocated on the caller's stream and held for the send stream, so a closed
    client leaves it in a pool the next client and the training step reuse;
  * every source the send stream copies is held for it, so a snapshot the caller drops on the
    engine's reply is not handed out again before the queued copy reads it;
  * the arena is reused across chunks with no host wait — a chunk's sends are settled two chunks
    later — and every chunk's broadcasts go out in declared order;
  * a deferred settle that passes the drain deadline, whether staging the next chunk or ending the
    sync, aborts the group and retires the client;
  * a chunk that fails before its event is recorded drains the send stream as a whole under the
    cleanup deadline, and the deadline aborts the group and retires the client;
  * the end of a sync settles what is outstanding and drops the arena, so it does not pin the sync
    GPU at the largest chunk's size between syncs.

    python tests/cpu/weight_sync/test_sglang_arena.py
"""

import contextlib

import pytest
import torch

import halo.weight_sync.clients.sglang as sglang_module
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from halo.weight_sync.transport import c10d_group
from tests.common.weight_sync import offline_client

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


class _PendingEvent:
    """The event of a chunk whose sends never retire: the engine stopped receiving."""

    def query(self) -> bool:
        return False


class _StreamContexts:
    """``torch.cuda.stream`` and ``current_stream`` stand-ins: entering a stream makes it current."""

    def __init__(self):
        self.active = [_FakeStream()]

    def current(self, device=None):
        return self.active[-1]

    @contextlib.contextmanager
    def enter(self, stream):
        self.active.append(stream)
        try:
            yield
        finally:
            self.active.pop()


@pytest.fixture
def streams() -> _StreamContexts:
    return _StreamContexts()


@pytest.fixture
def recorded_streams(monkeypatch) -> list:
    """Every ``record_stream`` call as ``(storage pointer, stream)``; the real one takes CUDA tensors only."""
    calls: list = []
    monkeypatch.setattr(
        torch.Tensor,
        "record_stream",
        lambda tensor, stream: calls.append((tensor.untyped_storage().data_ptr(), stream)),
    )
    return calls


@pytest.fixture
def client(monkeypatch, streams, recorded_streams):
    """A client staging on the CPU, with the CUDA stream/event calls stubbed and the engine's declare
    answered at once; the arena, the settle bookkeeping and the failure path run as in production."""
    handle = offline_client(SGLangWeightSyncClient)
    handle._sync_device = torch.device("cpu")
    handle._transport = (object(), None)
    handle._send_stream = _FakeStream()
    monkeypatch.setattr(sglang_module, "WEIGHT_SYNC_CHUNK_BYTES", _BUDGET)
    monkeypatch.setattr(torch.cuda, "stream", streams.enter)
    monkeypatch.setattr(torch.cuda, "current_stream", streams.current)
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(handle, "request", lambda method, path, **kwargs: None)
    monkeypatch.setattr(handle, "_lift_pause", lambda timeout, context: None)
    return handle


_DRAIN_DEADLINE = 1234.5


def _settle_probe(monkeypatch) -> list:
    """Records every settle as ``(event, timeout_s)``; the drain deadline is a sentinel."""
    settled: list = []
    monkeypatch.setattr(sglang_module, "CHUNK_SYNC_TIMEOUT_S", _DRAIN_DEADLINE)
    monkeypatch.setattr(
        sglang_module,
        "bounded_event_sync",
        lambda event, timeout_s, what, abort: settled.append((event, timeout_s)),
    )
    return settled


def _expired(waited: list):
    """A ``bounded_stream_sync`` whose deadline passes: it records ``(stream, timeout_s)``, aborts and raises."""

    def expired(stream, timeout_s, what, abort):
        waited.append((stream, timeout_s))
        abort()
        raise RuntimeError(f"{what} did not complete")

    return expired


def test_chunks_are_uploaded_into_one_arena_sized_to_the_budget(client, monkeypatch):
    """Per-tensor device allocations churn the allocator on the critical path and, once freed with a
    pending NCCL event, block the next upload inside the allocator; the arena sidesteps both."""
    settled = _settle_probe(monkeypatch)
    first = [torch.arange(6, dtype=torch.bfloat16).reshape(2, 3), torch.full((5,), 2.5, dtype=torch.float32)]

    views = list(client._upload_to_arena(first, torch.cuda.current_stream()))
    arena = client._arena
    assert arena is not None and arena.dtype == torch.uint8
    assert arena.numel() == _BUDGET, "the arena is sized to the chunk budget, not to the chunk"
    for view, tensor in zip(views, first, strict=True):
        assert view.dtype == tensor.dtype and view.shape == tensor.shape
        assert torch.equal(view, tensor)
        assert view.untyped_storage().data_ptr() == arena.untyped_storage().data_ptr(), "a view outside the arena"
    assert settled == [], "nothing was in flight, so nothing was waited on"

    pending = _FakeEvent()
    client._inflight.append(pending)
    oversized = [torch.zeros(_BUDGET + 1, dtype=torch.uint8)]
    views = list(client._upload_to_arena(oversized, torch.cuda.current_stream()))
    assert client._arena is not arena and client._arena.numel() >= oversized[0].numel(), (
        "a tensor above the budget must grow the arena"
    )
    assert settled == [(pending, _DRAIN_DEADLINE)], (
        "the arena was regrown while the previous chunk's sends could still be reading it"
    )
    assert torch.equal(views[0], oversized[0])


def test_the_arena_is_allocated_on_the_callers_stream_and_held_for_the_send_stream(
    client, monkeypatch, streams, recorded_streams
):
    """The caching allocator reuses a cached block only for the stream that allocated it, and the send
    stream is the client's own: an arena allocated under it stays reserved after the client closes,
    one per reconnect. Allocated under the caller's stream it returns to a pool the next client and
    the training step share; ``record_stream`` keeps that stream off the block until the sends that
    read it retire, which the close path relies on, since it drops the arena without settling."""
    _settle_probe(monkeypatch)
    monkeypatch.setattr(sglang_module.dist, "broadcast", lambda tensor, src, group: None)
    caller = streams.current()
    allocated_under: list = []
    waited_before_allocation: list = []
    real_empty = torch.empty

    def tracking_empty(*args, **kwargs):
        allocated_under.append(streams.current())
        waited_before_allocation.append(list(client._send_stream.waited_on))
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", tracking_empty)
    client._broadcast_chunk([("w", torch.ones(3, dtype=torch.bfloat16))], final=False)
    first = client._arena

    assert allocated_under == [caller], (
        f"the arena was allocated under {allocated_under}, not under the caller's stream {caller}"
    )
    assert waited_before_allocation == [[caller]], (
        "the send stream must wait on the caller's stream before the arena is allocated: the block may "
        f"last have been used there (waited on {waited_before_allocation})"
    )
    assert recorded_streams == [(first.untyped_storage().data_ptr(), client._send_stream)], (
        "the arena's use on the send stream was not recorded with the allocator"
    )

    client._broadcast_chunk([("big", torch.zeros(_BUDGET + 1, dtype=torch.uint8))], final=False)

    assert client._arena is not first and allocated_under == [caller, caller], "the oversized chunk did not regrow"
    assert recorded_streams[1:] == [(client._arena.untyped_storage().data_ptr(), client._send_stream)], (
        "a regrown arena must be recorded for the send stream again"
    )


def test_every_source_is_held_for_the_send_stream(client, monkeypatch):
    """The engine may reply before the send stream has copied a chunk, and the caller drops the chunk's
    snapshots on that reply; a block the allocator was not told is still being read goes to the next
    snapshot, which overwrites it before the queued copy runs."""
    _settle_probe(monkeypatch)
    monkeypatch.setattr(sglang_module.dist, "broadcast", lambda tensor, src, group: None)
    held: list = []
    monkeypatch.setattr(sglang_module, "hold_for_stream", lambda tensor, stream: held.append((tensor, stream)))
    chunk = [("a", torch.ones(3, dtype=torch.bfloat16)), ("b", torch.zeros(5, dtype=torch.float32))]

    client._broadcast_chunk(chunk, final=False)

    assert len(held) == len(chunk), f"{len(held)} of the chunk's {len(chunk)} sources were held"
    for (tensor, stream), (name, source) in zip(held, chunk, strict=True):
        assert tensor is source and stream is client._send_stream, (
            f"source {name!r} was not held for the send stream that copies it"
        )


def test_a_chunks_sends_are_settled_two_chunks_later_not_before_the_next_declaration(client, monkeypatch):
    settled = _settle_probe(monkeypatch)
    broadcasts: list[torch.Tensor] = []
    monkeypatch.setattr(sglang_module.dist, "broadcast", lambda tensor, src, group: broadcasts.append(tensor.clone()))
    chunks = [[(f"w{index}", torch.full((3,), float(index), dtype=torch.bfloat16))] for index in range(3)]

    client._broadcast_chunk(chunks[0], final=False)
    first = client._inflight[-1]
    assert isinstance(first, _FakeEvent) and first.stream is client._send_stream, (
        "the chunk's event must be recorded on the send stream"
    )
    assert len(client._send_stream.waited_on) == 1 and len(broadcasts) == 1, (
        "the send stream must be ordered behind the caller's stream before the chunk's first broadcast"
    )
    assert settled == [], "the first chunk's sends were waited for before anything else was declared"
    arena = client._arena

    client._broadcast_chunk(chunks[1], final=False)
    assert settled == [], "the second chunk's declaration waited on the first chunk's sends"
    assert client._arena is arena, "a chunk that fits must reuse the arena"

    client._broadcast_chunk(chunks[2], final=False)
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
    monkeypatch.setattr(c10d_group.c10d, "_abort_process_group", aborted.append)
    waited: list = []
    monkeypatch.setattr(sglang_module, "bounded_stream_sync", _expired(waited))
    # The arena of an earlier chunk.
    list(client._upload_to_arena([torch.ones(3, dtype=torch.bfloat16)], torch.cuda.current_stream()))
    previous = _FakeEvent()
    client._inflight.append(previous)  # the previous chunk, still in flight on the same peer

    with pytest.raises(RuntimeError, match="did not complete"):
        client._broadcast_chunk([("w", torch.ones(3, dtype=torch.bfloat16))], final=False)

    assert aborted == [client._transport[0]], "the failure path did not abort the weight-update group"
    assert client._aborted and not client._inflight
    assert waited == [(client._send_stream, sglang_module.CLEANUP_TIMEOUT_S)], (
        "the failure path must drain the whole send stream, not the previous chunk's event, under the "
        f"cleanup deadline rather than the drain deadline (waited on {waited})"
    )
    with pytest.raises(RuntimeError, match="aborted"):
        client._start_engine_update()


def test_a_drain_deadline_on_the_failure_path_still_reports_the_engines_own_error(client, monkeypatch):
    """The engine rejecting the chunk is what usually leaves a broadcast spinning: the drain then
    passes its deadline, and the error raised must still be the engine's, not the deadline's."""

    def rejected(method, path, **kwargs):
        raise RuntimeError("HTTP 500: the engine refused the declared shapes")

    monkeypatch.setattr(client, "request", rejected)

    def failing_broadcast(tensor, src, group):
        raise RuntimeError("NCCL error: peer answered with an error")

    monkeypatch.setattr(sglang_module.dist, "broadcast", failing_broadcast)
    monkeypatch.setattr(c10d_group.c10d, "_abort_process_group", lambda group: None)
    monkeypatch.setattr(sglang_module, "bounded_stream_sync", _expired([]))

    with pytest.raises(RuntimeError, match="refused the declared shapes") as excinfo:
        client._broadcast_chunk([("w", torch.ones(3, dtype=torch.bfloat16))], final=False)
    assert "/update_weights_from_distributed" in str(excinfo.value), "the failing engine route is not named"
    assert client._aborted, "the deadline must still have aborted the group"


def _stage_the_next_chunk(client) -> None:
    list(client._upload_to_arena([torch.ones(3, dtype=torch.bfloat16)], torch.cuda.current_stream()))


def _end_the_sync(client) -> None:
    client._release_arena()


@pytest.mark.parametrize("settle", [_stage_the_next_chunk, _end_the_sync], ids=["next-chunk", "end-of-sync"])
def test_a_deferred_settle_that_passes_the_drain_deadline_aborts_the_group(client, monkeypatch, settle):
    """The settle is the only host wait on a chunk's sends once the engine has acknowledged it; a peer
    that stopped receiving leaves those kernels spinning, and without the abort the client is not
    retired and the reconnect's destroy blocks behind them."""
    monkeypatch.setattr(sglang_module, "CHUNK_SYNC_TIMEOUT_S", 0.0)
    aborted: list = []
    monkeypatch.setattr(c10d_group.c10d, "_abort_process_group", aborted.append)
    client._inflight.extend(_PendingEvent() for _ in range(sglang_module._INFLIGHT_CHUNKS))

    with pytest.raises(RuntimeError, match="weight broadcast did not complete"):
        settle(client)

    assert aborted == [client._transport[0]], "the settle deadline did not abort the weight-update group"
    assert client._aborted, "an aborted group must keep close from destroying it"
    with pytest.raises(RuntimeError, match="aborted"):
        client._start_engine_update()


def test_the_end_of_a_sync_settles_the_outstanding_sends_and_drops_the_arena(client, monkeypatch):
    settled = _settle_probe(monkeypatch)
    list(client._upload_to_arena([torch.ones(3, dtype=torch.bfloat16)], torch.cuda.current_stream()))
    pending = [_FakeEvent(), _FakeEvent()]
    client._inflight.extend(pending)
    assert client._arena is not None, "nothing was uploaded — the test would be vacuous"

    client._finish_engine_update([])

    assert settled == [(event, _DRAIN_DEADLINE) for event in pending], (
        "every outstanding chunk's sends must be settled before the arena is released"
    )
    assert client._arena is None and not client._inflight, "a finished sync left the arena pinned"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
