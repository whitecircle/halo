#!/usr/bin/env python
"""A stream running behind the caller must still read the tensors it was handed.

The caller drops a chunk's snapshots once the engine replies, before those reads need to finish. Unless they are
recorded with the allocator, the next allocation on the caller's stream takes the freed block and overwrites it.
"""

from unittest import mock

import torch

import halo.weight_sync.clients.sglang as sglang_module
import halo.weight_sync.transport.packed_broadcast as packed_broadcast
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from halo.weight_sync.group import WeightSyncGroup
from halo.weight_sync.staging import ArenaChunkStager, PackedChunkStager
from halo.weight_sync.transport.packed_broadcast import PACKED_NUM_BUFFERS, broadcast_packed
from tests.common.harness import gpu_test_main, record_check
from tests.common.weight_sync import offline_client

_TENSORS = 8
_ELEMENTS = 1 << 20
# About 50 ms on current parts: far longer than the refill below takes on the caller's stream.
_SLEEP_CYCLES = 100_000_000


def _sources(device: torch.device) -> list[torch.Tensor]:
    return [torch.full((_ELEMENTS,), float(index + 1), device=device) for index in range(_TENSORS)]


def _refill_freed_blocks(device: torch.device) -> list[torch.Tensor]:
    """Same-size allocations on the caller's stream, which the allocator serves from freed blocks first."""
    return [torch.full((_ELEMENTS,), -1.0, device=device) for _ in range(_TENSORS)]


class _WireGroup:
    """Keeps what the wire carries: a copy taken on the sending stream."""

    def __init__(self):
        self.sent: list[torch.Tensor] = []

    def broadcast(self, tensor):
        self.sent.append(tensor.clone())

    def abort(self):
        """No peer to abort."""


def test_the_vllm_broadcast_holds_its_staged_buffers_for_the_pack_stream():
    device = torch.device("cuda", torch.cuda.current_device())
    sources = _sources(device)
    expected = torch.cat([source.view(torch.uint8) for source in sources])
    streams = [torch.cuda.Stream() for _ in range(PACKED_NUM_BUFFERS)]
    group = _WireGroup()
    drain = packed_broadcast.bounded_stream_sync

    def drained_then_held_back(stream, **kwargs):
        drain(stream, **kwargs)
        with torch.cuda.stream(stream):
            torch.cuda._sleep(_SLEEP_CYCLES)

    stager = PackedChunkStager()
    staged = stager.stage([(f"p{i}", s) for i, s in enumerate(sources)])
    with mock.patch.object(packed_broadcast, "bounded_stream_sync", drained_then_held_back):
        broadcast_packed(staged.wire, group, streams, ready=staged.ready)
    stager.release()
    del sources, staged
    _overwrites = _refill_freed_blocks(device)
    torch.cuda.synchronize()

    assert len(group.sent) == 1 and torch.equal(group.sent[0], expected), (
        "the pack stream read a staged buffer after its block was handed to the caller's next allocation "
        f"({int((group.sent[0] != expected).sum())} of {expected.numel()} bytes differ)"
    )


def _assert_sglang_holds_its_sources(*, through_group: bool) -> None:
    """The client sends on the caller's thread; a group sends on a worker thread, whose current stream is its own."""
    device = torch.device("cuda", torch.cuda.current_device())
    client = offline_client(SGLangWeightSyncClient)
    client._sync_device = device
    client._transport = (object(), None)
    client._send_stream = torch.cuda.Stream(device=device)
    # The copies into the arena run on the stager's stream, held back as the send stream used to be.
    client._stager._stream = torch.cuda.Stream(device=device)
    group = WeightSyncGroup([client], nodes=["node"], group_port=0, group_host=None)
    send = group._broadcast_chunk if through_group else client._broadcast_chunk
    sources = _sources(device)
    expected = [source.clone() for source in sources]
    sent: list[torch.Tensor] = []
    with torch.cuda.stream(client._stager._stream):
        torch.cuda._sleep(_SLEEP_CYCLES)
    with (
        mock.patch.object(sglang_module.dist, "broadcast", lambda tensor, src, group: sent.append(tensor.clone())),
        mock.patch.object(client, "request", return_value=None),
    ):
        send([(f"p{i}", s) for i, s in enumerate(sources)], final=False)
        del sources
        _overwrites = _refill_freed_blocks(device)
        client.release_send_buffers()
    torch.cuda.synchronize()

    corrupted = [
        index for index, (got, want) in enumerate(zip(sent, expected, strict=True)) if not torch.equal(got, want)
    ]
    assert not corrupted, (
        f"the send stream copied sources {corrupted} after their blocks were handed to the caller's next allocation"
    )


def test_the_sglang_client_holds_its_sources_for_the_send_stream():
    _assert_sglang_holds_its_sources(through_group=False)


def test_the_group_holds_its_sources_for_the_send_stream_it_sends_on_from_a_worker_thread():
    _assert_sglang_holds_its_sources(through_group=True)


def test_the_arena_waits_for_the_previous_chunks_readers_before_it_overwrites_them():
    device = torch.device("cuda", torch.cuda.current_device())
    stager = ArenaChunkStager()
    first = _sources(device)
    expected = [source.clone() for source in first]
    staged = stager.stage([(f"p{i}", s) for i, s in enumerate(first)])
    reader_stream = torch.cuda.Stream(device=device)
    reader_stream.wait_event(staged.ready)
    with torch.cuda.stream(reader_stream):
        torch.cuda._sleep(_SLEEP_CYCLES)
        read = [view.clone() for view in staged.wire]
        done = torch.cuda.Event()
        done.record(reader_stream)
    staged.add_reader(done)
    second = [torch.full((_ELEMENTS,), -2.0, device=device) for _ in range(_TENSORS)]
    stager.stage([(f"q{i}", s) for i, s in enumerate(second)])
    torch.cuda.synchronize()
    stager.release()

    corrupted = [
        index for index, (got, want) in enumerate(zip(read, expected, strict=True)) if not torch.equal(got, want)
    ]
    assert not corrupted, f"the next chunk overwrote the arena before the previous chunk's senders read {corrupted}"


@gpu_test_main(exact_world_size=1, prefix="weight_sync_stream_safety", partial_state=False)
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    record_check(
        checks,
        "vllm_broadcast_holds_its_buffers",
        test_the_vllm_broadcast_holds_its_staged_buffers_for_the_pack_stream,
    )
    record_check(
        checks, "sglang_client_holds_its_sources", test_the_sglang_client_holds_its_sources_for_the_send_stream
    )
    record_check(
        checks,
        "sglang_group_holds_its_sources",
        test_the_group_holds_its_sources_for_the_send_stream_it_sends_on_from_a_worker_thread,
    )
    record_check(
        checks,
        "arena_waits_for_readers",
        test_the_arena_waits_for_the_previous_chunks_readers_before_it_overwrites_them,
    )
    return {"checks": checks}


if __name__ == "__main__":
    run()
