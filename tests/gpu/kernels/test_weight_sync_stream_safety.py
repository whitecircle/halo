#!/usr/bin/env python
"""A send stream running behind the caller must still read the tensors it was handed.

Both weight-sync clients read the caller's snapshots on CUDA streams of their own (the vLLM
producer's pack streams, the SGLang client's send stream), and the caller drops a chunk's snapshots
once the engine has replied, which need not wait for those reads. Unless the reads are recorded with
the allocator, the next allocation on the caller's stream takes a freed block and overwrites it
before the queued read runs. Each case holds the reading stream back with a GPU sleep, drops the
sources, refills their blocks from the caller's stream and checks what reached the wire.

Run: torchrun --nproc_per_node=1 tests/gpu/kernels/test_weight_sync_stream_safety.py
"""

from unittest import mock

import torch

import src.weight_sync.clients.sglang as sglang_module
import src.weight_sync.transport.packed_broadcast as packed_broadcast
from src.weight_sync.clients.sglang import SGLangWeightSyncClient
from src.weight_sync.transport.packed_broadcast import PACKED_NUM_BUFFERS, packed_broadcast_producer
from tests.common.harness import gpu_test_main, record_check
from tests.common.weight_sync import offline_client

_TENSORS = 8
_ELEMENTS = 1 << 20
# About 50 ms on current parts: far longer than the refill below takes to run on the caller's stream.
_SLEEP_CYCLES = 100_000_000


def _sources(device: torch.device) -> list[torch.Tensor]:
    return [torch.full((_ELEMENTS,), float(index + 1), device=device) for index in range(_TENSORS)]


def _refill_freed_blocks(device: torch.device) -> list[torch.Tensor]:
    """Same-size allocations on the caller's stream, which the allocator serves from freed blocks first."""
    return [torch.full((_ELEMENTS,), -1.0, device=device) for _ in range(_TENSORS)]


class _WireGroup:
    """Broadcast sink keeping what the wire carries: a copy taken on the sending stream."""

    def __init__(self):
        self.sent: list[torch.Tensor] = []

    def broadcast(self, tensor):
        self.sent.append(tensor.clone())

    def abort(self):
        """No peer to abort."""


def test_the_vllm_producer_holds_its_inputs_for_the_pack_stream():
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

    with mock.patch.object(packed_broadcast, "bounded_stream_sync", drained_then_held_back):
        packed_broadcast_producer(iter([(f"p{i}", s) for i, s in enumerate(sources)]), group=group, streams=streams)
    del sources
    _overwrites = _refill_freed_blocks(device)
    torch.cuda.synchronize()

    assert len(group.sent) == 1 and torch.equal(group.sent[0], expected), (
        "the pack stream read an input after its block was handed to the caller's next allocation "
        f"({int((group.sent[0] != expected).sum())} of {expected.numel()} bytes differ)"
    )


def test_the_sglang_client_holds_its_sources_for_the_send_stream():
    device = torch.device("cuda", torch.cuda.current_device())
    client = offline_client(SGLangWeightSyncClient)
    client._sync_device = device
    client._transport = (object(), None)
    client._send_stream = torch.cuda.Stream(device=device)
    sources = _sources(device)
    expected = [source.clone() for source in sources]
    sent: list[torch.Tensor] = []
    with torch.cuda.stream(client._send_stream):
        torch.cuda._sleep(_SLEEP_CYCLES)
    with (
        mock.patch.object(sglang_module.dist, "broadcast", lambda tensor, src, group: sent.append(tensor.clone())),
        mock.patch.object(client, "request", return_value=None),
    ):
        client._broadcast_chunk([(f"p{i}", s) for i, s in enumerate(sources)], final=False)
        del sources
        _overwrites = _refill_freed_blocks(device)
        client._release_arena()
    torch.cuda.synchronize()

    corrupted = [
        index for index, (got, want) in enumerate(zip(sent, expected, strict=True)) if not torch.equal(got, want)
    ]
    assert not corrupted, (
        f"the send stream copied sources {corrupted} after their blocks were handed to the caller's next allocation"
    )


@gpu_test_main(exact_world_size=1, prefix="weight_sync_stream_safety", partial_state=False)
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    record_check(checks, "vllm_producer_holds_its_inputs", test_the_vllm_producer_holds_its_inputs_for_the_pack_stream)
    record_check(
        checks, "sglang_client_holds_its_sources", test_the_sglang_client_holds_its_sources_for_the_send_stream
    )
    return {"checks": checks}


if __name__ == "__main__":
    run()
