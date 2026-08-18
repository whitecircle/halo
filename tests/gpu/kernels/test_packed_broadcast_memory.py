#!/usr/bin/env python
"""Repeated packed_broadcast_producer calls with persistent streams must not ratchet reserved memory.

Per-call streams strand each sync's pack allocations (H2D uploads + the ~1GB cat) in per-stream
allocator pools no later stream can reuse, growing reserved memory by ~the model size per sync until
CUDA recycles stream handles — the production signature is 245 GiB reserved against a 72 GiB
allocation peak on the forwarding rank. This drives the producer the way VLLMWeightSyncClient does
(pinned host buffers re-uploaded via post_iter_func, client-owned streams) and asserts reserved
memory stops growing after the first sync.
"""

import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

import src.distributed.nccl.clients.vllm as vllm_client_module
from src.distributed.nccl.clients.vllm import VLLMWeightSyncClient
from src.distributed.nccl.transport.packed_tensor import DEFAULT_PACKED_NUM_BUFFERS, packed_broadcast_producer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

_SYNCS = 12
_PARAMS_PER_SYNC = 24
_PARAM_BYTES = 32 * 1024 * 1024  # 24 x 32MB per sync: several packs of mixed tensors per call
_PACK_BYTES = 64 * 1024 * 1024


class _SinkGroup:
    """Broadcast sink: consumes the packed buffer without a peer (memory behavior needs no wire)."""

    def broadcast(self, tensor, src=0):
        torch.cuda.current_stream().synchronize()

    def abort(self):
        """No peer to abort."""


def _run_syncs(streams):
    host_params = [
        (f"p{i}", torch.randn(_PARAM_BYTES // 4, device="cpu", pin_memory=True)) for i in range(_PARAMS_PER_SYNC)
    ]
    device = torch.device("cuda", torch.cuda.current_device())
    reserved = []
    for _ in range(_SYNCS):
        packed_broadcast_producer(
            iterator=iter(host_params),
            group=_SinkGroup(),
            src=0,
            post_iter_func=lambda item: item[1].to(device, non_blocking=True),
            buffer_size_bytes=_PACK_BYTES,
            streams=streams,
        )
        torch.cuda.synchronize()
        reserved.append(torch.cuda.memory_reserved())
    return reserved


def test_persistent_streams_bound_reserved_memory():
    streams = [torch.cuda.Stream() for _ in range(DEFAULT_PACKED_NUM_BUFFERS)]
    reserved = _run_syncs(streams)
    first, last = reserved[0], reserved[-1]
    # Identical per-sync allocation sequences on the same streams must reuse one pool: any growth
    # after the first sync means pack allocations are stranding blocks again.
    growth = last - first
    assert growth <= _PACK_BYTES, (
        f"reserved memory ratcheted across syncs with persistent streams: "
        f"{first / 2**20:.0f} MiB -> {last / 2**20:.0f} MiB (growth {growth / 2**20:.0f} MiB); "
        f"pack allocations are landing in unreusable per-stream pools again"
    )


def test_client_reuses_one_stream_pair_across_syncs():
    """The client half of the fix: VLLMWeightSyncClient must lazily create ONE stream pair and pass
    the same list into every producer call — reverting the wiring (dropping streams=, or re-creating
    per sync) reintroduces the production ratchet while the producer-contract test stays green."""
    client = object.__new__(VLLMWeightSyncClient)
    client._reset_buffer_state()  # the client's own initializer for the streamed-chunk state
    client._packed_streams = None
    client.base_url = "http://stub"
    client.communicator = SimpleNamespace(device=torch.device("cuda", torch.cuda.current_device()), aborted=False)

    captured = []

    def fake_producer(iterator, group, src=0, post_iter_func=None, streams=None, **kwargs):
        captured.append(streams)

    ok = SimpleNamespace(status_code=200)
    with (
        mock.patch.object(vllm_client_module, "packed_broadcast_producer", fake_producer),
        mock.patch.object(VLLMWeightSyncClient, "_post", return_value=ok),
        mock.patch.object(VLLMWeightSyncClient, "_post_once", return_value=ok),
    ):
        params = [("w", torch.zeros(4))]
        client.sync_model_weights(params)
        client.sync_model_weights(params)

    assert len(captured) == 2
    assert captured[0] is not None, "client did not create/pass persistent streams to the producer"
    assert captured[0] is captured[1], "client re-created its stream pair between syncs (per-sync churn)"
    assert captured[0] is client._packed_streams
    assert len(captured[0]) == DEFAULT_PACKED_NUM_BUFFERS


def main():
    test_persistent_streams_bound_reserved_memory()
    test_client_reuses_one_stream_pair_across_syncs()
    print("test_packed_broadcast_memory: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
