#!/usr/bin/env python
"""The vLLM weight-sync path must not ratchet reserved memory, across syncs or reconnects.

The allocator pools freed blocks per stream, and the producer packs on streams the client owns, so a closed client's
pool strands one sync's worth per reconnect.
"""

from types import SimpleNamespace
from unittest import mock

import torch

import halo.weight_sync.clients.vllm as vllm_client_module
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient
from halo.weight_sync.transport.packed_broadcast import PACKED_NUM_BUFFERS, packed_broadcast_producer
from tests.common.harness import gpu_test_main, record_check
from tests.common.weight_sync import offline_client, push

_SYNCS = 12
_PARAMS_PER_SYNC = 24
_PARAM_BYTES = 32 * 1024 * 1024  # 24 x 32MB per sync: several packs of mixed tensors per call.
_PACK_BYTES = 64 * 1024 * 1024
_CLIENTS = 4
# offline_client never runs init_communicator, so their snapshots stay on the host and the client lands them itself.
_HOST_PARAMS = 4


class _SinkGroup:
    """Consumes the packed buffer without a peer: memory behavior needs no wire."""

    def broadcast(self, tensor):
        torch.cuda.current_stream().synchronize()

    def abort(self):
        """No peer to abort."""


class _SinkCommunicator(_SinkGroup):
    """``PyNcclCommunicator`` stand-in on this rank's GPU: the client reads its device and abort state."""

    def __init__(self, device: torch.device):
        self.device = device
        self.aborted = False


def _run_syncs(streams):
    device = torch.device("cuda", torch.cuda.current_device())
    staged_params = [(f"p{i}", torch.randn(_PARAM_BYTES // 4, device=device)) for i in range(_PARAMS_PER_SYNC)]
    reserved = []
    for _ in range(_SYNCS):
        packed_broadcast_producer(
            iterator=iter(staged_params), group=_SinkGroup(), streams=streams, buffer_size_bytes=_PACK_BYTES
        )
        torch.cuda.synchronize()
        reserved.append(torch.cuda.memory_reserved())
    return reserved


def test_persistent_streams_bound_reserved_memory():
    streams = [torch.cuda.Stream() for _ in range(PACKED_NUM_BUFFERS)]
    reserved = _run_syncs(streams)
    first, last = reserved[0], reserved[-1]
    # Identical per-sync allocations on the same streams must reuse one pool; growth after the first means stranded
    # blocks.
    growth = last - first
    assert growth <= _PACK_BYTES, (
        f"reserved memory ratcheted across syncs with persistent streams: "
        f"{first / 2**20:.0f} MiB -> {last / 2**20:.0f} MiB (growth {growth / 2**20:.0f} MiB); "
        f"pack allocations are landing in unreusable per-stream pools again"
    )


def test_client_reuses_one_stream_pair_across_syncs():
    """Dropping ``streams=`` or re-creating them per sync reintroduces the ratchet while the producer-contract test
    stays green."""
    client = offline_client(VLLMWeightSyncClient)
    client._sync_device = torch.device("cuda", torch.cuda.current_device())
    client._transport = SimpleNamespace(device=client._sync_device, aborted=False)

    captured = []

    def fake_producer(iterator, group, streams, **kwargs):
        captured.append(streams)

    ok = SimpleNamespace(status_code=200)
    with (
        mock.patch.object(vllm_client_module, "packed_broadcast_producer", fake_producer),
        mock.patch.object(VLLMWeightSyncClient, "request", return_value=ok),
    ):
        params = [("w", torch.zeros(4))]
        push(client, params)
        push(client, params)

    assert len(captured) == 2
    assert captured[0] is not None, "client did not create/pass persistent streams to the producer"
    assert captured[0] is captured[1], "client re-created its stream pair between syncs (per-sync churn)"
    assert captured[0] is client._packed_streams
    assert len(captured[0]) == PACKED_NUM_BUFFERS


def test_reconnected_clients_reuse_one_pool():
    """Anything allocated on a closed client's stream pair stays reserved with nothing to reuse it, about one chunk per
    reconnect."""
    device = torch.device("cuda", torch.cuda.current_device())
    params = [(f"p{i}", torch.randn(_PARAM_BYTES // 4, device=device)) for i in range(_PARAMS_PER_SYNC)]
    params += [(f"h{i}", torch.randn(_PARAM_BYTES // 4)) for i in range(_HOST_PARAMS)]
    ok = SimpleNamespace(status_code=200)
    reserved = []
    with (
        mock.patch.object(VLLMWeightSyncClient, "request", return_value=ok),
    ):
        for _ in range(_CLIENTS):
            client = offline_client(VLLMWeightSyncClient)
            client._sync_device = device
            client._transport = _SinkCommunicator(device)
            push(client, params)
            client.close_communicator()
            torch.cuda.synchronize()
            reserved.append(torch.cuda.memory_reserved())
    growth = reserved[-1] - reserved[0]
    assert growth <= _PARAM_BYTES, (
        f"reserved memory ratcheted across {_CLIENTS} connect-sync-close cycles: "
        f"{[r >> 20 for r in reserved]} MiB (growth {growth >> 20} MiB); a closed client's pack buffer or "
        f"uploads stayed in its own streams' pools"
    )


@gpu_test_main(exact_world_size=1, prefix="packed_broadcast_memory", partial_state=False)
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    record_check(checks, "persistent_streams_bound_reserved_memory", test_persistent_streams_bound_reserved_memory)
    record_check(checks, "client_reuses_one_stream_pair_across_syncs", test_client_reuses_one_stream_pair_across_syncs)
    record_check(checks, "reconnected_clients_reuse_one_pool", test_reconnected_clients_reuse_one_pool)
    return {"checks": checks}


if __name__ == "__main__":
    run()
