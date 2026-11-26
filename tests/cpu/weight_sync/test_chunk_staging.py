#!/usr/bin/env python
"""A pool of servers stages each chunk once: the bytes the sending GPU holds for staging do not grow with the number of
servers, and every server broadcasts the same buffers."""

from __future__ import annotations

import threading

import pytest
import torch

import halo.weight_sync.pool as pool_module
import halo.weight_sync.staging as staging_module
from halo.weight_sync.chunking import resolve_sync_device
from halo.weight_sync.clients.server_calls import AsyncCall
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient
from halo.weight_sync.pool import WeightSyncPool
from halo.weight_sync.staging import ChunkStager, StagedChunk
from tests.common.weight_sync import push

_BUDGET = 4096
# Three bf16 tensors of 1,500 bytes fill a chunk's budget, so six make two chunks and a tail.
_PARAMS = [(f"layers.{i}.w", torch.full((750,), float(i + 1), dtype=torch.bfloat16)) for i in range(7)]
_ENGINES = {"vllm": VLLMWeightSyncClient, "sglang": SGLangWeightSyncClient}


class _Sightings:
    """What the sending side saw: stagings made, the most bytes all stagers held at once, and the buffers each client
    broadcast."""

    def __init__(self) -> None:
        self.stagings = 0
        self.peak_staged_bytes = 0
        self._stagers: dict[int, ChunkStager] = {}
        self.broadcasts: list[tuple[int, list[int]]] = []
        self._lock = threading.Lock()

    def note_staging(self) -> None:
        with self._lock:
            self.stagings += 1

    def note_broadcast(self, client: int, staged: StagedChunk, stager: ChunkStager) -> None:
        with self._lock:
            self._stagers[id(stager)] = stager
            live = sum(seen.staged_bytes for seen in self._stagers.values())
            self.peak_staged_bytes = max(self.peak_staged_bytes, live)
            self.broadcasts.append((client, [buffer.data_ptr() for buffer in staged.wire]))


def _counting(stage, sightings: _Sightings):
    def counted(stager, tensors, readers):
        sightings.note_staging()
        return stage(stager, tensors, readers)

    return counted


def _recording_client_class(engine: str, sightings: _Sightings) -> type:
    """The real client's staging and chunk accounting; the server and the NCCL transport are not there."""

    class Recording(_ENGINES[engine]):
        def check_server(self, total_timeout: float = 0.0) -> None:
            """No server to probe."""

        def init_communicator(self, device=0) -> None:
            self._sync_device = resolve_sync_device(device)

        def prepare_chunk_send(self, chunk) -> None:
            """Nothing to refuse."""

        def declare_chunk(self, chunk, final: bool) -> AsyncCall:
            return AsyncCall(lambda: None)

        def send_staged(self, staged: StagedChunk) -> None:
            sightings.note_broadcast(self.group_port, staged, self._stager)

        def _start_engine_update(self) -> None:
            """No engine to quiesce."""

        def resume_engine(self) -> None:
            """No engine to resume."""

        def require_resumed(self) -> None:
            """No engine to resume."""

        def _finish_engine_update(self, tail) -> None:
            self._transmit_chunk(tail, final=True)
            self.release_send_buffers()

    return Recording


def _push_through_pool(engine: str, servers: int, monkeypatch, *, shared: bool = True) -> _Sightings:
    sightings = _Sightings()
    with monkeypatch.context() as patch:
        patch.setattr(pool_module, "WEIGHT_SYNC_CHUNK_BYTES", _BUDGET)
        patch.setattr(staging_module, "WEIGHT_SYNC_CHUNK_BYTES", _BUDGET)
        for stager_class in (staging_module.PackedChunkStager, staging_module.ArenaChunkStager):
            patch.setattr(stager_class, "_stage", _counting(stager_class._stage, sightings))
        client_class = _recording_client_class(engine, sightings)
        if not shared:
            patch.setattr(client_class, "share_stager", lambda client, stager: None)
        pool = WeightSyncPool(
            server_configs=[{"url": f"http://localhost:{8000 + i}", "group_port": 51000 + i} for i in range(servers)],
            connection_timeout=0.0,
            client_cls=client_class,
            base_group_port=51000,
        )
        pool.init_communicator("cpu")
        push(pool, [(name, tensor.clone()) for name, tensor in _PARAMS])
    return sightings


@pytest.mark.parametrize("engine", sorted(_ENGINES))
def test_staging_memory_and_work_do_not_grow_with_the_number_of_servers(engine, monkeypatch):
    """Per server, each client used to pack or upload its own copy of every chunk, concurrently."""
    one = _push_through_pool(engine, 1, monkeypatch)
    four = _push_through_pool(engine, 4, monkeypatch)

    assert one.stagings >= 3, "the pushed payload must span several chunks, or the comparison is vacuous"
    assert four.stagings == one.stagings, f"{four.stagings} stagings for four servers against {one.stagings} for one"
    assert four.peak_staged_bytes == one.peak_staged_bytes > 0, (
        f"staging held {four.peak_staged_bytes} bytes for four servers and {one.peak_staged_bytes} for one"
    )
    per_client: dict[int, list[list[int]]] = {}
    for client, pointers in four.broadcasts:
        per_client.setdefault(client, []).append(pointers)
    sequences = list(per_client.values())
    assert len(sequences) == 4 and all(len(sequence) == one.stagings for sequence in sequences), per_client
    assert all(sequence == sequences[0] for sequence in sequences), (
        "every server must broadcast every chunk from the same buffers"
    )


@pytest.mark.parametrize("engine", sorted(_ENGINES))
def test_servers_that_stage_for_themselves_would_multiply_the_work(engine, monkeypatch):
    """The control for the test above: with the stager unshared, the same counts grow with the servers."""
    one = _push_through_pool(engine, 1, monkeypatch, shared=False)
    four = _push_through_pool(engine, 4, monkeypatch, shared=False)

    assert four.stagings == 4 * one.stagings, (four.stagings, one.stagings)
    assert four.peak_staged_bytes == 4 * one.peak_staged_bytes, (four.peak_staged_bytes, one.peak_staged_bytes)


@pytest.mark.parametrize("engine", sorted(_ENGINES))
def test_the_staged_wire_carries_every_tensor_of_the_chunk(engine):
    stager = _ENGINES[engine].CHUNK_STAGER()
    chunk = [("a", torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)), ("b", torch.full((5,), 2.5))]

    staged = stager.stage(chunk)
    sent = torch.cat([buffer.contiguous().view(torch.uint8).view(-1) for buffer in staged.wire])

    assert torch.equal(sent, torch.cat([tensor.contiguous().view(torch.uint8).view(-1) for _, tensor in chunk]))
    assert stager.stage(chunk) is staged, "a second caller of the same chunk must get the first caller's staging"
    stager.release()
    assert stager.staged_bytes == 0


def test_a_stale_staging_is_never_matched_to_a_later_syncs_tensors():
    """A snapshot freed and reallocated at the same address with the same name is a different chunk."""
    stager = VLLMWeightSyncClient.CHUNK_STAGER()
    first = stager.stage([("w", torch.zeros(4))])
    second = stager.stage([("w", torch.ones(4))])

    assert second is not first
    assert torch.equal(second.wire[0].view(torch.float32), torch.ones(4))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
