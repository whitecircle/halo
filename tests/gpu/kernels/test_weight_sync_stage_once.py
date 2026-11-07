#!/usr/bin/env python
"""One staging serves every sender: concurrent senders on their own streams read the same buffers, the GPU memory held
for staging does not grow with their number, and a sender still reading a chunk keeps the arena from being rewritten by
the next one."""

from unittest import mock

import torch

import halo.weight_sync.staging as staging_module
from halo.weight_sync.staging import ArenaChunkStager, ChunkStager, PackedChunkStager
from tests.common.harness import gpu_test_main, record_check

_TENSORS = 6
_ELEMENTS = 1 << 20
_ARENA_BYTES = 64 * 1024 * 1024
# About 50 ms on current parts: far longer than staging the next chunk takes on the caller's stream.
_SLEEP_CYCLES = 100_000_000
_SENDERS = 4


def _chunk(device: torch.device, value: float) -> list[tuple[str, torch.Tensor]]:
    return [(f"w{i}", torch.full((_ELEMENTS,), value + i, device=device)) for i in range(_TENSORS)]


def _bytes(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.contiguous().view(torch.uint8).view(-1) for tensor in tensors])


def _read_in_the_background(staged, streams: list[torch.cuda.Stream], *, capture: bool) -> list[torch.Tensor]:
    """What each sender's stream reads of the staged wire, after a delay, as a broadcast would; the copies are only
    taken when ``capture``, as they cost memory of their own."""
    captured = []
    for stream in streams:
        with torch.cuda.stream(stream):
            stream.wait_event(staged.ready)
            torch.cuda._sleep(_SLEEP_CYCLES)
            if capture:
                captured.append(_bytes(staged.wire))
            event = torch.cuda.Event()
            event.record(stream)
        staged.add_reader(event)
    return captured


def _staging_peak(stager_class: type[ChunkStager], senders: int) -> int:
    """Peak device bytes above the snapshots while ``senders`` streams read one staged chunk."""
    device = torch.device("cuda", torch.cuda.current_device())
    chunk = _chunk(device, 1.0)
    streams = [torch.cuda.Stream(device=device) for _ in range(senders)]
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    stager = stager_class()
    staged = stager.stage(chunk)
    _read_in_the_background(staged, streams, capture=False)
    # Every sender stages through the one stager, as a pool's clients do.
    for _ in range(senders - 1):
        assert stager.stage(chunk) is staged
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    del staged
    stager.release()
    return peak


def test_staging_memory_does_not_grow_with_the_number_of_senders():
    for stager_class in (PackedChunkStager, ArenaChunkStager):
        one = _staging_peak(stager_class, 1)
        many = _staging_peak(stager_class, _SENDERS)
        assert many <= one, (
            f"{stager_class.__name__}: staging peaked at {many >> 20} MiB for {_SENDERS} senders against "
            f"{one >> 20} MiB for one"
        )


def test_a_next_chunk_waits_for_every_sender_still_reading_the_arena():
    """The arena is reused: restaging while a sender's stream lags must not rewrite what it is yet to read."""
    device = torch.device("cuda", torch.cuda.current_device())
    with mock.patch.object(staging_module, "WEIGHT_SYNC_CHUNK_BYTES", _ARENA_BYTES):
        stager = ArenaChunkStager()
        first = _chunk(device, 1.0)
        expected = _bytes([tensor for _, tensor in first])
        staged = stager.stage(first)
        captured = _read_in_the_background(
            staged, [torch.cuda.Stream(device=device) for _ in range(_SENDERS)], capture=True
        )
        second = stager.stage(_chunk(device, 100.0))
        assert second is not staged and second.wire[0].untyped_storage().data_ptr() == (
            staged.wire[0].untyped_storage().data_ptr()
        ), "the second chunk did not reuse the arena, so the test would be vacuous"
        torch.cuda.synchronize()
    stale = [index for index, read in enumerate(captured) if not torch.equal(read, expected)]
    assert not stale, f"senders {stale} read the arena after the next chunk was copied over it"


@gpu_test_main(exact_world_size=1, prefix="weight_sync_stage_once", partial_state=False)
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    record_check(
        checks, "staging_memory_is_flat_in_senders", test_staging_memory_does_not_grow_with_the_number_of_senders
    )
    record_check(
        checks, "next_chunk_waits_for_readers", test_a_next_chunk_waits_for_every_sender_still_reading_the_arena
    )
    return {"checks": checks}


if __name__ == "__main__":
    run()
