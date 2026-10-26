"""Pack weights into ~1GB buffers before an NCCL broadcast, double-buffered on CUDA streams (vendored from vLLM
v0.18.0, Apache-2.0), and the send-buffer allocator bookkeeping the SGLang client's sends share: the caching
allocator reuses a block only for the stream that allocated it and knows another stream's reads only once recorded."""

from collections.abc import Iterator

import torch

from halo.weight_sync.transport.bounded_waits import BROADCAST_DRAIN_TIMEOUT_S, bounded_stream_sync
from halo.weight_sync.transport.pynccl import PyNcclCommunicator

PACKED_BUFFER_BYTES = 1024 * 1024 * 1024  # 1GB
PACKED_NUM_BUFFERS = 2


def allocate_send_buffer(
    nbytes: int,
    device: torch.device | None,
    *,
    pool_stream: torch.cuda.Stream,
    send_stream: torch.cuda.Stream,
) -> torch.Tensor:
    """A ``uint8`` send buffer drawn from ``pool_stream``'s pool and held for ``send_stream``.

    Drawn from the caller's stream, not the client's own, it returns to the pool the next client and the
    training step use instead of stranding one per reconnect. ``send_stream`` must already wait on
    ``pool_stream``: the block may last have been used there.
    """
    with torch.cuda.stream(pool_stream):
        buffer = torch.empty(nbytes, dtype=torch.uint8, device=device)
    buffer.record_stream(send_stream)
    return buffer


def hold_for_stream(tensor: torch.Tensor, stream: torch.cuda.Stream) -> None:
    """Keep a CUDA ``tensor`` from reuse until ``stream``'s queued reads of it retire: the caller drops a chunk's
    snapshots on the engine's reply, possibly before the send stream read them."""
    if tensor.is_cuda:
        tensor.record_stream(stream)


def _pack(
    tensors: list[torch.Tensor], nbytes: int, *, pool_stream: torch.cuda.Stream, send_stream: torch.cuda.Stream
) -> torch.Tensor:
    """Concatenate the flat ``uint8`` ``tensors`` into one buffer; called under ``send_stream``."""
    buffer = allocate_send_buffer(nbytes, send_stream.device, pool_stream=pool_stream, send_stream=send_stream)
    return torch.cat(tensors, dim=0, out=buffer)


def packed_broadcast_producer(
    iterator: Iterator[tuple[str, torch.Tensor]],
    group: PyNcclCommunicator,
    streams: list[torch.cuda.Stream],
    buffer_size_bytes: int = PACKED_BUFFER_BYTES,
) -> None:
    """Pack tensors into ~``buffer_size_bytes`` buffers and broadcast each as one NCCL op, one of ``streams`` per
    buffer. ``buffer_size_bytes`` and :data:`PACKED_NUM_BUFFERS` are wire parameters the consumer unpacks with.
    The tensors must already be on the streams' device: a copy made here would land in a pack stream's pool.
    """
    if len(streams) != PACKED_NUM_BUFFERS:
        raise ValueError(f"streams must match PACKED_NUM_BUFFERS ({PACKED_NUM_BUFFERS}), got {len(streams)}")
    # The buffers' pool, which every pack stream waits on: its blocks and the tensors were last written there.
    pool_stream = torch.cuda.current_stream(streams[0].device)
    for stream in streams:
        stream.wait_stream(pool_stream)
    buffer_idx = 0

    packing_tensor_list: list[list[torch.Tensor]] = [[] for _ in range(PACKED_NUM_BUFFERS)]
    packing_tensor_sizes: list[int] = [0 for _ in range(PACKED_NUM_BUFFERS)]
    packed_tensors: list[torch.Tensor] = [
        torch.empty(0, dtype=torch.uint8, device="cuda") for _ in range(PACKED_NUM_BUFFERS)
    ]

    while True:
        # Bounded sync: a server death mid-sync raises here instead of hanging the broadcast.
        bounded_stream_sync(
            streams[buffer_idx],
            timeout_s=BROADCAST_DRAIN_TIMEOUT_S,
            what="packed weight broadcast",
            abort=group.abort,
        )
        with torch.cuda.stream(streams[buffer_idx]):
            try:
                packing_tensor_list[buffer_idx] = []
                packing_tensor_sizes[buffer_idx] = 0
                while True:
                    tensor = next(iterator)[1].contiguous().view(torch.uint8).view(-1)
                    hold_for_stream(tensor, streams[buffer_idx])
                    packing_tensor_list[buffer_idx].append(tensor)
                    packing_tensor_sizes[buffer_idx] += tensor.numel()
                    if packing_tensor_sizes[buffer_idx] > buffer_size_bytes:
                        break
                packed_tensors[buffer_idx] = _pack(
                    packing_tensor_list[buffer_idx],
                    packing_tensor_sizes[buffer_idx],
                    pool_stream=pool_stream,
                    send_stream=streams[buffer_idx],
                )
                group.broadcast(packed_tensors[buffer_idx])
                buffer_idx = (buffer_idx + 1) % PACKED_NUM_BUFFERS
            except StopIteration:
                if len(packing_tensor_list[buffer_idx]) > 0:
                    packed_tensors[buffer_idx] = _pack(
                        packing_tensor_list[buffer_idx],
                        packing_tensor_sizes[buffer_idx],
                        pool_stream=pool_stream,
                        send_stream=streams[buffer_idx],
                    )
                    group.broadcast(packed_tensors[buffer_idx])
                break
