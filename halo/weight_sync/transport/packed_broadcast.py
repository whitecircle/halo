"""Pack weights into ~1GB buffers and broadcast each as one NCCL op, alternating CUDA streams (the wire format is
vendored from vLLM v0.18.0, Apache-2.0), and the send-buffer bookkeeping the staging shares: the caching allocator
reuses a block only for the stream that allocated it and knows another stream's reads only once recorded."""

from collections.abc import Sequence

import torch

from halo.weight_sync.transport.bounded_waits import CHUNK_SYNC_TIMEOUT_S, bounded_stream_sync
from halo.weight_sync.transport.pynccl import PyNcclCommunicator

PACKED_BUFFER_BYTES = 1024 * 1024 * 1024
PACKED_NUM_BUFFERS = 2


def allocate_send_buffer(
    nbytes: int,
    device: torch.device | None,
    *,
    pool_stream: torch.cuda.Stream,
    send_stream: torch.cuda.Stream,
) -> torch.Tensor:
    """A ``uint8`` send buffer drawn from ``pool_stream``'s pool and held for ``send_stream``, which must already wait
    on ``pool_stream``. Drawn from the caller's stream, it returns to the pool the next client and the training step
    use instead of stranding one per reconnect."""
    with torch.cuda.stream(pool_stream):
        buffer = torch.empty(nbytes, dtype=torch.uint8, device=device)
    buffer.record_stream(send_stream)
    return buffer


def hold_for_stream(tensor: torch.Tensor, stream: torch.cuda.Stream) -> None:
    """Keep a CUDA ``tensor`` from reuse until ``stream``'s queued reads retire, as the caller drops a chunk's
    snapshots on the engine's reply, possibly before the send stream read them."""
    if tensor.is_cuda:
        tensor.record_stream(stream)


def pack_tensors(tensors: Sequence[torch.Tensor], buffer_size_bytes: int = PACKED_BUFFER_BYTES) -> list[torch.Tensor]:
    """``tensors`` concatenated as bytes on the current stream, into buffers that each close once past
    ``buffer_size_bytes``. ``buffer_size_bytes`` and :data:`PACKED_NUM_BUFFERS` are wire parameters the consumer
    unpacks with."""
    groups: list[list[torch.Tensor]] = [[]]
    size = 0
    for tensor in tensors:
        flat = tensor.contiguous().view(torch.uint8).view(-1)
        groups[-1].append(flat)
        size += flat.numel()
        if size > buffer_size_bytes:
            groups.append([])
            size = 0
    return [torch.cat(group, dim=0) for group in groups if group]


def broadcast_packed(
    buffers: Sequence[torch.Tensor],
    group: PyNcclCommunicator,
    streams: list[torch.cuda.Stream],
    *,
    ready: torch.cuda.Event | None,
) -> None:
    """Broadcast each of ``buffers`` as one NCCL op, on one of ``streams`` per buffer in turn, once ``ready`` (recorded
    by the packing) has passed. The buffers are only read, so any number of communicators may send the same ones."""
    if len(streams) != PACKED_NUM_BUFFERS:
        raise ValueError(f"streams must match PACKED_NUM_BUFFERS ({PACKED_NUM_BUFFERS}), got {len(streams)}")
    for index, buffer in enumerate(buffers):
        stream = streams[index % PACKED_NUM_BUFFERS]
        # A server death mid-sync raises here instead of hanging the broadcast.
        bounded_stream_sync(stream, timeout_s=CHUNK_SYNC_TIMEOUT_S, what="packed weight broadcast", abort=group.abort)
        with torch.cuda.stream(stream):
            if ready is not None:
                stream.wait_event(ready)
            hold_for_stream(buffer, stream)
            group.broadcast(buffer)
