"""Pack weights into ~1GB buffers before NCCL broadcast to amortize per-tensor overhead; double-buffered with async CUDA streams. Vendored from vLLM v0.18.0, Apache-2.0."""

from collections.abc import Iterator

import torch

from src.weight_sync.transport.bounded_waits import BROADCAST_DRAIN_TIMEOUT_S, bounded_stream_sync
from src.weight_sync.transport.pynccl import PyNcclCommunicator
from src.weight_sync.transport.send_buffers import allocate_send_buffer, hold_for_stream

PACKED_BUFFER_BYTES = 1024 * 1024 * 1024  # 1GB
PACKED_NUM_BUFFERS = 2


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
    """Pack tensors into ~buffer_size_bytes buffers and broadcast each from the trainer as one NCCL op.

    ``buffer_size_bytes`` and :data:`PACKED_NUM_BUFFERS` are wire parameters: the update header
    announces both, and the consumer must unpack with the same pair.

    Each buffer is packed and broadcast on one of ``streams`` (one per buffer) but drawn from the
    caller's current stream's pool (:func:`allocate_send_buffer`), so a closed client's buffers are
    reused by the next one and by the training step. The tensors must already be contiguous on the
    streams' device, landed by the caller on its own stream: a copy made here would be allocated on a
    pack stream, in that stream's own pool. Each is held for the pack stream that reads it
    (:func:`hold_for_stream`).
    """
    if len(streams) != PACKED_NUM_BUFFERS:
        raise ValueError(f"streams must match PACKED_NUM_BUFFERS ({PACKED_NUM_BUFFERS}), got {len(streams)}")
    # The buffers' pool. Every pack stream waits on it: a block it hands back may last have been used
    # there, and the caller may have written the tensors there.
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
