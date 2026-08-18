"""Pack weights into ~1GB buffers before NCCL broadcast to amortize per-tensor overhead; double-buffered with async CUDA streams. Vendored from vLLM v0.18.0, Apache-2.0."""

from collections.abc import Callable, Iterator
from typing import Any

import torch

from src.distributed.nccl.transport.pynccl import bounded_stream_sync

DEFAULT_PACKED_BUFFER_SIZE_BYTES = 1024 * 1024 * 1024  # 1GB
DEFAULT_PACKED_NUM_BUFFERS = 2
# Deadline for draining one buffer's in-flight broadcast; sized for a full ~1GB buffer over a slow
# link, not for a wedged peer (that is what the deadline exists to surface).
_BROADCAST_SYNC_TIMEOUT_S = 600.0


def packed_broadcast_producer(
    iterator: Iterator[tuple[str, torch.Tensor]],
    group: Any,
    src: int = 0,
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor] | None = None,
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    streams: list[torch.cuda.Stream] | None = None,
) -> None:
    """Pack tensors into ~buffer_size_bytes buffers and broadcast each as one NCCL op.

    src=0 is the trainer. ``buffer_size_bytes`` and :data:`DEFAULT_PACKED_NUM_BUFFERS` are wire
    parameters — the update header announces both, and the consumer must unpack with the same pair.

    A caller invoking this repeatedly MUST pass its own persistent ``streams`` (one per buffer):
    every pack allocation (the H2D uploads and the ~1GB cat) lands on these streams, and the caching
    allocator's pools are per-stream — per-call streams strand each sync's whole transient footprint
    in a pool no later stream can reuse, ratcheting reserved memory by ~the model size per sync.
    Persistent streams give every sync the identical allocation sequence, so it reuses one pool.
    """
    if post_iter_func is None:

        def post_iter_func(x):
            return x[1]

    target_packed_tensor_size = buffer_size_bytes

    if streams is None:
        streams = [torch.cuda.Stream() for _ in range(DEFAULT_PACKED_NUM_BUFFERS)]
    elif len(streams) != DEFAULT_PACKED_NUM_BUFFERS:
        raise ValueError(
            f"streams must match DEFAULT_PACKED_NUM_BUFFERS ({DEFAULT_PACKED_NUM_BUFFERS}), got {len(streams)}"
        )
    buffer_idx = 0

    packing_tensor_list: list[list[torch.Tensor]] = [[] for _ in range(DEFAULT_PACKED_NUM_BUFFERS)]
    packing_tensor_sizes: list[int] = [0 for _ in range(DEFAULT_PACKED_NUM_BUFFERS)]
    packed_tensors: list[torch.Tensor] = [
        torch.empty(0, dtype=torch.uint8, device="cuda") for _ in range(DEFAULT_PACKED_NUM_BUFFERS)
    ]

    while True:
        # Bounded sync: a server death mid-sync raises here instead of hanging the broadcast.
        try:
            bounded_stream_sync(
                streams[buffer_idx], timeout_s=_BROADCAST_SYNC_TIMEOUT_S, what="packed weight broadcast"
            )
        except RuntimeError:
            group.abort()
            raise
        with torch.cuda.stream(streams[buffer_idx]):
            try:
                packing_tensor_list[buffer_idx] = []
                packing_tensor_sizes[buffer_idx] = 0
                while True:
                    tensor = post_iter_func(next(iterator)).contiguous().view(torch.uint8).view(-1)
                    packing_tensor_list[buffer_idx].append(tensor)
                    packing_tensor_sizes[buffer_idx] += tensor.numel()
                    if packing_tensor_sizes[buffer_idx] > target_packed_tensor_size:
                        break
                packed_tensors[buffer_idx] = torch.cat(packing_tensor_list[buffer_idx], dim=0)
                group.broadcast(packed_tensors[buffer_idx], src=src)
                buffer_idx = (buffer_idx + 1) % DEFAULT_PACKED_NUM_BUFFERS
            except StopIteration:
                if len(packing_tensor_list[buffer_idx]) > 0:
                    packed_tensors[buffer_idx] = torch.cat(packing_tensor_list[buffer_idx], dim=0)
                    group.broadcast(packed_tensors[buffer_idx], src=src)
                break
