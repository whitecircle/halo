"""Allocator bookkeeping both weight-sync clients share for the CUDA streams they send from.

The caching allocator files a block under the stream that allocated it, reuses it only for that
stream, and knows nothing of another stream's pending reads unless they are recorded with it.
"""

import torch


def allocate_send_buffer(
    nbytes: int,
    device: torch.device | None,
    *,
    pool_stream: torch.cuda.Stream,
    send_stream: torch.cuda.Stream,
) -> torch.Tensor:
    """A ``uint8`` staging buffer drawn from ``pool_stream``'s allocator pool and held for ``send_stream``.

    Allocated on a client's own send stream, a buffer stays reserved once the client closes, and every
    reconnect strands another. Drawn from the caller's stream, it returns to the pool the next client
    and the training step allocate from. ``record_stream`` keeps the block from being handed out again
    until the work ``send_stream`` has queued by the time it is freed retires.

    ``send_stream`` must already wait on ``pool_stream``: the block may last have been used there.
    """
    with torch.cuda.stream(pool_stream):
        buffer = torch.empty(nbytes, dtype=torch.uint8, device=device)
    buffer.record_stream(send_stream)
    return buffer


def hold_for_stream(tensor: torch.Tensor, stream: torch.cuda.Stream) -> None:
    """Keep a CUDA ``tensor`` the caller owns from being reused until ``stream``'s reads of it retire.

    A send stream may read a chunk's snapshots after the engine has replied, and the caller drops them
    on that reply: unrecorded, a block is handed to the next snapshot allocated on the caller's stream,
    which overwrites it before the queued read runs. Host tensors need no record: the pinned allocator
    tracks its own copies, and a pageable one is copied before ``copy_`` returns.
    """
    if tensor.is_cuda:
        tensor.record_stream(stream)
