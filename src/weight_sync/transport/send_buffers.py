"""Allocator bookkeeping for the CUDA streams the weight-sync clients send from: the caching allocator reuses a
block only for the stream that allocated it and knows another stream's reads only once recorded."""

import torch


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
