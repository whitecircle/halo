"""The Ulysses all-to-all: an autograd-aware swap of the sequence and head dims, one ``all_to_all_single`` each way."""

from __future__ import annotations

import torch
import torch.distributed as dist


def _require_divisible(tensor: torch.Tensor, dim: int, world_size: int) -> None:
    """Raise, on every rank ahead of the collective, unless ``world_size`` divides ``dim``: the swap's reshape would
    otherwise cut through a head or a token without an error."""
    if tensor.shape[dim] % world_size != 0:
        raise ValueError(
            f"Ulysses all-to-all: dim {dim} of size {tensor.shape[dim]} is not divisible by "
            f"context_parallel_size={world_size}."
        )


def _exchange(rows: list[torch.Tensor], process_group: dist.ProcessGroup) -> tuple[torch.Tensor, ...]:
    """COLLECTIVE over ``process_group``: ``rows`` are ``[world, ...]`` views, row ``r`` of each bound for group rank
    ``r``, sent side by side in one ``all_to_all_single``; row ``i`` of each result is what rank ``i`` sent here."""
    # A lone view that is already contiguous goes as it is, without the copy a cat makes.
    send = rows[0].contiguous() if len(rows) == 1 else torch.cat(rows, dim=-1)
    received = torch.empty_like(send)
    dist.all_to_all_single(received, send, group=process_group)
    return received.split([row.shape[-1] for row in rows], dim=-1)


def _swap_to_heads(tensors: tuple[torch.Tensor, ...], process_group: dist.ProcessGroup) -> tuple[torch.Tensor, ...]:
    """``[batch, seq/CP, heads, dim]`` each → ``[batch, seq, heads/CP, dim]``; the tensors share batch and sequence,
    each keeps its own head count and head dim."""
    world_size = dist.get_world_size(process_group)
    for tensor in tensors:
        _require_divisible(tensor, 2, world_size)
    batch, local_seq_len = tensors[0].shape[:2]
    rows = [tensor.reshape(batch, local_seq_len, world_size, -1).permute(2, 0, 1, 3) for tensor in tensors]
    # Row i of each holds group rank i's sequence chunk; in rank order they are the whole sequence.
    return tuple(
        part.transpose(0, 1).reshape(batch, world_size * local_seq_len, tensor.shape[2] // world_size, tensor.shape[3])
        for part, tensor in zip(_exchange(rows, process_group), tensors, strict=True)
    )


def _swap_to_sequence(tensors: tuple[torch.Tensor, ...], process_group: dist.ProcessGroup) -> tuple[torch.Tensor, ...]:
    """``[batch, seq, heads/CP, dim]`` each → ``[batch, seq/CP, heads, dim]``."""
    world_size = dist.get_world_size(process_group)
    for tensor in tensors:
        _require_divisible(tensor, 1, world_size)
    batch, seq_len = tensors[0].shape[:2]
    local_seq_len = seq_len // world_size
    rows = [tensor.reshape(batch, world_size, local_seq_len, -1).transpose(0, 1) for tensor in tensors]
    # Row i of each holds group rank i's heads; in rank order they are every head.
    return tuple(
        part.permute(1, 2, 0, 3).reshape(batch, local_seq_len, world_size * tensor.shape[2], tensor.shape[3])
        for part, tensor in zip(_exchange(rows, process_group), tensors, strict=True)
    )


class UlyssesAllToAll(torch.autograd.Function):
    """COLLECTIVE over ``process_group``: swap ``[batch, seq, heads, dim]`` tensors between sequence and head sharding
    in one ``all_to_all_single``; the backward swaps gradients the other way. An output can be a view of the received
    buffer (to heads at batch 1), which autograd refuses to modify in place."""

    @staticmethod
    def forward(
        ctx, process_group: dist.ProcessGroup, to_heads: bool, *tensors: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        ctx.process_group = process_group
        ctx.to_heads = to_heads
        return (_swap_to_heads if to_heads else _swap_to_sequence)(tensors, process_group)

    @staticmethod
    def backward(ctx, *grads: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        return None, None, *UlyssesAllToAll.apply(ctx.process_group, not ctx.to_heads, *grads)


def heads_to_sequence(tensor: torch.Tensor, process_group: dist.ProcessGroup) -> torch.Tensor:
    """COLLECTIVE over ``process_group``: ``[batch, seq, heads/CP, dim]`` → ``[batch, seq/CP, heads, dim]``."""
    if dist.get_world_size(process_group) == 1:
        return tensor
    (output,) = UlyssesAllToAll.apply(process_group, False, tensor)
    return output


def sequence_to_heads(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, process_group: dist.ProcessGroup
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """COLLECTIVE over ``process_group``: Q, K and V ``[batch, seq/CP, heads, dim]`` → ``[batch, seq, heads/CP,
    dim]`` in one all-to-all, each keeping its own head count (GQA) and head dim (MLA's V)."""
    if dist.get_world_size(process_group) == 1:
        return query, key, value
    return UlyssesAllToAll.apply(process_group, True, query, key, value)
