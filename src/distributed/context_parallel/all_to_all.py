"""The Ulysses all-to-all: an autograd-aware swap of the sequence and head dims, in both directions."""

from __future__ import annotations

import torch
import torch.distributed as dist


class UlyssesAllToAll(torch.autograd.Function):
    """COLLECTIVE over ``process_group``: scatter ``scatter_dim`` and gather ``gather_dim``; the backward swaps
    them back."""

    @staticmethod
    def forward(
        ctx, input: torch.Tensor, process_group: dist.ProcessGroup, scatter_dim: int, gather_dim: int
    ) -> torch.Tensor:
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim

        world_size = dist.get_world_size(process_group)
        if world_size == 1:
            return input

        if input.shape[scatter_dim] % world_size != 0:
            raise ValueError(
                f"Ulysses all-to-all: dim {scatter_dim} of size {input.shape[scatter_dim]} is not "
                f"divisible by context_parallel_size={world_size} — the ranged narrow below would "
                f"silently drop the tail elements."
            )
        chunk_size = input.shape[scatter_dim] // world_size
        input_list = [input.narrow(scatter_dim, i * chunk_size, chunk_size).contiguous() for i in range(world_size)]

        output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
        dist.all_to_all(output_list, input_list, group=process_group)

        return torch.cat(output_list, dim=gather_dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        return UlyssesAllToAll.apply(grad_output, ctx.process_group, ctx.gather_dim, ctx.scatter_dim), None, None, None


def heads_to_sequence(tensor: torch.Tensor, process_group: dist.ProcessGroup) -> torch.Tensor:
    """COLLECTIVE over ``process_group``: ``[batch, seq, heads/CP, dim]`` → ``[batch, seq/CP, heads, dim]``, the
    inverse of :func:`sequence_to_heads`."""
    return UlyssesAllToAll.apply(tensor, process_group, 1, 2)


def sequence_to_heads(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, process_group: dist.ProcessGroup
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """COLLECTIVE over ``process_group``: Q, K and V ``[batch, seq/CP, heads, dim]`` → ``[batch, seq, heads/CP,
    dim]``, K and V fused on the last dim (they share a head count) so two all-to-alls carry all three."""
    world_size = dist.get_world_size(process_group)
    if world_size == 1:
        return query, key, value

    kv_fused = torch.cat([key, value], dim=-1)

    query_out = UlyssesAllToAll.apply(query, process_group, 2, 1)
    kv_out = UlyssesAllToAll.apply(kv_fused, process_group, 2, 1)

    # The all-to-all scatters/gathers dims 2 and 1, never the fused last dim, so the split is exact.
    head_dim = key.shape[-1]
    return query_out, kv_out[..., :head_dim], kv_out[..., head_dim:]
