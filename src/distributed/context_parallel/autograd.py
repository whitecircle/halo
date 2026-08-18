"""Communication primitives used by Ulysses sequence parallelism: an autograd-aware
all-to-all that swaps the sequence and head dimensions, plus a fused Q+KV variant and
a RoPE (cos, sin) all-gather for the legacy path.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


class UlyssesAllToAll(torch.autograd.Function):
    """Autograd-aware all-to-all swapping sequence and head dimensions.

    Forward: ``[batch, seq/CP, heads, dim]`` → ``[batch, seq, heads/CP, dim]``.
    Backward: reverses the swap.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        process_group: dist.ProcessGroup,
        scatter_dim: int,
        gather_dim: int,
    ) -> torch.Tensor:
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim

        world_size = dist.get_world_size(process_group)
        if world_size == 1:
            return input

        if not input.is_contiguous():
            input = input.contiguous()

        if input.shape[scatter_dim] % world_size != 0:
            raise ValueError(
                f"Ulysses all-to-all: dim {scatter_dim} of size {input.shape[scatter_dim]} is not "
                f"divisible by cp_size={world_size} — the ranged narrow below would silently drop "
                f"the tail elements."
            )
        chunk_size = input.shape[scatter_dim] // world_size
        input_list = [input.narrow(scatter_dim, i * chunk_size, chunk_size).contiguous() for i in range(world_size)]

        output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
        dist.all_to_all(output_list, input_list, group=process_group)

        return torch.cat(output_list, dim=gather_dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        grad_input = UlyssesAllToAll.apply(
            grad_output,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return grad_input, None, None, None


def ulysses_all_to_all(
    tensor: torch.Tensor,
    process_group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
) -> torch.Tensor:
    """Swap sequence and head parallelism via Ulysses all-to-all.

    ``tensor`` is ``[batch, seq, heads, dim]``; scatter the head dim and gather the
    sequence dim (defaults).
    """
    return UlyssesAllToAll.apply(tensor, process_group, scatter_dim, gather_dim)


def ulysses_all_to_all_fused_kv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    process_group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """All-to-all for Q, K, V with K/V fusion (3 calls → 2).

    GQA K/V share head count, so they fuse on the last dim, halving one all-to-all.
    """
    world_size = dist.get_world_size(process_group)
    if world_size == 1:
        return query, key, value

    kv_fused = torch.cat([key, value], dim=-1)

    query_out = UlyssesAllToAll.apply(query, process_group, scatter_dim, gather_dim)
    kv_out = UlyssesAllToAll.apply(kv_fused, process_group, scatter_dim, gather_dim)

    # The all-to-all scatters/gathers dims 2 and 1, never the fused last dim, so the split is exact.
    head_dim = key.shape[-1]
    return query_out, kv_out[..., :head_dim], kv_out[..., head_dim:]


def gather_pos_embeddings(
    cos: torch.Tensor,
    sin: torch.Tensor,
    cp_group: dist.ProcessGroup,
    cp_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-gather position embeddings across a CP group.

    Used by the legacy path (RoPE after the all-to-all, so each rank needs cos/sin
    for the *full* sequence). Cos/sin fused on the last dim into one all-gather.

    Args:
        cos, sin: Per-rank slices ``[batch, chunk_size, dim]``.

    Returns:
        Full-sequence ``(cos, sin)`` of shape ``[batch, full_seq, dim]``.
    """
    if cp_size == 1:
        return cos, sin

    cos_sin_fused = torch.cat([cos, sin], dim=-1).contiguous()

    batch_size = cos_sin_fused.shape[0]
    chunk_size = cos_sin_fused.shape[1]
    fused_dim = cos_sin_fused.shape[2]
    full_seq_len = chunk_size * cp_size

    flat_input = cos_sin_fused.view(-1)
    full_flat = torch.empty(
        flat_input.numel() * cp_size,
        device=flat_input.device,
        dtype=flat_input.dtype,
    )
    dist.all_gather_into_tensor(full_flat, flat_input, group=cp_group)

    full_fused = full_flat.view(cp_size, batch_size, chunk_size, fused_dim)
    full_fused = full_fused.permute(1, 0, 2, 3).reshape(batch_size, full_seq_len, fused_dim)

    dim = cos.shape[-1]
    full_cos = full_fused[..., :dim].contiguous()
    full_sin = full_fused[..., dim:].contiguous()

    return full_cos, full_sin
