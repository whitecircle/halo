"""Mixture-of-Kittens (``moe_backend: mok``): one megakernel per direction runs the EP routing exchange, every routed
SwiGLU expert and the weighted combine over NVLink symmetric memory, in place of DeepEP plus the grouped GEMMs. Each
forward is padded to one fixed token count, as MoK sizes its buffers once and requires the same shape on every rank."""

from __future__ import annotations

import dataclasses
import functools

import torch
import torch.distributed as dist

from halo.kernels.batched_transpose import transpose_last_two

MOK_EP_SIZES = (1, 4, 8, 16, 32, 64)  # the EP sizes MoK's schedule is built for
MOK_ALIGN = 256  # MoK's hidden and intermediate multiple
# Its route buffer, tokens x top_k x 4 bytes, must split into 2048-byte chunks for any top_k.
_TOKEN_ALIGN = 512


@functools.lru_cache(maxsize=1)
def _mok():
    """``(mok.functional, its MoKConfig)``, or None when the package (Blackwell image only) does not import."""
    try:
        from mok import functional  # noqa: PLC0415
    except Exception:
        return None
    # A schedule sized to every routed row of the group: at the default half of that, a skewed router overflows it and
    # the kernel traps, killing the CUDA context with no message.
    return functional, functional.MoKConfig(schedule_capacity_multiplier=1.0)


def mok_available() -> bool:
    return _mok() is not None


def mok_token_capacity(tokens_per_rank: int) -> int:
    """The padded per-rank token count of every MoK forward for a run declaring ``tokens_per_rank``."""
    return max(_TOKEN_ALIGN, -(-tokens_per_rank // _TOKEN_ALIGN) * _TOKEN_ALIGN)


@functools.cache
def _zero_shared_expert(intermediate: int, hidden: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """MoK always runs a shared expert beside the routed ones; zero weights add nothing, and the layer's own shared
    expert (gated on some families) stays on its usual path."""
    up = torch.zeros(intermediate, hidden, device=device, dtype=torch.bfloat16)
    return up, up, torch.zeros(hidden, intermediate, device=device, dtype=torch.bfloat16)


def _mok_weights(weights: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MoK's ``F.linear``-layout ``gate``/``up`` ``[E, I, H]`` and ``down`` ``[E, H, I]`` from the layer's
    matmul-layout ``gate_up_proj`` ``[E, H, 2I]`` (gate first) or ``gate_proj``/``up_proj`` ``[E, H, I]``."""
    *glu, down = weights
    halves = glu[0].chunk(2, dim=2) if len(glu) == 1 else glu
    return tuple(transpose_last_two(w.to(torch.bfloat16)) for w in (*halves, down))


def _field_values(record) -> list:
    return [getattr(record, field.name) for field in dataclasses.fields(record)]


def _layer_grads(fused: bool, d_gate, d_up, d_down) -> tuple[torch.Tensor, ...]:
    """The layer-layout gradients of MoK's ``[E, I, H]`` / ``[E, H, I]`` ones; a fused bank's halves are written
    straight into one ``[E, H, 2I]`` gradient."""
    if not fused:
        return tuple(transpose_last_two(g) for g in (d_gate, d_up, d_down))
    experts, intermediate, hidden = d_gate.shape
    d_gate_up = d_gate.new_empty(experts, hidden, 2 * intermediate)
    for half, grad in zip(d_gate_up.chunk(2, dim=2), (d_gate, d_up), strict=True):
        transpose_last_two(grad, out=half)
    return d_gate_up, transpose_last_two(d_down)


class _MoKFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, router_weights, top_experts, capacity, group, num_experts, *weights):
        functional, config = _mok()
        tokens, hidden = x.shape
        top_k = top_experts.shape[1]
        x_pad = x.new_zeros(capacity, hidden, dtype=torch.bfloat16)
        x_pad[:tokens] = x
        weights_pad = router_weights.new_zeros(capacity, top_k, dtype=torch.float32)
        weights_pad[:tokens] = router_weights
        # Padding rows carry zero weight, so they move no output and no gradient; spread over every expert they cost
        # each one the same.
        experts_pad = torch.arange(capacity * top_k, device=x.device).view(capacity, top_k) % num_experts
        experts_pad[:tokens] = top_experts

        workspace = functional.get_workspace(
            config, group, device=x.device, num_local_tokens=capacity, hidden_size=hidden, topk=top_k
        )
        gate, up, down = _mok_weights(weights)
        schedule = functional.build_schedule(workspace, config, experts_pad, num_local_experts=gate.shape[0])
        shared = _zero_shared_expert(gate.shape[1], hidden, x.device)
        out, context = functional.forward(config, workspace, schedule, x_pad, weights_pad, *shared, gate, up, down)

        ctx.mok = (config, workspace, group, tokens, len(weights) == 2)
        ctx.save_for_backward(x_pad, weights_pad, *_field_values(schedule), *_field_values(context), *weights)
        return out[:tokens].to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        functional, _ = _mok()
        config, workspace, group, tokens, fused = ctx.mok
        x_pad, weights_pad, *saved = ctx.saved_tensors
        schedule = functional.MoKSchedule(*saved[:4])
        context = functional.MoKForwardContext(*saved[4:11])
        weights = saved[11:]
        grad_pad = grad_output.new_zeros(x_pad.shape, dtype=torch.bfloat16)
        grad_pad[:tokens] = grad_output
        gate, up, down = _mok_weights(weights)
        shared = _zero_shared_expert(gate.shape[1], x_pad.shape[1], x_pad.device)
        d_x, d_router, d_gate, d_up, d_down, *_ = functional.backward(
            config, workspace, schedule, context, grad_pad, x_pad, weights_pad, *shared, gate, up, down
        )
        grads = _layer_grads(fused, d_gate, d_up, d_down)
        return (
            d_x[:tokens].to(grad_output.dtype),
            d_router[:tokens],
            None,
            None,
            None,
            None,
            *(g.to(w.dtype) for g, w in zip(grads, weights, strict=True)),
        )


def mok_routed_experts(
    x: torch.Tensor,
    top_experts: torch.Tensor,
    router_weights: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
    *,
    capacity: int,
    group: dist.ProcessGroup,
    num_experts: int,
) -> torch.Tensor:
    """COLLECTIVE over ``group``. ``x`` ``[T, H]``'s routed-expert output ``[T, H]``, each token's ``top_experts`` (global
    ids, ``[T, k]``) weighted by ``router_weights``; ``weights`` are this rank's expert banks in the layer's layout.
    Raises when ``T`` exceeds ``capacity``, the token count every rank's buffers were sized to."""
    if x.shape[0] > capacity:
        raise RuntimeError(
            f"moe_backend='mok' sized its buffers to {capacity} tokens per rank from the run's declared budget "
            f"(per_device_train_batch_size x max_length), but a forward carried {x.shape[0]}. Raise max_length to "
            f"the longest row the run feeds."
        )
    return _MoKFn.apply(x, router_weights, top_experts, capacity, group, num_experts, *weights)
