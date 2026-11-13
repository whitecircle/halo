"""Mixture-of-Kittens (``moe_backend: mok``): one megakernel per direction runs the EP routing exchange, every routed
SwiGLU expert and the weighted combine over NVLink symmetric memory, in place of DeepEP plus the grouped GEMMs, in bf16
or MXFP8 (forward, dgrad and wgrad). Each forward is padded to one fixed token count, as MoK sizes its buffers once and
requires the same shape on every rank."""

from __future__ import annotations

import dataclasses
import functools

import torch
import torch.distributed as dist

from halo.kernels.batched_transpose import transpose_last_two
from halo.kernels.lowp.quantization import WeightVersionCache

MOK_EP_SIZES = (1, 4, 8, 16, 32, 64)  # the EP sizes MoK's schedule is built for
MOK_ALIGN = 256  # MoK's hidden and intermediate multiple
# Its route buffer, tokens x top_k x 4 bytes, must split into 2048-byte chunks for any top_k.
_TOKEN_ALIGN = 512

_mxfp8_cache = WeightVersionCache()  # expert bank -> its MXFP8 layouts, re-quantized once per optimizer step


@functools.lru_cache(maxsize=1)
def _mok():
    """``(mok.functional, its MoKConfig, mok.ops.mxfp8_quantize)``, or None when the package (Blackwell image only)
    does not import."""
    try:
        from mok import functional  # noqa: PLC0415
        from mok.ops import mxfp8_quantize  # noqa: PLC0415
    except Exception:
        return None
    # A schedule sized to every routed row of the group: at the default half of that, a skewed router overflows it and
    # the kernel traps, killing the CUDA context with no message.
    return functional, functional.MoKConfig(schedule_capacity_multiplier=1.0), mxfp8_quantize


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


def _mxfp8_weights(weights: tuple[torch.Tensor, ...], cacheable: bool) -> tuple[tuple[torch.Tensor, ...], ...]:
    """MoK's gate, up and down banks in MXFP8, each ``(data, scales, data_t, scales_t)``: the forward reads the first
    pair, the backward gate/up all four and down the transposed pair. Cached per optimizer step on the first bank's
    version unless ``cacheable`` is False, as FSDP2's pinned version counter would serve the step-0 cast forever."""
    key = weights[0]
    fingerprint = tuple(w._version for w in weights)
    cacheable = cacheable and key.is_leaf
    if cacheable and (cached := _mxfp8_cache.get(key, fingerprint)) is not None:
        return cached
    quantize = _mok()[2]
    banks = tuple(quantize(w, True, True) for w in _mok_weights(weights))
    if cacheable:
        _mxfp8_cache.put(key, fingerprint, banks)
    return banks


def _flatten(record) -> tuple[list[torch.Tensor], list[int]]:
    """``record``'s tensors in field order and each field's arity (0 for a bare tensor), for ``save_for_backward``."""
    tensors, arity = [], []
    for field in dataclasses.fields(record):
        value = getattr(record, field.name)
        tensors += value if isinstance(value, tuple) else [value]
        arity.append(len(value) if isinstance(value, tuple) else 0)
    return tensors, arity


def _unflatten(cls, tensors, arity: list[int]):
    values, start = [], 0
    for count in arity:
        values.append(tuple(tensors[start : start + count]) if count else tensors[start])
        start += count or 1
    return cls(*values)


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
    def forward(ctx, x, router_weights, top_experts, capacity, group, num_experts, mxfp8, weight_cacheable, *weights):
        functional, config, _ = _mok()
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
        banks = _mxfp8_weights(weights, weight_cacheable) if mxfp8 else _mok_weights(weights)
        experts, intermediate = banks[0][0].shape[:2] if mxfp8 else banks[0].shape[:2]
        schedule = functional.build_schedule(workspace, config, experts_pad, num_local_experts=experts)
        shared = _zero_shared_expert(intermediate, hidden, x.device)
        forward_banks = tuple(bank[:2] for bank in banks) if mxfp8 else banks
        out, context = functional.forward(config, workspace, schedule, x_pad, weights_pad, *shared, *forward_banks)

        schedule_tensors, schedule_arity = _flatten(schedule)
        context_tensors, context_arity = _flatten(context)
        # The bf16 backward re-transposes the banks rather than hold a second copy until then; the MXFP8 one keeps the
        # cast, which costs more to redo than its fp8 bytes.
        saved_banks = [t for bank in banks for t in bank] if mxfp8 else []
        ctx.mok = (config, workspace, group, tokens, len(weights) == 2, mxfp8, schedule_arity, context_arity)
        ctx.counts = (len(schedule_tensors), len(context_tensors), len(saved_banks))
        ctx.save_for_backward(x_pad, weights_pad, *schedule_tensors, *context_tensors, *saved_banks, *weights)
        return out[:tokens].to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        functional, _, _ = _mok()
        config, workspace, group, tokens, fused, mxfp8, schedule_arity, context_arity = ctx.mok
        n_schedule, n_context, n_banks = ctx.counts
        x_pad, weights_pad, *saved = ctx.saved_tensors
        schedule = _unflatten(functional.MoKSchedule, saved[:n_schedule], schedule_arity)
        context = _unflatten(functional.MoKForwardContext, saved[n_schedule : n_schedule + n_context], context_arity)
        weights = saved[n_schedule + n_context + n_banks :]
        if mxfp8:
            flat = saved[n_schedule + n_context : n_schedule + n_context + n_banks]
            gate, up, down = (tuple(flat[i : i + 4]) for i in range(0, n_banks, 4))
            down = down[2:]
            intermediate = gate[0].shape[1]
        else:
            gate, up, down = _mok_weights(weights)
            intermediate = gate.shape[1]
        grad_pad = grad_output.new_zeros(x_pad.shape, dtype=torch.bfloat16)
        grad_pad[:tokens] = grad_output
        shared = _zero_shared_expert(intermediate, x_pad.shape[1], x_pad.device)
        d_x, d_router, d_gate, d_up, d_down, *_ = functional.backward(
            config, workspace, schedule, context, grad_pad, x_pad, weights_pad, *shared, gate, up, down
        )
        grads = _layer_grads(fused, d_gate, d_up, d_down)
        return (
            d_x[:tokens].to(grad_output.dtype),
            d_router[:tokens],
            *(None,) * 6,
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
    mxfp8: bool = False,
    weight_cacheable: bool = True,
) -> torch.Tensor:
    """COLLECTIVE over ``group``. ``x`` ``[T, H]``'s routed-expert output ``[T, H]``, each token's ``top_experts`` (global
    ids, ``[T, k]``) weighted by ``router_weights``; ``weights`` are this rank's expert banks in the layer's layout.
    ``mxfp8`` runs the forward, dgrad and wgrad in MXFP8. Raises when ``T`` exceeds ``capacity``, the token count every
    rank's buffers were sized to."""
    if x.shape[0] > capacity:
        raise RuntimeError(
            f"moe_backend='mok' sized its buffers to {capacity} tokens per rank from the run's declared budget "
            f"(per_device_train_batch_size x max_length), but a forward carried {x.shape[0]}. Raise max_length to "
            f"the longest row the run feeds."
        )
    return _MoKFn.apply(
        x, router_weights, top_experts, capacity, group, num_experts, mxfp8, weight_cacheable, *weights
    )
