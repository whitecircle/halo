"""Mixture-of-Kittens (``moe_backend: mok``): one megakernel per direction runs the EP routing exchange, every routed
SwiGLU expert and the weighted combine over NVLink symmetric memory, in place of DeepEP plus the grouped GEMMs, in bf16
or MXFP8 (forward, dgrad and wgrad). Each forward is padded to one fixed token count, as MoK sizes its buffers once and
requires the same shape on every rank."""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable

from halo.kernels.batched_transpose import transpose_last_two
from halo.kernels.lowp.quantization import WeightVersionCache

if TYPE_CHECKING:
    from halo.distributed.expert_parallel.config import EPConfig

MOK_EP_SIZES = (1, 4, 8, 16, 32, 64)  # the EP sizes MoK's schedule is built for
MOK_ALIGN = 256  # MoK's hidden and intermediate multiple
# Its route buffer, tokens x top_k x 4 bytes, must split into 2048-byte chunks for any top_k.
_TOKEN_ALIGN = 512

_mxfp8_cache = WeightVersionCache()  # expert bank -> its MXFP8 casts, re-quantized once per optimizer step


# Comm SMs each way by expert intermediate size: a token's dispatch and combine scale with the hidden size and its
# expert compute with hidden x intermediate, so narrow experts are communication-bound and want more of the SMs.
_COMM_SMS_BY_INTERMEDIATE = ((1024, 48), (2048, 42))
_COMM_SMS_WIDE, _MAX_MINIBATCH = 36, 8192
_EXPERT_PADDING = 256  # MoK's schedule rounds each local expert's rows up to this (csrc/scheduler.cuh)
_BUILT_FOR = (10, 3)  # the image builds mok at its default arch, SM103, whose code an SM100 GPU cannot run
# MoK's ring of routed rows, in a rank's even share of the group's: every routed activation is allocated at the
# ring's size and saved for the backward, and a rank routed more than the ring holds replays it.
_RING_SHARES = 2


@functools.lru_cache(maxsize=1)
def _mok():
    """``mok.functional``, or None when the package is not installed. One that is installed and does not import
    raises: an ABI or CUDA mismatch must not read as an absent package."""
    try:
        from mok import functional  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        if exc.name != "mok":
            raise
        return None
    return functional


def _comm_sms(intermediate: int) -> int:
    return next((sms for bound, sms in _COMM_SMS_BY_INTERMEDIATE if intermediate <= bound), _COMM_SMS_WIDE)


def _schedule_capacity_multiplier(capacity: int, top_k: int, ep_size: int, local_experts: int) -> float:
    """MoK's ``schedule_capacity_multiplier`` for a schedule no routing can overflow. MoK sizes a rank's schedule to
    ``max(2, ceil(ep_size * multiplier))`` times ``capacity * top_k`` rows and traps in the kernel, killing the CUDA
    context with no message, when the rank's rows, each local expert's rounded up to 256, exceed it. A token sends
    one rank at most ``min(top_k, local_experts)`` rows, so the most a rank schedules is every such row of the group
    plus the rounding; MoK's default 0.5 holds half the group's rows, and 1.0 leaves the rounding out."""
    rows = ep_size * capacity * min(top_k, local_experts)
    worst = (rows + (_EXPERT_PADDING - 1) * local_experts) // _EXPERT_PADDING * _EXPERT_PADDING
    factor = -(-worst // (capacity * top_k))
    return (factor - 0.5) / ep_size  # the half keeps MoK's ceil at ``factor`` through the float division


@functools.cache
def _mok_config(capacity: int, intermediate: int, top_k: int, ep_size: int, local_experts: int):
    """The MoKConfig of a group of ``ep_size`` ranks padding every forward to ``capacity`` tokens each, for
    ``local_experts`` experts of width ``intermediate`` per rank."""
    minibatch = min(capacity, _MAX_MINIBATCH)
    sms = _comm_sms(intermediate)
    return _mok().MoKConfig(
        fwd_num_comm_sms=sms,
        bwd_num_comm_sms=sms,
        minibatch_size=minibatch,
        macrobatch_size=-(-_RING_SHARES * capacity * top_k // minibatch) * minibatch,
        schedule_capacity_multiplier=_schedule_capacity_multiplier(capacity, top_k, ep_size, local_experts),
    )


def is_mok_available() -> bool:
    return _mok() is not None


def require_mok():
    """``mok.functional``; raises where the package is not installed or this GPU is not the one it is built for."""
    functional = _mok()
    if functional is None:
        raise RuntimeError(
            "moe_backend='mok' needs the mok package, built into the Blackwell image only (its kernels are SM103). "
            "Run on that image, or set moe_backend='deepep'."
        )
    if torch.cuda.is_available() and torch.cuda.get_device_capability() != _BUILT_FOR:
        major, minor = torch.cuda.get_device_capability()
        raise RuntimeError(
            f"moe_backend='mok' runs on SM103 (B300) only: the image builds mok for it, and this GPU is "
            f"SM{major}{minor}. Set moe_backend='deepep'."
        )
    return functional


def mok_token_capacity(tokens_per_rank: int) -> int:
    """The padded per-rank token count of every MoK forward for a run declaring ``tokens_per_rank``."""
    return max(_TOKEN_ALIGN, -(-tokens_per_rank // _TOKEN_ALIGN) * _TOKEN_ALIGN)


@functools.cache
def _zero_shared_expert(intermediate: int, hidden: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """MoK always runs a shared expert beside the routed ones; zero weights add nothing, and the layer's own shared
    expert (gated on some families) stays on its usual path."""
    up = torch.zeros(intermediate, hidden, device=device, dtype=torch.bfloat16)
    return up, up, torch.zeros(hidden, intermediate, device=device, dtype=torch.bfloat16)


class _Banks(NamedTuple):
    """MoK's gate, up and down operands of a layer's banks, as its forward and its backward take them."""

    forward: tuple
    backward: tuple


def _layer_banks(weights: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``gate``, ``up`` ``[E, H, I]`` and ``down`` ``[E, I, H]`` from the layer's ``gate_up_proj`` ``[E, H, 2I]`` (gate
    first) or ``gate_proj``/``up_proj``."""
    *glu, down = weights
    return (*(glu[0].chunk(2, dim=2) if len(glu) == 1 else glu), down)


def _bf16_banks(weights: tuple[torch.Tensor, ...]) -> _Banks:
    """MoK's ``F.linear`` layout, ``gate``/``up`` ``[E, I, H]`` and ``down`` ``[E, H, I]``. Transposed on each call:
    the backward redoes it rather than hold a second copy of every bank until then."""
    banks = tuple(transpose_last_two(w.to(torch.bfloat16)) for w in _layer_banks(weights))
    return _Banks(banks, banks)


def _mxfp8_quantize():
    from mok.ops import mxfp8_quantize  # noqa: PLC0415

    return mxfp8_quantize


def _mxfp8_layer_cast(w: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """MoK's ``(data, scales, data_t, scales_t)`` of a layer-layout bank: cast in the layer's layout, whose transposed
    cast is MoK's normal one and back, so no bank is transposed first and the bits are those of casting MoK's layout."""
    data, scales, data_t, scales_t = _mxfp8_quantize()(w.to(torch.bfloat16).contiguous(), True, True)
    return data_t, scales_t, data, scales


def _mxfp8_banks(weights: tuple[torch.Tensor, ...]) -> _Banks:
    """Each bank cast to MXFP8 in both layouts, ``(data, scales, data_t, scales_t)``: the forward reads the first
    pair, the backward gate/up all four and down the transposed pair. Cached per optimizer step, keyed on the first
    bank and every bank's identity and version."""
    fingerprint = tuple((id(w), w._version) for w in weights)
    casts = _mxfp8_cache.get(weights[0], fingerprint)
    if casts is None:
        casts = tuple(_mxfp8_layer_cast(w) for w in _layer_banks(weights))
        _mxfp8_cache.put(weights[0], fingerprint, casts)
    gate, up, down = casts
    return _Banks(tuple(cast[:2] for cast in casts), (gate, up, down[2:]))


def _flatten(record) -> tuple[list[torch.Tensor], list[int]]:
    """``record``'s tensors in field order and each field's arity (0 for a bare tensor), for ``save_for_backward``:
    MoK's MXFP8 context holds tuples of tensors."""
    tensors, arity = [], []
    for field in dataclasses.fields(record):
        value = getattr(record, field.name)
        tensors += value if isinstance(value, tuple) else [value]
        arity.append(len(value) if isinstance(value, tuple) else 0)
    return tensors, arity


def _unflatten(cls, tensors: list[torch.Tensor], arity: list[int]):
    """The record :func:`_flatten` took apart, and the tensors past it."""
    values, start = [], 0
    for count in arity:
        values.append(tuple(tensors[start : start + count]) if count else tensors[start])
        start += count or 1
    return cls(*values), tensors[start:]


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
    def forward(ctx, x, router_weights, top_experts, capacity, group, num_experts, mxfp8, *weights):
        functional = _mok()
        tokens, hidden = x.shape
        top_k = top_experts.shape[1]
        local_experts, intermediate = weights[-1].shape[:2]  # down [E, I, H]
        config = _mok_config(capacity, intermediate, top_k, dist.get_world_size(group), local_experts)
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
        banks = (_mxfp8_banks if mxfp8 else _bf16_banks)(weights)
        schedule = functional.build_schedule(workspace, config, experts_pad, num_local_experts=local_experts)
        shared = _zero_shared_expert(intermediate, hidden, x.device)
        out, context = functional.forward(config, workspace, schedule, x_pad, weights_pad, *shared, *banks.forward)

        schedule_fields, context_fields = _flatten(schedule), _flatten(context)
        # The MXFP8 backward reads the forward's casts from here, which the cache holds for the step anyway: under
        # saved-tensor hooks (a non-reentrant checkpoint) its saved weights are other objects and would miss the cache.
        mxfp8_backward = banks.backward if mxfp8 else None
        ctx.mok = (config, workspace, group, tokens, mxfp8_backward, schedule_fields[1], context_fields[1])
        ctx.save_for_backward(x_pad, weights_pad, *schedule_fields[0], *context_fields[0], *weights)
        return out[:tokens].to(x.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        functional = _mok()
        config, workspace, group, tokens, mxfp8_backward, schedule_arity, context_arity = ctx.mok
        x_pad, weights_pad, *saved = ctx.saved_tensors
        schedule, saved = _unflatten(functional.MoKSchedule, saved, schedule_arity)
        context, weights = _unflatten(functional.MoKForwardContext, saved, context_arity)
        grad_pad = grad_output.new_zeros(x_pad.shape, dtype=torch.bfloat16)
        grad_pad[:tokens] = grad_output
        banks = mxfp8_backward or _bf16_banks(weights).backward
        shared = _zero_shared_expert(weights[-1].shape[1], x_pad.shape[1], x_pad.device)
        d_x, d_router, d_gate, d_up, d_down, *_ = functional.backward(
            config, workspace, schedule, context, grad_pad, x_pad, weights_pad, *shared, *banks
        )
        grads = _layer_grads(len(weights) == 2, d_gate, d_up, d_down)
        return (
            d_x[:tokens].to(grad_output.dtype),
            d_router[:tokens],
            *(None,) * 5,
            *(g.to(w.dtype) for g, w in zip(grads, weights, strict=True)),
        )


class MoKExperts:
    """A layer's routed experts on MoK, built in place of its DeepEP dispatcher."""

    def __init__(self, ep_config: EPConfig, num_experts: int, hidden_dim: int):
        self._ep_config, self._num_experts, self._hidden_dim = ep_config, num_experts, hidden_dim

    def __call__(
        self,
        x: torch.Tensor,
        top_experts: torch.Tensor,
        router_weights: torch.Tensor,
        banks: tuple[torch.Tensor, ...],
        mxfp8: bool = False,
    ) -> torch.Tensor:
        """COLLECTIVE over the EP group. ``x`` ``[T, H]``'s routed-expert output ``[T, H]``, each token's
        ``top_experts`` (global ids, ``[T, k]``) weighted by ``router_weights``; ``banks`` are this rank's expert
        weights in the layer's layout. ``mxfp8`` runs the forward, dgrad and wgrad in MXFP8. Raises when ``T`` exceeds
        the token count every rank's buffers were sized to."""
        capacity = self._ep_config.mok_token_capacity
        if x.shape[0] > capacity:
            raise RuntimeError(
                f"moe_backend='mok' sized its buffers to {capacity} tokens per rank from the run's declared budget "
                f"(the larger of the train and eval batch's rows x max_length), but a forward carried {x.shape[0]}. "
                f"Raise max_length to the longest row the run feeds."
            )
        group = self._ep_config.dispatch_ep_group
        return _MoKFn.apply(x, router_weights, top_experts, capacity, group, self._num_experts, mxfp8, *banks)

    def warm(self, banks: tuple[torch.Tensor, ...], top_k: int, mxfp8: bool = False) -> None:
        """COLLECTIVE over the EP group. One forward and backward over a zero token, so the bank transposes compile and
        MoK's kernels load before the first step's span."""
        device = banks[0].device
        # Grad on, inference mode off and identity saved-tensor hooks: a first forward under no_grad, inference mode
        # or a non-reentrant checkpoint still has to launch the backward's kernels.
        identity_hooks = torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t)
        with torch.inference_mode(False), torch.enable_grad(), identity_hooks:
            x = torch.zeros(1, self._hidden_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
            experts = torch.arange(top_k, device=device).view(1, top_k) % self._num_experts
            out = self(x, experts, torch.zeros(1, top_k, device=device), banks, mxfp8)
            torch.autograd.grad(out.sum(), x)

    def destroy(self) -> None:
        """COLLECTIVE. Release every MoK workspace of the process; a no-op once released."""
        _mok().clear_workspace_cache()
