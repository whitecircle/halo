"""Shared pieces of the distributed gradient-clip paths: argument normalization, the enable
predicate, and the clip coefficient.

Imports nothing from the trainer package, so both the mixin's EP/TP clips and the pipeline mixin's
whole-chain clip can use it. Each path reduces its own global norm (the collectives differ per
topology); the local accumulation under them is shared.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Guards the divide against a ~0 global norm (an all-zero gradient step), matching torch's own clip.
_CLIP_NORM_EPS = 1e-6


def clip_parameters(parameters) -> list[nn.Parameter]:
    """A clip call's ``parameters`` argument as a list, whether it arrived as a module or an iterable.

    HF/accelerate pass either; a generator is exhausted by the first pass over it, so it is
    materialized once here.
    """
    if hasattr(parameters, "parameters"):
        parameters = parameters.parameters()
    return list(parameters)


def trainable_clip_params(parameters) -> list[nn.Parameter]:
    """The trainable parameters of a clip call.

    An empty result is returned rather than short-circuited on: the distributed clips issue
    collectives, so the caller has to answer with a zero norm rather than skip.
    """
    return [p for p in clip_parameters(parameters) if p.requires_grad]


def clipping_enabled(max_norm) -> bool:
    """HF's convention: ``max_grad_norm <= 0`` (or unset) disables clipping.

    Scaling by ``0 / norm`` would otherwise zero every gradient.
    """
    return bool(max_norm) and max_norm > 0


def local_grad_norm_sq(shards: list[torch.Tensor], *, device) -> torch.Tensor:
    """Sum of the squared L2 norms of ``shards`` — this rank's contribution, 0-dim fp32 on ``device``.

    ``dtype=torch.float32`` accumulates each per-tensor norm in fp32 rather than in the shard's own
    bf16; a trailing ``.float()`` would only widen an already-rounded value. An empty list returns a
    zero rather than skipping: every caller reduces the result across ranks, and the collective must
    stay rank-uniform.
    """
    if not shards:
        return torch.zeros((), device=device, dtype=torch.float32)
    norms = torch._foreach_norm(shards, dtype=torch.float32)
    # ``to``: the sum lands on the shards' device, and the caller's reduce is issued on ``device``.
    return torch.linalg.vector_norm(torch.stack(norms)).pow(2).to(device)


def bucketed_grad_norm_sq(shards: dict[str, list[torch.Tensor]], *, device) -> dict[str, torch.Tensor]:
    """:func:`local_grad_norm_sq` per bucket, for the paths whose buckets reduce over different groups.

    Every declared bucket comes back, empty ones as zero, so the caller's reduces over them stay
    rank-uniform rather than gated on which parameters happened to carry a gradient.
    """
    return {name: local_grad_norm_sq(bucket, device=device) for name, bucket in shards.items()}


def scale_shards_to_max_norm_(
    shards: list[torch.Tensor], max_norm: float | torch.Tensor, total_norm: torch.Tensor
) -> None:
    """Scale local grad shards in place by the clamped clip coefficient.

    Device-resident and applied unconditionally so no path pays a host sync to decide whether to
    clip; ``max_norm`` is therefore taken as given rather than coerced with ``float()``, which would
    sync on a tensor threshold. ``nan_to_num`` runs before the clamp, since a NaN total norm would
    otherwise multiply into every gradient where 1.0 leaves them untouched. Callers pass local shards
    (``to_local()`` on DTensors) because ``_foreach_mul_`` refuses a mixed DTensor/plain list.
    """
    clip_coef = torch.nan_to_num(max_norm / (total_norm + _CLIP_NORM_EPS), nan=1.0).clamp(max=1.0)
    torch._foreach_mul_(shards, clip_coef)
