"""Parameter grouping for optimizer construction: the weight-decay split the custom-optimizer
builders pass to their optimizer, and the tensor-type split a stock optimizer needs under FSDP2 + EP.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Iterable, Sequence
from typing import Any

import torch
from torch.distributed.tensor import DTensor
from torch.nn import Parameter

from src.distributed.runtime import is_global_main_process

logger = logging.getLogger(__name__)


def decay_groups(
    named_parameters: Iterable[tuple[str, Parameter]],
    decay_parameters: Collection[str] | None,
    weight_decay: float,
    *,
    keep_empty: bool = False,
) -> list[dict[str, Any]]:
    """``[decay group, no-decay group]`` over the trainable params, each in iteration order.

    ``decay_parameters=None`` decays everything. ``keep_empty`` emits a group that came out empty:
    the group count defines the index space of a saved optimizer state and of a scheduler's per-group
    updates, so a builder whose optimizer always carries two groups must keep emitting two.
    """
    decay_names = None if decay_parameters is None else set(decay_parameters)
    decay: list[Parameter] = []
    no_decay: list[Parameter] = []
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        (decay if decay_names is None or name in decay_names else no_decay).append(param)

    return [
        {"params": params, "weight_decay": group_decay}
        for params, group_decay in ((decay, weight_decay), (no_decay, 0.0))
        if params or keep_empty
    ]


def build_tensor_type_grouped_optimizer(
    model: torch.nn.Module,
    args: Any,
    decay_parameters: Sequence[str],
    get_optimizer_cls_and_kwargs: Callable[[Any, torch.nn.Module], tuple[type, dict]],
):
    """Optimizer with param groups split by (weight decay, dtype, tensor type), single-tensor mode.

    FSDP2 parameters are DTensors while FSDP-ignored EP parameters are plain tensors. Foreach/fused
    optimizers cannot mix the two in one update, so split the groups and disable those paths. The
    dtype axis additionally handles ``fp32_non_ep_params``.
    """
    decay_names = set(decay_parameters)
    groups: dict[tuple, list] = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        wd = args.weight_decay if n in decay_names else 0.0
        groups.setdefault((wd, p.dtype, isinstance(p, DTensor)), []).append(p)

    grouped = [
        {"params": params, "weight_decay": wd, "foreach": False, "fused": False}
        for (wd, _dtype, _is_dt), params in groups.items()
    ]

    optimizer_cls, optimizer_kwargs = get_optimizer_cls_and_kwargs(args, model)
    optimizer_kwargs.pop("foreach", None)
    optimizer_kwargs.pop("fused", None)
    # Factory-style optimizers (GaLore/LOMO/layerwise) couple kwargs to the model or a fixed param
    # list; forwarding those alongside the grouped params here would bind the params twice.
    coupled = {"params", "model", "optimizer_dict"} & set(optimizer_kwargs)
    if coupled:
        raise ValueError(
            f"optim={getattr(args, 'optim', None)!r} builds with model-coupled kwargs "
            f"{sorted(coupled)}, which the tensor-type-grouped split cannot forward — pick a "
            f"plain optimizer for fp32_non_ep_params."
        )
    optimizer = optimizer_cls(grouped, **optimizer_kwargs)

    if is_global_main_process():
        for (wd, dtype, is_dt), params in groups.items():
            count = sum(p.numel() for p in params)
            logger.info(
                f"  Optimizer group: {dtype} {'DTensor' if is_dt else 'Tensor'}, wd={wd}, {count / 1e6:.1f}M params"
            )
    return optimizer
