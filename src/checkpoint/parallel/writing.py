"""The collective half of a checkpoint write."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from src.checkpoint.writing import save_dtype_caster
from src.distributed.tensor_parallel.plain_params import iter_hand_sliced_full
from src.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)


def gather_hand_sliced_params(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    retain: bool = True,
    keep_live_dtype: bool = False,
) -> None:
    """COLLECTIVE over the TP group: write ``model``'s hand-sliced TP params into ``state_dict`` as full CPU
    tensors. ``retain=False`` joins every gather but writes nothing (save-rank gating without N× host copies).

    Cast through the model-derived :func:`save_dtype_caster`, as the callers do for the params they resolve
    themselves: a blanket cast would push a norm / balancing / fp32-pinned tensor that happens to be
    hand-sliced down to bf16 while its gathered twin kept the trained dtype.
    """
    cast = save_dtype_caster(model, keep_live_dtype=keep_live_dtype)
    gathered_count = 0
    for name, full_tensor in iter_hand_sliced_full(model):
        if retain:
            state_dict[name] = cast(name, full_tensor.cpu())
        gathered_count += 1

    if gathered_count > 0 and is_global_main_process():
        logger.info(f"All-gathered {gathered_count} TP-sharded non-DTensor params")
