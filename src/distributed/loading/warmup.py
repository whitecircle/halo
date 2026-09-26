"""Collective half of the attention kernel warm-up: run the rank-local compiles, then barrier.

The compiles themselves are :func:`~src.models.patches.attention.warmup_fa4_kernels` (FA4) and
:func:`~src.models.patches.flex_sliding_attention.warmup_flex_sliding_kernels` (FlexAttention on sliding
layers); this module adds the world-size check and the fence around them.
"""

from __future__ import annotations

import torch
from accelerate.logging import get_logger
from transformers import PreTrainedModel

from src.distributed.runtime import barrier, get_global_world_size
from src.models.patches.attention import warmup_fa4_kernels
from src.models.patches.flex_sliding_attention import warmup_flex_sliding_kernels

logger = get_logger(__name__)


def warm_attention_kernels(model: PreTrainedModel, *, dtype: torch.dtype) -> None:
    """Pre-compile this model's FA4 or FlexAttention kernels on every rank, then barrier. No-op below world 2.

    A first-use JIT compile (~10s) on one rank lets its peers race ahead into the next collective.
    World>1 is the only condition every rank evaluates identically, so it is the one checked before
    the fence: the resolved backend and the presence of an FA4 build are per-rank, and returning on
    either would skip a barrier peers are already waiting in.
    """
    if get_global_world_size() <= 1:
        return
    try:
        warmup_fa4_kernels(model, dtype=dtype)
        warmup_flex_sliding_kernels(model, dtype=dtype)
    finally:
        try:
            barrier()
        except Exception as e:
            # Swallowed so a barrier failure cannot mask the warm-up's own error, but logged: a
            # failed teardown barrier is itself the rank skew this fence exists to prevent.
            logger.warning(
                f"Attention warmup teardown barrier failed ({type(e).__name__}: {e}) — ranks may be skewed "
                f"entering the next collective",
                main_process_only=False,
            )
