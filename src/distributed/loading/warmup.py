"""The collective half of the FA4 kernel warm-up: run it on every rank, then fence.

The compile itself is rank-local and lives with the rest of the attention dispatch
(:func:`~src.models.patches.attention.warmup_fa4_kernels`); only the world-size verdict and the
barrier belong under ``src.distributed``. A leaf so both loaders — the policy dispatcher and the
frozen auxiliary loader — reach one fence instead of restating it.
"""

from __future__ import annotations

import torch
from accelerate.logging import get_logger
from transformers import PreTrainedModel

from src.distributed.runtime import barrier, get_global_world_size
from src.models.patches.attention import warmup_fa4_kernels

logger = get_logger(__name__)


def warm_attention_kernels(model: PreTrainedModel, *, dtype: torch.dtype) -> None:
    """Pre-compile this model's FA4 kernels on every rank, then barrier. No-op below world 2.

    A first-use JIT compile (~10s) on one rank lets its peers race ahead into the next collective.
    World>1 is the only verdict every rank shares, so it is the one taken before the fence: every
    other input (the resolved backend, whether an FA4 build is present) is per rank, and a rank
    returning on one of those would skip the barrier its peers already wait in.
    """
    if get_global_world_size() <= 1:
        return
    try:
        warmup_fa4_kernels(model, dtype=dtype)
    finally:
        try:
            barrier()
        except Exception as e:
            # Swallowed so a barrier failure cannot mask the warm-up's own error, and logged because
            # a failed teardown barrier IS the rank skew this exists to prevent — the next collective
            # hangs, and without this line nothing points here.
            logger.warning(
                f"FA4 warmup teardown barrier failed ({type(e).__name__}: {e}) — ranks may be skewed "
                f"entering the next collective",
                main_process_only=False,
            )
