"""Attention-layer patching for Ulysses sequence parallelism.

Walks the model, swaps every supported attention module with the matching
:mod:`.layers` wrapper, and (optionally) validates the model first.
"""

from __future__ import annotations

import logging

import torch.distributed as dist
import torch.nn as nn

from src.distributed.context_parallel.layers.registry import CP_SUPPORTED_ATTENTION_CLASSES, WRAPPER_CLASS_MAP
from src.distributed.context_parallel.validation import (
    UlyssesConfigError,
    validate_model_for_ulysses,
)
from src.distributed.module_registry import swap_registered_modules

logger = logging.getLogger(__name__)


def patch_attention_for_ulysses(
    model: nn.Module,
    cp_group: dist.ProcessGroup,
    cp_size: int,
    validate: bool = True,
) -> int:
    """Patch the model's attention layers to use Ulysses sequence parallelism.

    Args:
        model: The model to patch.
        cp_group: NCCL process group for CP.
        cp_size: Number of ranks in the CP group.
        validate: If True, run :func:`validate_model_for_ulysses` first.

    Returns:
        The number of attention layers that were replaced (always > 0).

    Raises:
        UlyssesConfigError: If validation is enabled and the model is incompatible, or if no
            attention layer was patched (a CP run with zero wrapped layers would silently attend
            over local sequence chunks only).
    """
    if validate:
        validate_model_for_ulysses(model, cp_size)

    def build(path: str, attention: nn.Module, wrapper_cls: type) -> nn.Module:
        wrapper = wrapper_cls(attention, cp_group, cp_size)
        # The one place a fully-constructed wrapper is in hand, so the per-layer geometry is logged
        # here rather than once per family __init__.
        fields = ", ".join(f"{key}={value}" for key, value in wrapper.debug_fields().items())
        logger.debug(f"Patched {path} ({type(attention).__name__}) -> {wrapper_cls.__name__}: {fields}")
        return wrapper

    patched = len(swap_registered_modules(model, WRAPPER_CLASS_MAP, build, descend_into_match=True))

    if patched == 0:
        raise UlyssesConfigError(
            f"No attention layers were patched for Ulysses CP — the run would silently attend over "
            f"each rank's local sequence chunk only. The model has none of the supported attention "
            f"classes: {CP_SUPPORTED_ATTENTION_CLASSES}."
        )
    logger.info(f"✓ Patched {patched} attention layers for Ulysses CP")

    return patched
