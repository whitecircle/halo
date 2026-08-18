"""Liger-backed cross-entropy for the model families whose loss goes through
``transformers.loss.loss_utils``, plus the scoped patch that installs it there.

Every toolkit applier offers it (:mod:`src.kernels.liger.families`), so the patch is shared by the
whole roster rather than owned by any one family."""

from __future__ import annotations

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from liger_kernel.transformers import LigerCrossEntropyLoss
from transformers.loss import loss_utils

# Captured before any patch so the CPU / unsupported-kwarg fallback below cannot recurse.
_TORCH_CROSS_ENTROPY = F.cross_entropy


@functools.lru_cache(maxsize=1)
def _default_ce():
    return LigerCrossEntropyLoss(ignore_index=-100, reduction="mean")


def liger_cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    size_average: bool | None = None,
    ignore_index: int = -100,
    reduce: bool | None = None,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Drop-in replacement for ``F.cross_entropy`` backed by Liger.

    CPU tensors and deprecated ``size_average``/``reduce`` stay on torch (Liger is Triton/CUDA-only, and
    HF's loss path is also taken on CPU — dataset probes, CPU-only eval); the common
    ``(no weight, -100, 'mean')`` path reuses one cached loss module.
    """
    if not input.is_cuda or size_average is not None or reduce is not None:
        return _TORCH_CROSS_ENTROPY(
            input,
            target,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
    if weight is not None or (ignore_index, reduction, label_smoothing) != (-100, "mean", 0.0):
        return LigerCrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )(input, target)
    return _default_ce()(input, target)


class _AttributeOverride:
    """Read-through view of a module with selected attributes replaced.

    Lets a patch reach one importing module's view of a shared module instead of mutating the shared
    module itself. Writes go to the wrapped module, so a later writer — upstream Liger's own appliers
    reach ``torch.nn`` through ``from transformers.loss.loss_utils import nn``, which is this object once
    patched — mutates what it meant to and cannot silently shadow an override for the rest of the process.
    """

    def __init__(self, wrapped, **overrides):
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_overrides", overrides)

    def __getattr__(self, name: str):
        overrides = self.__dict__["_overrides"]
        return overrides[name] if name in overrides else getattr(self.__dict__["_wrapped"], name)

    def __setattr__(self, name: str, value) -> None:
        setattr(self.__dict__["_wrapped"], name, value)


# ``loss_utils`` reaches cross-entropy only as ``nn.functional.cross_entropy`` (its single call site,
# in ``fixed_cross_entropy``), so overriding its ``nn`` global redirects exactly that call. Assigning to
# ``torch.nn.functional.cross_entropy`` instead would swap the kernel under EVERY caller in the process
# — including CP's own fp32 loss and any 3-D or soft-target call, which this replacement cannot serve.
_LOSS_UTILS_NN = _AttributeOverride(
    nn, functional=_AttributeOverride(nn.functional, cross_entropy=liger_cross_entropy)
)


def patch_loss_utils_cross_entropy() -> None:
    """Route ``transformers.loss.loss_utils``' cross-entropy through Liger. Idempotent."""
    loss_utils.nn = _LOSS_UTILS_NN
