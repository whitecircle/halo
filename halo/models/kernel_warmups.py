"""The load-time kernel warm-ups a family's patches register, run once by the post-load finalization so no kernel
compiles mid-run."""

from collections.abc import Callable

from torch import nn

KernelWarmup = Callable[[nn.Module], None]

_warmups: list[KernelWarmup] = []


def register_kernel_warmup(warmup: KernelWarmup) -> None:
    """Idempotent. ``warmup(model)`` is rank-local, takes no collective and compiles only what ``model``'s own
    modules run."""
    if warmup not in _warmups:
        _warmups.append(warmup)


def warm_registered_kernels(model: nn.Module) -> None:
    for warmup in _warmups:
        warmup(model)
