"""Wrapper for Databricks' FlashAdamW: ~57% less per-param memory via 8-bit quantized moments plus
compressed master weights (24-bit default = bf16 param + 8-bit correction; ~5 B/param), with
quant/dequant fused into the Triton update step.
"""

import logging
from collections.abc import Collection, Sequence
from typing import Any

import torch
from torch.distributed.tensor import DTensor

from src.distributed.runtime import is_global_main_process
from src.optimizers.param_groups import decay_groups

try:
    from flashoptim import FlashAdamW
except ImportError:
    FlashAdamW = None

logger = logging.getLogger(__name__)


def _warn_if_uneven_shards(params) -> None:
    """Warn at build time when optimizer-state checkpointing cannot work.

    flashoptim's ``state_dict`` raises for any DTensor param whose sharded dim does not divide the
    mesh, so the trainer's all-or-nothing shard save skips optimizer state on every checkpoint and a
    resume warm-restarts. Training itself is unaffected: the step unwraps to the local shard.
    """
    uneven = [
        (name, tuple(p.shape))
        for name, p in params
        if isinstance(p, DTensor)
        and any(
            hasattr(pl, "dim") and p.shape[pl.dim] % p.device_mesh.size(dim) != 0
            for dim, pl in enumerate(p.placements)
        )
    ]
    if uneven and is_global_main_process():
        # Shapes are rank-uniform, so warning from one rank covers the job.
        name, shape = uneven[0]
        logger.warning(
            f"FlashAdamW cannot checkpoint optimizer state for this model: {len(uneven)} parameter(s) "
            f"are unevenly sharded (e.g. '{name}' with shape {shape}) and flashoptim's state_dict "
            f"refuses them, so every checkpoint will skip optimizer shards and any resume will "
            f"warm-restart the optimizer. Pick dims divisible by the shard world, or another optim."
        )


def _bump_version_counters_after_step(optimizer: torch.optim.Optimizer, args, kwargs) -> None:
    """Step post-hook: advance ``param._version``, which FlashAdamW's fused Triton update does not.

    Its raw-pointer stores are invisible to ATen, so the counter would stay at 0 for the whole run
    and ``cached_fake_quant``, keyed on ``(weight._version, fmt, axis)``, would serve the step-0
    quantization forever under any ``lowp_precision`` other than bf16. Only params carrying a grad
    are bumped: the step wrote no others, and a spurious bump evicts a still-valid cache entry.
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                torch.autograd.graph.increment_version(param)


def create_flash_adamw_optimizer(
    model: torch.nn.Module,
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    master_weight_bits: int | None = 24,
    decay_parameters: Collection[str] | None = None,
):
    """Create a FlashAdamW optimizer with weight-decay group splitting.

    ``master_weight_bits`` 24 or 32, or ``None`` to drop the master-weight correction (states stay
    8-bit quantized); ``decay_parameters=None`` applies ``weight_decay`` to every param.
    """
    if FlashAdamW is None:
        raise ImportError(
            "FlashAdamW requires the 'flashoptim' package. "
            "Install with: pip install flashoptim  (or the 'flash-optimizers' uv extra)"
        )

    param_groups = decay_groups(model.named_parameters(), decay_parameters, weight_decay)

    _warn_if_uneven_shards((n, p) for n, p in model.named_parameters() if p.requires_grad)

    optimizer = FlashAdamW(
        param_groups,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        master_weight_bits=master_weight_bits,
    )
    optimizer.register_step_post_hook(_bump_version_counters_after_step)
    return optimizer


def build_flash_adamw_optimizer(model: torch.nn.Module, args: Any, decay_parameters: Sequence[str]):
    """FlashAdamW with 8-bit quantized states + 24-bit master weights (~57% optimizer-memory cut)."""
    optimizer = create_flash_adamw_optimizer(
        model,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
        decay_parameters=decay_parameters,
    )
    if is_global_main_process():
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"FlashAdamW optimizer: {total / 1e6:.1f}M params (quantized states, 24-bit master weights)")
    return optimizer
