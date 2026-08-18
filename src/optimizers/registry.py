"""Toolkit optimizer names: the ``optim: <name>`` → builder map, and the HuggingFace
``OptimizerNames`` extension that lets a config select them.
"""

from collections.abc import Callable, Sequence
from typing import Any

import torch
from transformers.training_args import OptimizerNames

from src.optimizers.flash_adamw import build_flash_adamw_optimizer
from src.optimizers.muon import build_muon_optimizer

# HF's stock AdamW names: the ones whose foreach/fused kernels take ONE tensor type per group, so
# they are both what AdamWBF16 auto-replaces and what create_optimizer refuses over a mixed
# plain/DTensor parameter set. One home for the pair — two call sites read it.
STOCK_ADAMW_OPTIMIZERS: tuple[str, ...] = ("adamw_torch_fused", "adamw_torch")

# Custom optimizers selected by ``optim: <name>`` in YAML — single source of the name → (builder,
# description) mapping (``bf16`` and the ``fp32_non_ep_params`` split are branches in create_optimizer).
NAMED_OPTIMIZER_BUILDERS: dict[str, tuple[Callable[[torch.nn.Module, Any, Sequence[str]], Any], str]] = {
    "muon": (build_muon_optimizer, "Muon (Newton-Schulz orthogonalization)"),
    "flash_adamw": (build_flash_adamw_optimizer, "FlashAdamW (quantized states, ~5B/param)"),
}


def optim_name(optim) -> str:
    """``args.optim`` as its plain string value (post-init it may be an ``OptimizerNames`` member,
    whose repr and identity are the member, not the value the builder map is keyed on)."""
    return getattr(optim, "value", optim) or ""


def register_custom_optimizers():
    """Add genuine ``OptimizerNames`` members for the custom optimizers (``optim: muon|flash_adamw``).

    ``extend_enum``-style surgery, since transformers offers no extension point for optim *names*.
    A planted plain string is not enough: it makes ``OptimizerNames(name)`` return a ``str``, leaves
    ``OptimizerNames.MUON`` raising, and keeps the name out of the argparse ``choices`` built from
    member iteration — rejecting a pure-CLI ``--optim muon``.
    """
    for name in NAMED_OPTIMIZER_BUILDERS:
        if name in OptimizerNames._value2member_map_:
            continue
        member = str.__new__(OptimizerNames, name)
        member._name_ = name.upper()
        member._value_ = name
        # type.__setattr__ bypasses EnumType's reassignment guard for the class attribute; the
        # bookkeeping maps make the member reachable via call, index, iteration, and argparse.
        type.__setattr__(OptimizerNames, member._name_, member)
        OptimizerNames._member_map_[member._name_] = member
        OptimizerNames._value2member_map_[name] = member
        OptimizerNames._member_names_.append(member._name_)
