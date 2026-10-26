"""Toolkit optimizer names: the ``optim: <name>`` builder map and the ``OptimizerNames`` members that
let a config select them."""

from collections.abc import Callable, Sequence
from typing import Any

import torch
from transformers.training_args import OptimizerNames

from src.optimizers.flash_adamw import flash_adamw_optimizer_from_args
from src.optimizers.muon import muon_optimizer_from_args

# HF's stock AdamW names, whose foreach/fused kernels take one tensor type per group: what AdamWBF16
# auto-replaces, and what create_optimizer rejects over a mixed plain/DTensor parameter set.
STOCK_ADAMW_OPTIMIZERS: tuple[str, ...] = ("adamw_torch_fused", "adamw_torch")
# ``optim: <name>`` → (builder, description); AdamWBF16 and the ``fp32_non_ep_params`` split are
# create_optimizer branches instead.
NAMED_OPTIMIZER_BUILDERS: dict[str, tuple[Callable[[torch.nn.Module, Any, Sequence[str]], Any], str]] = {
    "muon": (muon_optimizer_from_args, "Muon (Newton-Schulz orthogonalization)"),
    "flash_adamw": (flash_adamw_optimizer_from_args, "FlashAdamW (quantized states, ~5B/param)"),
}


def optim_name(optim) -> str:
    """``args.optim`` as its plain string value (post-init it may be an ``OptimizerNames`` member,
    whose repr and identity are the member, not the value the builder map is keyed on)."""
    return getattr(optim, "value", optim) or ""


def register_custom_optimizers():
    """Add genuine ``OptimizerNames`` members for the custom optimizers (``optim: muon|flash_adamw``).

    transformers offers no extension point, and a plain string attribute is not enough: the argparse
    ``choices`` come from member iteration, so a pure-CLI ``--optim muon`` would be rejected.
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
