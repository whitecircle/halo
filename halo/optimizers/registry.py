"""The ``optim: <name>`` builder map and the ``OptimizerNames`` members that let a config select them."""

from collections.abc import Callable, Sequence
from typing import Any

import torch
from transformers.training_args import OptimizerNames

from halo.optimizers.flash_adamw import flash_adamw_optimizer_from_args
from halo.optimizers.muon import muon_optimizer_from_args

# Their foreach/fused kernels take one tensor type per group; AdamWBF16 auto-replaces them, and create_optimizer
# rejects them over a mixed plain/DTensor parameter set.
STOCK_ADAMW_OPTIMIZERS: tuple[str, ...] = ("adamw_torch_fused", "adamw_torch")
# AdamWBF16 and the ``fp32_non_ep_params`` split are create_optimizer branches instead.
NAMED_OPTIMIZER_BUILDERS: dict[str, tuple[Callable[[torch.nn.Module, Any, Sequence[str]], Any], str]] = {
    "muon": (muon_optimizer_from_args, "Muon (Newton-Schulz orthogonalization)"),
    "flash_adamw": (flash_adamw_optimizer_from_args, "FlashAdamW (quantized states, ~5B/param)"),
}


def optim_name(optim) -> str:
    """Post-init ``args.optim`` may be an ``OptimizerNames`` member; the builder map is keyed on its value."""
    return getattr(optim, "value", optim) or ""


def register_custom_optimizers():
    """Genuine members, as transformers offers no extension point and argparse ``choices`` come from member iteration,
    so a plain string attribute would reject a pure-CLI ``--optim muon``."""
    for name in NAMED_OPTIMIZER_BUILDERS:
        if name in OptimizerNames._value2member_map_:
            continue
        member = str.__new__(OptimizerNames, name)
        member._name_ = name.upper()
        member._value_ = name
        # type.__setattr__ bypasses EnumType's reassignment guard; the maps make the member reachable.
        type.__setattr__(OptimizerNames, member._name_, member)
        OptimizerNames._member_map_[member._name_] = member
        OptimizerNames._value2member_map_[name] = member
        OptimizerNames._member_names_.append(member._name_)
