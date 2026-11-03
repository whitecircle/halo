"""LoRA injected in place into an embedding backbone: the fold its saves write and the resume adapter it restores."""

import itertools
import logging
import os
from collections.abc import Iterator
from functools import partial

import torch
import torch.nn as nn
from peft import LoraConfig
from peft.tuners.tuners_utils import BaseTunerLayer

from halo.checkpoint.adapters import (
    adapter_layout_mismatch,
    missing_resume_adapter_reason,
    resume_adapter_on_own_weights_reason,
)
from halo.checkpoint.format import RESUME_ADAPTER_DIR
from halo.checkpoint.parallel.adapters import read_adapter_across_ranks
from halo.checkpoint.parallel.resume_weights import built_from_checkpoint_across_ranks
from halo.checkpoint.parallel.writing import chunked_saveable_tensors
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.models.param_roles import persistent_buffers
from halo.models.peft_adapters import lora_fold_targets, lora_folded_data, tuner_adapter_param_ids
from halo.models.unwrapping import strip_base_layer_segment
from halo.runtime.consensus import reject_across_ranks
from halo.runtime.dtensor import copy_full_tensor
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)


def has_injected_lora(backbone: nn.Module) -> bool:
    """Structural: a remote-code backbone's own ``lora_``-named parameters (jina's LoRA parametrizations) are base
    weights."""
    return any(isinstance(module, BaseTunerLayer) for module in backbone.modules())


def require_injected_lora_config(model: nn.Module) -> LoraConfig:
    for module in model.modules():
        configs = getattr(module, "peft_config", None)
        if isinstance(configs, dict) and isinstance(configs.get("default"), LoraConfig):
            return configs["default"]
    raise RuntimeError(
        "The backbone carries injected LoRA layers but no module holds the LoraConfig "
        "inject_adapter_in_model stamps, so the scaling the adapters train at cannot be recorded."
    )


def folded_backbone_chunks(backbone: nn.Module, *, retain: bool) -> Iterator[dict[str, torch.Tensor]]:
    """COLLECTIVE. Injected LoRA folded in out of place, under the base model's names, each target inside its chunk's
    gather in the chunk order every rank shares (a DTensor collective under FSDP2)."""
    adapters = tuner_adapter_param_ids(backbone)
    # The parameters themselves, not their data: the fold finds its targets by identity.
    params = ((name, param) for name, param in backbone.named_parameters() if id(param) not in adapters)
    base = itertools.chain(params, persistent_buffers(backbone))
    return chunked_saveable_tensors(
        backbone,
        retain=retain,
        items=((strip_base_layer_segment(name), tensor) for name, tensor in base),
        fold=partial(lora_folded_data, folds=lora_fold_targets(backbone)),
    )


def trainable_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    """Every parameter with ``requires_grad``, which a fold cannot give back. Structural, so rank-uniform in name and
    order."""
    return {name: param.data for name, param in model.named_parameters() if param.requires_grad}


def _resume_adapter_mismatch(saved: dict[str, torch.Tensor], live: dict[str, torch.Tensor], path: str) -> str | None:
    """Names and full shapes must match one for one: a tensor left out would resume from initialization and ``copy_``
    would broadcast a saved rank-1 adapter into a wider one."""
    mismatch = adapter_layout_mismatch(saved, {name: tuple(tensor.shape) for name, tensor in live.items()})
    return mismatch and (
        f"the resume adapter at {path} does not match this run's trainable tensors (was the run relaunched with "
        f"other lora_target_modules or lora_r?): {mismatch}"
    )


def restore_injected_lora(model: nn.Module, backbone: nn.Module, checkpoint: str) -> None:
    """COLLECTIVE. Bit-exact. Refuses, rank-uniformly, a backbone built from the checkpoint itself (its fold already
    holds the delta), a missing file, another LoRA scaling, and tensors differing in name or shape."""
    if built_from_checkpoint_across_ranks(backbone, checkpoint):
        raise ValueError(resume_adapter_on_own_weights_reason(checkpoint))
    saved, path = read_adapter_across_ranks(
        os.path.join(checkpoint, RESUME_ADAPTER_DIR),
        checkpoint=checkpoint,
        what="Resume adapter",
        attention=require_injected_lora_config(model).to_dict(),
    )
    if path is None:
        raise RuntimeError(missing_resume_adapter_reason(checkpoint))
    # Best-model loads follow an eval forward that left the FSDP2 tree unsharded.
    reshard_fsdp2_modules(model)
    live = trainable_tensors(model)
    reject_across_ranks(_resume_adapter_mismatch(saved, live, path), "Injected-LoRA resume", ValueError)
    with torch.no_grad():
        for name in sorted(live):
            copy_full_tensor(live[name], saved[name])
    del saved
    if is_global_main_process():
        logger.info(f"Restored {len(live)} injected-LoRA tensors from {path}")
