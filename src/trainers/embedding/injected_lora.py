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

from src.checkpoint.adapters import missing_resume_adapter_reason, resume_adapter_on_own_weights_reason
from src.checkpoint.format import RESUME_ADAPTER_DIR
from src.checkpoint.parallel.adapters import read_adapter_across_ranks
from src.checkpoint.parallel.resume_weights import built_from_checkpoint_across_ranks
from src.checkpoint.parallel.writing import chunked_saveable_tensors
from src.distributed.fsdp import reshard_fsdp2_modules
from src.log import KEY_PREVIEW_COUNT
from src.models.param_roles import persistent_buffers
from src.models.peft_adapters import lora_fold_targets, lora_folded_data, tuner_adapter_param_ids
from src.models.unwrapping import strip_base_layer_segment
from src.runtime.consensus import reject_across_ranks
from src.runtime.dtensor import copy_full_tensor
from src.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)


def has_injected_lora(backbone: nn.Module) -> bool:
    """True if ``inject_adapter_in_model`` added tuner layers to ``backbone`` (in place, so not a PeftModel).
    Structural: a remote-code backbone's own ``lora_``-named parameters (jina's LoRA parametrizations) are
    base weights, not injected adapters."""
    return any(isinstance(module, BaseTunerLayer) for module in backbone.modules())


def require_injected_lora_config(model: nn.Module) -> LoraConfig:
    """The ``LoraConfig`` ``inject_adapter_in_model`` stamped on the module of ``model`` it injected into."""
    for module in model.modules():
        configs = getattr(module, "peft_config", None)
        if isinstance(configs, dict) and isinstance(configs.get("default"), LoraConfig):
            return configs["default"]
    raise RuntimeError(
        "The backbone carries injected LoRA layers but no module holds the LoraConfig "
        "inject_adapter_in_model stamps, so the scaling the adapters train at cannot be recorded."
    )


def folded_backbone_chunks(backbone: nn.Module, *, retain: bool) -> Iterator[dict[str, torch.Tensor]]:
    """COLLECTIVE: :func:`chunked_saveable_tensors` over the backbone with its injected LoRA folded in out of
    place, under the base model's names: the tensors the tuner layers own are dropped by identity (a backbone's
    own ``lora_``-named parameters stay). Each target folds inside its chunk's gather, in the chunk order every
    rank shares (a DTensor collective under FSDP2)."""
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
    """The tensors an injected-LoRA run trains, which its fold cannot give back: every parameter with
    ``requires_grad``, the same set the optimizer steps. Structural, so rank-uniform in name and order."""
    return {name: param.data for name, param in model.named_parameters() if param.requires_grad}


def _resume_adapter_mismatch(saved: dict[str, torch.Tensor], live: dict[str, torch.Tensor], path: str) -> str | None:
    """Why the resume adapter at ``path`` cannot restore ``live`` exactly, or None.

    Names and full shapes must match one for one: a tensor left out would resume from
    initialization, and ``copy_`` would broadcast a saved rank-1 adapter into a wider one.
    """
    missing = sorted(live.keys() - saved.keys())
    unexpected = sorted(saved.keys() - live.keys())
    reshaped = [
        f"{name}: saved {tuple(saved[name].shape)}, live {tuple(live[name].shape)}"
        for name in sorted(live.keys() & saved.keys())
        if saved[name].shape != live[name].shape
    ]
    if not (missing or unexpected or reshaped):
        return None
    return (
        f"the resume adapter at {path} does not match this run's trainable tensors (was the run "
        f"relaunched with other lora_target_modules or lora_r?): missing {missing[:KEY_PREVIEW_COUNT]}, "
        f"unexpected {unexpected[:KEY_PREVIEW_COUNT]}, reshaped {reshaped[:KEY_PREVIEW_COUNT]}"
    )


def restore_injected_lora(model: nn.Module, backbone: nn.Module, checkpoint: str) -> None:
    """COLLECTIVE. Copy ``checkpoint``'s resume adapter into ``model``'s live trainable tensors, bit-exact.

    Refuses, rank-uniformly, a backbone built from the checkpoint itself (its fold already holds the
    delta), a missing adapter file, another LoRA scaling, and tensors that differ in name or shape.
    FSDP2 DTensors restore from mesh rank 0's copy through ``distribute_tensor``, in sorted key order.
    """
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
