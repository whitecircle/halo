"""Install each family's ``EP*MoELayer`` over the HuggingFace MoE blocks the EP family roster claims."""

import logging

import torch.nn as nn

from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.expert_parallel.deepep.capacity import register_forward_generation_hook
from halo.distributed.expert_parallel.families.roster import MOE_LAYER_MAP
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.runtime.ranks import is_global_main_process
from halo.subclass_registry import swap_registered_modules

logger = logging.getLogger(__name__)


def ep_claimed_blocks(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """``(path, block)`` for every MoE block :func:`apply_ep_layers` wraps, on a model not yet wrapped."""
    return [(path, module) for path, module in model.named_modules() if type(module).__name__ in MOE_LAYER_MAP]


def apply_ep_layers(model: nn.Module, ep_config: EPConfig, weights_already_sharded: bool = False) -> nn.Module:
    """Replace every claimed MoE block of ``model`` with its family's EP layer, in place. ``weights_already_sharded``:
    the expert tensors already hold this rank's range (the lazy loader), so each layer adopts them."""
    expert_tp_info = f", expert_tp_size={ep_config.expert_tp_size}" if ep_config.expert_tp_size > 1 else ""
    logger.info(f"Patching model for EP with DeepEP (ep_size={ep_config.ep_size}{expert_tp_info})")

    def build(path: str, block: nn.Module, layer_cls: type[EPMoELayerBase]) -> nn.Module:
        # recurse=True: only Gemma4 registers the expert tensors on the mapped class. Elsewhere they hang off a child
        # module, so a shallow scan would pass the quantized checkpoints this rejects.
        non_float = [n for n, p in block.named_parameters(recurse=True) if not p.is_floating_point()]
        if non_float:
            raise ValueError(
                f"Expert Parallelism requires a de-quantized (BF16) checkpoint, but MoE layer "
                f"'{path}' has non-float expert tensor(s) {non_float} — this is a natively "
                f"quantized model (e.g. MXFP4 gpt-oss). Load a BF16-dequantized checkpoint "
                f"(e.g. 'unsloth/gpt-oss-20b-BF16' or a local BF16 export) instead."
            )
        # Detect once, off the first matched block; a set ``num_experts`` means already finalized.
        if ep_config.num_experts is None:
            ep_config.finalize_expert_assignment(layer_cls.detect_num_experts(block))
        if is_global_main_process():
            logger.info(f"Patched {path} ({type(block).__name__}) -> {layer_cls.__name__}")
        return layer_cls(block, ep_config, weights_already_sharded=weights_already_sharded)

    # No descent into a wrapped block: Gemma4 registers the expert tensors on the mapped class
    # itself, so walking a match's subtree would re-match what was just wrapped.
    patched = len(swap_registered_modules(model, MOE_LAYER_MAP, build, descend_into_match=False))

    if patched == 0:
        if ep_config.ep_group_size > 1:
            # Unwrapped experts stay replicated and un-synced while the run reports a DP size that
            # assumes they are sharded.
            raise ValueError(
                f"Expert Parallelism requested (expert_parallel_size={ep_config.ep_size}, "
                f"expert_tensor_parallel_size={ep_config.expert_tp_size}) but NO MoE layer was "
                f"patched — none of the model's modules match a registered EP layer. "
                f"Supported HF MoE classes: {sorted(MOE_LAYER_MAP)}. Add an EP layer for "
                f"this family (see agent-docs/models/adding-a-model.md) or run without expert "
                f"parallelism."
            )
        # ep_group_size == 1 wraps only for grouped GEMM; unwrapped keeps stock HF expert compute.
        logger.warning(f"No MoE layers found. Supported: {sorted(MOE_LAYER_MAP)}")
    else:
        # Registered here so every EP-patch entry point gets it. Idempotent.
        register_forward_generation_hook(model)
        logger.info(f"Patched {patched} MoE layers for EP with DeepEP")

    return model
