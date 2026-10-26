"""Model patching for Expert Parallelism with DeepEP: wrap each supported HuggingFace MoE block with its
``EP*MoELayer`` wrapper (:data:`MOE_LAYER_MAP`).
"""

import logging

import torch.nn as nn

from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.deepep.capacity import register_forward_generation_hook
from src.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from src.runtime.ranks import is_global_main_process
from src.subclass_registry import build_class_claim_map, swap_registered_modules

# Registers every EP wrapper as an ``EPMoELayerBase`` subclass for ``build_moe_layer_map()`` to discover.
import src.distributed.expert_parallel.layers.roster  # noqa: F401  isort:skip

logger = logging.getLogger(__name__)


def build_moe_layer_map() -> dict[str, type[EPMoELayerBase]]:
    """HF MoE class name → EP wrapper class, derived from every registered wrapper's
    :attr:`~EPMoELayerBase.HF_MODULE_NAMES`.

    Walks the ``EPMoELayerBase`` subclass tree (concrete wrappers may sit under intermediate bases) so a
    new family self-registers on import. Raises on a duplicate HF name.
    """
    return build_class_claim_map(EPMoELayerBase, "HF MoE class name")


MOE_LAYER_MAP = build_moe_layer_map()


def ep_claimed_blocks(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """``(path, block)`` for every MoE block :func:`patch_moe_model_for_ep` wraps, on a model not yet
    patched: the modules whose parameters become EP parameters."""
    return [(path, module) for path, module in model.named_modules() if type(module).__name__ in MOE_LAYER_MAP]


def patch_moe_model_for_ep(
    model: nn.Module,
    ep_config: EPConfig,
    weights_already_sharded: bool = False,
) -> nn.Module:
    """Patch MoE model for Expert Parallelism with DeepEP.

    Args:
        model: The model to patch.
        ep_config: Expert parallelism configuration.
        weights_already_sharded: If True, expert weights are already sliced to this rank's local
            expert range (e.g. by the lazy loader), so wrapper ``__init__`` adopts existing
            parameters instead of slicing from the full expert tensors.
    """
    expert_tp_info = f", expert_tp_size={ep_config.expert_tp_size}" if ep_config.expert_tp_size > 1 else ""
    logger.info(f"Patching model for EP with DeepEP (ep_size={ep_config.ep_size}{expert_tp_info})")

    def build(path: str, block: nn.Module, wrapper_cls: type[EPMoELayerBase]) -> nn.Module:
        # recurse=True: only Gemma4 registers the expert tensors on the mapped class itself. Every
        # other family maps the MoE block, whose experts hang off a child module, so a shallow scan
        # would see nothing and pass on the quantized checkpoints this check exists to reject
        # (gpt-oss MXFP4 included).
        non_float = [n for n, p in block.named_parameters(recurse=True) if not p.is_floating_point()]
        if non_float:
            raise ValueError(
                f"Expert Parallelism requires a de-quantized (BF16) checkpoint, but MoE layer "
                f"'{path}' has non-float expert tensor(s) {non_float} — this is a natively "
                f"quantized model (e.g. MXFP4 gpt-oss). Load a BF16-dequantized checkpoint "
                f"(e.g. 'unsloth/gpt-oss-20b-BF16' or a local BF16 export) instead."
            )
        # Detect once, off the first matched block: ``finalize_expert_assignment`` sets
        # ``num_experts`` or raises, so a set count means already finalized — the spelling both lazy
        # loaders use.
        if ep_config.num_experts is None:
            ep_config.finalize_expert_assignment(wrapper_cls.detect_num_experts(block))
        if is_global_main_process():
            logger.info(f"Patched {path} ({type(block).__name__}) -> {wrapper_cls.__name__}")
        return wrapper_cls(block, ep_config, weights_already_sharded=weights_already_sharded)

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
                f"patched — none of the model's modules match a registered EP wrapper. "
                f"Supported HF MoE classes: {sorted(MOE_LAYER_MAP)}. Add an EP wrapper for "
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
