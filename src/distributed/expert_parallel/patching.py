"""Model patching for Expert Parallelism with DeepEP: wrap each supported HuggingFace MoE block with its
``EP*MoELayer`` wrapper (:data:`MOE_LAYER_MAP`), create DeepEP comm buffers, and enable EP-safe gradient
checkpointing.
"""

import functools
import logging

import torch.nn as nn

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.dispatcher import (
    register_forward_generation_hook,
    verify_rank_uniform_env,
)
from src.distributed.expert_parallel.gc_scope import install_ep_checkpoint_scopes
from src.distributed.module_registry import build_hf_module_name_map, swap_registered_modules
from src.distributed.runtime import is_global_main_process

# Registers every EP wrapper as an ``EPMoELayerBase`` subclass for ``build_moe_layer_map()`` to discover.
import src.distributed.expert_parallel.layers.roster  # noqa: F401  isort:skip

logger = logging.getLogger(__name__)


def build_moe_layer_map() -> dict[str, type[EPMoELayerBase]]:
    """HF MoE class name → EP wrapper class, derived from every registered wrapper's
    :attr:`~EPMoELayerBase.HF_MODULE_NAMES`.

    Walks the ``EPMoELayerBase`` subclass tree (concrete wrappers may sit under intermediate bases) so a
    new family self-registers on import. Raises on a duplicate HF name.
    """
    return build_hf_module_name_map(EPMoELayerBase, "MoE")


MOE_LAYER_MAP = build_moe_layer_map()


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
        # recurse=True: only Gemma4 registers the expert tensors on the mapped class itself.
        # Every other family maps the MoE *block*, whose experts hang off a child module, so a
        # shallow scan sees nothing and the check silently passes on the quantized checkpoints
        # it exists to refuse — gpt-oss MXFP4 included.
        non_float = [n for n, p in block.named_parameters(recurse=True) if not p.is_floating_point()]
        if non_float:
            raise ValueError(
                f"Expert Parallelism requires a de-quantized (BF16) checkpoint, but MoE layer "
                f"'{path}' has non-float expert tensor(s) {non_float} — this is a natively "
                f"quantized model (e.g. MXFP4 gpt-oss). Load a BF16-dequantized checkpoint "
                f"(e.g. 'unsloth/gpt-oss-20b-BF16' or a local BF16 export) instead."
            )
        # Detect once, off the first matched block: ``finalize_expert_assignment`` sets ``num_experts``
        # (or raises), so a set count is exactly "already finalized" — the spelling both lazy
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
            # assumes they are sharded — fail here, not 10k steps later.
            raise ValueError(
                f"Expert Parallelism requested (ep_size={ep_config.ep_size}, "
                f"expert_tp_size={ep_config.expert_tp_size}) but NO MoE layer was patched — none of "
                f"the model's modules match a registered EP wrapper. Supported HF MoE classes: "
                f"{sorted(MOE_LAYER_MAP)}. Add an EP wrapper for this family (see "
                f"agent-docs/models/adding-a-model.md) or run without expert parallelism."
            )
        # ep_group_size == 1 wraps only for grouped GEMM — unwrapped keeps stock HF expert compute.
        logger.warning(f"No MoE layers found. Supported: {sorted(MOE_LAYER_MAP)}")
    else:
        # Registered here (not create_ep_buffers) so every EP-patch entry point gets it. Idempotent.
        register_forward_generation_hook(model)
        logger.info(f"Patched {patched} MoE layers for EP with DeepEP")

    return model


def create_ep_buffers(model: nn.Module) -> int:
    """Create DeepEP communication buffers for all EP layers in the model.

    Collective — all EP ranks must call together, after all ranks have loaded and patched.

    Returns:
        Number of buffers created.
    """
    # Latched backstop: the entry scripts run this in distributed setup, before the weight load, so
    # here it is free. It still covers a caller that reaches DeepEP without that scaffold.
    verify_rank_uniform_env()

    created = 0
    for module in model.modules():
        if isinstance(module, EPMoELayerBase):
            module.dispatcher.create_buffer()
            created += 1

    if created > 0:
        logger.info(f"Created {created} DeepEP buffers")
    return created


def _split_every_n_layers(
    gradient_checkpointing_kwargs: dict | None, every_n_layers: int = 1
) -> tuple[dict | None, int]:
    """Lift ``every_n_layers`` out of the checkpoint kwargs, as HF's trainer does before enabling.

    It is ``gradient_checkpointing_enable``'s own keyword (which decoder layers checkpoint); left in
    the dict it reaches ``torch.utils.checkpoint`` and fails the first recompute. TRL's re-enable
    passes the args dict raw, so the split lives at the enable seam, not at the caller.
    """
    kwargs = dict(gradient_checkpointing_kwargs or {})
    every_n_layers = kwargs.pop("every_n_layers", every_n_layers)
    return kwargs or None, every_n_layers


def enable_ep_gradient_checkpointing(model: nn.Module, *, gradient_checkpointing_kwargs: dict) -> None:
    """Enable gradient checkpointing whose recompute replays EP's DeepEP dispatch/combine.

    The model's own ``gradient_checkpointing_enable`` resolves HF's policy (input-grad requirement,
    per-module flags, ``every_n_layers``), then every installed checkpoint function is re-pointed
    at its scoped wrapper so both passes of each invocation share one :class:`EPCheckpointScope`
    — the frame-local cache the EP layers replay from (:meth:`EPMoELayerBase._gc_dispatch`).
    """
    if not hasattr(model, "gradient_checkpointing_enable"):
        raise RuntimeError(
            f"gradient_checkpointing=True but {type(model).__name__} has no "
            f"gradient_checkpointing_enable(): the EP path has already disabled HF Trainer's own GC "
            f"setup, so returning here would train with NO gradient checkpointing at the memory "
            f"profile of a run that asked for it. Disable gradient_checkpointing for this model."
        )

    kwargs, every_n_layers = _split_every_n_layers(gradient_checkpointing_kwargs)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs, every_n_layers=every_n_layers)
    scoped = install_ep_checkpoint_scopes(model)
    if scoped == 0:
        raise RuntimeError(
            f"gradient_checkpointing=True but {type(model).__name__}.gradient_checkpointing_enable() "
            f"installed no checkpoint function: EP's dispatch replay hangs off that function, so the "
            f"run would issue a second DeepEP dispatch inside backward and corrupt every gradient."
        )
    _rescope_on_reenable(model)
    logger.info(
        f"Gradient checkpointing enabled on {scoped} modules (kwargs={kwargs}, every_n_layers={every_n_layers})"
    )


def _rescope_on_reenable(model: nn.Module) -> None:
    """Bind the EP scopes to the model's ``gradient_checkpointing_enable``, for every later caller.

    Anything that turns checkpointing back on calls that method, and it installs HF's BARE checkpoint
    function — dropping the scoped wrapper EP replays from. Online GRPO does exactly this every step:
    TRL wraps generation in ``disable_gradient_checkpointing``, whose exit re-enables from the args.
    The next backward then re-enters an MoE forward with no scope, and the layer raises rather than
    issue the second DeepEP dispatch that would corrupt every gradient. Restoring the kwargs is not
    enough — the scopes are not in them — so the two are made inseparable here instead, at the one
    seam every re-enable goes through.
    """
    enable = model.gradient_checkpointing_enable
    if getattr(enable, "_ep_scoped", False):
        return

    @functools.wraps(enable)
    def enable_with_ep_scopes(gradient_checkpointing_kwargs=None, every_n_layers: int = 1):
        kwargs, every_n_layers = _split_every_n_layers(gradient_checkpointing_kwargs, every_n_layers)
        result = enable(gradient_checkpointing_kwargs=kwargs, every_n_layers=every_n_layers)
        install_ep_checkpoint_scopes(model)  # idempotent: already-scoped functions are skipped
        return result

    enable_with_ep_scopes._ep_scoped = True
    model.gradient_checkpointing_enable = enable_with_ep_scopes
