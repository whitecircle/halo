"""Expert-parallel loads: each rank streams only its expert slice from safetensors, or, where the lazy path
cannot read the checkpoint, loads the whole model through ``from_pretrained`` and patches its MoE layers."""

from __future__ import annotations

import logging
from collections import Counter
from functools import partial

import torch.nn as nn
from transformers import AutoModelForCausalLM

from src.checkpoint.reading import has_whole_model_weight_file
from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.expert_fuser import ExpertFuser
from src.distributed.expert_parallel.lazy_plan import (
    FUSED_EXPERT_KEY_PATTERN,
    ExpertKeyLayout,
    LazyLoadPlan,
    is_lazy_loadable,
    lazy_load_prologue,
)
from src.distributed.expert_parallel.parallelize import apply_ep_layers
from src.distributed.parallelism_config import ParallelismConfig
from src.model_loading.buffer_repair import finalize_loaded_model
from src.model_loading.dtype import resolve_load_dtype
from src.model_loading.lazy_safetensors.materialization import WeightAction, WeightPlan
from src.model_loading.lazy_safetensors.meta_shell import instantiate_on_meta
from src.model_loading.parallel.lazy_load import load_planned_weights
from src.model_loading.parallel.model_source import ModelSource, reject_ep_sharded_checkpoint, resolve_lazy_checkpoint
from src.model_loading.parallel.whole_model import load_whole_model
from src.models.attention.selection import revalidate_attn_kwarg
from src.models.moe.config_fields import require_expert_count
from src.runtime.load_coordination import timed_global_load
from src.runtime.ranks import get_global_rank, is_global_main_process

logger = logging.getLogger(__name__)


def _check_fused_experts_mapped(base: LazyLoadPlan) -> None:
    """Fail loud when a fused checkpoint's expert weights all missed the model's namespace (a fused layout fuses
    nothing, so its plans are final): a silently empty expert plan would train fresh experts on real attention."""
    if base.expert_layout != ExpertKeyLayout.FUSED or any(p.action is WeightAction.EXPERT_SHARD for p in base.plans):
        return
    sample = next((key for key in base.weight_map if FUSED_EXPERT_KEY_PATTERN.search(key)), None)
    if sample is not None:
        raise RuntimeError(
            "The checkpoint contains fused expert weights, but none map to the instantiated model. "
            f"Example unmapped checkpoint key: {sample!r}. Check the checkpoint-to-model namespace mapping."
        )


def load_ep_model_lazy(
    model_name_or_path: str,
    ep_config: EPConfig,
    config,
    dtype=None,
    trust_remote_code: bool = True,
    model_class=None,
    keep_fp32_params: bool = False,
    preserve_checkpoint_precision: bool = False,
    **model_kwargs,
) -> nn.Module:
    """COLLECTIVE: a MoE streamed from the safetensors checkpoint at ``model_name_or_path``, each rank reading
    only its own expert slice (fused-3D hubs) or fusing only its own experts (per-expert hubs), then EP-patched.

    Configured fp32 masters keep their stored values on their first read or fusion;
    ``preserve_checkpoint_precision`` (a resume) also requires every one restored.
    """
    model_class = model_class or AutoModelForCausalLM
    rank = get_global_rank()
    logger.info(f"[Rank {rank}] EP lazy loader: {model_name_or_path}")
    revalidate_attn_kwarg(model_kwargs, config)

    base = lazy_load_prologue(
        model_name_or_path,
        config,
        ep_config,
        dtype=dtype,
        meta_init_phase="ep_lazy_meta_init",
        build_meta_shell=lambda _incomplete: instantiate_on_meta(
            model_name_or_path, model_class, config, dtype=dtype, trust_remote_code=trust_remote_code, **model_kwargs
        ),
    )
    # Before the absent-module sweep, which would otherwise re-initialize the whole model first.
    _check_fused_experts_mapped(base)

    def report(plans: list[WeightPlan], fused_keys: set[str]) -> None:
        if fused_keys:
            logger.info(f"[Rank {rank}] Fused {len(fused_keys)} expert parameters from individual keys")
        counts = Counter(plan.action for plan in plans)
        logger.info(
            f"[Rank {rank}] Loading: {counts[WeightAction.EXPERT_SHARD]} expert-sharded, "
            f"{counts[WeightAction.REPLICATE]} replicated, {counts[WeightAction.IGNORE]} ignored, "
            f"{len(fused_keys)} fused"
        )

    weights_already_sharded = load_planned_weights(
        base,
        base.plans,
        model_path=model_name_or_path,
        ep_config=ep_config,
        keep_fp32_params=keep_fp32_params,
        preserve_checkpoint_precision=preserve_checkpoint_precision,
        what=f"EP lazy load from {model_name_or_path}",
        label=f"EP lazy load (rank {rank})",
        report=report,
        fuser=(
            ExpertFuser(ep_config.expert_start_idx, ep_config.expert_end_idx)
            if base.expert_layout == ExpertKeyLayout.PER_EXPERT
            else None
        ),
    )

    logger.info(f"[Rank {rank}] Applying EP patching...")
    # Derived, never asserted: a checkpoint whose expert keys match no pattern is planned REPLICATE, and
    # claiming "already sharded" there would give every rank global experts 0..E_local-1.
    model = apply_ep_layers(base.model, ep_config, weights_already_sharded=weights_already_sharded)
    finalize_loaded_model(model)
    logger.info(f"[Rank {rank}] EP lazy loading complete")
    return model


def _load_ep_model_huggingface(
    source: ModelSource,
    pc: ParallelismConfig,
    ep_config: EPConfig,
    model_class,
    from_pretrained_kwargs: dict,
    *,
    preserve_checkpoint_precision: bool,
) -> nn.Module:
    """COLLECTIVE: the whole model through ``from_pretrained``, a node's ranks at a time, EP-patched before it
    moves to the device.

    Every conversion-mapping result (a fused ``gate_up_proj``, a stacked per-expert bank) is an anonymous CPU
    tensor rather than a page-cache mmap view, so a converted family stages its whole expert set per rank.
    """
    kwargs = dict(from_pretrained_kwargs)
    config = kwargs["config"]
    num_experts = require_expert_count(config)
    if is_global_main_process():
        logger.info(f"[Rank 0] Loading model for EP: {source.name}")
        logger.info(f"[Rank 0] Model has {num_experts} experts")
        logger.info("[Rank 0] Waiting for sequential model loading...")
    if ep_config.num_experts is None:
        ep_config.finalize_expert_assignment(num_experts)
    kwargs["dtype"] = resolve_load_dtype(kwargs.get("dtype"), config)
    revalidate_attn_kwarg(kwargs, config)
    model = load_whole_model(
        model_class,
        source,
        load_phase="EP model load",
        max_concurrent_loading=pc.max_concurrent_loading,
        ep_config=ep_config,
        keep_fp32_non_ep=pc.fp32_non_ep_params,
        ep_wrapped=True,
        preserve_checkpoint_precision=preserve_checkpoint_precision,
        before_move=partial(apply_ep_layers, ep_config=ep_config),
        **kwargs,
    )
    if is_global_main_process():
        logger.info("[Rank 0] Model loaded successfully")
    return model


def load_ep_model(
    source: ModelSource,
    pc: ParallelismConfig,
    model_class,
    from_pretrained_kwargs: dict,
    *,
    preserve_checkpoint_precision: bool,
) -> nn.Module:
    """COLLECTIVE: a MoE with its experts distributed by EP, streamed lazily where ``pc.ep_lazy_loading`` allows
    and the checkpoint is a lazily readable safetensors one, else through ``from_pretrained``.

    ``preserve_checkpoint_precision`` marks a full-finetune resume and makes fp32-master coverage strict.
    """
    # First: ep_config() builds the EP groups on every rank, which no per-rank filesystem probe may precede.
    ep_config = pc.ep_config()
    main = is_global_main_process()
    if not pc.ep_lazy_loading:
        reject_ep_sharded_checkpoint(source)
        if main:
            logger.info(
                "[Rank 0] EP lazy loading disabled — using HuggingFace from_pretrained + EP patch "
                "(higher CPU RAM than lazy safetensors path)"
            )
        method = "huggingface_ep_lazy_disabled"
    elif resolve_lazy_checkpoint(source, lazy_requested=True, layout_supported=is_lazy_loadable):
        if main:
            logger.info(f"[Rank 0] Using lazy safetensors loader from {source.local_dir}")
        return timed_global_load(
            "EP",
            "lazy_safetensors",
            lambda: load_ep_model_lazy(
                source.local_dir,
                ep_config,
                model_class=model_class,
                keep_fp32_params=pc.fp32_non_ep_params,
                preserve_checkpoint_precision=preserve_checkpoint_precision,
                **from_pretrained_kwargs,
            ),
        )
    else:
        # An unmappable expert layout is a family property (from_pretrained is its route); missing
        # safetensors is an environment problem costing every rank the full checkpoint in RAM.
        if not has_whole_model_weight_file(source.local_dir, safetensors_only=True):
            fallback_reason, log = "no_safetensors", logger.warning
        else:
            fallback_reason, log = "lazy_incompatible_expert_layout", logger.info
        if main:
            log(f"[Rank 0] Using HuggingFace from_pretrained + EP patch (reason={fallback_reason})")
        method = f"huggingface_ep_fallback_{fallback_reason}"
    return timed_global_load(
        "EP",
        method,
        lambda: _load_ep_model_huggingface(
            source,
            pc,
            ep_config,
            model_class,
            from_pretrained_kwargs,
            preserve_checkpoint_precision=preserve_checkpoint_precision,
        ),
    )
