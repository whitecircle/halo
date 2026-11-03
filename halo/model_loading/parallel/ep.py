"""Expert-parallel loads: each rank streams only its expert slice from safetensors, or, where the lazy path
cannot read the checkpoint, loads the whole model through ``from_pretrained`` and patches its MoE layers."""

from __future__ import annotations

import logging
from collections import Counter
from functools import partial

import torch.nn as nn
from transformers import AutoModelForCausalLM

from halo.checkpoint.reading import has_whole_model_weight_file
from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.dtype import resolve_load_dtype
from halo.model_loading.finalization import finalize_loaded_model
from halo.model_loading.lazy_safetensors.materialization import WeightAction, WeightPlan
from halo.model_loading.lazy_safetensors.meta_shell import instantiate_on_meta
from halo.model_loading.parallel.expert_fuser import ExpertFuser
from halo.model_loading.parallel.lazy_load import load_planned_weights
from halo.model_loading.parallel.lazy_plan import (
    FUSED_EXPERT_KEY_PATTERN,
    ExpertKeyLayout,
    LazyLoadPlan,
    agree_lazy_load,
    lazy_load_prologue,
)
from halo.model_loading.parallel.model_source import ModelSource, reject_ep_sharded_checkpoint
from halo.model_loading.parallel.whole_model import load_whole_model
from halo.models.moe.config_fields import require_expert_count
from halo.runtime.load_coordination import timed_global_load
from halo.runtime.ranks import get_global_rank, is_global_main_process

logger = logging.getLogger(__name__)


def _check_fused_experts_mapped(base: LazyLoadPlan) -> None:
    """Fail loud when a fused checkpoint's expert weights all missed the model's namespace; a silently empty expert
    plan would train fresh experts on real attention."""
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
    """COLLECTIVE: stream a MoE from safetensors, each rank reading only its expert slice or fusing only its experts,
    then EP-patch. ``preserve_checkpoint_precision`` (a resume) requires every fp32 master restored."""
    model_class = model_class or AutoModelForCausalLM
    rank = get_global_rank()
    logger.info(f"[Rank {rank}] EP lazy loader: {model_name_or_path}")

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
    # Derived: a checkpoint whose expert keys match no pattern is planned REPLICATE, and asserting sharded
    # there would give every rank global experts 0..E_local-1.
    model = apply_ep_layers(base.model, ep_config, weights_already_sharded=weights_already_sharded)
    finalize_loaded_model(model, weights_dir=model_name_or_path)
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
    """COLLECTIVE: the whole model through ``from_pretrained``, a node's ranks at a time, EP-patched before it moves to
    the device. Conversion results are anonymous CPU tensors, not mmap views, so a converted family stages its whole
    expert set per rank."""
    kwargs = dict(from_pretrained_kwargs)
    config = kwargs["config"]
    num_experts = require_expert_count(config)
    if is_global_main_process():
        logger.info(f"[Rank 0] Loading model for EP: {source.name}")
        logger.info(f"[Rank 0] Model has {num_experts} experts")
        logger.info("[Rank 0] Waiting for the per-node model load throttle...")
    if ep_config.num_experts is None:
        ep_config.finalize_expert_assignment(num_experts)
    kwargs["dtype"] = resolve_load_dtype(kwargs.get("dtype"), config)
    model = load_whole_model(
        model_class,
        source,
        pc,
        load_phase="EP model load",
        ep_config=ep_config,
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
    """COLLECTIVE: a MoE distributed by EP, streamed lazily where allowed and readable, else through
    ``from_pretrained``. ``preserve_checkpoint_precision`` marks a full-finetune resume and makes fp32-master coverage
    strict."""
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
    elif agree_lazy_load(source):
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
