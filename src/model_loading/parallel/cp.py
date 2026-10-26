"""Context Parallelism loads: the model wrapped for Ulysses, loaded whole or through the EP loader."""

from __future__ import annotations

import gc
import logging

import torch
import torch.nn as nn

from src.distributed.context_parallel.wrapper import patch_model_for_cp
from src.distributed.expert_parallel.loading import load_ep_model
from src.distributed.expert_parallel.patching import patch_moe_model_for_ep
from src.distributed.parallelism_config import ParallelismConfig
from src.model_loading.parallel.model_source import ModelSource
from src.model_loading.parallel.whole_model import load_whole_model
from src.models.attention.selection import revalidate_attn_kwarg
from src.runtime.ranks import get_global_rank

logger = logging.getLogger(__name__)


def load_cp_model(
    source: ModelSource,
    pc: ParallelismConfig,
    model_class,
    from_pretrained_kwargs: dict,
    *,
    preserve_checkpoint_precision: bool,
    wraps_experts: bool,
) -> nn.Module:
    """COLLECTIVE: load a model for Ulysses CP with no expert distribution.

    ``wraps_experts`` gives a MoE its grouped-GEMM expert wrappers before the CP wrap, on the inner HF
    model as under EP+CP; without them a MoE under pure CP pays the Liger swiglu/geglu force-off without
    the speedup it buys.
    """
    cp_config = pc.cp_config()
    ep_config = pc.ep_config() if wraps_experts else None
    rank = get_global_rank()
    logger.info(f"[Rank {rank}] Loading model for Ulysses CP: cp={cp_config.cp_size}")

    kwargs = dict(from_pretrained_kwargs)
    revalidate_attn_kwarg(kwargs, kwargs["config"])
    # Finalized before the CP wrap, on the inner HF model — the wrapper carries no tie_weights.
    model = load_whole_model(
        model_class,
        source.name,
        load_phase="CP model load",
        max_concurrent_loading=pc.max_concurrent_loading,
        ep_config=ep_config,
        keep_fp32_non_ep=pc.fp32_non_ep_params,
        ep_wrapped=wraps_experts,
        preserve_checkpoint_precision=preserve_checkpoint_precision,
        **kwargs,
    )
    if wraps_experts:
        model = patch_moe_model_for_ep(model, ep_config)

    model = patch_model_for_cp(model, cp_config)
    logger.info(f"[Rank {rank}] ✓ Model loaded with CP")
    return model


def load_ep_cp_model(
    source: ModelSource,
    pc: ParallelismConfig,
    model_class,
    from_pretrained_kwargs: dict,
    *,
    preserve_checkpoint_precision: bool,
) -> nn.Module:
    """COLLECTIVE: load a MoE with its experts distributed by EP and its sequences split by Ulysses CP.

    Router gradients averaged over ``world_size`` stay correct under CP: each rank's local gradient is
    ``cp_size ×`` the batch gradient (mean over local tokens), so ``all_reduce(SUM) / world_size``
    recovers the batch mean.
    """
    ep_config = pc.ep_config()
    cp_config = pc.cp_config()
    rank = get_global_rank()
    logger.info(f"[Rank {rank}] Loading model for EP+Ulysses CP: ep={ep_config.ep_size}, cp={cp_config.cp_size}")

    model = load_ep_model(
        source,
        ep_config,
        model_class=model_class,
        max_concurrent_loading=pc.max_concurrent_loading,
        lazy=pc.ep_lazy_loading,
        keep_fp32_params=pc.fp32_non_ep_params,
        preserve_checkpoint_precision=preserve_checkpoint_precision,
        **from_pretrained_kwargs,
    )
    model = patch_model_for_cp(model, cp_config)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model
