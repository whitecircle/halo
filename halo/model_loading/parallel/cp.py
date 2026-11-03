"""Context Parallelism loads: the model wrapped for Ulysses, loaded whole or through the EP loader."""

from __future__ import annotations

import gc
import logging

import torch
import torch.nn as nn

from halo.distributed.context_parallel.ulysses_model import UlyssesModel
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.ep import load_ep_model
from halo.model_loading.parallel.model_source import ModelSource
from halo.model_loading.parallel.whole_model import load_whole_model
from halo.models.attention.selection import reselect_attn_kwarg
from halo.runtime.ranks import get_global_rank

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
    """COLLECTIVE: load a model for Ulysses CP with no expert distribution. ``wraps_experts`` gives a MoE its
    grouped-GEMM EP layers before the CP wrap; without them pure CP pays the Liger swiglu/geglu force-off without the
    speedup it buys."""
    cp_config = pc.cp_config()
    ep_config = pc.ep_config() if wraps_experts else None
    rank = get_global_rank()
    logger.info(f"[Rank {rank}] Loading model for Ulysses CP: cp={cp_config.cp_size}")

    kwargs = dict(from_pretrained_kwargs)
    reselect_attn_kwarg(kwargs, kwargs["config"])
    # Finalized before the CP wrap, as the wrapper carries no tie_weights.
    model = load_whole_model(
        model_class,
        source,
        pc,
        load_phase="CP model load",
        ep_config=ep_config,
        preserve_checkpoint_precision=preserve_checkpoint_precision,
        **kwargs,
    )
    if wraps_experts:
        model = apply_ep_layers(model, ep_config)

    model = UlyssesModel(model, cp_config)
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
    """COLLECTIVE: load a MoE with experts distributed by EP and sequences split by CP. Router gradients averaged over
    ``world_size`` stay correct: each rank's gradient is ``cp_size ×`` the batch gradient, so
    ``all_reduce(SUM) / world_size`` recovers the batch mean."""
    ep_config = pc.ep_config()
    cp_config = pc.cp_config()
    rank = get_global_rank()
    logger.info(f"[Rank {rank}] Loading model for EP+Ulysses CP: ep={ep_config.ep_size}, cp={cp_config.cp_size}")

    model = load_ep_model(
        source, pc, model_class, from_pretrained_kwargs, preserve_checkpoint_precision=preserve_checkpoint_precision
    )
    model = UlyssesModel(model, cp_config)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model
