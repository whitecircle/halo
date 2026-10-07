"""Model loaders for Context Parallelism (CP and EP+CP).

Mirrors :mod:`src.distributed.expert_parallel.loading`: high-level entry point that
loads the model from disk, applies optional EP, and wraps it for Ulysses CP.
"""

from __future__ import annotations

import gc
import logging

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from src.distributed.context_parallel.config import CPConfig
from src.distributed.context_parallel.wrapper import patch_model_for_cp
from src.distributed.expert_parallel.loading import load_ep_model, load_through_cpu
from src.distributed.expert_parallel.patching import create_ep_buffers, patch_moe_model_for_ep
from src.distributed.runtime import get_global_rank
from src.models.patches.attention import revalidate_attn_kwarg

logger = logging.getLogger(__name__)


def load_model_for_cp(
    model_name_or_path: str,
    cp_config: CPConfig,
    config,
    dtype: torch.dtype | None = None,
    trust_remote_code: bool = True,
    model_class=None,
    max_concurrent_loading: int | None = None,
    ep_config=None,
    keep_fp32_params: bool = False,
    preserve_checkpoint_precision: bool = False,
    **model_kwargs,
) -> nn.Module:
    """Load a model with Ulysses CP support only (no expert distribution).

    For non-MoE models, and for MoE models taking CP without EP.

    Args:
        config: the resolved HF config, REQUIRED — like the EP and PP loaders. It is what
            :func:`validate_attn_implementation` judges the requested backend against, so an
            optional one would let a caller that omitted it install an UNVALIDATED implementation.
        model_class: defaults to ``AutoModelForCausalLM``.
        max_concurrent_loading: ranks that may load weights in parallel within a node
            (passed to :func:`~src.distributed.filesystem.joined_node_load`).
        ep_config: when given (an ``ep_size == 1`` config), the MoE blocks get the grouped-GEMM
            expert wrappers before the CP wrap — same ordering as EP+CP, so the generation hook and
            the state-dict expert paths land on the inner HF model, not on the CP wrapper. Without
            it a MoE under pure CP pays the Liger swiglu/geglu force-off without the grouped-GEMM
            speedup it buys.
        keep_fp32_params: :func:`cast_loaded_parameters`' ``keep_fp32`` (the run keeps fp32 masters,
            ``fp32_non_ep_params``).
    """
    if model_class is None:
        model_class = AutoModelForCausalLM

    rank = get_global_rank()
    logger.info(f"[Rank {rank}] Loading model for Ulysses CP: cp={cp_config.cp_size}")

    revalidate_attn_kwarg(model_kwargs, config)

    # Finalized before the CP wrap, on the inner HF model — the wrapper carries no tie_weights.
    model = load_through_cpu(
        model_class,
        model_name_or_path,
        load_phase="CP model load",
        max_concurrent_loading=max_concurrent_loading,
        ep_config=ep_config,
        keep_fp32=keep_fp32_params,
        ep_wrapped=ep_config is not None,
        preserve_checkpoint_precision=preserve_checkpoint_precision,
        config=config,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        **model_kwargs,
    )

    if ep_config is not None:
        model = patch_moe_model_for_ep(model, ep_config)
        create_ep_buffers(model)

    model = patch_model_for_cp(model, cp_config)
    logger.info(f"[Rank {rank}] ✓ Model loaded with CP")

    return model


def load_model_for_ep_cp(
    model_name_or_path: str,
    ep_config,
    cp_config: CPConfig,
    config,
    dtype: torch.dtype | None = None,
    trust_remote_code: bool = True,
    model_class=None,
    max_concurrent_loading: int | None = None,
    lazy: bool = True,
    revision: str | None = None,
    keep_fp32_params: bool = False,
    preserve_checkpoint_precision: bool = False,
    **model_kwargs,
) -> nn.Module:
    """Load a MoE model with both EP and Ulysses CP support.

    Args:
        config: the resolved HF config, REQUIRED — see :func:`load_model_for_cp`.
        lazy: passed to :func:`load_ep_model`. ``False`` falls back to
            ``from_pretrained`` + EP patching (vs lazy safetensors).
        revision: Hub revision pin, passed to :func:`load_ep_model` so the lazy
            snapshot resolution reads the pinned checkpoint.
        keep_fp32_params: passed to :func:`load_ep_model`.

    Note:
        Router grads averaged by ``world_size``, correct for CP: each rank's
        ``local_grad = cp_size × batch_grad`` (mean over local tokens), so
        ``all_reduce(SUM)`` / ``world_size`` recovers the batch-mean gradient.
    """
    rank = get_global_rank()
    logger.info(f"[Rank {rank}] Loading model for EP+Ulysses CP: ep={ep_config.ep_size}, cp={cp_config.cp_size}")

    model = load_ep_model(
        model_name_or_path,
        ep_config,
        config=config,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        model_class=model_class,
        max_concurrent_loading=max_concurrent_loading,
        lazy=lazy,
        revision=revision,
        keep_fp32_params=keep_fp32_params,
        preserve_checkpoint_precision=preserve_checkpoint_precision,
        **model_kwargs,
    )

    model = patch_model_for_cp(model, cp_config)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model
