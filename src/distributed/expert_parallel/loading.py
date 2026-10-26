"""Load MoE models for Expert Parallelism.

:func:`load_ep_model` is the entry point: refuses an EP-sharded checkpoint and dispatches
to the lazy safetensors loader (default, :mod:`.lazy_loader`) or the sequential ``from_pretrained`` +
EP-patch fallback (:func:`_load_ep_model_huggingface`). EP is orthogonal to DP (Megatron-LM style).
"""

import time
from collections.abc import Callable
from functools import partial
from typing import TypeVar

import torch
from accelerate.logging import get_logger
from transformers import AutoModelForCausalLM

from src.distributed.expert_parallel.config import EPConfig, get_num_experts
from src.distributed.expert_parallel.lazy_loader import (
    lazy_loader_supports_checkpoint,
    load_ep_model_lazy,
)
from src.distributed.expert_parallel.patching import patch_moe_model_for_ep
from src.model_loading.lazy_safetensors.weights import (
    has_safetensors_checkpoint,
    resolve_run_dtype,
)
from src.model_loading.parallel.model_source import (
    ModelSource,
    reject_ep_sharded_checkpoint,
    resolve_lazy_checkpoint,
)
from src.model_loading.parallel.whole_model import load_whole_model
from src.models.attention.selection import revalidate_attn_kwarg
from src.runtime.load_coordination import log_global_load_duration_seconds
from src.runtime.ranks import get_global_rank, get_local_rank

_T = TypeVar("_T")

logger = get_logger(__name__)


def _timed_ep_load(method: str, load_fn: Callable[[], _T]) -> _T:
    """Run an EP load path and log global wall span (earliest start → latest end across ranks)."""
    t_start_wall = time.time()
    loaded = load_fn()
    # Deliberately not in a ``finally``: this is a collective, so running it while one rank is
    # unwinding an exception turns that rank's traceback into an NCCL timeout on all the others.
    log_global_load_duration_seconds(
        tag="EP",
        method=method,
        t_start_wall=t_start_wall,
        t_end_wall=time.time(),
    )
    return loaded


def load_ep_model(
    source: ModelSource,
    ep_config: EPConfig,
    config,
    dtype=None,
    trust_remote_code: bool = True,
    model_class=None,
    max_concurrent_loading: int | None = None,
    lazy: bool = True,
    revision: str | None = None,
    keep_fp32_params: bool = False,
    preserve_checkpoint_precision: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """Load a MoE model for EP training.

    The checkpoint format (``huggingface`` vs ``ep_sharded``) is auto-detected. With
    ``lazy=True`` (default) and a resolvable safetensors checkpoint, each rank
    reads only its expert slice in parallel (:mod:`.lazy_loader`); otherwise falls
    back to per-rank ``from_pretrained`` + EP patch. ``source`` is the agreed checkpoint
    (:func:`~src.model_loading.parallel.model_source.resolve_model_source`). ``config`` is the
    caller's already-loaded model config, required so that no loader re-reads it per rank.
    ``keep_fp32_params`` is :func:`~src.model_loading.parallel.whole_model.cast_loaded_parameters`' ``keep_fp32``.
    Configured masters retain stored FP32 values for every checkpoint start.
    ``preserve_checkpoint_precision`` identifies a full-finetune resume and makes master coverage
    strict; it does not decide whether a fresh stage preserves the source's precision.
    """
    rank = get_global_rank()
    huggingface_load = partial(
        _load_ep_model_huggingface,
        source.name,
        ep_config,
        config,
        dtype,
        trust_remote_code,
        model_class=model_class,
        max_concurrent_loading=max_concurrent_loading,
        revision=revision,
        keep_fp32_params=keep_fp32_params,
        preserve_checkpoint_precision=preserve_checkpoint_precision,
        **model_kwargs,
    )
    if not lazy:
        reject_ep_sharded_checkpoint(source)
        logger.info(
            f"[Rank {rank}] EP lazy loading disabled — using HuggingFace from_pretrained + EP patch "
            f"(higher CPU RAM than lazy safetensors path)"
        )
        return _timed_ep_load("huggingface_ep_lazy_disabled", huggingface_load)

    # Lazy first; HF fallback when there are no safetensors or the expert layout is unmappable.
    if resolve_lazy_checkpoint(source, lazy_requested=True, layout_supported=lazy_loader_supports_checkpoint):
        logger.info(f"[Rank {rank}] Using lazy safetensors loader from {source.local_dir}")
        return _timed_ep_load(
            "lazy_safetensors",
            lambda: load_ep_model_lazy(
                source.local_dir,
                ep_config,
                config,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                model_class=model_class,
                keep_fp32_params=keep_fp32_params,
                preserve_checkpoint_precision=preserve_checkpoint_precision,
                **model_kwargs,
            ),
        )

    # Diagnostic only, and derived here rather than beside the gate so the probes cost nothing on the
    # lazy path. Rank-local: a rank that disagreed with rank 0's verdict reports its own reason. The
    # level separates the two kinds: an unmappable expert layout is a class-declared family property
    # (the HF path IS its supported route), while missing safetensors is an environment problem that
    # silently costs every rank the full checkpoint in CPU RAM.
    if not has_safetensors_checkpoint(source.local_dir):
        fallback_reason, log = "no_safetensors", logger.warning
    else:
        fallback_reason, log = "lazy_incompatible_expert_layout", logger.info
    log(f"[Rank {rank}] Using HuggingFace from_pretrained + EP patch (reason={fallback_reason})")
    return _timed_ep_load(f"huggingface_ep_fallback_{fallback_reason}", huggingface_load)


def _load_ep_model_huggingface(
    model_name_or_path: str,
    ep_config: EPConfig,
    config,
    dtype=None,
    trust_remote_code: bool = True,
    model_class=None,
    max_concurrent_loading: int | None = None,
    revision: str | None = None,
    keep_fp32_params: bool = False,
    preserve_checkpoint_precision: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """Load EP model from HuggingFace checkpoint.

    Files must be downloaded/cached first. ``device_map="cpu"`` keeps unconverted bf16 keys as
    safetensors mmap views (page cache, shared across ranks), but every conversion-mapping result —
    a fused ``gate_up_proj`` or a stacked per-expert bank — is an anonymous CPU tensor, so a converted
    MoE family stages its whole expert set per rank here; ``max_concurrent_loading`` bounds how many
    ranks do so at once.
    """
    if model_class is None:
        model_class = AutoModelForCausalLM

    rank = get_global_rank()
    logger.info(f"[Rank {rank}] Loading model for EP: {model_name_or_path}")

    num_experts = get_num_experts(config)
    logger.info(f"[Rank {rank}] Model has {num_experts} experts")

    if ep_config.num_experts is None:
        ep_config.finalize_expert_assignment(num_experts)

    dtype = resolve_run_dtype(dtype, config)

    revalidate_attn_kwarg(model_kwargs, config)

    # Bounded ranks per node at a time — an unbounded fan-in CPU-OOMs on large MoE.
    logger.info(f"[Rank {rank}] Waiting for sequential model loading (local_rank={get_local_rank()})...")
    model = load_whole_model(
        model_class,
        model_name_or_path,
        load_phase="EP model load",
        max_concurrent_loading=max_concurrent_loading,
        ep_config=ep_config,
        keep_fp32_non_ep=keep_fp32_params,
        ep_wrapped=True,
        preserve_checkpoint_precision=preserve_checkpoint_precision,
        before_move=partial(patch_moe_model_for_ep, ep_config=ep_config),
        config=config,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        revision=revision,
        **model_kwargs,
    )

    logger.info(f"[Rank {rank}] Model loaded successfully")
    return model
