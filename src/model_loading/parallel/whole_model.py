"""Whole-model loads onto one rank: the node-throttled ``from_pretrained`` with its run-dtype cast and
FP32-master restore, the trainer-constructor load, and the attention warm-up every load ends in."""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from functools import partial

import torch
from transformers import PreTrainedModel

from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.parallelize import ep_claimed_blocks
from src.hardware import current_device
from src.model_loading.buffer_repair import finalize_loaded_model
from src.model_loading.checkpoint_coverage import from_pretrained_verified
from src.model_loading.dtype import cast_parameters_to_run_dtype, normalize_model_init_dtype
from src.model_loading.model_preparation import auto_load_model
from src.model_loading.parallel.fp32_masters import restore_fp32_master_parameters
from src.model_loading.parallel.model_source import ModelSource, resolve_model_source
from src.models.attention.backend_setup import warm_fa4_kernels
from src.models.attention.sdpa_flex_sliding import warm_flex_sliding_kernels
from src.models.attention.tree import warm_tree_kernels
from src.models.attention.tree_plan import FullAttentionForm
from src.models.config_access import set_config_field_run_scoped
from src.runtime.collectives import barrier
from src.runtime.consensus import DeferredRankFailure
from src.runtime.load_coordination import joined_node_load
from src.runtime.ranks import get_global_world_size

logger = logging.getLogger(__name__)


def cast_loaded_parameters(model: torch.nn.Module, dtype, *, keep_fp32: bool, ep_wrapped: bool) -> None:
    """:func:`cast_parameters_to_run_dtype` on a freshly loaded model, EP-aware.

    ``keep_fp32`` (``fp32_non_ep_params``) keeps stored fp32 values for the parameters the trainer's fp32
    upcast covers, which skips EP parameters: with ``ep_wrapped`` a parameter inside a block the EP
    wrappers take trains at the run dtype.
    """
    cast_parameters_to_run_dtype(model, dtype, keep_fp32=keep_fp32)
    if keep_fp32 and ep_wrapped:
        for _path, block in ep_claimed_blocks(model):
            cast_parameters_to_run_dtype(block, dtype)


def move_model_to_local_device(model: torch.nn.Module) -> torch.nn.Module:
    """Move ``model`` to this rank's CUDA device (or CPU) and free the stale CPU copy."""
    model = model.to(current_device())
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def load_whole_model(
    model_class,
    source: ModelSource,
    *,
    load_phase: str,
    max_concurrent_loading: int | None,
    ep_config: EPConfig | None,
    keep_fp32_non_ep: bool,
    ep_wrapped: bool,
    preserve_checkpoint_precision: bool = False,
    place_on_device: bool = True,
    before_move: Callable[[torch.nn.Module], torch.nn.Module] | None = None,
    **from_pretrained_kwargs,
) -> torch.nn.Module:
    """COLLECTIVE: ``from_pretrained`` a node's ranks at a time, cast to the run dtype with the FP32
    masters restored.

    Loads on CPU unless ``from_pretrained_kwargs`` names a ``device_map``; with ``place_on_device`` the
    model then moves onto this rank's device after ``before_move`` (the EP patch). A failed restore
    raises on every rank only once all have left the node-serialized region, where a world collective
    would wait on queued ranks.
    """
    from_pretrained_kwargs.setdefault("device_map", "cpu")
    precision_guard = DeferredRankFailure(f"FP32-master checkpoint load from {source.name}")
    with joined_node_load(f"{load_phase} from {source.name}", max_concurrent_loading):
        model = from_pretrained_verified(model_class, source.name, **from_pretrained_kwargs)
        cast_loaded_parameters(
            model, from_pretrained_kwargs.get("dtype"), keep_fp32=keep_fp32_non_ep, ep_wrapped=ep_wrapped
        )
        precision_guard.run(
            lambda: restore_fp32_master_parameters(
                model,
                source.local_dir,
                ep_config,
                keep_non_ep=keep_fp32_non_ep,
                strict=preserve_checkpoint_precision,
            )
        )
        if place_on_device and precision_guard.reason is None:
            if before_move is not None:
                model = before_move(model)
            model = move_model_to_local_device(model)
    precision_guard.reject()
    finalize_loaded_model(model)
    return model


def resolve_trainer_model(model, args, *, model_class=None, keep_fp32_non_ep: bool):
    """COLLECTIVE-EQUIVALENT for a path string: the model a trainer constructor was handed, loaded from
    the agreed source and cast when it is a path.

    ``model_class`` None takes the widest Auto class. An instantiated model must come without
    ``model_init_kwargs``.
    """
    if not isinstance(model, str):
        if args is not None and getattr(args, "model_init_kwargs", None) is not None:
            raise ValueError(
                "model_init_kwargs can only be used when model is a string path, but the model is already instantiated."
            )
        return model
    model_id = model
    model_init_kwargs = getattr(args, "model_init_kwargs", None) or {}
    normalize_model_init_dtype(model_init_kwargs)
    revision = resolve_model_source(model_id, model_init_kwargs.get("revision"), tag="trainer_model").revision

    # Run-scoped after the load, never through from_pretrained: a kwarg write lands in the exported
    # config.json and defaults the KV cache off for every downstream consumer.
    run_scoped_cache_off = (
        args is not None and getattr(args, "gradient_checkpointing", False) and "use_cache" not in model_init_kwargs
    )

    # Unthrottled: each rank keeps its copy in host memory until the trainer places it, so admitting the
    # ranks in batches would bound nothing.
    with joined_node_load(f"Model load from {model_id}", max_concurrent=0):
        model = auto_load_model(model_id, model_class=model_class, **{**model_init_kwargs, "revision": revision})
    # An unset or "auto" dtype loads at the checkpoint's own dtype (recorded on the config); the cast
    # unifies the fp32 pins to it.
    requested = model_init_kwargs.get("dtype")
    cast_parameters_to_run_dtype(
        model,
        requested if isinstance(requested, torch.dtype) else model.config.dtype,
        keep_fp32=keep_fp32_non_ep,
    )
    finalize_loaded_model(model)
    if run_scoped_cache_off:
        set_config_field_run_scoped(model.config, "use_cache", False)
    return model


def _warm_then_barrier(*warmups: Callable[[], None]) -> None:
    """COLLECTIVE: run rank-local kernel warm-ups on every rank, then barrier. No-op below world 2.

    A first-use JIT compile (~10s) on one rank lets its peers race ahead into the next collective. World
    size is the only condition every rank evaluates identically: the resolved backend and an FA4 build
    are per-rank, and returning on either would skip a barrier peers already wait in.
    """
    if get_global_world_size() <= 1:
        return
    try:
        for warmup in warmups:
            warmup()
    finally:
        try:
            barrier()
        except Exception as e:
            # Swallowed so a barrier failure cannot mask the warm-up's own error, but logged: a
            # failed teardown barrier is itself the rank skew this fence exists to prevent.
            logger.warning(
                f"Attention warmup teardown barrier failed ({type(e).__name__}: {e}) — ranks may be skewed "
                f"entering the next collective",
            )


def warm_attention_kernels(model: PreTrainedModel, *, dtype: torch.dtype) -> None:
    """COLLECTIVE: pre-compile this model's FA4 or FlexAttention kernels on every rank, then barrier."""
    _warm_then_barrier(
        partial(warm_fa4_kernels, model, dtype=dtype), partial(warm_flex_sliding_kernels, model, dtype=dtype)
    )


def warm_tree_attention_kernels(model: PreTrainedModel, *, dtype: torch.dtype, full_form: FullAttentionForm) -> None:
    """COLLECTIVE: pre-compile the kernels tree attention adds on every rank, then barrier."""
    _warm_then_barrier(partial(warm_tree_kernels, model, dtype=dtype, full_form=full_form))
