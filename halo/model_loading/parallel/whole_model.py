"""Whole-model loads onto one rank: the node-throttled ``from_pretrained`` with its run-dtype cast and
FP32-master restore, the trainer-constructor load, and the attention warm-up every load ends in."""

from __future__ import annotations

import gc
from collections.abc import Callable
from functools import partial

import torch
from transformers import PretrainedConfig, PreTrainedModel

from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.expert_parallel.parallelize import ep_claimed_blocks
from halo.distributed.parallelism_config import ParallelismConfig
from halo.hardware import current_device
from halo.model_loading.checkpoint_coverage import from_pretrained_verified
from halo.model_loading.dtype import cast_parameters_to_run_dtype, normalize_model_init_dtype
from halo.model_loading.finalization import finalize_loaded_model
from halo.model_loading.model_preparation import apply_family_patches, auto_load_model, resolve_build_config
from halo.model_loading.parallel.fp32_masters import restore_fp32_master_parameters
from halo.model_loading.parallel.model_source import ModelSource, resolve_model_source
from halo.models.attention.backend_setup import warm_fa4_kernels
from halo.models.attention.masked_matmul import warm_masked_matmul_kernels
from halo.models.attention.sdpa_flex_sliding import warm_flex_sliding_kernels
from halo.models.attention.tree import warm_tree_kernels
from halo.models.attention.tree_plan import FullAttentionForm
from halo.models.introspection.config_access import set_config_field_run_scoped
from halo.runtime.consensus import DeferredRankFailure, barrier_on_exit
from halo.runtime.load_coordination import joined_node_load
from halo.runtime.ranks import get_global_world_size

# ``from_pretrained`` arguments that also name a config field: the load reads them itself, never off the config.
_LOAD_ARGUMENTS_NAMING_CONFIG_FIELDS = frozenset({"dtype", "torch_dtype", "quantization_config"})


def cast_loaded_parameters(model: torch.nn.Module, dtype, *, keep_fp32: bool, ep_wrapped: bool) -> None:
    """:func:`cast_parameters_to_run_dtype` on a fresh model, EP-aware: ``keep_fp32`` skips EP parameters in the
    trainer's upcast, so with ``ep_wrapped`` those blocks train at the run dtype."""
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
    pc: ParallelismConfig,
    *,
    load_phase: str,
    ep_config: EPConfig | None,
    preserve_checkpoint_precision: bool = False,
    place_on_device: bool = True,
    before_move: Callable[[torch.nn.Module], torch.nn.Module] | None = None,
    **from_pretrained_kwargs,
) -> torch.nn.Module:
    """COLLECTIVE: ``from_pretrained`` a node's ranks at a time, cast to the run dtype with fp32 masters restored, on
    CPU unless ``device_map`` is named. A failed restore raises on every rank only after all leave the node-serialized
    region, where a world collective would wait on queued ranks."""
    from_pretrained_kwargs.setdefault("device_map", "cpu")
    precision_guard = DeferredRankFailure(f"FP32-master checkpoint load from {source.name}")
    with joined_node_load(f"{load_phase} from {source.name}", pc.max_concurrent_loading):
        model = from_pretrained_verified(model_class, source.name, **from_pretrained_kwargs)
        cast_loaded_parameters(
            model,
            from_pretrained_kwargs.get("dtype"),
            keep_fp32=pc.fp32_non_ep_params,
            ep_wrapped=ep_config is not None,
        )
        precision_guard.run(
            lambda: restore_fp32_master_parameters(
                model,
                source.local_dir,
                ep_config,
                keep_non_ep=pc.fp32_non_ep_params,
                strict=preserve_checkpoint_precision,
            )
        )
        if place_on_device and precision_guard.reason is None:
            if before_move is not None:
                model = before_move(model)
            model = move_model_to_local_device(model)
    precision_guard.reject()
    finalize_loaded_model(model, weights_dir=source.local_dir)
    return model


def _apply_config_field_kwargs(config: PretrainedConfig, model_init_kwargs: dict) -> dict:
    """Set each ``model_init_kwargs`` entry naming a config field, as ``from_pretrained`` does, and return the rest,
    since a build handed ``config`` passes every keyword it does not read to ``__init__``."""
    fields = {
        key: value
        for key, value in model_init_kwargs.items()
        if hasattr(config, key) and key not in _LOAD_ARGUMENTS_NAMING_CONFIG_FIELDS
    }
    for key, value in fields.items():
        setattr(config, key, value)
    return {key: value for key, value in model_init_kwargs.items() if key not in fields}


def resolve_trainer_model(model, args, *, parallelism_config: ParallelismConfig | None, model_class=None):
    """COLLECTIVE for a path string: a trainer constructor's model, loaded from the agreed source with family patches,
    cast and warmed. An instantiated model must come without ``model_init_kwargs``."""
    if not isinstance(model, str):
        if args is not None and getattr(args, "model_init_kwargs", None) is not None:
            raise ValueError(
                "model_init_kwargs can only be used when model is a string path, but the model is already instantiated."
            )
        return model
    model_id = model
    model_init_kwargs = getattr(args, "model_init_kwargs", None) or {}
    normalize_model_init_dtype(model_init_kwargs)
    source = resolve_model_source(model_id, model_init_kwargs.get("revision"), tag="trainer_model")
    revision = source.revision
    config = resolve_build_config(
        model_id, revision=revision, trust_remote_code=model_init_kwargs.get("trust_remote_code", False)
    )
    load_kwargs = _apply_config_field_kwargs(config, model_init_kwargs)
    # Unpinned, transformers picks the backend at the build and only the backend-independent patches apply.
    if attn_implementation := apply_family_patches(config, load_kwargs.get("attn_implementation")):
        load_kwargs["attn_implementation"] = attn_implementation

    # Run-scoped after the load, never through from_pretrained: a kwarg write lands in the exported
    # config.json and defaults the KV cache off for every downstream consumer.
    run_scoped_cache_off = (
        args is not None and getattr(args, "gradient_checkpointing", False) and "use_cache" not in model_init_kwargs
    )

    # Unthrottled: each rank keeps its copy in host memory until the trainer places it, so admitting the
    # ranks in batches would bound nothing.
    with joined_node_load(f"Model load from {model_id}", max_concurrent=0):
        model = auto_load_model(
            model_id, model_class=model_class, config=config, **{**load_kwargs, "revision": revision}
        )
    # An unset or "auto" dtype loads at the checkpoint's own dtype (recorded on the config); the cast
    # unifies the fp32 pins to it.
    requested = load_kwargs.get("dtype")
    run_dtype = requested if isinstance(requested, torch.dtype) else model.config.dtype
    cast_parameters_to_run_dtype(
        model, run_dtype, keep_fp32=parallelism_config is not None and parallelism_config.fp32_non_ep_params
    )
    finalize_loaded_model(model, weights_dir=source.local_dir)
    if run_scoped_cache_off:
        set_config_field_run_scoped(model.config, "use_cache", False)
    warm_attention_kernels(model, dtype=run_dtype)
    return model


def _warm_then_join(*warmups: Callable[[], None]) -> None:
    """COLLECTIVE: run rank-local warm-ups behind :func:`barrier_on_exit`, a no-op below world 2. A first-use JIT
    (~10s) lets peers race into the next collective; world size is the only condition every rank evaluates
    identically."""
    if get_global_world_size() <= 1:
        return
    with barrier_on_exit("Attention kernel warm-up"):
        for warmup in warmups:
            warmup()


def warm_attention_kernels(model: PreTrainedModel, *, dtype: torch.dtype) -> None:
    """COLLECTIVE: pre-compile FA4, FlexAttention or masked matmul kernels, joined. A failed FA4 warm-up only warns, and
    that rank compiles at first use; any other failure reaches the join and raises on every rank."""
    _warm_then_join(
        partial(warm_fa4_kernels, model, dtype=dtype),
        partial(warm_flex_sliding_kernels, model, dtype=dtype),
        partial(warm_masked_matmul_kernels, model, dtype=dtype),
    )


def warm_tree_attention_kernels(model: PreTrainedModel, *, dtype: torch.dtype, full_form: FullAttentionForm) -> None:
    """COLLECTIVE: pre-compile the kernels tree attention adds, joined. A failed warm-up only warns, and that rank
    compiles at first use."""
    _warm_then_join(partial(warm_tree_kernels, model, dtype=dtype, full_form=full_form))
