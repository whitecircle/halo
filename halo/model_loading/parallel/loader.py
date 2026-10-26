"""``load_distributed_model``: the policy load under a ``ParallelismConfig``, from the agreed source and
the run's attention through the per-mode loader to the post-load policies."""

from __future__ import annotations

import logging

import torch
from transformers import AutoConfig, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from halo.checkpoint.weights_source import LOADED_WEIGHTS_FROM_ATTR
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.distributed.rank_uniform_env import agree_rank_uniform_env
from halo.model_loading.buffer_repair import finalize_loaded_model
from halo.model_loading.dtype import configure_float32_matmul_precision
from halo.model_loading.liger.dispatch import apply_liger_kernel
from halo.model_loading.model_preparation import (
    apply_family_patches,
    apply_run_model_policies,
    resolve_auto_model_class,
)
from halo.model_loading.parallel.cp import load_cp_model, load_ep_cp_model
from halo.model_loading.parallel.ep import load_ep_model
from halo.model_loading.parallel.load_rules import (
    reject_unsupported_load,
    reject_unsupported_model,
    reject_unsupported_run,
    reject_unsupported_structure,
)
from halo.model_loading.parallel.lowp_compute import apply_lowp_compute
from halo.model_loading.parallel.model_source import ModelSource, resolve_model_source
from halo.model_loading.parallel.pp import load_pp_stage_model
from halo.model_loading.parallel.tp import (
    load_ep_tp_model,
    load_tp_model,
    load_tp_moe_model,
    uses_attention_only_tp,
)
from halo.model_loading.parallel.whole_model import cast_loaded_parameters, load_whole_model, warm_attention_kernels
from halo.model_loading.remote_code import apply_remote_code_compat_shims
from halo.models.attention.backend_setup import patch_flex_attention_compile
from halo.models.attention.gpt_oss_sinks import SinksPolicy, uses_gpt_oss_sinks
from halo.models.attention.selection import (
    is_fa4_backward_nan_prone,
    is_flash_attn_implementation,
    preferred_flash_impl,
    resolve_attn_implementation,
    validate_attn_implementation,
)
from halo.models.config_access import apply_config_overrides
from halo.models.moe.config_fields import has_routed_experts
from halo.runtime.consensus import reject_across_ranks
from halo.runtime.load_coordination import joined_node_load
from halo.runtime.ranks import get_local_rank, is_global_main_process

logger = logging.getLogger(__name__)


def _backend_sinks_policy(sinks_policy: SinksPolicy, init_from_scratch: bool) -> SinksPolicy:
    """The sinks policy the attention backend is chosen under: a from-scratch build has no pretrained sinks to
    neutralize, so its backend must carry them."""
    return SinksPolicy.LIVE if init_from_scratch else sinks_policy


def resolve_policy_attn_implementation(
    model_config,
    requested: str | None,
    dtype: torch.dtype,
    *,
    sinks_policy: SinksPolicy,
    init_from_scratch: bool,
    fa4_nan_fallback: str = "sdpa",
) -> str:
    """The attention backend :func:`load_distributed_model` resolves for a policy from its config, dtype and GPU,
    before the Context Parallelism swap: what a run can check before any weight loads. ``fa4_nan_fallback`` is
    :func:`resolve_attn_implementation`'s."""
    return resolve_attn_implementation(
        model_config,
        requested,
        dtype,
        sinks_policy=_backend_sinks_policy(sinks_policy, init_from_scratch),
        fa4_nan_fallback=fa4_nan_fallback,
    )


def _resolve_run_attention(
    model_config,
    pc: ParallelismConfig,
    requested: str | None,
    dtype: torch.dtype,
    *,
    sinks_policy: SinksPolicy,
    reset_sinks: bool,
    init_from_scratch: bool,
) -> str:
    """The attention implementation the model is built with: resolved against this model and dtype,
    swapped to FlashAttention under CP, and passed through the family patches."""
    # The kernel Ulysses runs, whatever the label: it vetoes FA4 for the FA4-backward-NaN families itself.
    cp_flash = preferred_flash_impl(allow_fa4=not is_fa4_backward_nan_prone(model_config)) if pc.is_cp_mode else None
    attn_implementation = resolve_policy_attn_implementation(
        model_config,
        requested,
        dtype,
        sinks_policy=sinks_policy,
        init_from_scratch=init_from_scratch,
        fa4_nan_fallback=cp_flash or "sdpa",
    )

    # CP (Ulysses) always runs flash_attn_func; label the config with the kernel it will actually use.
    if attn_implementation == "flex_attention" and pc.is_cp_mode:
        if dtype == torch.float32:
            raise ValueError(
                "Context Parallelism runs attention through FlashAttention (half-precision only), "
                "but this model resolved to flex_attention under dtype=float32 (fp32 training). "
                "Train in bf16/fp16, or drop Context Parallelism for fp32 runs."
            )
        if is_global_main_process():
            logger.warning(
                f"flex_attention is not supported with Context Parallelism (Ulysses always uses "
                f"flash_attn). Switching to {cp_flash}."
            )
        # Re-validated, not assigned: the substitute must clear the same sink-capability matrix, or a
        # run keeping live sinks (reset_sinks=false) silently moves to a sink-dropping kernel.
        attn_implementation = validate_attn_implementation(
            model_config, cp_flash, _backend_sinks_policy(sinks_policy, init_from_scratch)
        )
        if not is_flash_attn_implementation(attn_implementation):
            raise ValueError(
                f"Context Parallelism runs attention through flash_attn regardless of the config's "
                f"label, but this model accepts neither {cp_flash} nor any other FlashAttention "
                f"implementation (it fell back to '{attn_implementation}'). The labelled config "
                f"would not describe the kernel Ulysses actually calls. Train this model without "
                f"Context Parallelism."
            )

    # EP recompiles deadlock with NCCL all-to-all; FSDP2 + trainable sinks NaN in the compiled backward.
    if attn_implementation == "flex_attention":
        if pc.has_ep_group:
            patch_flex_attention_compile("EP recompilation deadlock avoidance")
        elif uses_gpt_oss_sinks(model_config) and reset_sinks:
            patch_flex_attention_compile("FSDP2 + trainable attention sinks NaN gradient fix")

    # Gemma 4 builds `sdpa_flex_sliding` for `sdpa`, whose FlexAttention graphs warm_attention_kernels
    # compiles on every rank before training.
    return apply_family_patches(model_config, attn_implementation)


def _build_random_init(model_class, pc: ParallelismConfig, from_pretrained_kwargs: dict, *, wraps_experts: bool):
    """COLLECTIVE: a random-weight model built from the config onto this rank's GPU, a node's ranks at a time."""
    # from_config materializes on CPU before .to(cuda), so it needs the same concurrency gate.
    with joined_node_load(f"Model build from {model_class.__name__} config", pc.max_concurrent_loading):
        model = model_class.from_config(
            from_pretrained_kwargs["config"],
            dtype=from_pretrained_kwargs.get("dtype"),
            attn_implementation=from_pretrained_kwargs.get("attn_implementation"),
            trust_remote_code=from_pretrained_kwargs.get("trust_remote_code", True),
        )
        model = model.to(f"cuda:{get_local_rank()}")
        cast_loaded_parameters(
            model, from_pretrained_kwargs.get("dtype"), keep_fp32=pc.fp32_non_ep_params, ep_wrapped=wraps_experts
        )
    # A random init has no checkpoint masters to restore; the join stays in the load's collective
    # sequence, as after a checkpoint load.
    reject_across_ranks(None, f"FP32-master checkpoint load from {model_class.__name__} config")
    finalize_loaded_model(model)
    return model


def _load_undistributed_model(
    source: ModelSource,
    pc: ParallelismConfig,
    model_class,
    from_pretrained_kwargs: dict,
    *,
    preserve_checkpoint_precision: bool,
    wraps_experts: bool,
    init_from_scratch: bool,
) -> PreTrainedModel:
    """COLLECTIVE: load a whole model onto this rank for plain DDP/FSDP data parallelism.

    ``wraps_experts`` adds the EP layers anyway: at ``ep_group_size == 1`` they distribute nothing
    and exist for the grouped-GEMM expert compute and the FSDP-managed ep1 expert shard.
    """
    # Creating the groups must precede the node-serialized rank-local checkpoint replay.
    ep_config = pc.ep_config() if wraps_experts else None
    if init_from_scratch:
        model = _build_random_init(model_class, pc, from_pretrained_kwargs, wraps_experts=wraps_experts)
    else:
        model = load_whole_model(
            model_class,
            source,
            pc,
            load_phase="Model load",
            ep_config=ep_config,
            preserve_checkpoint_precision=preserve_checkpoint_precision,
            place_on_device=False,
            device_map={"": get_local_rank()},
            **from_pretrained_kwargs,
        )
    if wraps_experts:
        model = apply_ep_layers(model, ep_config)
    mode = "Grouped GEMM EP layers (no EP distribution)" if wraps_experts else "standard DDP"
    if is_global_main_process():
        logger.info(f"Model loaded with {mode}")
    return model


def _dispatch_model_loading(
    source: ModelSource,
    pc: ParallelismConfig,
    model_class,
    model_config,
    dtype: torch.dtype,
    *,
    attn_implementation: str,
    trust_remote_code: bool,
    quantization_config,
    preserve_checkpoint_precision: bool,
    wraps_experts: bool,
    init_from_scratch: bool,
    model_kwargs: dict,
) -> PreTrainedModel:
    """The per-mode loader ``pc`` selects, given the run's ``from_pretrained`` kwargs."""
    from_pretrained_kwargs = dict(
        config=model_config,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
        **model_kwargs,
    )
    if source.revision is not None:
        from_pretrained_kwargs["revision"] = source.revision
    if quantization_config is not None:
        from_pretrained_kwargs["quantization_config"] = quantization_config
        if is_global_main_process():
            logger.info(f"Using quantization config: {quantization_config}")
    if init_from_scratch and is_global_main_process():
        logger.info("init_from_scratch=True — building model with RANDOM weights from config (no checkpoint load).")

    rung = (source, pc, model_class, from_pretrained_kwargs)
    # First: a stage owns only its layer slice, so every other loader materializes pp_size× too much.
    if pc.is_pp_mode:
        model = load_pp_stage_model(
            *rung, preserve_checkpoint_precision=preserve_checkpoint_precision, wraps_experts=wraps_experts
        )
        if is_global_main_process():
            logger.info(
                f"Model loaded for PP stage {pc.pp_rank}/{pc.pp_size} (ep={pc.ep_size}, dp={pc.data_parallel_size})"
            )
        return model
    if uses_attention_only_tp(pc, model_config):
        if pc.has_ep_group:
            return load_ep_tp_model(*rung, preserve_checkpoint_precision=preserve_checkpoint_precision)
        return load_tp_moe_model(
            *rung, preserve_checkpoint_precision=preserve_checkpoint_precision, wraps_experts=wraps_experts
        )
    if pc.is_tp_mode:
        return load_tp_model(*rung, preserve_checkpoint_precision=preserve_checkpoint_precision)
    if pc.has_ep_group and pc.is_cp_mode:
        model = load_ep_cp_model(*rung, preserve_checkpoint_precision=preserve_checkpoint_precision)
        if is_global_main_process():
            logger.info(f"Model loaded with EP+CP (ep={pc.ep_size}, cp={pc.cp_size})")
        return model
    if pc.has_ep_group:
        model = load_ep_model(*rung, preserve_checkpoint_precision=preserve_checkpoint_precision)
        if is_global_main_process():
            logger.info(
                f"Model loaded with EP (ep_size={pc.ep_size}, {'node-local' if pc.ep_scope == 'node' else 'cross-node'})"
            )
        return model
    if pc.is_cp_mode:
        model = load_cp_model(
            *rung, preserve_checkpoint_precision=preserve_checkpoint_precision, wraps_experts=wraps_experts
        )
        if is_global_main_process():
            logger.info(f"Model loaded with CP (cp_size={pc.cp_size}, grouped_gemm_experts={wraps_experts})")
        return model
    return _load_undistributed_model(
        *rung,
        preserve_checkpoint_precision=preserve_checkpoint_precision,
        wraps_experts=wraps_experts,
        init_from_scratch=init_from_scratch,
    )


def _resolve_run_config(
    model_name_or_path: str,
    revision: str | None,
    pc: ParallelismConfig,
    dtype: torch.dtype,
    *,
    trust_remote_code: bool,
    model_config_overrides: dict | None,
    sinks_policy: SinksPolicy,
    quantization_config,
    init_from_scratch: bool,
    model_class,
    text_only_model: bool,
) -> tuple[ModelSource, AutoConfig, type, bool]:
    """COLLECTIVE: the agreed source, the run's config, its model class and whether the EP layers apply, with
    every refusal config.json answers raised before the process groups and the meta shell: a second, not minutes."""
    apply_remote_code_compat_shims()
    # Every read below — config, tokenizer, weights — takes this source's revision, so a Hub source loads
    # one agreed commit on every rank, straight from the cache.
    source = resolve_model_source(model_name_or_path, revision, tag="policy")
    model_config = AutoConfig.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code, revision=source.revision
    )
    apply_config_overrides(model_config, model_config_overrides)
    wraps_experts = pc.needs_ep_layers and has_routed_experts(model_config)
    reject_unsupported_model(pc, model_config, sinks_policy=sinks_policy, wraps_experts=wraps_experts)
    reject_unsupported_load(
        pc,
        dtype,
        quantization_config=quantization_config,
        init_from_scratch=init_from_scratch,
        wraps_experts=wraps_experts,
    )

    # Multimodal wrappers need AutoModelForImageTextToText (CausalLM drops the vision tower);
    # text_only_model opts into that drop deliberately (and restores config-honored aux loss).
    if model_class is None:
        model_class = resolve_auto_model_class(model_config, text_only=text_only_model)
    elif text_only_model and is_global_main_process():
        logger.warning(
            f"text_only_model=True is ignored: the caller pinned model_class={model_class} "
            f"(task-head scripts resolve their own class). The flag only affects auto-resolution."
        )
    return source, model_config, model_class, wraps_experts


def load_distributed_model(
    model_name_or_path: str,
    parallelism_config: ParallelismConfig,
    dtype: torch.dtype = torch.bfloat16,
    *,
    trust_remote_code: bool,
    attn_implementation: str | None = None,
    model_config_overrides: dict | None = None,
    use_liger_kernel: bool = False,
    liger_kernel_config: dict | None = None,
    model_class=None,
    quantization_config=None,
    reset_sinks: bool = True,
    train_sinks: bool = False,
    init_from_scratch: bool = False,
    revision: str | None = None,
    text_only_model: bool = False,
    preserve_checkpoint_precision: bool = False,
    **model_kwargs,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """COLLECTIVE: load the policy and its tokenizer under ``parallelism_config`` (EP, CP, TP, PP or
    combinations).

    ``attn_implementation`` and ``model_class`` resolve from the config when None;
    ``reset_sinks``/``train_sinks`` resolve to the GptOss :class:`SinksPolicy`; ``text_only_model`` picks
    the text-only class of a multimodal checkpoint. Configured FP32 masters are restored before any
    wrap for every checkpoint start; ``preserve_checkpoint_precision`` marks a full-finetune resume
    source and also requires complete master coverage.
    """
    # Latched: free after the entry script's own pre-load call, a backstop for a caller without it.
    agree_rank_uniform_env()
    reject_unsupported_run(parallelism_config, dtype)
    sinks_policy = SinksPolicy.from_flags(reset_sinks=reset_sinks, train_sinks=train_sinks)

    # True fp32 matmuls before any forward — the image's TF32 default corrupts long-context RoPE.
    configure_float32_matmul_precision()

    source, model_config, model_class, wraps_experts = _resolve_run_config(
        model_name_or_path,
        revision,
        parallelism_config,
        dtype,
        trust_remote_code=trust_remote_code,
        model_config_overrides=model_config_overrides,
        sinks_policy=sinks_policy,
        quantization_config=quantization_config,
        init_from_scratch=init_from_scratch,
        model_class=model_class,
        text_only_model=text_only_model,
    )

    attn_implementation = _resolve_run_attention(
        model_config,
        parallelism_config,
        attn_implementation,
        dtype,
        sinks_policy=sinks_policy,
        reset_sinks=reset_sinks,
        init_from_scratch=init_from_scratch,
    )
    reject_unsupported_structure(
        parallelism_config,
        source,
        model_class,
        model_config,
        attn_implementation,
        dtype,
        trust_remote_code=trust_remote_code,
    )

    if use_liger_kernel:
        apply_liger_kernel(
            model_config,
            liger_kernel_config=liger_kernel_config,
            needs_ep_layers=parallelism_config.needs_ep_layers,
            tp_size=parallelism_config.tp_size,
            cp_size=parallelism_config.cp_size,
            pp_size=parallelism_config.pp_size,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code, revision=source.revision
    )
    # Training pads right. Half the collators force their side per call, the other half inherit this
    # one — so a left-padding tokenizer (GLM-4.7-Flash, Gemma 4) makes the packing collator read
    # leading pads as an attended document and drop the last real document's labels.
    tokenizer.padding_side = "right"

    model = _dispatch_model_loading(
        source,
        parallelism_config,
        model_class,
        model_config,
        dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
        quantization_config=quantization_config,
        preserve_checkpoint_precision=preserve_checkpoint_precision,
        wraps_experts=wraps_experts,
        init_from_scratch=init_from_scratch,
        model_kwargs=model_kwargs,
    )

    # Where weights were actually READ, or None for a random init: the resume loaders key their
    # "already holds these weights" skip on this, since ``_name_or_path`` survives init_from_scratch.
    setattr(model, LOADED_WEIGHTS_FROM_ATTR, None if init_from_scratch else model_name_or_path)
    apply_run_model_policies(
        model,
        model_config,
        sinks_policy=sinks_policy,
        attn_implementation=attn_implementation,
        sinks_pretrained=not init_from_scratch,
    )
    if parallelism_config.lowp.enabled:
        apply_lowp_compute(
            model, parallelism_config.lowp, expert_weight_cacheable=not parallelism_config.experts_fsdp_managed
        )
    warm_attention_kernels(model, dtype=dtype)
    return model, tokenizer
