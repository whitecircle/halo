"""``load_distributed_model``: the policy load under a ``ParallelismConfig``, from the agreed source and
the run's attention through the per-mode loader to the post-load policies."""

from __future__ import annotations

import torch
from accelerate.logging import get_logger
from transformers import AutoConfig, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from src.checkpoint.weights_source import LOADED_WEIGHTS_FROM_ATTR
from src.distributed.expert_parallel.parallelize import apply_ep_layers
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.rank_uniform_env import agree_rank_uniform_env
from src.hardware import is_blackwell_gpu
from src.model_loading.buffer_repair import finalize_loaded_model
from src.model_loading.dtype import configure_float32_matmul_precision
from src.model_loading.liger.dispatch import apply_liger_kernel
from src.model_loading.model_preparation import (
    apply_family_patches,
    apply_run_model_policies,
    resolve_auto_model_class,
)
from src.model_loading.parallel.cp import load_cp_model, load_ep_cp_model
from src.model_loading.parallel.ep import load_ep_model
from src.model_loading.parallel.load_rules import reject_unsupported_model, reject_unsupported_run
from src.model_loading.parallel.lowp_compute import apply_lowp_compute
from src.model_loading.parallel.model_source import ModelSource, resolve_model_source
from src.model_loading.parallel.pp import load_pp_stage_model
from src.model_loading.parallel.tp import load_ep_tp_model, load_tp_model, load_tp_moe_model
from src.model_loading.parallel.whole_model import cast_loaded_parameters, load_whole_model, warm_attention_kernels
from src.model_loading.remote_code import apply_remote_code_compat_shims
from src.models.attention.backend_setup import patch_flex_attention_compile
from src.models.attention.gpt_oss_sinks import SinksPolicy, uses_gpt_oss_sinks
from src.models.attention.selection import resolve_attn_implementation, validate_attn_implementation
from src.models.config_access import apply_config_overrides
from src.models.moe.config_fields import has_routed_experts
from src.runtime.consensus import reject_across_ranks
from src.runtime.load_coordination import joined_node_load
from src.runtime.ranks import get_local_rank

logger = get_logger(__name__)


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
    # A from-scratch build has no pretrained sinks to neutralize, so its backend must carry them.
    attn_sinks_policy = SinksPolicy.LIVE if init_from_scratch else sinks_policy
    attn_implementation = resolve_attn_implementation(model_config, requested, dtype, sinks_policy=attn_sinks_policy)

    # CP (Ulysses) always runs flash_attn_func; label the config with the kernel it will actually use.
    if attn_implementation == "flex_attention" and pc.is_cp_mode:
        if dtype == torch.float32:
            raise ValueError(
                "Context Parallelism runs attention through FlashAttention (half-precision only), "
                "but this model resolved to flex_attention under dtype=float32 (fp32 training). "
                "Train in bf16/fp16, or drop Context Parallelism for fp32 runs."
            )
        cp_fa = "flash_attention_4" if is_blackwell_gpu() else "flash_attention_2"
        logger.warning(
            f"flex_attention is not supported with Context Parallelism (Ulysses always uses "
            f"flash_attn). Switching to {cp_fa}. A native flex_attention Ulysses path may be added later."
        )
        # Re-validated, not assigned: the substitute must clear the same sink-capability matrix, or a
        # run keeping live sinks (reset_sinks=false) silently moves to a sink-dropping kernel.
        attn_implementation = validate_attn_implementation(model_config, cp_fa, attn_sinks_policy)
        if not attn_implementation.startswith("flash_attention"):
            raise ValueError(
                f"Context Parallelism runs attention through flash_attn regardless of the config's "
                f"label, but this model accepts neither {cp_fa} nor any other FlashAttention "
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

    ``wraps_experts`` adds the EP MoE wrappers anyway: at ``ep_group_size == 1`` they distribute nothing
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
            load_phase="Model load",
            max_concurrent_loading=pc.max_concurrent_loading,
            ep_config=ep_config,
            keep_fp32_non_ep=pc.fp32_non_ep_params,
            ep_wrapped=wraps_experts,
            preserve_checkpoint_precision=preserve_checkpoint_precision,
            place_on_device=False,
            device_map={"": get_local_rank()},
            **from_pretrained_kwargs,
        )
    if wraps_experts:
        model = apply_ep_layers(model, ep_config)
    mode = "Grouped GEMM wrappers (no EP distribution)" if wraps_experts else "standard DDP"
    logger.info(f"Model loaded with {mode}")
    return model


def _dispatch_model_loading(
    source: ModelSource,
    pc: ParallelismConfig,
    model_class,
    from_pretrained_kwargs: dict,
    *,
    preserve_checkpoint_precision: bool,
    wraps_experts: bool,
    init_from_scratch: bool,
) -> PreTrainedModel:
    """The per-mode loader ``pc`` selects."""
    rung = (source, pc, model_class, from_pretrained_kwargs)
    # First: a stage owns only its layer slice, so every other loader materializes pp_size× too much.
    if pc.is_pp_mode:
        model = load_pp_stage_model(
            *rung, preserve_checkpoint_precision=preserve_checkpoint_precision, wraps_experts=wraps_experts
        )
        logger.info(
            f"Model loaded for PP stage {pc.pp_rank}/{pc.pp_size} (ep={pc.ep_size}, dp={pc.data_parallel_size})"
        )
        return model
    if pc.has_ep_group and pc.is_tp_mode:
        return load_ep_tp_model(*rung, preserve_checkpoint_precision=preserve_checkpoint_precision)
    if pc.is_tp_mode and has_routed_experts(from_pretrained_kwargs["config"]):
        return load_tp_moe_model(
            *rung, preserve_checkpoint_precision=preserve_checkpoint_precision, wraps_experts=wraps_experts
        )
    if pc.is_tp_mode:
        return load_tp_model(*rung, preserve_checkpoint_precision=preserve_checkpoint_precision)
    if pc.has_ep_group and pc.is_cp_mode:
        model = load_ep_cp_model(*rung, preserve_checkpoint_precision=preserve_checkpoint_precision)
        logger.info(f"Model loaded with EP+CP (ep={pc.ep_size}, cp={pc.cp_size})")
        return model
    if pc.has_ep_group:
        model = load_ep_model(*rung, preserve_checkpoint_precision=preserve_checkpoint_precision)
        logger.info(
            f"Model loaded with EP (ep_size={pc.ep_size}, {'node-local' if pc.ep_scope == 'node' else 'cross-node'})"
        )
        return model
    if pc.is_cp_mode:
        model = load_cp_model(
            *rung, preserve_checkpoint_precision=preserve_checkpoint_precision, wraps_experts=wraps_experts
        )
        logger.info(f"Model loaded with CP (cp_size={pc.cp_size}, grouped_gemm_experts={wraps_experts})")
        return model
    return _load_undistributed_model(
        *rung,
        preserve_checkpoint_precision=preserve_checkpoint_precision,
        wraps_experts=wraps_experts,
        init_from_scratch=init_from_scratch,
    )


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

    apply_remote_code_compat_shims()
    # Every read below — config, tokenizer, weights — takes this source's revision, so a Hub source loads
    # one agreed commit on every rank, straight from the cache.
    source = resolve_model_source(model_name_or_path, revision, tag="policy")

    model_config = AutoConfig.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code, revision=source.revision
    )
    apply_config_overrides(model_config, model_config_overrides)
    wraps_experts = parallelism_config.needs_ep_wrappers and has_routed_experts(model_config)
    # Before the process groups and the meta shell: config.json is all the sharding arithmetic needs,
    # so failing here costs a second instead of minutes.
    reject_unsupported_model(parallelism_config, model_config, sinks_policy=sinks_policy, wraps_experts=wraps_experts)

    # Multimodal wrappers need AutoModelForImageTextToText (CausalLM drops the vision tower);
    # text_only_model opts into that drop deliberately (and restores config-honored aux loss).
    if model_class is None:
        model_class = resolve_auto_model_class(model_config, text_only=text_only_model)
    elif text_only_model:
        logger.warning(
            f"text_only_model=True is ignored: the caller pinned model_class={model_class} "
            f"(task-head scripts resolve their own class). The flag only affects auto-resolution."
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

    if use_liger_kernel:
        apply_liger_kernel(
            model_config,
            liger_kernel_config=liger_kernel_config,
            needs_ep_wrappers=parallelism_config.needs_ep_wrappers,
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
        # Also gated on MoE: a dense QLoRA run routing to DDP (or to CP, which keeps Params4bit) must pass.
        if (
            parallelism_config.has_ep_group
            or parallelism_config.is_tp_mode
            or parallelism_config.is_pp_mode
            or wraps_experts
        ):
            raise ValueError(
                "Quantized loading (QLoRA / bitsandbytes) is not supported with EP / TP / PP "
                "or grouped-GEMM MoE loaders: they materialize plain de-quantized weights, so "
                "bitsandbytes Params4bit are lost and PEFT's 4-bit adapter dispatch fails (PP "
                "rejects PEFT outright). For QLoRA use plain data parallelism (on a MoE model with "
                "`use_grouped_gemm: false`, under torchrun and accelerate launch alike) or CP on a "
                "dense model, which preserves quantization. Under EP/TP use plain LoRA (no "
                "quantization)."
            )
        from_pretrained_kwargs["quantization_config"] = quantization_config
        logger.info(f"Using quantization config: {quantization_config}")

    if init_from_scratch:
        # Identical per-rank seeds give consistent FSDP2 shards; EP/TP/CP would need a sharded random-init.
        if (
            parallelism_config.has_ep_group
            or parallelism_config.is_tp_mode
            or parallelism_config.is_cp_mode
            or parallelism_config.is_expert_tp_mode
            or parallelism_config.is_pp_mode
        ):
            raise NotImplementedError(
                "init_from_scratch is currently supported only for dense data-parallel / "
                "FSDP2 training (no EP/TP/CP/ETP/PP). PP additionally has nothing to be "
                "stage-aware about without a checkpoint, so every rank would build the whole "
                "model. To train from scratch WITH parallelism, materialize the random-init "
                "checkpoint once outside the distributed job (AutoModelForCausalLM.from_config "
                "+ save_pretrained), point model_name_or_path at it, and drop init_from_scratch "
                "— see 'Large or MoE models' in agent-docs/training-methods/pretraining.md."
            )
        if quantization_config is not None:
            raise ValueError("init_from_scratch is incompatible with quantization (QLoRA loads pretrained weights).")
        logger.info("init_from_scratch=True — building model with RANDOM weights from config (no checkpoint load).")

    model = _dispatch_model_loading(
        source,
        parallelism_config,
        model_class,
        from_pretrained_kwargs,
        preserve_checkpoint_precision=preserve_checkpoint_precision,
        wraps_experts=wraps_experts,
        init_from_scratch=init_from_scratch,
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
        if quantization_config is not None:
            raise ValueError(
                "lowp_precision (low-precision compute) is incompatible with a quantization_config "
                "(QLoRA/bitsandbytes): the weights are already quantized and not plain nn.Linear. "
                "Use one or the other."
            )
        if dtype == torch.float16:
            raise ValueError(
                "lowp_precision requires bf16/fp32 master weights, but dtype=float16. "
                "Set bf16 (the toolkit default) — low precision affects only the GEMM operands."
            )
        apply_lowp_compute(
            model, parallelism_config.lowp, expert_weight_cacheable=not parallelism_config.experts_fsdp_managed
        )

    warm_attention_kernels(model, dtype=dtype)

    return model, tokenizer
