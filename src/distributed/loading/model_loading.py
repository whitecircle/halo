"""Model loading: distributed (EP / CP / TP / combinations) and standard.

``load_distributed_model`` dispatches to per-mode ``_load_*`` loaders by ``ParallelismConfig``.
"""

from __future__ import annotations

import gc
import os
import time
from typing import TYPE_CHECKING

import torch
from accelerate.logging import get_logger
from huggingface_hub import snapshot_download
from torch.distributed.tensor import DTensor
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DistributedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.checkpoint.config_export import LOADED_WEIGHTS_FROM_ATTR
from src.distributed.context_parallel.loading import load_model_for_cp, load_model_for_ep_cp
from src.distributed.expert_parallel.expert_weights import ep_layer_classes_for_config
from src.distributed.expert_parallel.lazy_loader import (
    lazy_loader_supports_checkpoint,
    load_ep_model_lazy,
)
from src.distributed.expert_parallel.loading import (
    decide_lazy_loadable,
    load_ep_model,
    reject_ep_sharded_checkpoint,
    resolve_hub_or_local_dir,
)
from src.distributed.expert_parallel.patching import create_ep_buffers, patch_moe_model_for_ep
from src.distributed.filesystem import fs_aware_main_first, sequential_load_within_node
from src.distributed.loading.warmup import warm_attention_kernels
from src.distributed.mesh import create_dp_tp_mesh, get_tp_submesh
from src.distributed.parallelism_config import accelerate_launch_rejection
from src.distributed.pipeline_parallel.lazy_loader import load_pp_stage_model
from src.distributed.runtime import (
    fs_aware_load_rank,
    get_global_rank,
    get_local_rank,
    log_global_load_duration_seconds,
)
from src.distributed.tensor_parallel.parallelize_attention import (
    apply_tp_to_attention_only,
    retarget_hf_replicated_grad_hooks,
)
from src.distributed.tensor_parallel.state_dict import reject_plan_sharded_plain_params, tp_plan_shards_params
from src.distributed.tensor_parallel.tie_plan import (
    config_ties_word_embeddings,
    consistent_tied_tp_plan,
    validate_tied_pair_consistent,
)
from src.env import is_accelerate_launch
from src.hardware import is_blackwell_gpu
from src.kernels.liger.orchestrator import apply_liger_kernel
from src.kernels.lowp.mixed_precision import apply_mixed_precision_compute
from src.models.loading.checkpoint_coverage import from_pretrained_verified
from src.models.loading.config_levels import configs_declaring, set_config_field_run_scoped
from src.models.loading.dtype import configure_float32_matmul_precision, resolve_model_dtype
from src.models.loading.model_preparation import (
    apply_family_attention_patches,
    auto_load_model,
    finalize_run_model,
    resolve_auto_model_class,
)
from src.models.moe_balancing import config_has_experts
from src.models.patches.attention import (
    model_has_sinks,
    patch_flex_attention_compile,
    resolve_attn_implementation,
    validate_attn_implementation,
)
from src.models.patches.buffer_fixes import finalize_loaded_model
from src.models.patches.gpt_oss_sinks import SinksPolicy
from src.models.patches.remote_code_compat import apply_remote_code_compat_shims

if TYPE_CHECKING:
    from src.distributed.parallelism_config import ParallelismConfig

logger = get_logger(__name__)

# Precision is a run knob, not a model-config field: every loader passes the run's dtype explicitly
# and it overrides whatever the config carries, so these override keys would be accepted and then
# discarded. Both spellings, since transformers 5 renamed ``torch_dtype`` to ``dtype`` and keeps the
# old name as a deprecated alias, so a config declares both.
_REJECTED_CONFIG_OVERRIDE_KEYS = ("dtype", "torch_dtype")

# Keys the lazy loaders take as their own named parameters, so passing them again out of
# ``common_kwargs`` is a duplicate-argument TypeError.
_LAZY_LOADER_NAMED_KWARGS = ("config", "dtype", "trust_remote_code")


def _validate_launch_method_for_parallelism(pc: ParallelismConfig) -> None:
    """Reject EP/CP/TP/PP under any ``accelerate launch`` (FSDP and MULTI_GPU/DDP alike — all need
    torchrun).

    Same axis set as the trainer's ``ParallelismValidationMixin._validate_parallelism_modes``, but
    reached first, so a PP job is rejected before stage construction while the message can still name
    the launcher. The grouped-GEMM-only case is gated in ``_validate_gmm_launch_method``, which needs
    the loaded config: those wrappers activate only on a MoE.
    """
    if message := accelerate_launch_rejection(pc):
        raise ValueError(message)


def _validate_fp32_non_ep_params(pc: ParallelismConfig, model_config) -> None:
    """Reject ``fp32_non_ep_params`` for a family whose EP layer declares it unsupported.

    Class-declared (``_supports_fp32_non_ep_params``) and resolved through the model-type registry.
    Without this gate the failure is DeepEP's C++ assert at the first dispatch, after the whole
    multi-GPU load, naming neither fp32 nor the knob.
    """
    if not (pc.fp32_non_ep_params and pc.ep_size > 1):
        return
    for layer_cls in ep_layer_classes_for_config(model_config):
        if not layer_cls._supports_fp32_non_ep_params:
            family = "/".join(layer_cls.HF_MODEL_TYPES)
            raise ValueError(
                f"fp32_non_ep_params=True is not supported for {family} under "
                f"Expert Parallelism (ep_size={pc.ep_size}): the combination is unvalidated for this "
                f"family and documented to fail at the first dispatch (see its model page). Train it "
                f"experts-distributed in plain bf16 — AdamWBF16's stochastic rounding is the supported "
                f"precision floor there."
            )


def _validate_gmm_launch_method(pc: ParallelismConfig, model_config) -> None:
    """Reject grouped-GEMM MoE expert wrappers under any ``accelerate launch`` (need torchrun).

    Runs after the model config is loaded so it only fires when the wrappers would actually activate
    (MoE model). Under accelerate MULTI_GPU the wrapped experts take over accelerate's DDP management
    (custom accelerator, unrequested FSDP2 grad sync); under accelerate FSDP they are unvalidated
    against the plugin's wrapping.
    """
    if pc.is_ep_mode or pc.is_cp_mode or pc.is_tp_mode:
        return  # already rejected above for accelerate launches
    if not (pc.needs_ep_wrappers and config_has_experts(model_config) and is_accelerate_launch()):
        return

    raise ValueError(
        "Grouped GEMM for MoE experts (use_grouped_gemm=True, the default on "
        "SM90+ hardware) swaps in custom expert-layer wrappers that require "
        "'torchrun', not 'accelerate launch'. Two ways forward:\n"
        "\n"
        "  1. Launch with torchrun (keeps the grouped-GEMM speedup, no EP needed):\n"
        "       torchrun --nproc_per_node=<N> \\\n"
        "           scripts/training/sft.py <config>\n"
        "\n"
        "  2. Train this MoE under 'accelerate launch' without grouped GEMM:\n"
        "       set `use_grouped_gemm: false` in the YAML (or pass --use_grouped_gemm false)\n"
        "\n"
        "See agent-docs/optimization/grouped-gemm.md (Standalone grouped GEMM mode)."
    )


def _ensure_model_downloaded(model_name_or_path: str, revision: str | None = None) -> None:
    """Download model files to cache before all processes load.

    FS-aware: shared input FS → global rank 0 downloads, others wait; non-shared → each node's
    local rank 0. Peers wait on the c10d store (hours-scale timeout), not a NCCL barrier — a 100B+
    ``snapshot_download`` outlives the NCCL watchdog (``DIST_NCCL_TIMEOUT_MINUTES``).
    """
    with fs_aware_main_first("model_download"):
        if fs_aware_load_rank() and not os.path.isdir(model_name_or_path):
            _download_model_to_cache(model_name_or_path, revision=revision)


def _download_model_to_cache(model_name_or_path: str, revision: str | None = None) -> None:
    """Download model to cache via ``snapshot_download``."""
    logger.info(f"Downloading model to cache: {model_name_or_path}")
    snapshot_download(model_name_or_path, revision=revision)
    logger.info("Model downloaded to cache")


def _apply_config_overrides(model_config: AutoConfig, overrides: dict | None) -> None:
    """Apply model config overrides from a dict; unknown keys raise, since a skipped typo would
    train with the stock config.

    Each key is written to every config level that declares it. On a composite config (Qwen3.5/3.6,
    Gemma 4, VLM wrappers) the decoder knobs a run overrides (``output_router_logits``,
    ``router_aux_loss_coef``) exist only on ``text_config``, so a top-level-only write would either
    reject a valid override or land where the decoder never reads it.

    Precision keys (:data:`_REJECTED_CONFIG_OVERRIDE_KEYS`) are rejected rather than written: they
    exist on every config and would then be overruled by the explicit ``dtype=`` the loader passes.
    """
    if not overrides:
        return
    # Before any write, so a rejected key cannot leave the config half-overridden.
    rejected = sorted(key for key in overrides if key in _REJECTED_CONFIG_OVERRIDE_KEYS)
    if rejected:
        raise ValueError(
            f"model_config_overrides may not set {rejected}. Every loader passes the run's dtype "
            f"explicitly, resolved from the run's own precision flags, and that argument wins over "
            f"the config — so the override would be silently discarded and the run would train in "
            f"the precision it did not ask for. Set the run's dtype with `bf16: true` or "
            f"`fp16: true` in the training config (bf16 is the toolkit default), and fp32 master "
            f"weights with fp32_non_ep_params / fp32_experts."
        )
    logger.info(f"Applying model config overrides: {overrides}")
    for key, value in overrides.items():
        targets = configs_declaring(model_config, key)
        if not targets:
            raise ValueError(
                f"model_config_overrides key '{key}' does not exist on this model's config "
                f"({type(model_config).__name__}) or its text sub-config — a skipped override "
                f"would silently train with the stock value. Fix or remove the key."
            )
        for target in targets:
            setattr(target, key, value)
        logger.info(f"  Set {key} = {value} on {', '.join(type(t).__name__ for t in targets)}")


def load_distributed_model(
    model_name_or_path: str,
    parallelism_config: ParallelismConfig,
    dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
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
    **model_kwargs,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model with a ``ParallelismConfig`` (EP, CP, TP, or combinations).

    ``attn_implementation``/``model_class`` auto-resolved when None. ``reset_sinks``/``train_sinks``
    resolve to the GptOss :class:`SinksPolicy` (neutralized / live-frozen / trainable).
    ``text_only_model`` steers the auto-resolution to the text-only CausalLM class for a multimodal
    checkpoint (ignored, with a warning, when the caller pins ``model_class``).
    """
    _validate_launch_method_for_parallelism(parallelism_config)
    sinks_policy = SinksPolicy.from_flags(reset_sinks=reset_sinks, train_sinks=train_sinks)

    # DeepEP's dispatch buffer is sized for 2-byte tokens; fp32 activations trip a C++ assert in
    # `get_dispatch_buffer_size` at the first dispatch, naming neither fp32 nor the config. Keyed on
    # ep_size rather than is_ep_mode: pure ETP folds into is_ep_mode but never reaches the transport.
    if dtype == torch.float32 and parallelism_config.ep_size > 1:
        raise ValueError(
            f"fp32 training is not supported under Expert Parallelism (ep_size="
            f"{parallelism_config.ep_size}): DeepEP's dispatch buffer is sized for 2-byte tokens and "
            f"asserts on 4-byte ones. Train in bf16 (the toolkit default) — for fp32 master weights "
            f"without fp32 compute use fp32_non_ep_params / fp32_experts, which keep the "
            f"dispatched activations bf16 (except Gemma 4, whose norms re-emit fp32 activations "
            f"into the dispatch — it is refused separately)."
        )

    # Full fp32 matmul precision before any forward: the image's TF32 default degrades long-context RoPE.
    configure_float32_matmul_precision()

    apply_remote_code_compat_shims()
    _ensure_model_downloaded(model_name_or_path, revision=revision)

    model_config = AutoConfig.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code, revision=revision
    )
    _apply_config_overrides(model_config, model_config_overrides)
    # Before the process groups and the meta shell: config.json is all the sharding arithmetic
    # needs, so failing here costs a second instead of minutes.
    parallelism_config.validate_against_model_config(model_config)
    _validate_gmm_launch_method(parallelism_config, model_config)
    if parallelism_config.is_cp_mode and model_has_sinks(model_config) and sinks_policy.live:
        raise ValueError(
            f"GptOss under Context Parallelism requires reset_sinks: true (requested policy: {sinks_policy}): "
            "the Ulysses attention kernels never receive the sink column, so live sinks would give a wrong "
            "softmax normalization in every layer."
        )
    _validate_fp32_non_ep_params(parallelism_config, model_config)

    # Multimodal wrappers need AutoModelForImageTextToText (CausalLM drops the vision tower);
    # text_only_model opts into that drop deliberately (and restores config-honored aux loss).
    if model_class is None:
        model_class = resolve_auto_model_class(model_config, text_only=text_only_model)
    elif text_only_model:
        logger.warning(
            f"text_only_model=True is ignored: the caller pinned model_class={model_class} "
            f"(task-head scripts resolve their own class). The flag only affects auto-resolution."
        )

    attn_implementation = resolve_attn_implementation(
        model_config,
        attn_implementation,
        dtype,
        sinks_reset=(reset_sinks and not init_from_scratch),
    )

    # CP (Ulysses) always runs flash_attn_func; label the config with the kernel it will actually use.
    if attn_implementation == "flex_attention" and parallelism_config.is_cp_mode:
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
        # Re-validated rather than assigned: the substitute must clear the same sink-capability
        # matrix, or a run keeping live sinks (reset_sinks=false) moves to a sink-dropping kernel.
        attn_implementation = validate_attn_implementation(
            model_config, cp_fa, sinks_reset=(reset_sinks and not init_from_scratch)
        )
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
        if parallelism_config.is_ep_mode:
            patch_flex_attention_compile("EP recompilation deadlock avoidance")
        elif model_has_sinks(model_config) and reset_sinks:
            patch_flex_attention_compile("FSDP2 + trainable attention sinks NaN gradient fix")

    apply_family_attention_patches(model_config, attn_implementation)

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
        model_name_or_path, trust_remote_code=trust_remote_code, revision=revision
    )
    # Training pads right. Collators that don't set their own side inherit this, so a left-padding
    # tokenizer (GLM-4.7-Flash, Gemma 4) would make the packing collator read leading pads as an
    # attended document and drop the last real document's labels.
    tokenizer.padding_side = "right"

    common_kwargs = dict(
        config=model_config,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
        **model_kwargs,
    )
    if revision is not None:
        common_kwargs["revision"] = revision
    if quantization_config is not None:
        # Also gated on MoE: a dense QLoRA run routing to DDP (or to CP, which keeps Params4bit) must pass.
        if (
            parallelism_config.is_ep_mode
            or parallelism_config.is_tp_mode
            or parallelism_config.is_pp_mode
            or (parallelism_config.needs_ep_wrappers and config_has_experts(model_config))
        ):
            raise ValueError(
                "Quantized loading (QLoRA / bitsandbytes) is not supported with EP / TP / PP "
                "or grouped-GEMM MoE loaders: they materialize plain de-quantized weights, so "
                "bitsandbytes Params4bit are lost and PEFT's 4-bit adapter dispatch fails (PP "
                "rejects PEFT outright). Use QLoRA with standard DDP/FSDP (accelerate launch), "
                "plain LoRA (no quantization) for EP/TP, or QLoRA + CP (which preserves "
                "quantization)."
            )
        common_kwargs["quantization_config"] = quantization_config
        logger.info(f"Using quantization config: {quantization_config}")

    if init_from_scratch:
        # Identical per-rank seeds give consistent FSDP2 shards; EP/TP/CP would need a sharded random-init.
        if (
            parallelism_config.is_ep_mode
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
        common_kwargs["_init_from_scratch"] = True
        logger.info("init_from_scratch=True — building model with RANDOM weights from config (no checkpoint load).")

    model = _dispatch_model_loading(model_name_or_path, parallelism_config, model_class, common_kwargs)

    # Where the weights were read from, or None for a random init: the resume loaders key their
    # "already holds these weights" skip on this, since ``_name_or_path`` survives init_from_scratch.
    setattr(model, LOADED_WEIGHTS_FROM_ATTR, None if init_from_scratch else model_name_or_path)

    finalize_run_model(
        model,
        model_config,
        sinks_policy=sinks_policy,
        attn_implementation=attn_implementation,
        sinks_pretrained=not init_from_scratch,
    )

    if parallelism_config.lowp_precision != "bf16":
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
        apply_mixed_precision_compute(
            model,
            precision=parallelism_config.lowp_precision,
            apply_dense_mlp=parallelism_config.lowp_apply_dense_mlp,
            apply_moe_experts=parallelism_config.lowp_apply_moe_experts,
            keep_first_n_blocks=parallelism_config.lowp_keep_first_blocks,
            keep_last_n_blocks=parallelism_config.lowp_keep_last_blocks,
            expert_weight_cacheable=not parallelism_config.experts_fsdp_managed,
        )

    warm_attention_kernels(model, dtype=dtype)

    return model, tokenizer


def _dispatch_model_loading(
    model_name_or_path: str,
    pc: ParallelismConfig,
    model_class,
    common_kwargs: dict,
) -> PreTrainedModel:
    """Pick the appropriate per-mode loader based on ``pc``."""
    local_rank = get_local_rank()
    model_config = common_kwargs.get("config")
    is_moe = config_has_experts(model_config)

    # First, because a stage holds only its layer slice: any other loader materializes pp_size x too much.
    if pc.is_pp_mode:
        return _load_pp_stage_model(model_name_or_path, pc, model_class, common_kwargs, is_moe)

    if pc.is_ep_tp_mode:
        return _load_ep_tp_model(model_name_or_path, pc, model_class, common_kwargs, local_rank)

    if pc.is_tp_mode:
        if is_moe:
            return _load_tp_moe_model(model_name_or_path, pc, model_class, common_kwargs, local_rank)
        return _load_tp_model(model_name_or_path, pc, model_class, common_kwargs)

    if pc.is_ep_mode and pc.is_cp_mode:
        return _load_ep_cp_model(model_name_or_path, pc, model_class, common_kwargs)

    if pc.is_ep_mode:
        return _load_ep_model(model_name_or_path, pc, model_class, common_kwargs)

    if pc.is_cp_mode:
        return _load_cp_model(model_name_or_path, pc, model_class, common_kwargs)

    # is_ep_mode returned above, so nothing below distributes experts across ranks.
    return _load_undistributed_model(
        model_name_or_path,
        pc,
        model_class,
        common_kwargs,
        local_rank,
        ep_wrappers=pc.needs_ep_wrappers and is_moe,
    )


def _sequential_load_to_cuda(
    model_name_or_path: str,
    model_class,
    local_rank: int,
    max_concurrent: int,
    common_kwargs: dict,
) -> PreTrainedModel:
    """Load to CPU one rank at a time (low CPU peak), move to this rank's GPU, then
    free CPU memory. Shared by the EP+TP sequential fallback and the TP-MoE loader."""
    with sequential_load_within_node(max_concurrent=max_concurrent):
        model = from_pretrained_verified(
            model_class,
            model_name_or_path,
            device_map="cpu",
            **common_kwargs,
        )
        model = model.to(f"cuda:{local_rank}")
        gc.collect()
        torch.cuda.empty_cache()
    finalize_loaded_model(model)
    return model


def _from_pretrained_on_local_gpu(
    model_name_or_path: str,
    model_class,
    local_rank: int,
    max_concurrent: int,
    common_kwargs: dict,
) -> PreTrainedModel:
    """``from_pretrained`` straight onto this rank's GPU, one rank at a time per node.

    With ``_init_from_scratch`` in ``common_kwargs``, builds from config with random weights instead.
    """
    if common_kwargs.pop("_init_from_scratch", False):
        config = common_kwargs.get("config")
        # from_config materializes on CPU before .to(cuda), so it needs the same concurrency gate.
        with sequential_load_within_node(max_concurrent=max_concurrent):
            model = model_class.from_config(
                config,
                dtype=common_kwargs.get("dtype"),
                attn_implementation=common_kwargs.get("attn_implementation"),
                trust_remote_code=common_kwargs.get("trust_remote_code", True),
            )
            model = model.to(f"cuda:{local_rank}")
    else:
        with sequential_load_within_node(max_concurrent=max_concurrent):
            ddp_kwargs = {"device_map": {"": local_rank}, **common_kwargs}
            model = from_pretrained_verified(model_class, model_name_or_path, **ddp_kwargs)
    finalize_loaded_model(model)
    return model


def _apply_attention_only_tp(model: PreTrainedModel, rank: int, tp_size: int, dp_size: int) -> None:
    """Shard attention via DTensor TP (MoE experts untouched), stamping the mesh onto the model.
    Shared by the EP+TP and TP-MoE loaders.

    ``apply_tp_to_attention_only`` already raises (with the model's actual attention classes) when
    ``tp_size > 1`` shards nothing.
    """
    device_mesh = create_dp_tp_mesh(tp_size=tp_size, dp_size=dp_size)
    num_tp_modules = apply_tp_to_attention_only(model, device_mesh)
    logger.info(f"[Rank {rank}] TP applied to {num_tp_modules} attention modules")
    model._device_mesh = device_mesh


def _lazy_loader_passthrough(common_kwargs: dict) -> dict:
    """``common_kwargs`` minus the keys the lazy loaders already take by name."""
    return {k: v for k, v in common_kwargs.items() if k not in _LAZY_LOADER_NAMED_KWARGS}


def _apply_ep_wrappers(model: PreTrainedModel, ep_config) -> PreTrainedModel:
    """Replace MoE blocks with EP/grouped-GEMM wrappers and create the DeepEP buffers.
    Shared by the EP+TP, TP-MoE and grouped-GEMM loaders."""
    model = patch_moe_model_for_ep(model, ep_config)
    create_ep_buffers(model)
    return model


def _load_pp_stage_model(
    model_name_or_path: str,
    pc: ParallelismConfig,
    model_class,
    common_kwargs: dict,
    is_moe: bool,
) -> PreTrainedModel:
    """Load this pipeline stage's decoder layers only, streamed per rank from safetensors.

    Returns a normal ``*ForCausalLM`` whose layer list is already this stage's slice re-based to 0;
    ``build_pipeline_stage`` finishes the split in the trainer. MoE stages get their EP /
    grouped-GEMM wrappers here, exactly as the EP loaders do.
    """
    rank = get_global_rank()
    t_start_wall = time.time()
    # Built before the checkpoint probe below: create_ep_config() issues new_group on every rank, so
    # a per-rank filesystem decision must not come first.
    ep_config = pc.create_ep_config() if (is_moe and pc.needs_ep_wrappers) else None  # is_ep_mode implies it

    local_dir = resolve_hub_or_local_dir(model_name_or_path, revision=common_kwargs.get("revision"))
    # Same gate the EP loaders run, rank-0-decided for the reason decide_lazy_loadable documents.
    reject_ep_sharded_checkpoint(local_dir, model_name_or_path)

    loadable = decide_lazy_loadable(local_dir, lazy_loader_supports_checkpoint)
    if not loadable:
        raise ValueError(
            f"Pipeline parallelism needs a safetensors checkpoint the stage-aware loader can read, "
            f"but {model_name_or_path!r} has none. Falling back to a whole-model load would make "
            f"every rank hold all pp_size={pc.pp_size} stages — the ceiling PP exists to remove. "
            f"Either the checkpoint is not safetensors (convert it), or this family's EP layer "
            f"declares _supports_lazy_loading=False / nests its expert index under an extra module "
            f"(pre-5.14 Zaya checkpoints), in which case the model cannot use PP at all (see "
            f"src/distributed/expert_parallel/layers/ and agent-docs/parallelism/pipeline-parallelism.md)."
        )
    if local_dir is None:
        raise RuntimeError(
            f"[Rank {rank}] Global rank 0 resolved a stage-loadable checkpoint for "
            f"{model_name_or_path!r} but this rank could not resolve it to a local directory — a "
            f"partially populated cache on a non-shared filesystem. Pre-download the checkpoint on "
            f"every node."
        )

    model = load_pp_stage_model(
        local_dir,
        pc.pp_rank,
        pc.pp_size,
        config=common_kwargs["config"],
        pp_split=pc.pp_split,
        ep_config=ep_config,
        dtype=common_kwargs.get("dtype"),
        trust_remote_code=common_kwargs.get("trust_remote_code", True),
        model_class=model_class,
        **_lazy_loader_passthrough(common_kwargs),
    )

    # Deliberately not in a ``finally``: this collective would turn one rank's traceback into a
    # peer-wide timeout.
    log_global_load_duration_seconds(
        tag="PP",
        method="pp_stage_lazy",
        t_start_wall=t_start_wall,
        t_end_wall=time.time(),
    )
    logger.info(f"Model loaded for PP stage {pc.pp_rank}/{pc.pp_size} (ep={pc.ep_size}, dp={pc.data_parallel_size})")
    return model


def _load_ep_tp_model(
    model_name_or_path: str,
    pc: ParallelismConfig,
    model_class,
    common_kwargs: dict,
    local_rank: int,
) -> PreTrainedModel:
    """Load model with EP+TP: attention sharded via DTensor, experts distributed via DeepEP.

    ``ep_lazy_loading`` uses lazy safetensors for experts then applies TP post-hoc; falls back to
    sequential ``from_pretrained`` if unavailable.
    """
    rank = get_global_rank()
    t_start_wall = time.time()
    load_method = "ep_tp_huggingface"
    logger.info(f"[Rank {rank}] Loading model for EP+TP mode...")
    logger.info(f"  EP size: {pc.ep_size}, TP size: {pc.tp_size}, DP size: {pc.data_parallel_size}")

    # Built before the branch: create_ep_config() issues new_group on every rank while the lazy-path
    # predicate below is a per-rank FS probe, so branching first orders the groups per rank (deadlock).
    ep_config = pc.create_ep_config()

    # Parity with load_ep_model: an ep_sharded dir would otherwise reach the lazy loader, whose keys
    # match no expert pattern, leaving every expert on meta. Rank-0-decided like the lazy gate below;
    # both sit in front of the same world collectives.
    sharded_probe = resolve_hub_or_local_dir(model_name_or_path, revision=common_kwargs.get("revision"))
    reject_ep_sharded_checkpoint(sharded_probe, model_name_or_path)

    local_dir = sharded_probe if pc.ep_lazy_loading else None
    use_lazy = decide_lazy_loadable(local_dir, lazy_loader_supports_checkpoint)

    if use_lazy:
        if local_dir is None:
            raise RuntimeError(
                f"[Rank {rank}] Global rank 0 resolved a lazy-loadable safetensors checkpoint for "
                f"{model_name_or_path!r} but this rank could not resolve it to a local directory — a "
                f"partially populated cache on a non-shared filesystem. Pre-download the checkpoint "
                f"on every node, or pass ep_lazy_loading=False."
            )
        logger.info(f"[Rank {rank}] Using lazy safetensors loader for EP+TP")
        model = load_ep_model_lazy(
            local_dir,
            ep_config,
            config=common_kwargs["config"],
            dtype=common_kwargs.get("dtype"),
            trust_remote_code=common_kwargs.get("trust_remote_code", True),
            model_class=model_class,
            **_lazy_loader_passthrough(common_kwargs),
        )
        _apply_attention_only_tp(model, rank, pc.tp_size, pc.data_parallel_size)
        load_method = "ep_tp_lazy"
    else:
        if pc.ep_lazy_loading:
            logger.info(f"[Rank {rank}] Lazy path unavailable, falling back to sequential loading")
        model = _sequential_load_to_cuda(
            model_name_or_path,
            model_class,
            local_rank,
            pc.max_concurrent_loading,
            common_kwargs,
        )
        _apply_attention_only_tp(model, rank, pc.tp_size, pc.data_parallel_size)
        model = _apply_ep_wrappers(model, ep_config)

    # Deliberately not in a ``finally``: this collective would turn one rank's traceback into a
    # peer-wide timeout.
    log_global_load_duration_seconds(
        tag="EP+TP",
        method=load_method,
        t_start_wall=t_start_wall,
        t_end_wall=time.time(),
    )
    logger.info(f"Model loaded with EP+TP (tp={pc.tp_size}, ep={pc.ep_size}, dp={pc.data_parallel_size})")
    return model


def _load_tp_model(
    model_name_or_path: str,
    pc: ParallelismConfig,
    model_class,
    common_kwargs: dict,
) -> PreTrainedModel:
    """Load model with TP: HF-native ``tp_plan`` weight sharding."""
    tp_size, dp_size = pc.tp_size, pc.data_parallel_size

    device_mesh = create_dp_tp_mesh(tp_size=tp_size, dp_size=dp_size)
    tp_mesh = get_tp_submesh(device_mesh)
    logger.info(f"Created device mesh ({'DP x TP' if dp_size > 1 else 'TP only'}): {device_mesh}")

    # No max_concurrent_loading throttle here, unlike the other loaders: this load ends in
    # ``tie_weights``, whose equality check on a tied pair of DTensors is an all-reduce on the default
    # process group, which a rank-serialized region deadlocks. The throttle would buy little anyway:
    # transformers streams the checkpoint key by key straight onto each rank's GPU.
    with consistent_tied_tp_plan(model_class, common_kwargs.get("config")):
        model = from_pretrained_verified(
            model_class,
            model_name_or_path,
            # The 5.16 spelling of tp_plan="auto" (removal slated for 5.18); with an explicit
            # device_mesh the mesh is authoritative for the load. The config's derived tp_size is
            # read only by the model's own save_pretrained, which no toolkit path calls on a live
            # TP model: every TP save goes through save_tp_model.
            distributed_config=DistributedConfig(tp_plan="auto"),
            device_mesh=tp_mesh,
            **common_kwargs,
        )
    # tp_plan="auto" resolves to an empty plan for architectures shipping no base_model_tp_plan
    # (Qwen3-VL); transformers only warns, and the run becomes tp_size replicas at 1/tp_size
    # throughput. Checked against the applied plan, so a plan covering no parameter (or covering them
    # only with transform-only styles) is named directly.
    applied_plan = getattr(model, "_tp_plan", None) or {}
    if tp_size > 1 and not any(tp_plan_shards_params(name, applied_plan) for name, _ in model.named_parameters()):
        raise ValueError(
            f"tensor_parallel_size={tp_size} sharded ZERO parameters: this architecture ships no "
            f"base_model_tp_plan, so TP would run {tp_size} full replicas at 1/{tp_size} throughput. "
            f"Drop tensor_parallel_size for this model."
        )
    if tp_size > 1:
        reject_plan_sharded_plain_params(model, applied_plan)
    if config_ties_word_embeddings(model.config):
        validate_tied_pair_consistent(model, applied_plan)
        if tp_size > 1 and not isinstance(model.get_input_embeddings().weight, DTensor):
            logger.warning(
                "This architecture plans only one end of its tied embedding/head pair, so the pair "
                "stays REPLICATED: budget a full vocab-sized weight, gradient and optimizer state on "
                "every TP rank."
            )
    model._device_mesh = device_mesh
    retarget_hf_replicated_grad_hooks(model)
    # After the tie check above, since the seam re-ties and would mask a load that left the pair untied.
    finalize_loaded_model(model)

    logger.info(f"Model loaded with TP (tp_size={tp_size})")
    if dp_size > 1:
        logger.info(f"  FSDP2 will be used for DP (dp_size={dp_size})")
    return model


def _load_tp_moe_model(
    model_name_or_path: str,
    pc: ParallelismConfig,
    model_class,
    common_kwargs: dict,
    local_rank: int,
) -> PreTrainedModel:
    """Load MoE model with TP: attention sharded via DTensor, MoE experts replicated.

    HF's ``tp_plan="auto"`` wrongly shards MoE expert biases, so apply TP to attention only and
    optionally grouped-GEMM wrappers for the MoE layers.
    """
    rank = get_global_rank()
    tp_size, dp_size = pc.tp_size, pc.data_parallel_size
    logger.info(f"[Rank {rank}] Loading MoE model for TP-only mode (attention-only TP)...")
    logger.info(f"  TP size: {tp_size}, DP size: {dp_size}")

    model = _sequential_load_to_cuda(
        model_name_or_path,
        model_class,
        local_rank,
        pc.max_concurrent_loading,
        common_kwargs,
    )
    _apply_attention_only_tp(model, rank, tp_size, dp_size)

    if pc.needs_ep_wrappers:
        model = _apply_ep_wrappers(model, pc.create_ep_config())
        logger.info("Applied grouped GEMM wrappers to MoE layers")

    logger.info(f"Model loaded with TP on MoE (tp={tp_size}, dp={dp_size})")
    return model


def _load_ep_cp_model(
    model_name_or_path: str,
    pc: ParallelismConfig,
    model_class,
    common_kwargs: dict,
) -> PreTrainedModel:
    """Load model with EP+CP: experts distributed + sequences split."""
    model = load_model_for_ep_cp(
        model_name_or_path,
        pc.create_ep_config(),
        pc.create_cp_config(),
        model_class=model_class,
        max_concurrent_loading=pc.max_concurrent_loading,
        lazy=pc.ep_lazy_loading,
        **common_kwargs,
    )
    logger.info(f"Model loaded with EP+CP (ep={pc.ep_size}, cp={pc.cp_size})")
    return model


def _load_ep_model(
    model_name_or_path: str,
    pc: ParallelismConfig,
    model_class,
    common_kwargs: dict,
) -> PreTrainedModel:
    """Load model with EP: all-to-all expert routing."""
    model = load_ep_model(
        model_name_or_path,
        pc.create_ep_config(),
        model_class=model_class,
        max_concurrent_loading=pc.max_concurrent_loading,
        lazy=pc.ep_lazy_loading,
        **common_kwargs,
    )
    scope_str = "node-local" if pc.is_node_local_ep else "cross-node"
    logger.info(f"Model loaded with EP (ep_size={pc.ep_size}, {scope_str})")
    return model


def _load_cp_model(
    model_name_or_path: str,
    pc: ParallelismConfig,
    model_class,
    common_kwargs: dict,
) -> PreTrainedModel:
    """Load model with CP: Ulysses attention (+ grouped-GEMM expert wrappers on a MoE)."""
    # Else a MoE under pure CP pays the Liger swiglu/geglu force-off while its experts keep the
    # stock per-expert loop.
    needs_wrappers = pc.needs_ep_wrappers and config_has_experts(common_kwargs.get("config"))
    model = load_model_for_cp(
        model_name_or_path,
        pc.create_cp_config(),
        model_class=model_class,
        max_concurrent_loading=pc.max_concurrent_loading,
        ep_config=pc.create_ep_config() if needs_wrappers else None,
        **common_kwargs,
    )
    logger.info(f"Model loaded with CP (cp_size={pc.cp_size}, grouped_gemm_experts={needs_wrappers})")
    return model


def _load_undistributed_model(
    model_name_or_path: str,
    pc: ParallelismConfig,
    model_class,
    common_kwargs: dict,
    local_rank: int,
    *,
    ep_wrappers: bool,
) -> PreTrainedModel:
    """Load a whole model onto this rank — plain DDP/FSDP data parallelism, no expert distribution.

    ``ep_wrappers`` adds the EP MoE wrappers anyway: with ``ep_group_size == 1`` they distribute
    nothing and exist for the grouped-GEMM expert compute (and, under ``fsdp_shard_ep1_experts``,
    for the FSDP-managed ep1 expert shard).
    """
    model = _from_pretrained_on_local_gpu(
        model_name_or_path,
        model_class,
        local_rank,
        pc.max_concurrent_loading,
        common_kwargs,
    )
    if ep_wrappers:
        model = _apply_ep_wrappers(model, pc.create_ep_config())
    mode = "Grouped GEMM wrappers (no EP distribution)" if ep_wrappers else "standard DDP"
    logger.info(f"Model loaded with {mode}")
    return model


def load_model_from_pretrained(
    model,
    args=None,
    model_cls=None,
):
    """Load a model from a pretrained path string, resolving its dtype; returns ``(model, model_id)``.

    An already-instantiated model must have ``model_init_kwargs`` unset. ``model_cls`` None → resolved
    via `resolve_auto_model_class()`.
    """
    if isinstance(model, str):
        model_id = model
        model_init_kwargs = getattr(args, "model_init_kwargs", None) or {}
        resolve_model_dtype(model_init_kwargs)

        # Applied run-scoped after the load rather than through from_pretrained: a kwarg write lands
        # in the artifact's config, so the exported config.json would default the KV cache off for
        # downstream consumers. An explicit use_cache in model_init_kwargs still wins.
        run_scoped_cache_off = (
            args is not None
            and getattr(args, "gradient_checkpointing", False)
            and "use_cache" not in model_init_kwargs
        )

        model = auto_load_model(model, model_class=model_cls, **model_init_kwargs)
        finalize_loaded_model(model)
        if run_scoped_cache_off:
            set_config_field_run_scoped(model.config, "use_cache", False)
        return model, model_id

    model_id = getattr(model.config, "_name_or_path", None)
    if args is not None and getattr(args, "model_init_kwargs", None) is not None:
        raise ValueError(
            "model_init_kwargs can only be used when model is a string path, but the model is already instantiated."
        )
    return model, model_id
