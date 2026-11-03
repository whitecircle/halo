"""The refusals a load runs before it reads a weight, since ``ParallelismConfig`` is model-blind and these faults
otherwise surface minutes and several collectives into a job."""

from __future__ import annotations

import copy

import torch

from halo.distributed.context_parallel.model_rules import reject_unsupported_ulysses_model
from halo.distributed.expert_parallel.deepep.capacity import (
    ep_dispatch_capacity,
    padded_wire_hidden,
    reject_oversized_dispatch,
)
from halo.distributed.parallelism_config import ParallelismConfig
from halo.distributed.parallelism_rules import reject_custom_parallelism_under_accelerate
from halo.distributed.tensor_parallel.parallelize_attention import (
    reject_unshardable_tp_model,
    validate_tp_head_divisibility,
)
from halo.env import is_accelerate_launch
from halo.model_loading.lazy_safetensors.meta_shell import MetaShellBuildError, instantiate_on_meta
from halo.model_loading.parallel.model_source import ModelSource
from halo.model_loading.parallel.tp import uses_attention_only_tp
from halo.models.attention.gpt_oss_sinks import SinksPolicy, reject_trainable_sinks_without_sinks, uses_gpt_oss_sinks
from halo.models.config_access import text_config
from halo.models.context_window import context_window_from_config
from halo.models.external_tables import EXTERNAL_TABLES_STAMP
from halo.models.moe.config_fields import (
    has_routed_experts,
    resolve_expert_count,
    resolve_expert_ffn_width,
    resolve_router_topk,
)
from halo.models.moe.family_specs import find_moe_family_spec_for_config


def _divisors_up_to(value: int, ceiling: int) -> list[int]:
    """Divisors of ``value`` in ``2..ceiling``, the sizes a rejection may suggest; 1 means no expert
    parallelism, which is no answer to "pick a working ep_size"."""
    return [d for d in range(2, min(value, ceiling) + 1) if value % d == 0]


def _reject_undispatchable_token_budget(pc: ParallelismConfig, model_config) -> None:
    """Refuse a per-rank token budget the first MoE dispatch would refuse, sized off
    ``ep_rows_per_device × max_length``. ``max_length: null`` resolves against ``config.json``, the largest budget a
    run can declare, so it cannot switch the gate off."""
    if pc.ep_size <= 1 or pc.ep_rows_per_device <= 0:
        return
    text = text_config(model_config)
    hidden = getattr(text, "hidden_size", 0)
    num_topk = resolve_router_topk(text)
    if not hidden or not num_topk:
        return
    max_length = pc.ep_declared_max_length or context_window_from_config(model_config)
    if not max_length:
        # max_length: null on a config that states no window leaves nothing to judge against.
        return
    # CP is the one axis that shrinks a rank's token count; dividing by no other keeps this an upper bound. The rows
    # are one forward's, so one microbatch's under PP.
    budget = pc.ep_rows_per_device * max_length
    tokens_per_rank = budget // pc.cp_size
    capacity = ep_dispatch_capacity(tokens_per_rank)
    try:
        reject_oversized_dispatch(
            capacity,
            num_topk=num_topk,
            padded_hidden=padded_wire_hidden(hidden),
            spans_domains=pc.requires_rdma,
        )
    except ValueError as exc:
        cp_note = f" ({budget} / context_parallel_size={pc.cp_size})" if pc.cp_size > 1 else ""
        length_note = "" if pc.ep_declared_max_length else " (no length cap → the model's context window)"
        rows_lever = "raise pipeline_microbatches" if pc.pp_size > 1 else "lower per_device_train_batch_size"
        raise ValueError(
            f"Either {rows_lever} or lower the row-length cap (max_length, or "
            f"max_prompt_length + max_completion_length where those two bound a row) — or, for the "
            f"cross-node Gin ceiling only, raise HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK. This run's "
            f"declared token budget cannot be dispatched: {pc.ep_rows_per_device} row(s)/forward × "
            f"{max_length} tokens/row{length_note} gives {tokens_per_rank} tokens/rank"
            f"{cp_note}, sized to a {capacity}-token DeepEP arena on an EP group of "
            f"{pc.ep_group_size} rank(s) (ep_scope={pc.ep_scope}, nvlink_domain_size="
            f"{pc.nvlink_domain_size}, "
            f"{'cross-domain — Gin/RDMA' if pc.requires_rdma else 'within one NVLink domain'}). "
            f"{exc}"
        ) from exc


def reject_unsupported_run(pc: ParallelismConfig, dtype: torch.dtype) -> None:
    """Refuse EP/CP/TP/PP under ``accelerate launch`` (the trainer's own axis set, reached first so the
    message can name the launcher), and fp32 training under Expert Parallelism."""
    reject_custom_parallelism_under_accelerate(pc)
    # DeepEP's dispatch buffer is sized for 2-byte tokens, so fp32 trips a raw C++ assert naming neither fp32 nor the
    # config. Keyed on ep_size, as pure ETP folds into has_ep_group but never reaches the transport.
    if dtype == torch.float32 and pc.ep_size > 1:
        raise ValueError(
            f"fp32 training is not supported under Expert Parallelism "
            f"(expert_parallel_size={pc.ep_size}): DeepEP's dispatch buffer is sized "
            f"for 2-byte tokens and asserts on 4-byte ones. Train in bf16 (the toolkit default) — "
            f"for fp32 master weights without fp32 compute use fp32_non_ep_params / fp32_experts, "
            f"which keep the dispatched activations bf16 (except Gemma 4, whose norms re-emit fp32 "
            f"activations into the dispatch — it is refused separately)."
        )


def reject_unsupported_model(
    pc: ParallelismConfig, model_config, *, sinks_policy: SinksPolicy, wraps_experts: bool
) -> None:
    """Refuse what this model's ``config.json`` cannot take under ``pc``, before the process groups."""
    reject_trainable_sinks_without_sinks(model_config, sinks_policy)
    # Read only by the EP layers; a model with none trains at the run dtype whatever these say.
    if (pc.ep_fp32_router or pc.ep_fp32_experts) and not wraps_experts:
        raise ValueError(
            "fp32_router and fp32_experts are read by the EP layers, and this run builds none: the model has no "
            "routed experts with an EP layer, or expert_parallel_size is 1 with use_grouped_gemm off. Remove them, or "
            "use fp32_non_ep_params for FP32 masters on the rest of the model."
        )
    if pc.ep_size > 1:
        num_experts = resolve_expert_count(model_config)
        if num_experts and num_experts % pc.ep_size != 0:
            raise ValueError(
                f"expert_parallel_size={pc.ep_size} does not divide this model's {num_experts} "
                f"routed experts: DeepEP dispatch maps tokens to ranks by a uniform expert-to-rank "
                f"division, so an uneven assignment would route tokens to the wrong experts or "
                f"drop them. expert_parallel_size values dividing {num_experts} that fit this job: "
                f"{', '.join(str(e) for e in _divisors_up_to(num_experts, pc.stage_world_size))} "
                f"(the topology gates narrow this further — a single-domain job takes 2 or the "
                f"domain width)."
            )
    if pc.expert_tp_size > 1:
        width = resolve_expert_ffn_width(model_config)
        if width and width % pc.expert_tp_size != 0:
            raise ValueError(
                f"expert_tensor_parallel_size={pc.expert_tp_size} does not divide this model's "
                f"expert FFN width ({width}): expert-TP shards that dimension, so the shards would "
                f"be ragged. Use an expert_tensor_parallel_size dividing {width}."
            )
    if pc.fp32_non_ep_params and pc.experts_fsdp_managed and has_routed_experts(model_config):
        # At config time and rank-symmetric, as under PP a stage may hold no MoE layer: a per-rank walk would raise on
        # MoE stages while dense stages hang in the wrap.
        remedy = (
            "drop fp32_non_ep_params (fsdp_shard_ep1_experts=false is refused under PP/TP/CP)"
            if pc.pp_size > 1 or pc.tp_size > 1 or pc.cp_size > 1
            else "set fsdp_shard_ep1_experts: false (see agent-docs/models/bailing.md#ling-30) or drop fp32_non_ep_params"
        )
        raise ValueError(
            f"fp32_non_ep_params=True cannot combine with fsdp_shard_ep1_experts=True at "
            f"expert_parallel_size=1 on a MoE model: dense params go fp32 while the "
            f"FSDP-managed experts stay bf16 in the same shard group, and FSDP2 asserts "
            f"'uniform original parameter dtype' at the first forward. Remedy: {remedy}."
        )
    # Dense TP goes through HF's tp_plan="auto", which validates no head count of its own, so
    # without this it fails on the first forward's reshape, after the full load.
    validate_tp_head_divisibility(text_config(model_config), pc.tp_size)
    _reject_undispatchable_token_budget(pc, model_config)

    # Under accelerate MULTI_GPU the wrapped experts would silently hijack accelerate's DDP management, and under
    # accelerate FSDP they are unvalidated against the plugin's wrapping.
    if wraps_experts and is_accelerate_launch() and not (pc.has_ep_group or pc.is_cp_mode or pc.is_tp_mode):
        raise ValueError(
            "Grouped GEMM for MoE experts (use_grouped_gemm=True, the default on "
            "SM90+ hardware) swaps in custom EP layers that require "
            "'torchrun', not 'accelerate launch'. Two ways forward:\n"
            "\n"
            "  1. Launch with torchrun (keeps the grouped-GEMM speedup, no EP needed):\n"
            "       torchrun --nproc_per_node=<N> \\\n"
            "           <training script> <config>\n"
            "\n"
            "  2. Train this MoE under 'accelerate launch' without grouped GEMM:\n"
            "       set `use_grouped_gemm: false` in the YAML (or pass --use_grouped_gemm=false)\n"
            "\n"
            "See agent-docs/optimization/grouped-gemm.md (Standalone grouped GEMM mode)."
        )
    if pc.is_cp_mode and uses_gpt_oss_sinks(model_config) and sinks_policy.live:
        raise ValueError(
            f"GptOss under Context Parallelism requires reset_sinks: true (requested policy: {sinks_policy}): "
            "the Ulysses attention kernels never receive the sink column, so live sinks would give a wrong "
            "softmax normalization in every layer."
        )
    # Declared on the family spec (``supports_fp32_non_ep_params``); else DeepEP's C++ assert fires at first dispatch.
    spec = find_moe_family_spec_for_config(model_config)
    if pc.fp32_non_ep_params and pc.ep_size > 1 and spec is not None and not spec.supports_fp32_non_ep_params:
        family = "/".join(spec.model_types)
        raise ValueError(
            f"fp32_non_ep_params=True is not supported for {family} under Expert Parallelism "
            f"(expert_parallel_size={pc.ep_size}): the combination is unvalidated for this family "
            f"and documented to fail at the first dispatch (see its model page). Train it "
            f"experts-distributed in plain bf16 — AdamWBF16's stochastic rounding is the supported "
            f"precision floor there."
        )


def reject_unsupported_load(
    pc: ParallelismConfig, dtype: torch.dtype, *, quantization_config, init_from_scratch: bool, wraps_experts: bool
) -> None:
    """Refuse a QLoRA, from-scratch or low-precision load ``pc`` and ``dtype`` cannot take, before any weight read."""
    if quantization_config is not None and (pc.has_ep_group or pc.is_tp_mode or pc.is_pp_mode or wraps_experts):
        # Also gated on MoE: a dense QLoRA run routing to DDP (or to CP, which keeps Params4bit) must pass.
        raise ValueError(
            "Quantized loading (QLoRA / bitsandbytes) is not supported with EP / TP / PP "
            "or grouped-GEMM MoE loaders: they materialize plain de-quantized weights, so "
            "bitsandbytes Params4bit are lost and PEFT's 4-bit adapter dispatch fails (PP "
            "rejects PEFT outright). For QLoRA use plain data parallelism (on a MoE model with "
            "`use_grouped_gemm: false`, under torchrun and accelerate launch alike) or CP on a "
            "dense model, which preserves quantization. Under EP/TP use plain LoRA (no "
            "quantization)."
        )
    if init_from_scratch:
        # Identical per-rank seeds give consistent FSDP2 shards; EP/TP/CP would need a sharded random-init.
        if pc.has_ep_group or pc.is_tp_mode or pc.is_cp_mode or pc.is_expert_tp_mode or pc.is_pp_mode:
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
    if pc.lowp.enabled:
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


def reject_random_init_tables(model_config, *, init_from_scratch: bool) -> None:
    """Refuse a random-weight build of a model whose config builds an external table: the table is frozen and comes
    only from a checkpoint, so its attach would pair a pretrained table with random weights."""
    if init_from_scratch and getattr(model_config, EXTERNAL_TABLES_STAMP, False):
        raise ValueError(
            f"init_from_scratch builds {getattr(model_config, 'model_type', 'this model')} with random weights, but "
            f"its external tables are frozen and come only from the checkpoint its config names, so the run would "
            f"train random weights against a pretrained table. Train from the checkpoint: drop init_from_scratch."
        )


def reject_unsupported_structure(
    pc: ParallelismConfig,
    source: ModelSource,
    model_class,
    model_config,
    attn_implementation: str,
    dtype: torch.dtype,
    *,
    trust_remote_code: bool,
) -> None:
    """Refuse a module tree CP or selective TP cannot run, on a meta shell before any weight read. Config-only, so
    every rank reaches the same verdict."""
    attention_only_tp = uses_attention_only_tp(pc, model_config)
    if not (pc.is_cp_mode or attention_only_tp):
        return
    # On a copy: the build writes its dtype and attention onto the config it is handed. Built with the run's
    # attention, since remote code may pick its attention class by it (Bailing's per-backend classes).
    try:
        shell = instantiate_on_meta(
            source.local_dir,
            model_class,
            copy.deepcopy(model_config),
            dtype,
            trust_remote_code=trust_remote_code,
            config_only=True,
            attn_implementation=attn_implementation,
            revision=source.revision,
        )
    except MetaShellBuildError as e:
        mode = "Context Parallelism" if pc.is_cp_mode else "Attention-only tensor parallelism"
        raise RuntimeError(
            f"{e} {mode} checks the module tree on this shell before any weight is read, so the architecture "
            f"cannot run under {mode}."
        ) from e
    if pc.is_cp_mode:
        reject_unsupported_ulysses_model(shell, pc.cp_size)
    if attention_only_tp:
        reject_unshardable_tp_model(shell, pc.tp_size)
