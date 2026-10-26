"""The refusals a load runs before it reads a weight: the run against its launcher and dtype, the model's
``config.json`` against the parallelism shape, then the quantized, from-scratch and low-precision options.

``ParallelismConfig`` is model-blind, so these rules are otherwise reached only minutes and several
collectives into a job (expert assignment after the EP groups exist, the first DeepEP dispatch).
"""

from __future__ import annotations

import torch

from src.distributed.expert_parallel.deepep.capacity import (
    ep_dispatch_capacity,
    padded_wire_hidden,
    reject_oversized_dispatch,
)
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.parallelism_rules import reject_accelerate_launch
from src.distributed.tensor_parallel.parallelize_attention import validate_tp_head_divisibility
from src.env import is_accelerate_launch
from src.models.attention.gpt_oss_sinks import SinksPolicy, uses_gpt_oss_sinks
from src.models.config_access import text_config
from src.models.context_window import context_window_from_config
from src.models.moe.config_fields import (
    has_routed_experts,
    resolve_expert_count,
    resolve_expert_ffn_width,
    resolve_router_topk,
)
from src.models.moe.family_specs import find_moe_family_spec_for_config


def _divisors_up_to(value: int, ceiling: int) -> list[int]:
    """Divisors of ``value`` in ``2..ceiling``, the sizes a rejection may suggest; 1 means no expert
    parallelism, which is no answer to "pick a working ep_size"."""
    return [d for d in range(2, min(value, ceiling) + 1) if value % d == 0]


def _reject_undispatchable_token_budget(pc: ParallelismConfig, model_config) -> None:
    """Refuse a per-rank token budget the first MoE dispatch would refuse.

    The dispatcher sizes its arena from the all-reduced MAX tokens/rank of the EP group, and every rank
    of a group declares the same per-device shape, so that MAX is ``ep_rows_per_device × max_length``
    here. ``max_length: null`` resolves against ``config.json``: it is the largest budget a run can
    declare, so treating it as undeclared would switch the gate off exactly where it binds.
    """
    if pc.ep_size <= 1 or pc.ep_rows_per_device <= 0:
        return
    text = text_config(model_config)
    hidden = getattr(text, "hidden_size", 0)
    num_topk = resolve_router_topk(text)
    if not hidden or not num_topk:
        return
    max_length = pc.ep_declared_max_length or context_window_from_config(model_config)
    if not max_length:
        # max_length: null on a config that states no window — nothing to judge against.
        return
    # CP is the one axis that shrinks a rank's token count (it holds one sequence chunk). EP is
    # orthogonal to DP, TP replicates the batch, and a PP microbatch is a fraction of it — so
    # dividing by none of them keeps this an upper bound.
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
        raise ValueError(
            f"Lower per_device_train_batch_size or the row-length cap (max_length, or "
            f"max_prompt_length + max_completion_length where those two bound a row) — or, for the "
            f"cross-node Gin ceiling only, raise HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK. This run's "
            f"declared token budget cannot be dispatched: {pc.ep_rows_per_device} row(s)/device × "
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
    reject_accelerate_launch(pc)
    # DeepEP's dispatch buffer is sized for 2-byte tokens; fp32 activations trip a raw C++ assert in
    # `get_dispatch_buffer_size` at the first dispatch, naming neither fp32 nor the config. Keyed on
    # ep_size, not has_ep_group: pure ETP folds into has_ep_group but never reaches the transport.
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
    """Refuse what this model's ``config.json`` cannot take under ``pc``, before the process groups.

    ``wraps_experts`` is whether the load swaps in the EP MoE wrappers (``needs_ep_wrappers`` on a MoE).
    """
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
        # Config-time and rank-symmetric on purpose: under PP a hybrid stack can give one stage
        # no MoE layer, so the equivalent check on the trainer's per-rank module walk would
        # raise on MoE stages while dense stages walked into the wrap and hung the job.
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
    # without this it fails on the first forward's reshape — after the full load.
    validate_tp_head_divisibility(text_config(model_config), pc.tp_size)
    _reject_undispatchable_token_budget(pc, model_config)

    # Under accelerate MULTI_GPU the wrapped experts would silently hijack accelerate's DDP management,
    # and under accelerate FSDP they are unvalidated against the plugin's wrapping; EP/CP/TP launches
    # under accelerate were already refused by reject_unsupported_run.
    if wraps_experts and is_accelerate_launch() and not (pc.has_ep_group or pc.is_cp_mode or pc.is_tp_mode):
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
    # Declared on the family spec (``supports_fp32_non_ep_params``); otherwise DeepEP's raw C++ assert fires at
    # the first dispatch.
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
