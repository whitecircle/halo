"""Trainer construction refusals as pure reject_* functions over explicit inputs, called by the trainer spine and
the trainers' own hooks."""

from __future__ import annotations

from halo.config_validation import differs_from_default
from halo.data.collators.packing import DataCollatorWithFlattening, DataCollatorWithPacking
from halo.distributed.context_parallel.config import CPConfigError
from halo.models.attention.gpt_oss_sinks import has_live_attention_sinks
from halo.models.moe.config_fields import has_routed_experts
from halo.models.moe.family_specs import wraps_experts
from halo.models.segment_markers import reject_compressed_kv_rows
from halo.models.unwrapping import find_peft_model
from halo.optimizers.registry import NAMED_OPTIMIZER_BUILDERS, STOCK_ADAMW_OPTIMIZERS, optim_name

# The FSDP2 shaping knobs only the trainer's own wrap implements; each one's "not requested" value is its
# ParallelismConfig default.
FSDP_SHAPING_KNOBS = ("use_hsdp", "fsdp_reshard_after_forward", "fsdp_reshard_after_backward", "fsdp_defer_grad_sync")
# Why the router aux loss is inert on a trainer whose loss is not built by ``*ForCausalLM.forward`` from labels.
INERT_AUX_LOSS_DETAIL = (
    "{trainer} computes its loss from the model's outputs (no ``labels`` forward), so the HuggingFace router aux "
    "loss is never added: expert load is left unbalanced and the run still materializes a [B*S, num_experts] "
    "router-logit tensor per MoE layer per forward."
)


def reject_cp_trainer_args(
    training_args,
    compute_metrics=None,
    preprocess_logits_for_metrics=None,
) -> None:
    """Refuse, with ``CPConfigError``, trainer settings that re-derive the loss or a metric from ``(logits, labels)``
    outside the CP wrapper, pairing this rank's chunk of logits with the full-length labels."""
    if getattr(training_args, "label_smoothing_factor", 0.0):
        raise CPConfigError(
            "label_smoothing_factor > 0 is not supported with Context Parallelism: the Trainer pops "
            "`labels` before the model call and smooths the loss itself, so the CP wrapper returns "
            "no loss and the smoother compares FULL-length labels against this rank's "
            "sequence-chunk logits. Set label_smoothing_factor=0, or train without CP."
        )

    if getattr(training_args, "loss_type", None) == "dft":
        raise CPConfigError(
            "loss_type='dft' is not supported with Context Parallelism: TRL implements it either by "
            "installing a compute_loss_func (the Trainer then pops `labels`, leaving the CP wrapper "
            "with no loss and the func with full-length labels) or, under Liger, by a "
            "`use_token_scaling` forward flag the CP wrapper's own loss ignores. Use "
            "loss_type='nll', or train without CP."
        )

    # Not gated on eval_strategy: a trainer built with eval_strategy='no' can still reach the
    # misaligned metric path through a direct evaluate()/predict() call.
    if compute_metrics is not None or preprocess_logits_for_metrics is not None:
        raise CPConfigError(
            "Evaluation under Context Parallelism is loss-only: compute_metrics / "
            "preprocess_logits_for_metrics receive this rank's sequence-chunk logits gathered "
            "against the full-length labels, so the two are misaligned. Drop them (eval_loss still "
            "reports), or evaluate from a saved checkpoint without CP."
        )


def reject_conflicting_optimizer_choice(bf16_optimizer: bool, training_args) -> None:
    """Refuse ``bf16_optimizer=True`` beside a toolkit-named ``optim``, which the bf16 path would silently discard."""
    requested_optim = optim_name(training_args.optim if training_args is not None else None)
    if bf16_optimizer and requested_optim in NAMED_OPTIMIZER_BUILDERS:
        raise ValueError(
            f"bf16_optimizer=True and optim={requested_optim!r} both select an optimizer, and "
            f"the bf16 path wins — {requested_optim} would be silently discarded with nothing "
            f"in the log to say so. Pick one: drop bf16_optimizer to run {requested_optim}, or "
            f"drop optim to run AdamWBF16."
        )


def reject_stock_adamw_on_mixed_params(optim: str, config, model_config) -> None:
    """Refuse the stock fused/foreach AdamW where plain expert tensors coexist with FSDP2 DTensor peers.

    ``aten._fused_adamw_`` raises 'mixed torch.Tensor and DTensor' at the first step over that set. Read off
    the config so every PP stage agrees (a module walk would refuse on MoE stages while dense stages walked into
    the collective); ep1 experts FSDP2 shards, and a MoE family with no EP layer class, are uniformly DTensor.
    """
    if optim not in STOCK_ADAMW_OPTIMIZERS:
        return  # named/per-parameter optimizers carry their own gates
    if not (
        config.ep_group_size > 1
        or (
            wraps_experts(model_config, ep_layers_requested=config.needs_ep_layers) and not config.experts_fsdp_managed
        )
    ):
        return
    raise ValueError(
        f"optim={optim!r} cannot mix the plain-tensor expert FFN params (EP rank-local experts, "
        f"or the grouped-GEMM gate_proj_gmm/up_proj_gmm split at expert_parallel_size=1) with the "
        f"FSDP2 DTensor non-expert params: aten._fused_adamw_ raises 'mixed torch.Tensor and DTensor' "
        f"at the first step. Use AdamWBF16 (bf16=true, the production default, or "
        f"bf16_optimizer=true — its per-parameter path is DTensor-aware), or "
        f"fp32_non_ep_params=true for fp32 masters on the non-expert params only (experts stay "
        f"bf16). The stock AdamW runs where the parameters are uniform (dense models, and "
        f"expert_parallel_size=1 MoE without expert TP at the default fsdp_shard_ep1_experts), over "
        f"the parameters at the run's dtype: bf16 masters under bf16=true, not fp32."
    )


def reject_trainer_router_aux_loss_coef(training_args) -> None:
    """Refuse a ``router_aux_loss_coef`` on the trainer config: the balancing mode owns the term."""
    if training_args.router_aux_loss_coef is not None:
        raise ValueError(
            f"router_aux_loss_coef={training_args.router_aux_loss_coef} is set on the trainer config, but the "
            "router aux loss is owned by moe_balancing and the model config's own router_aux_loss_coef "
            "(model_init_kwargs). Unset it here."
        )


def reject_ep_layers_under_accelerate(ep_layers) -> None:
    """Refuse EP layers on an ``accelerate launch``: they need the mixin-managed FSDP2 path."""
    if ep_layers:
        raise ValueError(
            "The model carries EP layers but this is an "
            "'accelerate launch' run: the EP layers require the mixin-managed FSDP2 path "
            "(torchrun). Launch with torchrun, or load the model with "
            "use_grouped_gemm=false to train it under accelerate."
        )


def reject_sync_ref_model_on_sharded_policy(training_args, *, sharded: bool) -> None:
    """Refuse TRL's ``sync_ref_model`` on an FSDP2/EP/TP policy: its callback zips policy and reference params
    with plain ``.data`` ops, which crash on a DTensor/EP policy."""
    if sharded and getattr(training_args, "sync_ref_model", False):
        raise ValueError(
            "sync_ref_model is not supported under FSDP2/EP/TP: TRL's SyncRefModelCallback "
            "assumes an unwrapped policy. Disable sync_ref_model (use PEFT with ref_model=None, "
            "or precompute_ref_log_probs where available)."
        )


def reject_fsdp_knobs_under_qlora(config) -> None:
    """Refuse the FSDP-shaping knobs under QLoRA, which skips FSDP2 (``fully_shard`` cannot wrap bnb's
    ``Params4bit``): a multi-node ``use_hsdp`` would otherwise run a flat replicated all-reduce."""
    set_knobs = [name for name in FSDP_SHAPING_KNOBS if differs_from_default(config, name)]
    if set_knobs:
        raise ValueError(
            f"{', '.join(set_knobs)} cannot take effect under QLoRA: quantized 4-bit base "
            f"weights skip FSDP2 entirely (fully_shard cannot wrap bnb Params4bit), so the run "
            f"would silently use plain replicated gradient sync instead of the sharding the "
            f"config names. Remove the flag(s), or use plain LoRA / full fine-tuning."
        )


def reject_cp_packed_collator(collator, *, is_cp_mode: bool) -> None:
    """Refuse a packing / padding-free collator handed to a trainer under CP: Ulysses drops ``attention_mask``
    and calls flash-attn with no ``cu_seqlens``, so packed documents attend across each other. The collator
    factory applies the same rule; the two roots cover the completion-only variants."""
    if is_cp_mode and isinstance(collator, DataCollatorWithPacking | DataCollatorWithFlattening):
        raise ValueError(
            f"{type(collator).__name__} is incompatible with context parallelism: the Ulysses "
            f"path has no per-document boundaries, so packed sequences attend across each other. "
            f"Use a padding collator (padding_free/packing off) under CP."
        )


def reject_compressed_kv_collator(collator, model_config) -> None:
    """Refuse a packing / padding-free collator handed to a trainer for a model with compressed-KV layers
    (:func:`~halo.models.segment_markers.reject_compressed_kv_rows`, which the collator factory runs too)."""
    if isinstance(collator, DataCollatorWithPacking):
        reject_compressed_kv_rows(model_config, "packing")
    elif isinstance(collator, DataCollatorWithFlattening):
        reject_compressed_kv_rows(model_config, "padding_free")


def reject_reference_sinks_mismatch(model, ref_model, alternatives: str) -> None:
    """Refuse a reference whose attention sinks differ from the policy's: the pair compute different log-probs
    for identical tokens, which biases the KL under every sharding. ``alternatives`` names the trainer's way
    around holding a reference."""
    if model is None or has_live_attention_sinks(model) == has_live_attention_sinks(ref_model):
        return
    policy_sinks, reference_sinks = (
        "live" if has_live_attention_sinks(side) else "not live" for side in (model, ref_model)
    )
    raise ValueError(
        f"The reference model and the policy disagree on attention sinks (policy: {policy_sinks}, "
        f"reference: {reference_sinks}), so they compute different log-probs for identical tokens and "
        "the KL term is biased on every token. Load the reference through load_frozen_reference_model "
        "with the policy's reset_sinks, as the training scripts do; a reference TRL builds itself keeps "
        f"the pretrained sinks live. Or hold none: {alternatives}"
    )


def reject_inert_aux_loss_balancing(trainer_name: str, moe_balancing: str) -> None:
    """Refuse ``moe_balancing='aux_loss'`` on a trainer whose loss never adds the router aux term, for a model
    whose forward would compute one."""
    if moe_balancing == "aux_loss":
        raise ValueError(
            f"moe_balancing='aux_loss' balances nothing on this trainer. "
            f"{INERT_AUX_LOSS_DETAIL.format(trainer=trainer_name)} Use "
            f"moe_balancing='bias_update' (aux-loss-free DeepSeek-V3 bias update, on families "
            f"whose EP layers support it) or moe_balancing='none'."
        )


def reject_unreloadable_best_model(training_args, config, model) -> None:
    """Refuse ``load_best_model_at_end`` on a full fine-tune whose end-of-run reload the checkpoint loader refuses.

    Decidable only once PEFT is applied (adapter-only runs reload in place). Under EP/CP base weights load only
    at construction; FSDP2 over TP stacks a 2-D placement ``distribute_tensor`` does not invert for packed
    projections, while pure TP reloads into the live DTensor placements.
    """
    if not training_args.load_best_model_at_end:
        return
    # merge_expert_lora_on_save folds the adapter into a full base checkpoint, which the loader refuses.
    adapter_only = find_peft_model(model) is not None or config.expert_lora is not None
    if adapter_only and not config.merge_expert_lora_on_save:
        return
    is_moe = has_routed_experts(getattr(model, "config", None))
    if config.cp_size > 1 or (is_moe and config.needs_ep_layers):
        raise ValueError(
            "load_best_model_at_end is unsupported for this shape as a full fine-tune: under "
            "EP/CP base weights only load at construction, so the end-of-run best-model reload "
            "would be refused after the whole run, and the export would otherwise carry the "
            "LAST weights. Drop load_best_model_at_end and export the best checkpoint directly."
        )
    if config.tp_size > 1 and config.data_parallel_size > 1:
        raise ValueError(
            "load_best_model_at_end is unsupported under TP+DP as a full fine-tune: the best-model "
            "reload distributes each checkpoint tensor into the live DTensor placements, and FSDP2 "
            "over TP stacks a strided dp shard on the tp shard — a 2-D placement distribute_tensor "
            "does not invert for packed projections (right shape, wrong rows), so the end-of-run "
            "reload is refused. Drop load_best_model_at_end and export the best checkpoint "
            "directly, or run pure TP (tensor_parallel_size == world size), where the reload is "
            "supported."
        )


def reject_ep_gradient_checkpointing(ep_layers) -> None:
    """Reject gradient checkpointing over EP layers whose family declares it unsupported."""
    unsupported = sorted(
        {type(module).__name__ for module in ep_layers if not module.FAMILY.supports_gradient_checkpointing}
    )
    if unsupported:
        raise ValueError(
            f"gradient_checkpointing=True is not supported with Expert Parallelism for "
            f"{', '.join(unsupported)}: the family declares supports_gradient_checkpointing="
            f"False (see the EP layer class docstring for the mechanism — e.g. Zaya's EDA/CCA "
            f"cross-layer state makes checkpoint recompute recurse polynomially). Disable "
            f"gradient_checkpointing for this model."
        )


def reject_batch_rebalance_on_dp_sharded_loader(training_args) -> None:
    """Refuse HF's ``batch_rebalance`` sampler on the DP-sharded loader: it shards batches over every
    process, not over data-parallel replicas."""
    if training_args.train_sampling_strategy != "batch_rebalance":
        return
    raise ValueError(
        "train_sampling_strategy='batch_rebalance' shards batches across every process, while this "
        "run's loader shards across data-parallel replicas (TP/CP/ETP/PP, or a pre-sharded "
        "dataset): ranks sharing a batch would draw different rows. Use 'random', 'sequential' "
        "or 'group_by_length'."
    )


def reject_dispatch_batches(dispatch_batches: bool | None, data_parallel_size: int, world_size: int) -> None:
    """Refuse accelerate's dispatching loader on a loader sharded fewer ways than the world: it slices
    rank 0's batches by global rank."""
    if not dispatch_batches or data_parallel_size == world_size:
        return
    raise ValueError(
        f"accelerator_config.dispatch_batches: true slices rank 0's batches by global rank, but "
        f"this loader is sharded {data_parallel_size} ways over {world_size} ranks "
        f"(TP/CP/ETP siblings and pipeline peers share a data-parallel shard; a pre-sharded "
        f"dataset is each rank's own), so ranks would get the wrong rows. Set "
        f"accelerator_config.dispatch_batches: false."
    )
