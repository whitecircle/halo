"""Construction refusals of packed micro-batches (``packed_microbatch_tokens``) shared by the on-policy GRPO trainers:
the options, models and backends under which a packed row would change the objective or cross documents, and the
budgets no EP dispatch carries."""

from halo.data.collators.collator_rules import reject_multi_document_rows
from halo.distributed.expert_parallel.deepep.capacity import (
    aligned_dispatch_ceiling,
    ep_dispatch_capacity,
    padded_wire_hidden,
    reject_oversized_dispatch,
)
from halo.models.attention.gpt_oss_sinks import has_live_attention_sinks
from halo.models.attention.selection import is_flash_attn_implementation
from halo.models.config_access import text_config
from halo.models.moe.config_fields import resolve_dispatch_width, resolve_router_topk
from halo.models.moe.family_specs import attention_reads_row_positions
from halo.trainers.grpo.row_order import MICROBATCH_INVARIANT_LOSS_TYPES
from halo.trainers.training_args_overrides import uses_evaluation

PACKED_KNOB = "packed_microbatch_tokens"


def reject_unpackable_trainer(trainer: type, refusal: str | None) -> None:
    """Refuse a trainer class that declared why its rows cannot be packed (``PACKED_ROWS_REFUSAL``)."""
    if refusal is not None:
        raise ValueError(f"{PACKED_KNOB} is not supported by {trainer.__name__}: {refusal}. Unset {PACKED_KNOB}.")


def reject_unpackable_options(
    args, *, use_chunked_logprobs: bool, length_ordered: bool | None, misaligned_accumulation: bool
) -> None:
    """Refuse the options under which packed micro-batches would change the objective or leave their rows unscored by
    the packed branch: every one is named with its remedy, none falls back to the per-row path. ``length_ordered`` is
    the trainer's ``length_ordered_microbatches`` (``None`` where it has none)."""
    knob = PACKED_KNOB
    if not use_chunked_logprobs:
        raise ValueError(
            f"{knob} needs use_chunked_grpo_logprobs: true: a packed micro-batch would otherwise materialize a "
            f"[tokens, vocab] logits plane, and TRL's full-logits scorer does not take packed rows."
        )
    if args.per_device_train_batch_size != 1:
        raise ValueError(
            f"{knob} needs per_device_train_batch_size: 1, got {args.per_device_train_batch_size}: the plan, not a "
            f"batch size, picks the rows of a micro-batch. Raise gradient_accumulation_steps or {knob} instead."
        )
    if args.loss_type not in MICROBATCH_INVARIANT_LOSS_TYPES:
        raise ValueError(
            f"{knob} is refused with loss_type={args.loss_type!r}: only the losses in "
            f"{list(MICROBATCH_INVARIANT_LOSS_TYPES)} are the same however rows share a bin (bnpo normalizes each "
            f"micro-batch by its own token count, so its objective would depend on which rows share one). Use grpo "
            f"or dapo."
        )
    if args.top_entropy_quantile < 1.0:
        raise ValueError(
            f"{knob} is refused with top_entropy_quantile={args.top_entropy_quantile}: the entropy threshold is taken "
            f"over each micro-batch's tokens, which a packed bin would widen from one row to several. Set it to 1.0."
        )
    if length_ordered:
        raise ValueError(
            f"{knob} replaces the length-ordered deal (length_ordered_microbatches: true): the plan already orders "
            f"the round's rows. Unset length_ordered_microbatches."
        )
    if misaligned_accumulation:
        raise ValueError(
            f"{knob} maps a generation round onto the optimizer step's window, but gradient_accumulation_steps "
            f"({args.gradient_accumulation_steps}) is not a multiple of steps_per_generation × num_iterations "
            f"({args.steps_per_generation} × {args.num_iterations}). Align them."
        )
    if uses_evaluation(args) and args.per_device_eval_batch_size != 1:
        raise ValueError(
            f"{knob} keeps evaluation rounds on the per-row path, which needs per_device_eval_batch_size: 1, got "
            f"{args.per_device_eval_batch_size}: a padded multi-row evaluation forward would run left-padded RoPE "
            f"positions."
        )


def reject_unpackable_model(model, attn_impl: str | None) -> None:
    """Refuse a model or backend whose packed row would cross documents silently, change the objective, or lose
    attention sinks. No opt-in: an RL step that mixes documents corrupts the policy gradient without a trace."""
    config = model.config
    knob = PACKED_KNOB
    if attention_reads_row_positions(config):
        raise ValueError(
            f"{knob} is refused for model_type={getattr(config, 'model_type', None)!r}: its attention reads a token's "
            f"row index, which a flattened row changes. Unset {knob}."
        )
    if not is_flash_attn_implementation(attn_impl):
        raise ValueError(
            f"{knob} needs a varlen flash attention backend, but this model resolved to attn_implementation="
            f"{attn_impl!r}: a dense kernel builds an [N, N] mask over the flattened micro-batch and, for several "
            f"families, attends across its rows. Use flash_attention_2/3/4 or unset {knob}."
        )
    reject_multi_document_rows(
        config,
        "packing",
        attn_impl=attn_impl,
        allow_cross_document_linear_attention=False,
        keeps_packed_rows=False,
        varlen_rows=True,
    )
    if has_live_attention_sinks(model) and attn_impl != "flash_attention_4":
        raise ValueError(
            f"{knob} with live attention sinks needs flash_attention_4 (its varlen entry carries them), but the "
            f"model resolved to {attn_impl!r}, which drops them. Unset full_determinism, or set reset_sinks: true, or "
            f"unset {knob}."
        )


def _dispatch_shape(parallelism_config, model_config) -> tuple[int, int] | None:
    """``(num_topk, padded wire hidden)`` of the run's EP dispatch, ``None`` without one."""
    if parallelism_config.ep_size <= 1:
        return None
    text = text_config(model_config)
    width, num_topk = resolve_dispatch_width(text), resolve_router_topk(text)
    return (num_topk, padded_wire_hidden(width)) if width and num_topk else None


def resolve_dispatch_ceiling(parallelism_config, model_config) -> int | None:
    """Tokens one EP dispatch can carry per rank on this run (the 32-bit wire index, and the proxy-GIN ceiling on a
    group spanning NVLink domains), rounded down to the arena's block; ``None`` where there is no dispatch."""
    shape = _dispatch_shape(parallelism_config, model_config)
    if shape is None:
        return None
    return aligned_dispatch_ceiling(
        num_topk=shape[0], padded_hidden=shape[1], spans_domains=parallelism_config.requires_rdma
    )


def reject_undispatchable_budget(parallelism_config, model_config, budget: int) -> None:
    """Refuse a ``packed_microbatch_tokens`` whose arena the first dispatch would refuse, before the first rollout."""
    shape = _dispatch_shape(parallelism_config, model_config)
    if shape is None:
        return
    try:
        reject_oversized_dispatch(
            ep_dispatch_capacity(budget),
            num_topk=shape[0],
            padded_hidden=shape[1],
            spans_domains=parallelism_config.requires_rdma,
        )
    except ValueError as exc:
        raise ValueError(f"{PACKED_KNOB}={budget} cannot be dispatched: lower it. {exc}") from exc
