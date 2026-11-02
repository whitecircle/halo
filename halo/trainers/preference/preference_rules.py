"""Construction refusals of the SMPO, DPO and KTO trainers, as pure functions of what they read."""

from collections.abc import Callable, Mapping
from typing import Any

from halo.distributed.parallelism_config import ParallelismConfig
from halo.trainers.parallelism.pipeline_rules import reject_pp_peft

# Why each DPO loss type that couples the whole batch cannot be split into pipeline microbatches.
_PP_BATCH_COUPLED_DPO_LOSSES = {
    "aot": "it sorts the paired log-ratios across the whole batch (torch.sort over dim 0), so a "
    "microbatch split re-pairs the sorted quantiles",
    "aot_unpaired": "it sorts the chosen and rejected log-ratios across the whole batch "
    "(torch.sort over dim 0), so a microbatch split re-pairs the sorted quantiles",
    "sft": "its cross-entropy is a mean over the whole batch's chosen completion tokens — a "
    "batch-global token denominator that changes with the microbatch split",
}


def reject_smpo_options(args, parallelism_config: ParallelismConfig | None) -> None:
    """Refuse ``padding_free`` under context parallelism."""
    if args.padding_free and parallelism_config and parallelism_config.cp_size > 1:
        raise ValueError(
            "padding_free mode is incompatible with Context Parallelism (CP). "
            "CP requires fixed-length sequences for collective synchronization. "
            "Please disable padding_free when using CP."
        )


def reject_smpo_vlm_options(args, parallelism_config: ParallelismConfig | None, *, is_vlm: bool) -> None:
    """Refuse the options a VLM run cannot honor; a text run, on any checkpoint, keeps them.

    ``padding_free`` flattens the batch into one sequence, dropping the prompt-tensor alignment the vision
    tower maps patches through; CP scatters a prompt's placeholder span while every rank keeps the pixels.
    """
    if not is_vlm:
        return
    if args.padding_free:
        raise ValueError(
            "SMPO VLM mode is incompatible with padding_free: flattening the batch breaks the "
            "image-placeholder ↔ pixel-patch alignment. Disable padding_free for VLM training."
        )
    if parallelism_config is not None and parallelism_config.cp_size > 1:
        raise ValueError(
            "SMPO VLM mode is incompatible with Context Parallelism (CP): sequence splitting "
            "scatters expanded image placeholder spans across CP ranks. Drop "
            "context_parallel_size for VLM training."
        )


def reject_smpo_pp_options(
    args, parallelism_config: ParallelismConfig | None, peft_config: Any | None, *, is_vlm: bool
) -> None:
    """Refuse the SMPO options pipeline parallelism cannot honor, the explicit ctor parameters included
    (the mixin's kwargs-reading hook cannot see them)."""
    if parallelism_config is None or not parallelism_config.is_pp_mode:
        return
    if is_vlm:
        raise ValueError(
            "SMPO VLM mode is not supported under pipeline parallelism: vision tensors ride the "
            "chosen|rejected concat into the vision tower of the full multimodal forward, but a "
            "pipeline stage's forward takes hidden_states and the fixed-shape P2P contract "
            "carries no patch-space tensors. Train VLM SMPO without PP."
        )
    if args.padding_free:
        raise ValueError(
            "padding_free is not supported under pipeline parallelism: it flattens each batch "
            "into one variable-length row, and the pipeline's P2P buffer shapes are frozen "
            "after the first step. Disable padding_free."
        )
    if args.lower_clip_percentile is not None or args.upper_clip_percentile is not None:
        raise ValueError(
            "lower_clip_percentile/upper_clip_percentile are not supported under pipeline "
            "parallelism: the clip bound is a token quantile over the WHOLE batch half, and a "
            "microbatch loss sees only its own tokens — the bound (and thus the objective) "
            "would change with pipeline_microbatches. Set both to null under PP "
            "(min_log_prob, a per-token constant, stays supported)."
        )
    reject_pp_peft(peft_config, explicit_param_trainer="SMPO")


def reject_dpo_pp_objective(args, *, supported: Mapping[str, Callable]) -> None:
    """Refuse a DPO objective the PP last-stage loss does not reproduce; ``supported`` maps the loss
    types it re-implements."""
    loss_types = args.loss_type if isinstance(args.loss_type, list) else [args.loss_type]
    for loss_type in loss_types:
        if loss_type in supported:
            continue
        mechanism = _PP_BATCH_COUPLED_DPO_LOSSES.get(
            loss_type,
            "its per-pair loss is not re-implemented in the PP last-stage loss, and running it "
            "unaudited would train a silently different objective",
        )
        raise ValueError(
            f"loss_type '{loss_type}' is not supported under pipeline parallelism: {mechanism}. "
            f"Supported under PP: {', '.join(supported)}."
        )
    if args.f_divergence_type != "reverse_kl":
        raise ValueError(
            f"f_divergence_type='{args.f_divergence_type}' is not supported under "
            f"pipeline parallelism: the PP last-stage loss implements the standard reverse-KL "
            f"scores only; another divergence would silently be computed as reverse-KL."
        )
    if args.use_weighting:
        raise ValueError(
            "use_weighting (WPO) is not supported under pipeline parallelism: the per-pair "
            "weights need a full-vocab logsumexp over the shifted logits inside the loss, which "
            "the PP last-stage loss does not implement."
        )
    if args.ld_alpha is not None:
        raise ValueError(
            "ld_alpha (length-desensitized DPO) is not supported under pipeline parallelism: it "
            "reshapes the sequence log-probs themselves, the PP last-stage loss does not "
            "implement that shaping, and the precomputed reference columns would have to be "
            "produced with the identical shaping to stay comparable."
        )


def reject_kto_pp_loss_type(args) -> None:
    """Refuse every KTO loss but the KL-free one a pipeline microbatch can compute."""
    if args.loss_type != "apo_zero_unpaired":
        raise ValueError(
            f"loss_type '{args.loss_type}' is not supported under pipeline "
            f"parallelism: the default 'kto' loss needs a world-global detached KL baseline "
            f"(per-rank batch mean, all-gathered, clamped) that no microbatch can compute, "
            f"plus a per-batch no-grad policy forward over the KL sequences — a full-model "
            f"pass even with precomputed reference log-probs. Only the KL-free "
            f"'apo_zero_unpaired' loss is decomposable and supported under PP."
        )


def reject_policy_from_checkpoint_without_resume(kwargs: Mapping[str, Any]) -> None:
    """Refuse a policy built from a checkpoint the trainer is not told about."""
    if kwargs.get("policy_from_checkpoint", False) and kwargs.get("resume_checkpoint") is None:
        raise ValueError("policy_from_checkpoint=True needs the resume_checkpoint the policy was built from.")
