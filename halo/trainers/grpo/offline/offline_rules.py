"""Offline GRPO's construction refusals, run before the trainer loads its model or tokenizes its data."""

from collections.abc import Mapping

from datasets import Dataset

from halo.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN
from halo.model_loading.tokenizer_setup import resolve_tokenizer
from halo.models.context_window import is_bounded_length
from halo.runtime.consensus import reject_across_ranks
from halo.trainers.grpo.offline.reference import ReferenceMode
from halo.trainers.parallelism.pipeline_rules import reject_pp_compute_metrics, reject_pp_peft


def require_tokenizer(processing_class):
    """The tokenizer behind ``processing_class`` (a VLM ``ProcessorMixin`` keeps its token ids on an inner one)."""
    if processing_class is None:
        raise ValueError(
            "OfflineGRPOTrainer requires processing_class: the stored prompts and completions are "
            "tokenized with it at construction, and its pad id pads every batch."
        )
    return resolve_tokenizer(processing_class)


def reject_offline_pp_options(args, parallelism_config, peft_config, compute_metrics) -> None:
    """Its own gate rather than the spine's ``_reject_unsupported_pp_options`` hook, which never sees ``peft_config``
    and ``compute_metrics``, explicit constructor parameters."""
    if parallelism_config is None or not parallelism_config.is_pp_mode:
        return
    if args.use_chunked_grpo_logprobs:
        raise ValueError(
            "use_chunked_grpo_logprobs does nothing under pipeline parallelism: the pipeline drives "
            "stages through the PP loss adapter and never calls _get_per_token_logps, so the last "
            "stage still materializes its own [rows, max_length, vocab] logits plane. Remove it, and "
            "lower per_device_train_batch_size or raise pipeline_parallel_size to fit the plane."
        )
    reject_pp_compute_metrics(
        compute_metrics,
        (
            "this trainer's own "
            "prediction_step returns the loss and NO predictions (loss, None, None) in every "
            "mode, so compute_metrics is never called off-PP either — there is no prediction "
            "convention for the pipeline to reproduce. Serving it under PP would broadcast the "
            "last stage's raw [rows, max_length, vocab] logits to every rank in the chain and "
            "feed metrics a tensor no non-PP run ever produces. Drop compute_metrics; the "
            "objective's own statistics are logged as metrics already."
        ),
    )
    reject_pp_peft(peft_config, explicit_param_trainer="OfflineGRPOTrainer")
    if not is_bounded_length(args.max_prompt_length) or not is_bounded_length(args.max_completion_length):
        raise ValueError(
            f"Pipeline parallelism requires bounded sequences (the P2P buffer shapes freeze on "
            f"the first step): set both max_prompt_length (got {args.max_prompt_length!r}) and "
            f"max_completion_length (got {args.max_completion_length!r}) to positive values."
        )


def resolve_pp_max_length(args, parallelism_config) -> int | None:
    """The fixed shape every pipeline batch pads to, which bounds the P2P buffers: the two caps' sum. ``None`` off PP."""
    if parallelism_config is None or not parallelism_config.is_pp_mode:
        return None
    return args.max_prompt_length + args.max_completion_length


def reject_offline_cp_options(parallelism_config, ref_model, reference_mode: ReferenceMode) -> None:
    if parallelism_config is None or not parallelism_config.is_cp_mode:
        return
    if reference_mode is not ReferenceMode.RUN_START:
        raise ValueError(
            "Offline GRPO with CP requires a full policy fine-tune: PEFT and native expert LoRA are "
            "not validated under offline-GRPO CP (their adapter checkpoint path, and at kl_beta > 0 "
            "their adapter-disabled reference). Fully fine-tune the policy, or drop context parallelism."
        )
    if ref_model is not None:
        raise ValueError(
            "Offline GRPO with CP scores its run-start reference through the distributed policy. "
            "An explicit ref_model would never be read. Drop ref_model."
        )


def reject_encoder_decoder(model) -> None:
    """The objective scores each completion as the continuation of its prompt in one causal sequence."""
    if model.config.is_encoder_decoder:
        raise ValueError(
            f"OfflineGRPOTrainer trains decoder-only models; {type(model).__name__} is an encoder-decoder. "
            "The objective scores each completion as the continuation of its prompt in one causal sequence."
        )


def reject_unsupported_reference_input(
    train_dataset, eval_dataset, *, active: bool = True, presharded: bool = False
) -> None:
    """COLLECTIVE; inert unless ``active``. Refuse the splits a run-start reference sweep cannot score."""
    if not active:
        return
    reason = None
    if presharded:
        reason = (
            "Offline GRPO reference precompute is not supported with a pre-sharded dataset; load the same "
            "unsharded dataset on every rank."
        )
    elif not isinstance(train_dataset, Dataset):
        reason = (
            "Offline GRPO KL reference preparation requires a finite datasets.Dataset training split, not "
            "streaming or schema-less input."
        )
    elif isinstance(eval_dataset, Mapping):
        reason = (
            "Offline GRPO constructor KL reference preparation requires one finite evaluation Dataset. Use "
            "evaluate() for named tokenized splits after construction."
        )
    elif eval_dataset is not None and not isinstance(eval_dataset, Dataset):
        reason = "Offline GRPO KL reference preparation requires a finite datasets.Dataset evaluation split."
    elif any(
        REF_PER_TOKEN_LOGPS_COLUMN in split.column_names
        for split in (train_dataset, eval_dataset)
        if split is not None
    ):
        reason = (
            "Supplied ref_per_token_logps are not supported by offline GRPO full-finetuning KL; load an "
            "unsharded dataset and let the trainer prepare its checkpointed run-start reference."
        )
    reject_across_ranks(reason, "Validating offline GRPO reference inputs", exc_type=ValueError)
