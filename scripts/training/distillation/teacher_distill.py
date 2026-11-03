#!/usr/bin/env python
"""Distributed off-policy distillation (text or VLM): the student is fit to a frozen teacher's token distribution."""

import logging

from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase

from halo.configs.distillation import TeacherDistillationConfig, TeacherDistillScriptArguments
from halo.configs.parallelism_arguments import ParallelismArguments
from halo.configs.script_arguments import ModelArguments
from halo.data.collators.collator_rules import evaluation_batch_size, rows_reach_varlen_kernel
from halo.data.collators.factory import select_data_collator
from halo.data.collators.padded import DataCollatorForCausalLMWithPadding
from halo.entrypoints.data_setup import (
    load_script_datasets,
    log_script_dataset_examples,
    prepare_chat_text_split,
    prepare_raw_vlm_split,
    resolve_vlm_run,
)
from halo.entrypoints.model_setup import (
    apply_max_length,
    apply_text_path_padding_side,
    default_padded_attn_implementation,
    install_resolved_tokenizer,
    load_policy_model,
    log_model_info,
    reject_fa4_varlen_request,
)
from halo.entrypoints.script import (
    configure_balancing_and_build_callbacks,
    distributed_trainer_kwargs,
    init_training_script,
    parse_script_args,
    run_trainer,
    with_distributed_teardown,
)
from halo.model_loading.dtype import resolve_training_dtype
from halo.model_loading.parallel.frozen_models import load_frozen_auxiliary_model
from halo.model_loading.parallel.peft_setup import configure_adapter_training
from halo.models.attention.gpt_oss_sinks import (
    NEUTRALIZED_TEACHER_SINKS_COST,
    SINK_CARRYING_ATTN_HINT,
    SinksPolicy,
    uses_gpt_oss_sinks,
)
from halo.runtime.collectives import barrier
from halo.runtime.filesystem import hub_metadata_main_first
from halo.runtime.ranks import is_global_main_process
from halo.trainers.distillation.divergences import resolve_shared_vocab_width
from halo.trainers.distillation.teacher_distillation import DistributedTeacherDistillationTrainer

logger = logging.getLogger(__name__)


def _load_distill_teacher(
    *, args, model_config, training_config, requested_attn: str | None, is_vlm: bool, local_rank: int
) -> PreTrainedModel:
    """At ``teacher_model_revision`` (the student's pin names a commit in another repo), on the student's request,
    which under live sinks falls back to auto-detection: a sink-dropping teacher shifts every logprob."""
    return load_frozen_auxiliary_model(
        args.teacher_model,
        # The student's target, so in the run's precision: an fp32 run against a bf16 teacher fits rounded targets.
        dtype=resolve_training_dtype(training_config),
        # Empty string is an unset pin, not a branch name the hub can resolve.
        revision=args.teacher_model_revision or None,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=requested_attn,
        reset_sinks=args.reset_sinks,
        is_vlm=is_vlm,
        # VLM only: the text teacher lands in host memory and the trainer moves it (place_frozen_scorer).
        device_map={"": local_rank} if is_vlm else None,
        download_tag="teacher_model",
    )


def _reject_neutralized_teacher_sinks(teacher_config: PretrainedConfig, *, reset_sinks: bool) -> None:
    """Refuse ``reset_sinks: true`` with a GptOss teacher, before any weight loads. A frozen teacher never trains back
    (:data:`~halo.models.attention.gpt_oss_sinks.NEUTRALIZED_TEACHER_SINKS_COST`), so the student would fit a
    degenerate distribution."""
    if reset_sinks and uses_gpt_oss_sinks(teacher_config):
        raise ValueError(
            f"reset_sinks: true would neutralize the attention sinks of the GptOss teacher too, and "
            f"{NEUTRALIZED_TEACHER_SINKS_COST}. Set reset_sinks: false: both models keep the sinks their checkpoints "
            f"hold, frozen, on a sink-carrying attention implementation ({SINK_CARRYING_ATTN_HINT})."
        )


def _load_teacher_metadata(*, args, model_config) -> tuple[PreTrainedTokenizerBase, PretrainedConfig]:
    """The teacher repo's own tokenizer and config, at the teacher's pin."""

    def fetch(local_files_only: bool):
        kwargs = {
            "revision": args.teacher_model_revision or None,
            "trust_remote_code": model_config.trust_remote_code,
            "local_files_only": local_files_only,
        }
        return (
            AutoTokenizer.from_pretrained(args.teacher_model, **kwargs),
            AutoConfig.from_pretrained(args.teacher_model, **kwargs),
        )

    return hub_metadata_main_first("teacher_metadata", fetch)


def _text_distill_collator(args, tokenizer, model_config):
    """The completions-only collator, else the padded causal-LM one, which restores a turn-ending EOS where pad == eos;
    a plain LM collator masks it, leaving no stop signal to learn."""
    return select_data_collator(
        tokenizer=tokenizer,
        train_on_completions_only=args.train_on_completions_only,
        assistant_message_template=args.assistant_message_template,
        model_config=model_config,
    ) or DataCollatorForCausalLMWithPadding(tokenizer=tokenizer, mlm=False)


def main():
    args, training_config, model_config, dist_args = parse_script_args(
        TeacherDistillScriptArguments, TeacherDistillationConfig, ModelArguments, ParallelismArguments
    )

    context = init_training_script(
        args,
        training_config,
        model_config,
        dist_args,
        script_prefix="distill",
        trainer_cls=DistributedTeacherDistillationTrainer,
    )
    parallelism_config = context.parallelism_config

    ds, dataset_presharded = load_script_datasets(args, parallelism_config, conversation_field=args.conversation_field)
    is_vlm = resolve_vlm_run(args, model_config, ds, text_only_model=args.text_only_model)

    teacher_tokenizer, teacher_config = _load_teacher_metadata(args=args, model_config=model_config)
    _reject_neutralized_teacher_sinks(teacher_config, reset_sinks=args.reset_sinks)
    # One request for both loads, so the forwards compared token by token take the same backend where both accept it.
    attn_default = default_padded_attn_implementation(
        model_config, sinks_reset=args.reset_sinks, companion_configs=(teacher_config,)
    )
    # More than one padded row per batch reaches the varlen call.
    reject_fa4_varlen_request(
        model_config,
        attn_default,
        varlen_rows=rows_reach_varlen_kernel(
            packing=False,
            padding_free=False,
            per_device_train_batch_size=training_config.per_device_train_batch_size,
            per_device_eval_batch_size=evaluation_batch_size(training_config),
        ),
        dtype=resolve_training_dtype(training_config),
        sinks_policy=SinksPolicy.from_flags(reset_sinks=args.reset_sinks, train_sinks=args.train_sinks),
    )
    policy = load_policy_model(context, training_config, model_config, args, vlm_run=is_vlm, attn_default=attn_default)
    student_model = policy.model
    tokenizer = apply_max_length(training_config, args, student_model, policy.tokenizer)
    processing_class = install_resolved_tokenizer(policy.processing_class, tokenizer)
    apply_text_path_padding_side(tokenizer, is_vlm)

    peft_config = configure_adapter_training(args, student_model, model_config, "CAUSAL_LM")

    # Before any teacher weight loads (the trainer checks again).
    resolve_shared_vocab_width(student_model.config, teacher_config, tokenizer, teacher_tokenizer)
    teacher_model = _load_distill_teacher(
        args=args,
        model_config=model_config,
        training_config=training_config,
        requested_attn=attn_default,
        is_vlm=policy.is_vlm,
        local_rank=context.local_rank,
    )

    log_model_info(student_model, tokenizer)
    if is_global_main_process():
        teacher_params = sum(p.numel() for p in teacher_model.parameters()) / 1e9
        logger.info(f"Teacher model: {args.teacher_model} ({teacher_params:.2f}B params)")

    if is_vlm:
        train_dataset, eval_dataset, data_collator = prepare_raw_vlm_split(
            ds,
            args,
            training_config,
            processing_class,
            tokenizer,
            student_model.config,
            desc="Processing VLM distillation dataset",
        )
    else:
        train_dataset, eval_dataset = prepare_chat_text_split(
            ds, tokenizer, args, training_config, desc="Processing distillation dataset", use_padding=False
        )
        data_collator = _text_distill_collator(args, tokenizer, student_model.config)

    log_script_dataset_examples({"train": train_dataset, "test": eval_dataset}, tokenizer, args, training_config)

    callbacks = configure_balancing_and_build_callbacks(args, training_config, student_model, context)

    barrier()

    trainer = DistributedTeacherDistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processing_class,
        callbacks=callbacks,
        peft_config=peft_config,
        **distributed_trainer_kwargs(args, parallelism_config, dataset_presharded=dataset_presharded),
    )
    run_trainer(
        trainer, context, method_name="Distillation", extra_start_log=[f"modality: {'vlm' if is_vlm else 'text'}"]
    )


if __name__ == "__main__":
    with_distributed_teardown(main)()
