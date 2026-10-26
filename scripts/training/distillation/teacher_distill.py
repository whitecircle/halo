#!/usr/bin/env python
"""Distributed off-policy distillation (text or VLM): the student is fit to a frozen teacher's token distribution.

Usage: torchrun --nproc_per_node=8 scripts/training/distillation/teacher_distill.py \\
    examples/distillation/qwen3_5/distill-qwen3.5-9b-from-qwen3.6-35b-a3b.yaml
"""

from accelerate.logging import get_logger
from transformers import AutoConfig, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from src.configs.distillation import DistillationConfig, TeacherDistillScriptArguments
from src.configs.parallelism_arguments import ParallelismArguments
from src.configs.script_arguments import ModelArguments
from src.data.collators.factory import select_data_collator
from src.data.collators.padded import DataCollatorForCausalLMWithPadding
from src.entrypoints.data_setup import (
    load_script_datasets,
    log_script_dataset_examples,
    prepare_chat_text_split,
    prepare_raw_vlm_split,
    resolve_vlm_run,
)
from src.entrypoints.model_setup import (
    apply_max_length,
    apply_text_path_padding_side,
    default_padded_attn_implementation,
    install_resolved_tokenizer,
    load_policy_model,
    log_model_info,
)
from src.entrypoints.script import (
    configure_balancing_and_build_callbacks,
    distributed_trainer_kwargs,
    init_training_script,
    parse_script_args,
    run_trainer,
    with_distributed_teardown,
)
from src.model_loading.dtype import resolve_training_dtype
from src.model_loading.parallel.frozen_models import load_frozen_auxiliary_model
from src.model_loading.parallel.peft_setup import configure_adapter_training
from src.runtime.collectives import barrier
from src.runtime.filesystem import hub_metadata_main_first
from src.runtime.ranks import is_global_main_process
from src.trainers.distillation.divergences import resolve_shared_vocab_width
from src.trainers.distillation.teacher_distillation import DistributedDistillationTrainer

logger = get_logger(__name__, log_level="INFO")


def _load_distill_teacher(*, args, model_config, training_config, is_vlm: bool, local_rank: int) -> PreTrainedModel:
    """Load the frozen teacher at ``teacher_model_revision`` (the student's pin names a commit in another repo), in
    the run's dtype, on the shared padded-workload backend request, which under live sinks falls back to
    auto-detection: a sink-dropping teacher would shift every logprob it scores against its student."""
    requested_attn = default_padded_attn_implementation(model_config, sinks_reset=args.reset_sinks)
    return load_frozen_auxiliary_model(
        args.teacher_model,
        # The teacher's logits are the target the student is fit to, so it loads in the run's own
        # precision: an fp32 run scored against a bf16 teacher fits rounded targets.
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


def _load_teacher_tokenizer(*, args, model_config, student_config, student_tokenizer) -> PreTrainedTokenizerBase:
    """The teacher repo's own tokenizer, at the teacher's pin, checked with its config against the
    student's token ids and logit rows before any teacher weight loads (the trainer checks again).
    """

    def fetch():
        kwargs = {"revision": args.teacher_model_revision or None, "trust_remote_code": model_config.trust_remote_code}
        return (
            AutoTokenizer.from_pretrained(args.teacher_model, **kwargs),
            AutoConfig.from_pretrained(args.teacher_model, **kwargs),
        )

    teacher_tokenizer, teacher_config = hub_metadata_main_first("teacher_metadata", fetch)
    resolve_shared_vocab_width(student_config, teacher_config, student_tokenizer, teacher_tokenizer)
    return teacher_tokenizer


def _text_distill_collator(args, tokenizer, model_config):
    """The completions-only collator, else the padded causal-LM one; the losses mask on its ``labels``.

    The padded collator restores a turn-ending EOS where pad == eos, which a plain LM collator masks as
    padding, leaving the student no stop signal to learn.
    """
    return select_data_collator(
        tokenizer=tokenizer,
        train_on_completions_only=args.train_on_completions_only,
        assistant_message_template=args.assistant_message_template,
        model_config=model_config,
    ) or DataCollatorForCausalLMWithPadding(tokenizer=tokenizer, mlm=False)


def main():
    args, training_config, model_config, dist_args = parse_script_args(
        TeacherDistillScriptArguments, DistillationConfig, ModelArguments, ParallelismArguments
    )

    context = init_training_script(
        args,
        training_config,
        model_config,
        dist_args,
        script_prefix="distill",
        trainer_cls=DistributedDistillationTrainer,
    )
    parallelism_config = context.parallelism_config

    ds, dataset_presharded = load_script_datasets(args, parallelism_config, conversation_field=args.conversation_field)
    is_vlm = resolve_vlm_run(args, model_config, ds, text_only_model=args.text_only_model)

    # Same padded-workload request the teacher load makes: the two forwards are compared token by
    # token, so a backend split between them biases the distillation targets.
    policy = load_policy_model(
        context,
        training_config,
        model_config,
        args,
        vlm_run=is_vlm,
        attn_default=default_padded_attn_implementation(model_config, sinks_reset=args.reset_sinks),
    )
    student_model = policy.model
    tokenizer = apply_max_length(training_config, args, student_model, policy.tokenizer)
    processing_class = install_resolved_tokenizer(policy.processing_class, tokenizer)
    apply_text_path_padding_side(tokenizer, is_vlm)

    peft_config = configure_adapter_training(args, student_model, model_config, "CAUSAL_LM")

    teacher_tokenizer = _load_teacher_tokenizer(
        args=args, model_config=model_config, student_config=student_model.config, student_tokenizer=tokenizer
    )
    teacher_model = _load_distill_teacher(
        args=args,
        model_config=model_config,
        training_config=training_config,
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

    trainer = DistributedDistillationTrainer(
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
