#!/usr/bin/env python
"""Distributed off-policy distillation (text or VLM).

Trains a student to match a frozen teacher's token distribution. One script serves both text and
vision-language models: ``load_policy_model`` auto-detects the student modality and the
teacher is loaded with the matching class. The distillation trainer forwards ``model(**inputs)`` /
``teacher(**inputs)``, so ``pixel_values`` thread to both models. Teacher and student must share
processor/tokenizer geometry (full-vocabulary logit alignment).

CP is not supported (the teacher is unparallelized); use EP and/or TP.

Usage:
    torchrun --nproc_per_node=8 scripts/training/distillation/teacher_distill.py \\
        examples/distillation/qwen3_5/distill-qwen3.5-9b-from-qwen3.6-35b-a3b.yaml

EP applies to the student, so the dense-student config above takes no --expert_parallel_size.
"""

from accelerate.logging import get_logger
from transformers import AutoConfig, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from trl import ModelConfig

from src.configs.distillation import DistillationConfig, TeacherDistillScriptArguments
from src.configs.parallelism_arguments import ParallelismArguments
from src.data.collators.factory import select_data_collator
from src.data.collators.packing import DataCollatorForCausalLMWithPadding
from src.data.collators.vlm import VLMDataCollator
from src.data.pipeline.processing import process_dataset_with_map_and_filter, resolve_map_num_proc
from src.data.pipeline.row_processors import create_llm_processor, text_render_kwargs
from src.data.pipeline.vlm_dataset import prepare_vlm_dataset
from src.entrypoints.data_setup import load_script_datasets, log_script_dataset_examples, resolve_vlm_run
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
from src.trainers.distillation.losses import shared_vocab_width
from src.trainers.distillation.teacher_distillation import DistributedDistillationTrainer

logger = get_logger(__name__, log_level="INFO")


def _load_distill_teacher(*, args, model_config, training_config, is_vlm: bool, local_rank: int) -> PreTrainedModel:
    """Load the frozen teacher: its own pinned revision, the run's dtype, and a backend resolved
    against the teacher's own config under the student's sinks policy.

    ``teacher_model_revision`` pins the teacher repo; the student's ``model_revision`` names a commit
    in a different repo and would 404.

    Its batches are padded, so the backend request is the shared padded-workload one, which under
    live sinks falls back to auto-detection: a teacher on a sink-dropping backend would shift every
    logprob it scores against its student.
    """
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
        # VLM only: the text teacher lands in host memory and the trainer moves it (_setup_teacher_model).
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
    shared_vocab_width(student_config, teacher_config, student_tokenizer, teacher_tokenizer)
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


def _prepare_text_distill_data(ds, args, training_config, tokenizer, model_config):
    """SFT's chat row processor: each conversation is templated and tokenized once, and one over
    ``max_length`` is dropped rather than truncated mid-turn, measured on the ids it trains on. The
    collator derives the labels the losses mask on."""
    render_kwargs = text_render_kwargs(args)
    ds = process_dataset_with_map_and_filter(
        ds,
        create_llm_processor(
            tokenizer=tokenizer, max_length=training_config.max_length, use_padding=False, **render_kwargs
        ),
        # sorted: this list feeds the coordinated-map cache key — set order is hash-randomized per process.
        remove_columns=sorted(set(ds["train"].column_names)),
        desc="Processing distillation dataset",
        cache_key_extras={**render_kwargs, "add_generation_prompt": False},
        num_proc=resolve_map_num_proc(training_config.dataset_num_proc),
    )
    return ds["train"], ds["test"], _text_distill_collator(args, tokenizer, model_config)


def _prepare_vlm_distill_data(ds, args, training_config, processor, tokenizer, model_config):
    """Map raw conversations to history/images and collate with VLMDataCollator."""
    num_proc = resolve_map_num_proc(training_config.dataset_num_proc)
    ds = prepare_vlm_dataset(
        ds,
        args,
        processor,
        tokenizer,
        training_config.max_length,
        num_proc,
        desc="Processing VLM distillation dataset",
    )
    # The mapped columns (history/images/…) are not model-forward kwargs, so HF's default
    # remove_unused_columns=True would strip them before they reach the collator.
    training_config.remove_unused_columns = False
    collator = VLMDataCollator(
        processor,
        tokenizer,
        training_config.max_length,
        response_prompt_template=args.assistant_message_template if args.train_on_completions_only else None,
        train_on_completions_only=args.train_on_completions_only,
        model_config=model_config,
    )
    return ds["train"], ds["test"], collator


def main():
    args, training_config, model_config, dist_args = parse_script_args(
        TeacherDistillScriptArguments, DistillationConfig, ModelConfig, ParallelismArguments
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

    ds, dataset_presharded = load_script_datasets(
        args,
        parallelism_config,
        conversation_field=args.conversation_field,
    )
    # The data path follows the run, not the checkpoint class: a natively-multimodal student
    # distilled on text-only rows is a text run (see is_vlm_run). Decided before the model load,
    # which requires the checkpoint's processor for an image run.
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
        train_dataset, eval_dataset, data_collator = _prepare_vlm_distill_data(
            ds, args, training_config, processing_class, tokenizer, student_model.config
        )
    else:
        train_dataset, eval_dataset, data_collator = _prepare_text_distill_data(
            ds, args, training_config, tokenizer, student_model.config
        )

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
        trainer,
        context,
        method_name="Distillation",
        extra_start_log=[f"modality: {'vlm' if is_vlm else 'text'}"],
    )


if __name__ == "__main__":
    with_distributed_teardown(main)()
