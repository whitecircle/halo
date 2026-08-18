#!/usr/bin/env python
"""Distributed off-policy distillation (text or VLM) with EP and TP support.

Trains a student to match a frozen teacher's token distribution. One script serves both text and
vision-language models — ``load_model_for_training`` auto-detects the student modality and the
teacher is loaded with the matching class. The distillation trainer forwards ``model(**inputs)`` /
``teacher(**inputs)``, so ``pixel_values`` thread to both models. Teacher and student must share
processor/tokenizer geometry (full-vocabulary logit alignment).

CP is not supported (the teacher is unparallelized); use EP and/or TP.

Usage:
    torchrun --nproc_per_node=8 scripts/training/distillation/teacher_distill.py \\
        examples/distillation/qwen3_5/distill-qwen3.5-9b-from-qwen3.6-35b-a3b.yaml

EP applies to the STUDENT, so the dense-student config above takes no --expert_parallel_size.
"""

from accelerate.logging import get_logger
from transformers import DataCollatorForLanguageModeling, PreTrainedModel
from trl import ModelConfig

from src.args.distill_args import DistillScriptArguments
from src.args.distributed_args import DistributedArguments
from src.configs.distillation_config import DistillationConfig
from src.data.collators.factory import select_data_collator
from src.data.collators.vlm import VLMDataCollator
from src.data.pipeline.processing import coordinated_map, filter_by_length, resolve_map_num_proc
from src.data.pipeline.rendered import tokenize_rendered
from src.data.pipeline.row_processors import apply_chat_template_to_conversations
from src.data.pipeline.vlm_dataset import prepare_vlm_dataset
from src.data.vlm import is_vlm_run
from src.distributed.loading.frozen_models import load_frozen_auxiliary_model
from src.distributed.loading.peft_setup import prepare_peft_model, setup_peft_model
from src.distributed.loading.vlm_setup import load_model_for_training
from src.distributed.runtime import barrier, is_global_main_process
from src.models.loading.dtype import resolve_training_dtype
from src.models.loading.model_preparation import log_model_info
from src.trainers.distillation.teacher_distillation import DistributedDistillationTrainer
from src.training.environment import run_training
from src.training.parser import H4ArgumentParser
from src.training.script_runner import (
    apply_distributed_trainer_config,
    apply_max_length,
    build_training_callbacks,
    distributed_trainer_kwargs,
    enforce_text_path_padding_side,
    init_training_script,
    install_resolved_tokenizer,
    load_script_datasets,
    log_script_dataset_examples,
    padded_workload_attn_implementation,
    reject_images_under_text_only_model,
    run_trainer,
)

logger = get_logger(__name__, log_level="INFO")


def _load_distill_teacher(
    *, args, model_config, training_config, dist_args, is_vlm: bool, local_rank: int
) -> PreTrainedModel:
    """Load the frozen teacher: its own pinned revision, the run's dtype, and a backend resolved
    against the teacher's own config under the student's sinks policy.

    ``teacher_model_revision`` pins the TEACHER repo — the student's ``model_revision`` names a commit
    in a different repo and would 404.

    Its batches are padded, so the backend request is the shared padded-workload one — which under
    live sinks drops back to auto-detection, because a teacher on a sink-dropping backend would
    shift every logprob it scores by nats against its student.
    """
    requested_attn = padded_workload_attn_implementation(model_config, sinks_reset=dist_args.reset_sinks)
    return load_frozen_auxiliary_model(
        args.teacher_model,
        # The teacher's logits are the target the student is fit to, so it loads in the run's own
        # precision: an fp32 run scored against a bf16 teacher fits rounded targets.
        dtype=resolve_training_dtype(training_config),
        # Empty string is an unset pin, not a branch name the hub can resolve.
        revision=args.teacher_model_revision or None,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=requested_attn,
        reset_sinks=dist_args.reset_sinks,
        is_vlm=is_vlm,
        # VLM only: the text teacher lands in host memory and the trainer moves it (_setup_teacher_model).
        device_map={"": local_rank} if is_vlm else None,
        download_tag="teacher_model",
    )


def _prepare_text_distill_data(ds, args, training_config, tokenizer, model_config):
    """Chat-template → length-filter → tokenize; the collator derives the labels the losses mask on."""
    num_proc_kwargs = {"num_proc": resolve_map_num_proc(training_config.dataset_num_proc)}
    ds = coordinated_map(
        ds,
        lambda row: {
            "text": apply_chat_template_to_conversations(
                row,
                tokenizer,
                conversation_field=args.conversation_field,
                system_prompt=args.system_prompt,
                model_supports_system_role=args.model_supports_system_role,
                tools_field=args.tools_field,
                interleaved_thinking=args.interleaved_thinking,
            )
        },
        desc="Applying chat template",
        cache_key_extras={
            "conversation_field": args.conversation_field,
            "system_prompt": args.system_prompt,
            "model_supports_system_role": args.model_supports_system_role,
            "tools_field": args.tools_field,
            "interleaved_thinking": args.interleaved_thinking,
        },
        **num_proc_kwargs,
    )
    train_dataset = filter_by_length(ds["train"], training_config.max_length, tokenizer, **num_proc_kwargs)
    # Same policy as train: drop over-length conversations instead of truncating them mid-turn.
    eval_dataset = filter_by_length(ds["test"], training_config.max_length, tokenizer, **num_proc_kwargs)

    def _tokenize(row):
        # tokenize_rendered applies the tokenizer's own special-token post-processing exactly once.
        enc = tokenize_rendered(tokenizer, row["text"], truncation=True, max_length=training_config.max_length)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    train_dataset = coordinated_map(
        train_dataset,
        _tokenize,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
        cache_key_extras={"max_length": training_config.max_length},
        **num_proc_kwargs,
    )
    eval_dataset = coordinated_map(
        eval_dataset,
        _tokenize,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing eval",
        cache_key_extras={"max_length": training_config.max_length},
        **num_proc_kwargs,
    )
    # Both distillation terms mask on ``labels`` (the distill KL as well as the CLM term), so
    # completions-only masking reaches the same tokens here as it does on the VLM path. The factory
    # returns None when nothing is masked; the losses read ``inputs["labels"]``, so the CLM collator
    # that derives them from input_ids is the floor.
    collator = select_data_collator(
        tokenizer=tokenizer,
        train_on_completions_only=args.train_on_completions_only,
        assistant_message_template=args.assistant_message_template,
        model_config=model_config,
    ) or DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return train_dataset, eval_dataset, collator


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
    # The mapped columns (history/images/…) are not model-forward kwargs — HF's default
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
    parser = H4ArgumentParser((DistillScriptArguments, DistillationConfig, ModelConfig, DistributedArguments))
    args, training_config, model_config, dist_args = parser.parse()

    runtime = init_training_script(
        args,
        training_config,
        model_config,
        dist_args,
        script_prefix="distill",
        supports_cp=False,
        supports_pp=False,
    )
    parallelism_config = runtime.parallelism_config
    local_rank = runtime.local_rank

    # Same padded-workload request the teacher load makes: the two forwards are compared token by
    # token, so a backend split between them biases the distillation targets.
    student_model, processing_class, tokenizer, is_vlm_checkpoint = load_model_for_training(
        model_config,
        training_config,
        parallelism_config,
        attn_default=padded_workload_attn_implementation(model_config, sinks_reset=dist_args.reset_sinks),
        reset_sinks=dist_args.reset_sinks,
        train_sinks=dist_args.train_sinks,
        weights_source=runtime.model_source,
        text_only_model=dist_args.text_only_model,
    )
    tokenizer = apply_max_length(training_config, args, student_model, tokenizer)
    processing_class = install_resolved_tokenizer(processing_class, tokenizer, is_vlm_checkpoint)

    # The distillation trainer is a plain Trainer (no peft_config kwarg), so apply PEFT here via the canonical
    # prepare_peft_model (k-bit prep before the wrap + bf16 adapter cast).
    peft_config = setup_peft_model(args, student_model, model_config, "CAUSAL_LM")
    if peft_config is not None:
        student_model, _ = prepare_peft_model(student_model, peft_config, training_config)

    teacher_model = _load_distill_teacher(
        args=args,
        model_config=model_config,
        training_config=training_config,
        dist_args=dist_args,
        is_vlm=is_vlm_checkpoint,
        local_rank=local_rank,
    )

    log_model_info(student_model, tokenizer)
    if is_global_main_process():
        teacher_params = sum(p.numel() for p in teacher_model.parameters()) / 1e9
        logger.info(f"Teacher model: {args.teacher_model} ({teacher_params:.2f}B params)")

    ds, dataset_presharded = load_script_datasets(
        args,
        parallelism_config,
        conversation_field=args.conversation_field,
    )
    # The DATA path follows the run, not the checkpoint class: a natively-multimodal student
    # distilled on text-only rows is a text run (see is_vlm_run).
    reject_images_under_text_only_model(args, ds, text_only_model=dist_args.text_only_model)
    is_vlm = is_vlm_run(args, model_config.model_name_or_path, ds, config=student_model.config)
    enforce_text_path_padding_side(tokenizer, is_vlm)
    if is_vlm:
        train_dataset, eval_dataset, data_collator = _prepare_vlm_distill_data(
            ds, args, training_config, processing_class, tokenizer, student_model.config
        )
    else:
        train_dataset, eval_dataset, data_collator = _prepare_text_distill_data(
            ds, args, training_config, tokenizer, student_model.config
        )

    log_script_dataset_examples({"train": train_dataset, "test": eval_dataset}, tokenizer, args, training_config)

    apply_distributed_trainer_config(training_config, parallelism_config)

    callbacks = build_training_callbacks(args, training_config, student_model, parallelism_config)

    barrier()

    trainer = DistributedDistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processing_class,
        callbacks=callbacks,
        **distributed_trainer_kwargs(args, dist_args, parallelism_config, dataset_presharded=dataset_presharded),
    )
    run_trainer(
        trainer,
        runtime,
        method_name="Distillation",
        extra_start_log=[f"modality: {'vlm' if is_vlm else 'text'}"],
    )


if __name__ == "__main__":
    run_training(main)()
