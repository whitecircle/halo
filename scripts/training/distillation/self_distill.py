#!/usr/bin/env python
"""SDPG self-distillation (text or VLM) with distributed parallelism.

Privileged-context self-distillation on top of SFT (:class:`DistributedSelfDistillationTrainer`):
one model is both the student (prompt only) and a privileged teacher (prompt + a hint revealing
the gold answer), and the teacher's full-vocabulary distribution supervises the student on the
shared response tokens (``L = L_sft + beta(k)·L_OPD + alpha·L_ref``). One script serves both text
and vision-language models — ``load_model_for_training`` auto-detects the modality and selects the
text or VLM self-distillation collator. This is the offline SDPG approximation (scores a fixed
dataset; the faithful on-policy SDPG runs via ``online_grpo/rlvr.py --use_sdpg``). For
off-policy teacher→student distillation use ``teacher_distill.py``.

Dataset: raw SFT conversations (``conversation_field``) plus a privileged answer field (VLM adds an
``images`` column). The collator tokenizes the student and teacher (hinted) branches at collation
time, so a pre-tokenized dataset is not usable here.

Usage:
    torchrun --nproc_per_node=8 scripts/training/distillation/self_distill.py \\
        examples/distillation/qwen3_5/self-distill-qwen3.5-9b.yaml

That student is dense, so it takes no --expert_parallel_size; the MoE configs pin it themselves.
"""

from dataclasses import fields

from accelerate.logging import get_logger
from transformers import PreTrainedModel
from trl import ModelConfig, SFTConfig

from src.args.distributed_args import DistributedArguments
from src.args.mixins import SDPGArguments
from src.args.self_distill_args import SelfDistillationArguments
from src.data.collators.self_distill import SelfDistillTextCollator, audit_self_distill_row
from src.data.collators.vlm import SelfDistillVLMDataCollator
from src.data.pipeline.processing import coordinated_map, resolve_map_num_proc
from src.data.pipeline.vlm_dataset import prepare_vlm_dataset
from src.data.vlm import is_vlm_run
from src.distributed.loading.frozen_models import load_frozen_auxiliary_model
from src.distributed.loading.peft_setup import setup_peft_model
from src.distributed.loading.vlm_setup import load_model_for_training
from src.distributed.runtime import barrier, is_global_main_process
from src.models.loading.dtype import resolve_training_dtype
from src.models.loading.model_preparation import log_model_info
from src.trainers.distillation.self_distillation import DistributedSelfDistillationTrainer
from src.training.environment import run_training
from src.training.parser import H4ArgumentParser
from src.training.script_runner import (
    apply_distributed_trainer_config,
    apply_max_length,
    build_training_callbacks,
    disable_trl_dataset_prep,
    distributed_trainer_kwargs,
    enforce_text_path_padding_side,
    init_training_script,
    install_resolved_tokenizer,
    load_script_datasets,
    log_script_dataset_examples,
    reject_images_under_text_only_model,
    reject_non_default_args,
    reject_unsupported_args,
    run_trainer,
)

logger = get_logger(__name__, log_level="INFO")


def _build_text_dataset_and_collator(ds, args, tokenizer, max_length, model_config, num_proc):
    train_dataset = ds["train"]
    eval_dataset = ds.get("test")
    if args.conversation_field not in train_dataset.column_names:
        raise ValueError(
            f"self_distillation requires a RAW conversation dataset with a '{args.conversation_field}' "
            f"field plus the privileged answer field ('{args.sdpg_answer_field}'); "
            f"got columns {train_dataset.column_names}."
        )
    collator = SelfDistillTextCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        conversation_field=args.conversation_field,
        hint_template=args.sdpg_hint_template,
        answer_field=args.sdpg_answer_field,
        solution_field=args.privileged_solution_field,
        confidence_field=args.confidence_field,
        confidence_power=args.confidence_power,
        response_prompt_template=args.assistant_message_template if args.train_on_completions_only else None,
        train_on_completions_only=args.train_on_completions_only,
        system_prompt=args.system_prompt,
        model_supports_system_role=args.model_supports_system_role,
        tools_field=args.tools_field,
        interleaved_thinking=args.interleaved_thinking,
        model_config=model_config,
    )
    # Enforce the collator's length contract now, where the raise is world-uniform: at collate time
    # only the rank drawing the over-length row raises, and its peers block in the step's collectives
    # until the watchdog fires.
    coordinated_map(
        train_dataset,
        audit_self_distill_row,
        desc="Auditing self-distill row lengths",
        num_proc=num_proc,
        fn_kwargs={"collator": collator},
        cache_key_extras=collator.cache_signature(),
    )
    if eval_dataset is not None:
        coordinated_map(
            eval_dataset,
            audit_self_distill_row,
            desc="Auditing self-distill eval row lengths",
            num_proc=num_proc,
            fn_kwargs={"collator": collator},
            cache_key_extras=collator.cache_signature(),
        )
    return train_dataset, eval_dataset, collator


def _build_vlm_dataset_and_collator(ds, args, processor, tokenizer, max_length, num_proc, model_config):
    """VLM path: map raw conversations to history/images (keeping the privileged fields) + collator."""
    privileged = tuple(f for f in (args.sdpg_answer_field, args.privileged_solution_field, args.confidence_field) if f)
    ds = prepare_vlm_dataset(
        ds,
        args,
        processor,
        tokenizer,
        max_length,
        num_proc,
        keep=privileged,
        desc="Processing VLM self-distillation dataset",
    )
    collator = SelfDistillVLMDataCollator(
        processor,
        tokenizer,
        max_length,
        response_prompt_template=args.assistant_message_template if args.train_on_completions_only else None,
        train_on_completions_only=args.train_on_completions_only,
        hint_template=args.sdpg_hint_template,
        answer_field=args.sdpg_answer_field,
        solution_field=args.privileged_solution_field,
        confidence_field=args.confidence_field,
        confidence_power=args.confidence_power,
        model_config=model_config,
    )
    return ds["train"], ds.get("test"), collator


def _load_sdpg_reference(*, args, model_config, sft_config, dist_args, is_vlm: bool) -> PreTrainedModel | None:
    """Load the frozen reference for the ``L_ref`` KL anchor, or ``None`` when the anchor is off.

    A dense unparallelized replica is accepted under EP/TP (unlike DPO/KTO's ref_model rejection): the
    SDPG reference is only scored — frozen, never trained or synced — so each rank holds a full copy
    with no sharding-consistency requirement (same shape as offline-GRPO's dense KL reload).

    It goes through the shared frozen loader because ``L_ref`` is a divergence between this model's
    distribution and the policy's: a reference on a different attention backend, or carrying live
    sinks the policy reset, moves the anchor rather than the model.
    """
    if not args.reference_kl_coef or args.reference_kl_coef <= 0:
        return None
    ref_path = args.reference_model_name_or_path or model_config.model_name_or_path
    reference_model = load_frozen_auxiliary_model(
        ref_path,
        dtype=resolve_training_dtype(sft_config),
        # The pin names a commit in the policy's repo, so it applies only when the reference is that
        # same repo; a separate reference repo takes its own main.
        revision=model_config.model_revision if ref_path == model_config.model_name_or_path else None,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        reset_sinks=dist_args.reset_sinks,
        is_vlm=is_vlm,
        download_tag="reference_model",
    )
    if is_global_main_process():
        logger.info(f"Loaded frozen reference model from '{ref_path}' (alpha={args.reference_kl_coef})")
    return reference_model


def main():
    parser = H4ArgumentParser((SelfDistillationArguments, SFTConfig, ModelConfig, DistributedArguments))
    args, sft_config, model_config, distributed_args = parser.parse()

    # Inherited from SFTScriptArguments but unreachable: GenerateExamplesCallback needs a tokenized
    # "generate" split and self-distillation keeps the dataset raw. num_eval_examples rides along on
    # the default-comparing form, since its own default (50) is truthy.
    reject_non_default_args("Self-distillation", args, "generate_eval_examples", "num_eval_examples")

    # Reject inherited options the SelfDistill collators do not implement; ignoring them would train
    # something other than the config states.
    if args.train_on_last_assistant_only:
        raise ValueError(
            "train_on_last_assistant_only is not implemented for self-distillation (the SelfDistill "
            "collators mask all assistant turns when train_on_completions_only is set). Remove it "
            "from the YAML."
        )
    if sft_config.packing or sft_config.padding_free:
        raise ValueError(
            "packing / padding_free are not supported for self-distillation: the SelfDistill "
            "collators tokenize the student and teacher branches at collation time and always "
            "right-pad, so both are forced off below. Remove them from the YAML."
        )
    # TRL applies these inside its own dataset prep + default collator, both replaced here (the
    # SelfDistill collators mask through train_on_completions_only), so they would parse and mask nothing.
    reject_unsupported_args(
        "Self-distillation",
        # Tri-state: an explicit False ("train on the full sequence") is ignored the same as True.
        completion_only_loss=sft_config.completion_only_loss is not None,
        assistant_only_loss=sft_config.assistant_only_loss,
    )

    # The SelfDistill collator tokenizes the raw branches at collation time, so the raw conversation
    # and privileged columns have to survive HF's remove_unused_columns=True, which would strip them.
    sft_config.remove_unused_columns = False

    # supports_cp=False: the privileged-teacher pass uses a separate, longer sequence, so CP must be
    # rejected at config-build time (mirrors teacher_distill; the trainer also sets _supports_cp=False).
    runtime = init_training_script(
        args,
        sft_config,
        model_config,
        distributed_args,
        script_prefix="self-distill",
        supports_cp=False,
        supports_pp=False,
        sync_tokens=("eos_token", "pad_token"),
    )
    parallelism_config = runtime.parallelism_config

    model, processing_class, tokenizer, is_vlm_checkpoint = load_model_for_training(
        model_config,
        sft_config,
        parallelism_config,
        reset_sinks=distributed_args.reset_sinks,
        train_sinks=distributed_args.train_sinks,
        weights_source=runtime.model_source,
        text_only_model=distributed_args.text_only_model,
    )
    tokenizer = apply_max_length(sft_config, args, model, tokenizer)
    processing_class = install_resolved_tokenizer(processing_class, tokenizer, is_vlm_checkpoint)
    peft_config = setup_peft_model(args, model, model_config, "CAUSAL_LM")
    log_model_info(model, tokenizer)

    ds, dataset_presharded = load_script_datasets(args, parallelism_config, conversation_field=args.conversation_field)
    # The data path follows the run, not the checkpoint class: a natively-multimodal student
    # distilled on text-only rows is a text run (see is_vlm_run).
    reject_images_under_text_only_model(args, ds, text_only_model=distributed_args.text_only_model)
    is_vlm = is_vlm_run(args, model_config.model_name_or_path, ds, config=model.config)
    enforce_text_path_padding_side(tokenizer, is_vlm)
    if is_vlm:
        num_proc = resolve_map_num_proc(sft_config.dataset_num_proc)
        train_dataset, eval_dataset, collator = _build_vlm_dataset_and_collator(
            ds, args, processing_class, tokenizer, sft_config.max_length, num_proc, model.config
        )
    else:
        train_dataset, eval_dataset, collator = _build_text_dataset_and_collator(
            ds, args, tokenizer, sft_config.max_length, model.config, resolve_map_num_proc(sft_config.dataset_num_proc)
        )

    log_script_dataset_examples({"train": train_dataset, "test": eval_dataset}, tokenizer, args, sft_config)

    reference_model = _load_sdpg_reference(
        args=args,
        model_config=model_config,
        sft_config=sft_config,
        dist_args=distributed_args,
        is_vlm=is_vlm_checkpoint,
    )

    disable_trl_dataset_prep(sft_config)
    apply_distributed_trainer_config(sft_config, parallelism_config)

    callbacks = build_training_callbacks(args, sft_config, model, parallelism_config)

    barrier()

    trainer = DistributedSelfDistillationTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processing_class,
        peft_config=peft_config,
        data_collator=collator,
        callbacks=callbacks,
        **distributed_trainer_kwargs(
            args, distributed_args, parallelism_config, dataset_presharded=dataset_presharded
        ),
        **{
            f.name: getattr(args, f.name)
            for f in fields(SDPGArguments)
            if f.name not in SelfDistillationArguments.DATASET_SIDE_SDPG_FIELDS
        },
        reference_model=reference_model,
        reference_kl_coef=args.reference_kl_coef,
        reference_kl_loss=args.reference_kl_loss,
        confidence_weight_opd=args.confidence_weight_opd,
        opd_exclude_eos=args.opd_exclude_eos,
    )
    run_trainer(
        trainer,
        runtime,
        method_name="Self-distillation",
        extra_start_log=[f"modality: {'vlm' if is_vlm else 'text'}"],
    )


if __name__ == "__main__":
    run_training(main)()
