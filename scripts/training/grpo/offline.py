#!/usr/bin/env python
"""Distributed offline GRPO: group-relative policy optimization over pre-computed completions and rewards."""

from functools import partial

from halo.callbacks.generate_examples import GenerateExamplesCallback
from halo.configs.grpo_offline import OfflineGRPOConfig, OfflineGRPOScriptArguments
from halo.configs.parallelism_arguments import ParallelismArguments
from halo.configs.script_arguments import ModelArguments
from halo.data.conversation import chat_template_kwargs, reject_image_content
from halo.data.pipeline.coordinated_map import coordinated_map, resolve_map_num_proc
from halo.data.pipeline.prompt_completion import render_prompt_completions
from halo.data.pipeline.text_rows import prepare_generative_row
from halo.data.vlm_detection import reject_image_columns
from halo.entrypoints.data_setup import load_script_datasets, log_script_dataset_examples
from halo.entrypoints.model_setup import (
    apply_prompt_completion_window,
    default_padded_attn_implementation,
    load_export_processor,
    load_model_and_tokenizer,
    log_model_info,
)
from halo.entrypoints.script import (
    ScriptContext,
    configure_balancing_and_build_callbacks,
    distributed_trainer_kwargs,
    init_training_script,
    parse_script_args,
    run_trainer,
    with_distributed_teardown,
)
from halo.model_loading.parallel.frozen_models import load_frozen_reference_model
from halo.model_loading.parallel.peft_setup import configure_adapter_training, wraps_in_peft_model
from halo.runtime.collectives import barrier
from halo.trainers.grpo.offline.offline_rules import reject_unsupported_reference_input
from halo.trainers.grpo.offline.trainer import OfflineGRPOTrainer


def render_offline_grpo_row(row, tokenizer, tools_field: str | None):
    reject_image_content(row["prompt"], "offline GRPO field 'prompt'")
    for completion in row["completions"]:
        reject_image_content(completion, "offline GRPO field 'completions'")
    template_kwargs = chat_template_kwargs(row, interleaved_thinking=False, tools_field=tools_field)
    row["prompt"], row["completions"] = render_prompt_completions(
        partial(tokenizer.apply_chat_template, tokenize=False, **template_kwargs),
        row["prompt"],
        [("completions", completion) for completion in row["completions"]],
    )
    return row


def _load_kl_reference(
    args,
    context: ScriptContext,
    offline_grpo_config: OfflineGRPOConfig,
    model_config: ModelArguments,
    *,
    policy,
    tokenizer,
    peft_config,
    attn_default: str | None,
):
    """The frozen KL reference the trainer cannot derive from ``policy``, or ``None``. Full fine-tuning sweeps raw
    run-start scores once; only native expert LoRA needs a live, unadapted base, loaded from the configured model, not
    a trained resume checkpoint."""
    if not OfflineGRPOTrainer.requires_ref_model(policy, offline_grpo_config, context.parallelism_config, peft_config):
        return None
    return load_frozen_reference_model(
        args,
        model_config,
        offline_grpo_config,
        tokenizer,
        model_config.model_name_or_path,
        is_vlm=False,
        reset_sinks=args.reset_sinks,
        attn_default=attn_default,
    )


def main():
    args, offline_grpo_config, model_config, dist_args = parse_script_args(
        OfflineGRPOScriptArguments, OfflineGRPOConfig, ModelArguments, ParallelismArguments
    )

    context = init_training_script(
        args,
        offline_grpo_config,
        model_config,
        dist_args,
        script_prefix="offline-grpo",
        trainer_cls=OfflineGRPOTrainer,
    )
    parallelism_config = context.parallelism_config

    ds, dataset_presharded = load_script_datasets(
        args,
        parallelism_config,
        conversation_field="prompt",
    )
    reject_image_columns(ds, "Offline GRPO")
    # The trainer refuses the same splits at construction; checked here, before the model load.
    reject_unsupported_reference_input(
        ds["train"],
        ds["test"],
        active=OfflineGRPOTrainer.sweeps_run_start_reference(
            model_config.model_name_or_path,
            offline_grpo_config.kl_beta,
            parallelism_config,
            peft=wraps_in_peft_model(model_config),
        ),
        presharded=dataset_presharded,
    )

    # Padded batches: the CP collator right-pads full rows, the non-CP layout left-pads prompts.
    requested_attn = default_padded_attn_implementation(
        model_config, sinks_reset=args.reset_sinks, context_parallel=parallelism_config.is_cp_mode
    )
    model, tokenizer = load_model_and_tokenizer(
        context, offline_grpo_config, model_config, args, attn_default=requested_attn
    )

    # Both budgets are truncation caps here, and either may be null ("no cap"), so the tokenizer is
    # pinned only when they jointly bound the sequence.
    tokenizer, _ = apply_prompt_completion_window(
        args,
        model,
        tokenizer,
        max_prompt_length=offline_grpo_config.max_prompt_length,
        max_completion_length=offline_grpo_config.max_completion_length,
    )

    peft_config = configure_adapter_training(args, model, model_config, "CAUSAL_LM")

    ref_model = _load_kl_reference(
        args,
        context,
        offline_grpo_config,
        model_config,
        policy=model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        attn_default=requested_attn,
    )

    log_model_info(model, tokenizer)

    map_kwargs = {"num_proc": resolve_map_num_proc(offline_grpo_config.dataset_num_proc)}

    ds_processed = coordinated_map(
        ds,
        partial(render_offline_grpo_row, tokenizer=tokenizer, tools_field=args.tools_field),
        desc="Applying chat templates",
        **map_kwargs,
    )

    # Only when the callback will consume it: the prep tokenizes the whole test split.
    generate_dataset = (
        coordinated_map(
            ds["test"],
            partial(
                prepare_generative_row,
                tokenizer=tokenizer,
                max_length=offline_grpo_config.max_prompt_length,
                tools_field=args.tools_field,
            ),
            desc="Preparing generation dataset",
            **map_kwargs,
        )
        if args.generate_eval_examples
        else None
    )

    train_dataset = ds_processed["train"]
    eval_dataset = ds_processed["test"]

    log_script_dataset_examples(
        {"train": train_dataset, "test": eval_dataset, "generate": generate_dataset},
        tokenizer,
        args,
        offline_grpo_config,
    )

    callbacks = configure_balancing_and_build_callbacks(
        args,
        offline_grpo_config,
        model,
        context,
        generate_callback=GenerateExamplesCallback.from_config(args, offline_grpo_config, generate_dataset, tokenizer),
    )

    barrier()

    export_processor = load_export_processor(model_config, args, tokenizer)
    trainer = OfflineGRPOTrainer(
        model=model,
        args=offline_grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        ref_model=ref_model,
        callbacks=callbacks,
        resume_checkpoint=context.resume_checkpoint,
        **distributed_trainer_kwargs(args, parallelism_config, dataset_presharded=dataset_presharded),
    )
    trainer.export_processing_class = export_processor
    run_trainer(trainer, context, method_name="Offline GRPO")


if __name__ == "__main__":
    with_distributed_teardown(main)()
