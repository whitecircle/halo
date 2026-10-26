#!/usr/bin/env python
"""Distributed offline GRPO training.

Group-relative policy optimization over pre-computed completions and rewards — no live generation.

Usage:
    torchrun --nproc_per_node=8 scripts/training/offline_grpo.py \\
        examples/grpo/offline/qwen3_5/offline-grpo-qwen3.6-35b-a3b-gsm8k.yaml
"""

from functools import partial

from trl import ModelConfig

from src.callbacks.generate_examples import GenerateExamplesCallback
from src.configs.grpo_offline import OfflineGRPOConfig, OfflineGRPOScriptArguments
from src.configs.parallelism_arguments import ParallelismArguments
from src.data.pipeline.conversation import chat_template_kwargs, reject_image_content
from src.data.pipeline.preferences import split_rendered_completion
from src.data.pipeline.processing import coordinated_map, resolve_map_num_proc
from src.data.pipeline.row_processors import prepare_generative_row
from src.data.sources.loading import reject_image_columns
from src.entrypoints.data_setup import load_script_datasets, log_script_dataset_examples
from src.entrypoints.model_setup import (
    apply_prompt_completion_window,
    default_padded_attn_implementation,
    load_model_and_tokenizer,
    log_model_info,
)
from src.entrypoints.script import (
    ScriptContext,
    apply_distributed_trainer_config,
    configure_balancing_and_build_callbacks,
    distributed_trainer_kwargs,
    init_training_script,
    parse_script_args,
    run_trainer,
    with_distributed_teardown,
)
from src.model_loading.parallel.frozen_models import load_frozen_reference_model
from src.model_loading.parallel.peft_setup import configure_adapter_training, wraps_in_peft_model
from src.runtime.collectives import barrier
from src.trainers.grpo.mixins.offline_reference import reject_unsupported_reference_input
from src.trainers.grpo.offline import OfflineGRPOTrainer


def build_chat_template_row_fn(tokenizer, tools_field: str | None):
    """Row transform rendering ``prompt``/``completions`` message lists to text.

    Completions are rendered as ``template(prompt + completion)`` minus the rendered-prompt prefix,
    not standalone: strict templates (Qwen3.5) raise on assistant-only message lists, and
    BOS-emitting templates would inject a mid-sequence BOS into every completion.

    Offline GRPO is text-only, so an image content part is refused per row rather than rendered as
    placeholder tokens with no pixels behind them.
    """

    def apply_chat_templates(row):
        template_kwargs = chat_template_kwargs(row, interleaved_thinking=False, tools_field=tools_field)
        prompt_messages = row["prompt"]
        reject_image_content(prompt_messages, "offline GRPO field 'prompt'")
        for completion in row["completions"]:
            reject_image_content(completion, "offline GRPO field 'completions'")
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, **template_kwargs)
        row["completions"] = [
            split_rendered_completion(
                prompt_text,
                tokenizer.apply_chat_template(prompt_messages + completion, tokenize=False, **template_kwargs),
                "completions",
            )
            for completion in row["completions"]
        ]
        row["prompt"] = prompt_text
        return row

    return apply_chat_templates


def _load_kl_reference(
    args,
    context: ScriptContext,
    offline_grpo_config: OfflineGRPOConfig,
    model_config: ModelConfig,
    *,
    policy,
    tokenizer,
    peft_config,
    attn_default: str | None,
):
    """The frozen KL reference the trainer cannot derive from ``policy``, or ``None`` where it can.

    Full fine-tuning sweeps raw run-start scores once and restores them from each checkpoint.
    Only native expert LoRA needs a live, unadapted base; PEFT can disable its adapters in place.
    Load that base from the configured model, not a trained resume checkpoint.
    """
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
        OfflineGRPOScriptArguments, OfflineGRPOConfig, ModelConfig, ParallelismArguments
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

    apply_chat_templates = build_chat_template_row_fn(tokenizer, args.tools_field)

    chat_template_cache_extras = {"tools_field": args.tools_field}
    map_kwargs = {"num_proc": resolve_map_num_proc(offline_grpo_config.dataset_num_proc)}

    ds_processed = coordinated_map(
        ds,
        apply_chat_templates,
        desc="Applying chat templates",
        cache_key_extras=chat_template_cache_extras,
        **map_kwargs,
    )

    # Only when the callback will consume it — the prep tokenizes the whole test split.
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
            cache_key_extras=chat_template_cache_extras,
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
        policy_gradient_loss=True,
    )

    apply_distributed_trainer_config(offline_grpo_config, parallelism_config)

    barrier()

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
    run_trainer(trainer, context, method_name="Offline GRPO")


if __name__ == "__main__":
    with_distributed_teardown(main)()
