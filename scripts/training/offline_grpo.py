#!/usr/bin/env python
"""Distributed offline GRPO training with Expert and Tensor Parallelism support.

Group-relative policy optimization over pre-computed completions and rewards — no live generation.

CP is not supported (the trainer uses the ``logits_to_keep`` optimization); use EP and/or TP.

Usage:
    torchrun --nproc_per_node=8 scripts/training/offline_grpo.py \\
        examples/grpo/offline/qwen3_5/offline-grpo-qwen3.6-35b-a3b-gsm8k.yaml --expert_parallel_size=8
"""

from functools import partial

from trl import ModelConfig

from src.args.distributed_args import DistributedArguments
from src.args.offline_grpo_args import OfflineGRPOScriptArguments
from src.callbacks.generate_examples import GenerateExamplesCallback
from src.configs.offline_grpo_config import OfflineGRPOConfig
from src.data.pipeline.conversation import chat_template_kwargs, reject_image_content
from src.data.pipeline.preferences import split_rendered_completion
from src.data.pipeline.processing import coordinated_map, resolve_map_num_proc
from src.data.pipeline.row_processors import prepare_generative_row
from src.data.sources.loading import reject_image_columns
from src.distributed.loading.peft_setup import setup_peft_model
from src.distributed.runtime import barrier
from src.models.loading.model_preparation import log_model_info
from src.trainers.grpo.offline import OfflineGRPOTrainer
from src.training.environment import run_training
from src.training.parser import H4ArgumentParser
from src.training.script_runner import (
    apply_distributed_trainer_config,
    apply_prompt_completion_window,
    build_training_callbacks,
    distributed_trainer_kwargs,
    init_training_script,
    load_script_datasets,
    load_script_model,
    log_script_dataset_examples,
    padded_workload_attn_implementation,
    run_trainer,
)


def build_chat_template_row_fn(tokenizer, tools_field: str | None):
    """Row transform rendering ``prompt``/``completions`` message lists to text.

    Completions are rendered as ``template(prompt + completion)`` minus the rendered-prompt prefix
    — never standalone: strict templates (Qwen3.5) raise on assistant-only message lists, and
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


def main():
    parser = H4ArgumentParser((OfflineGRPOScriptArguments, OfflineGRPOConfig, ModelConfig, DistributedArguments))
    args, offline_grpo_config, model_config, dist_args = parser.parse()

    runtime = init_training_script(
        args,
        offline_grpo_config,
        model_config,
        dist_args,
        script_prefix="offline-grpo",
        supports_cp=False,
    )
    parallelism_config = runtime.parallelism_config

    # Every batch is padded here: prompts left, completions right.
    requested_attn = padded_workload_attn_implementation(model_config, sinks_reset=dist_args.reset_sinks)
    model, tokenizer = load_script_model(
        runtime, offline_grpo_config, model_config, dist_args, attn_implementation=requested_attn
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

    peft_config = setup_peft_model(args, model, model_config, "CAUSAL_LM")

    log_model_info(model, tokenizer)

    ds, dataset_presharded = load_script_datasets(
        args,
        parallelism_config,
        conversation_field="prompt",
    )
    reject_image_columns(ds, "Offline GRPO")

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

    callbacks = build_training_callbacks(
        args,
        offline_grpo_config,
        model,
        parallelism_config,
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
        callbacks=callbacks,
        **distributed_trainer_kwargs(args, dist_args, parallelism_config, dataset_presharded=dataset_presharded),
    )
    run_trainer(trainer, runtime, method_name="Offline GRPO")


if __name__ == "__main__":
    run_training(main)()
