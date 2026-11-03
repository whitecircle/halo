#!/usr/bin/env python
"""Distributed Bradley-Terry reward-model training on preference pairs, text or vision-language. The model class
follows the checkpoint, the data path the run.

A multimodal run needs a multimodal sequence-classification head: transformers ships few, and
halo/models/seq_cls_heads.py registers Gemma 4 and MoE Qwen3.5/3.6; any other is refused before the load. CP is not
supported (the score head pools the complete sequence). PP admits a multimodal checkpoint only without images."""

from transformers import AutoModelForSequenceClassification
from trl import RewardConfig

from halo.config_validation import reject_unsupported_args
from halo.configs.parallelism_arguments import ParallelismArguments
from halo.configs.reward import RewardScriptArguments
from halo.configs.script_arguments import ModelArguments
from halo.data.render_columns import alias_images_column, alias_tools_column
from halo.data.vlm_detection import is_vlm_run
from halo.entrypoints.data_setup import load_script_datasets, log_script_dataset_examples
from halo.entrypoints.model_setup import (
    apply_max_length,
    default_padded_attn_implementation,
    install_resolved_tokenizer,
    load_model_and_tokenizer,
    load_vlm_processor,
    log_model_info,
    reject_multimodal_model_without_seq_cls_head,
)
from halo.entrypoints.script import (
    configure_balancing_and_build_callbacks,
    distributed_trainer_kwargs,
    init_training_script,
    parse_script_args,
    run_trainer,
    with_distributed_teardown,
)
from halo.model_loading.parallel.peft_setup import configure_adapter_training
from halo.runtime.collectives import barrier
from halo.trainers.reward.bradley_terry import DistributedRewardTrainer


def main():
    args, reward_config, model_config, dist_args = parse_script_args(
        RewardScriptArguments, RewardConfig, ModelArguments, ParallelismArguments
    )

    # text_only_model: the score head is pinned to AutoModelForSequenceClassification. log_decoded_samples: rows reach
    # the trainer untokenized, so there is nothing to decode. Rejected, as neither can take effect.
    reject_unsupported_args(
        "Reward modeling",
        text_only_model=args.text_only_model,
        log_decoded_samples=args.log_decoded_samples,
    )

    context = init_training_script(
        args,
        reward_config,
        model_config,
        dist_args,
        script_prefix="reward",
        trainer_cls=DistributedRewardTrainer,
    )
    parallelism_config = context.parallelism_config

    # Before the load: a family with no score head fails inside Auto* resolution on every rank, naming no alternative.
    # After the distributed init, so the hub read is ordered main-rank-first.
    reject_multimodal_model_without_seq_cls_head(model_config)

    model, tokenizer = load_model_and_tokenizer(
        context,
        reward_config,
        model_config,
        args,
        model_class=AutoModelForSequenceClassification,
        model_config_overrides={"num_labels": 1},
        attn_default=default_padded_attn_implementation(model_config, sinks_reset=args.reset_sinks),
    )
    tokenizer = apply_max_length(reward_config, args, model, tokenizer)

    peft_config = configure_adapter_training(args, model, model_config, "SEQ_CLS")

    log_model_info(model, tokenizer)

    ds, dataset_presharded = load_script_datasets(args, parallelism_config)
    # Ahead of the dispatch: is_vlm_run reads images_field while the vision map reads the aliased
    # column, so the alias keeps the two verdicts identical and a mistyped name detectable.
    ds = alias_images_column(ds, args.images_field, str(args.dataset))
    # TRL's RewardTrainer renders the column literally named ``tools`` into the chat template and no
    # other, so a declared tools column is aliased onto that spelling rather than rendered toolless.
    ds = alias_tools_column(ds, args.tools_field, str(args.dataset))

    # Decided now the dataset is known: a multimodal checkpoint on text-only pairs is a text run, class unchanged.
    is_vlm = is_vlm_run(args, model_config.model_name_or_path, ds, config=model.config)
    if is_vlm:
        # The VLM pair render (halo/data/pipeline/preferences.py) templates the prompt without
        # `tools=`, so the aliased column would render toolless on this arm.
        reject_unsupported_args("Reward modeling (VLM)", tools_field=args.tools_field)
    # The processor only on the VLM branch: TRL's text tokenize map calls tokenizer-only methods, and the trainer reads
    # this object's modality as the run's verdict. Its inner tokenizer is the right-padded one TRL and the head expect.
    processing_class = tokenizer
    if is_vlm:
        processing_class = install_resolved_tokenizer(load_vlm_processor(model_config), tokenizer)

    # No pre-tokenization on either branch: TRL tokenizes the raw columns and filters over-length rows, and the VLM
    # branch renders pairs in its own map.
    train_dataset = ds["train"]
    eval_dataset = ds["test"]

    log_script_dataset_examples({"train": train_dataset, "test": eval_dataset}, tokenizer, args, reward_config)

    barrier()

    callbacks = configure_balancing_and_build_callbacks(args, reward_config, model, context)

    trainer = DistributedRewardTrainer(
        model=model,
        processing_class=processing_class,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        callbacks=callbacks,
        **distributed_trainer_kwargs(args, parallelism_config, dataset_presharded=dataset_presharded),
    )
    run_trainer(
        trainer,
        context,
        method_name="Reward",
        extra_start_log=[f"modality: {'vlm' if is_vlm else 'text'}"],
    )


if __name__ == "__main__":
    with_distributed_teardown(main)()
