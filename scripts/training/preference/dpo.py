#!/usr/bin/env python
"""Distributed DPO training (text or VLM): ``load_policy_model`` auto-detects the modality, and a VLM's processor flips
TRL 1.14's DPOTrainer into vision mode.

CP is not supported (``concatenated_forward`` needs full sequences). Under EP/TP use PEFT (``ref_model=None``) or
``precompute_ref_log_probs``, as the reference is not parallelized; PP runs text-only full fine-tuning of the per-pair
loss types (sigmoid, hinge, ipo) with ``precompute_ref_log_probs`` only."""

from trl import DPOConfig

from halo.config_validation import reject_unsupported_args
from halo.configs.parallelism_arguments import ParallelismArguments
from halo.configs.preference import DPOScriptArguments
from halo.configs.script_arguments import ModelArguments
from halo.data.render_columns import alias_images_column
from halo.data.vlm_detection import reject_unsupported_media_across_ranks
from halo.entrypoints.data_setup import (
    apply_vision_options,
    load_script_datasets,
    prepare_script_preference_data,
    resolve_preference_vlm_run,
)
from halo.entrypoints.model_setup import apply_text_path_padding_side, load_preference_models
from halo.entrypoints.script import (
    configure_balancing_and_build_callbacks,
    distributed_trainer_kwargs,
    init_training_script,
    parse_script_args,
    run_trainer,
    with_distributed_teardown,
)
from halo.runtime.collectives import barrier
from halo.trainers.preference.dpo import DistributedDPOTrainer


def main():
    args, dpo_config, model_config, dist_args = parse_script_args(
        DPOScriptArguments, DPOConfig, ModelArguments, ParallelismArguments
    )

    context = init_training_script(
        args,
        dpo_config,
        model_config,
        dist_args,
        script_prefix="dpo",
        trainer_cls=DistributedDPOTrainer,
    )
    parallelism_config = context.parallelism_config

    ds, dataset_presharded = load_script_datasets(args, parallelism_config)
    # Ahead of the dispatch: is_vlm_run_across_ranks reads images_field while TRL's vision probe reads the column
    # name, so a declared column has to carry TRL's spelling before either verdict is taken.
    ds = alias_images_column(ds, args.images_field, str(args.dataset))

    # Vision routing keys on the dataset: natively-multimodal models train on text preference data through the text
    # pipeline, and only an image-carrying dataset takes TRL's vision path, rows untouched. Before the model load,
    # which needs the checkpoint's processor.
    is_vlm_data = resolve_preference_vlm_run(
        args, model_config, ds, text_only_model=args.text_only_model, method="DPO", trl_renders_rows=True
    )
    if is_vlm_data:
        # TRL's DataCollatorForVisionPreference templates the rows without `tools=`, so a declared
        # tools column would survive the signature filter and render toolless.
        reject_unsupported_args("DPO VLM mode", tools_field=args.tools_field)
        reject_unsupported_media_across_ranks(ds, ("prompt", "chosen", "rejected"), "DPO")

    models = load_preference_models(context, dpo_config, model_config, args, vlm_run=is_vlm_data, method="DPO")
    apply_vision_options(args, models.model, model_config, models.processing_class, is_vlm=is_vlm_data)
    tokenizer = models.tokenizer

    apply_text_path_padding_side(tokenizer, is_vlm_data)
    train_dataset, eval_dataset, generate_callback = prepare_script_preference_data(
        args,
        dpo_config,
        ds,
        tokenizer,
        is_vlm_data=is_vlm_data,
        generation_max_prompt_length=args.generation_max_prompt_length,
    )

    callbacks = configure_balancing_and_build_callbacks(
        args, dpo_config, models.model, context, generate_callback=generate_callback
    )

    barrier()

    trainer = DistributedDPOTrainer(
        model=models.model,
        args=dpo_config,
        ref_model=models.ref_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=models.processing_class,
        peft_config=models.peft_config,
        callbacks=callbacks,
        resume_checkpoint=context.resume_checkpoint,
        policy_from_checkpoint=context.policy_from_checkpoint,
        **distributed_trainer_kwargs(args, parallelism_config, dataset_presharded=dataset_presharded),
    )
    run_trainer(
        trainer,
        context,
        method_name="DPO",
        extra_start_log=[f"modality: {'vlm' if is_vlm_data else 'text'}"],
    )


if __name__ == "__main__":
    with_distributed_teardown(main)()
