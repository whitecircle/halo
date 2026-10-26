#!/usr/bin/env python
"""Distributed DPO training (text or VLM).

Direct Preference Optimization on preference pairs (chosen vs rejected). One script serves both
text and vision-language models: ``load_policy_model`` auto-detects the modality. For a VLM
the processor is the ``processing_class``, which flips TRL 1.14's DPOTrainer into vision mode and
auto-selects ``DataCollatorForVisionPreference`` for an images-bearing dataset; for a text model
the repo tokenizes the pairs and runs generation-eval examples.

CP is not supported (``concatenated_forward`` needs full sequences). Under EP/TP use PEFT
(``ref_model=None``) or ``precompute_ref_log_probs`` — the reference is not parallelized; PP runs
text-only full fine-tuning of the per-pair loss types (sigmoid, hinge, ipo) with
``precompute_ref_log_probs`` only.

Usage:
    # Text or VLM (auto-detected) — EP / TP via torchrun
    torchrun --nproc_per_node=8 scripts/training/preference/dpo.py \\
        examples/preference/gptoss/dpo-gptoss-20b-tulu3-prefmix-ep.yaml

Dataset: text → {"prompt", "chosen", "rejected"} message lists; VLM → the same plus an
``images``/``image`` column (TRL applies the chat template and threads pixel_values), or
``images_field`` naming the column a hub dataset stores them in.
"""

from trl import DPOConfig

from src.config_validation import reject_unsupported_args
from src.configs.parallelism_arguments import ParallelismArguments
from src.configs.preference import DPOScriptArguments
from src.configs.script_arguments import ModelArguments
from src.data.render_columns import alias_images_column
from src.entrypoints.data_setup import load_script_datasets, prepare_script_preference_data, resolve_vlm_run
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
from src.model_loading.parallel.frozen_models import load_reference_model_for_preference
from src.model_loading.parallel.peft_setup import configure_adapter_training
from src.runtime.collectives import barrier
from src.trainers.preference.dpo import DistributedDPOTrainer


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
    # Ahead of the dispatch: is_vlm_run reads images_field while TRL's vision probe reads the column
    # name, so a declared column has to carry TRL's spelling before either verdict is taken.
    ds = alias_images_column(ds, args.images_field, str(args.dataset))

    # Vision routing keys on the dataset, not the model: natively-multimodal models train on text
    # preference data through the normal text pipeline, and only an image-carrying dataset takes TRL's
    # vision path. There the rows pass through untouched: TRL tokenizes them, auto-selects
    # DataCollatorForVisionPreference and applies no hub-shape normalization of its own. Decided
    # before the model load, which requires the checkpoint's processor for a vision run.
    is_vlm_data = resolve_vlm_run(args, model_config, ds, text_only_model=args.text_only_model)
    if is_vlm_data:
        # TRL's DataCollatorForVisionPreference templates the rows without `tools=`, so a declared
        # tools column would survive the signature filter and render toolless.
        reject_unsupported_args("DPO VLM mode", tools_field=args.tools_field)

    # --- Model (text or VLM, auto-detected); padded preference takes the shared padded-workload
    # backend (SDPA, dropped under live sinks). The reference load uses the same binding: a logratio
    # whose halves came from different kernels is biased.
    attn_default = default_padded_attn_implementation(model_config, sinks_reset=args.reset_sinks)
    policy = load_policy_model(context, dpo_config, model_config, args, vlm_run=is_vlm_data, attn_default=attn_default)
    model = policy.model

    peft_config = configure_adapter_training(args, model, model_config, "CAUSAL_LM")

    # Reference model: under PEFT the reference is the adapter-free base; full finetune loads a copy.
    model_ref = load_reference_model_for_preference(
        args,
        model_config,
        dpo_config,
        parallelism_config,
        policy.tokenizer,
        is_vlm=policy.is_vlm,
        method="DPO",
        reset_sinks=args.reset_sinks,
        attn_default=attn_default,
    )

    tokenizer = apply_max_length(dpo_config, args, model, policy.tokenizer)
    processing_class = install_resolved_tokenizer(policy.processing_class, tokenizer)
    log_model_info(model, tokenizer)

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
        args, dpo_config, model, context, generate_callback=generate_callback
    )

    barrier()

    trainer = DistributedDPOTrainer(
        model=model,
        args=dpo_config,
        ref_model=model_ref,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processing_class,
        peft_config=peft_config,
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
