#!/usr/bin/env python
"""Distributed SMPO training (text or VLM).

Smooth Margin Preference Optimization — reference-model-free preference training. One script serves
both text and vision-language models. The model class follows the checkpoint
(``load_policy_model`` auto-detects it); the data path follows the run (``is_vlm_run``), so a
natively-multimodal checkpoint trained on text-only pairs takes the text path and keeps CP,
padding_free and PP. A VLM run passes its raw rows through: the trainer normalizes and
chat-templates them itself and auto-selects ``DataCollatorForVLMSMPO`` (image processing at
collation, vision tensors threaded through the chosen/rejected concat). A text run templates the
pairs here and runs generation-eval examples.

Supported Parallelism Modes: EP, CP, TP, ETP, PP and their allowlisted combinations
(``SUPPORTED_AXIS_SETS``; TP+CP unsupported; a VLM run supports none of CP, PP or padding_free).

Usage:
    torchrun --nproc_per_node=8 scripts/training/preference/smpo.py \\
        examples/preference/gptoss/smpo-gptoss-20b-tulu3-prefmix-ep.yaml

Dataset: text → {"prompt", "chosen", "rejected"} message lists; VLM → the same plus an
``images``/``image`` column (single image or list), which is what declares the VLM run.
"""

from trl import ModelConfig

from src.config_validation import reject_unsupported_args
from src.configs.parallelism_arguments import ParallelismArguments
from src.configs.preference import SmoothMarginPOConfig, SMPOScriptArguments
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
from src.model_loading.parallel.peft_setup import configure_adapter_training
from src.runtime.collectives import barrier
from src.trainers.preference.smpo import SmoothMarginPOTrainer


def main():
    args, smpo_config, model_config, dist_args = parse_script_args(
        SMPOScriptArguments, SmoothMarginPOConfig, ModelConfig, ParallelismArguments
    )

    context = init_training_script(
        args, smpo_config, model_config, dist_args, script_prefix="smpo", trainer_cls=SmoothMarginPOTrainer
    )
    parallelism_config = context.parallelism_config

    ds, dataset_presharded = load_script_datasets(args, parallelism_config)

    # The run's data path, not the checkpoint's modality: a multimodal checkpoint carrying text-only
    # pairs is a text run, and CP / padding_free / PP stay legal for it. Decided before the model
    # load, which needs the checkpoint's processor for a VLM run, so the CP and PP refusals below
    # raise before any weight is read (the trainer repeats them and adds padding_free).
    is_vlm = resolve_vlm_run(args, model_config, ds, text_only_model=args.text_only_model)
    if dist_args.context_parallel_size > 1 and is_vlm:
        raise ValueError("SMPO VLM mode does not support Context Parallelism — drop --context_parallel_size.")
    if dist_args.pipeline_parallel_size > 1 and is_vlm:
        raise ValueError("SMPO VLM mode does not support Pipeline Parallelism — drop --pipeline_parallel_size.")
    if is_vlm:
        # The trainer's VLM row render (tokenize_vlm_preference_row) templates the prompt without
        # `tools=`, so a declared tools column would render toolless.
        reject_unsupported_args("SMPO VLM mode", tools_field=args.tools_field)

    # --- Model (class follows the checkpoint); padded preference takes the shared padded-workload
    # backend (SDPA, dropped under live sinks and under CP) ---
    policy = load_policy_model(
        context,
        smpo_config,
        model_config,
        args,
        vlm_run=is_vlm,
        attn_default=default_padded_attn_implementation(
            model_config, sinks_reset=args.reset_sinks, context_parallel=parallelism_config.is_cp_mode
        ),
    )
    model = policy.model
    tokenizer = apply_max_length(smpo_config, args, model, policy.tokenizer)
    processing_class = install_resolved_tokenizer(policy.processing_class, tokenizer)
    apply_text_path_padding_side(tokenizer, is_vlm)

    peft_config = configure_adapter_training(args, model, model_config, "CAUSAL_LM")

    log_model_info(model, tokenizer)

    # On the VLM arm the rows pass through untouched: the trainer normalizes and chat-templates them
    # itself, through the processor it holds as processing_class.
    train_dataset, eval_dataset, generate_callback = prepare_script_preference_data(
        args,
        smpo_config,
        ds,
        tokenizer,
        is_vlm_data=is_vlm,
        generation_max_prompt_length=smpo_config.max_prompt_length,
    )

    callbacks = configure_balancing_and_build_callbacks(
        args, smpo_config, model, context, generate_callback=generate_callback
    )

    barrier()

    trainer = SmoothMarginPOTrainer(
        model=model,
        args=smpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processing_class,
        peft_config=peft_config,
        callbacks=callbacks,
        **distributed_trainer_kwargs(args, parallelism_config, dataset_presharded=dataset_presharded),
        is_vlm=is_vlm,
    )
    run_trainer(
        trainer,
        context,
        method_name="SMPO",
        extra_start_log=[f"modality: {'vlm' if is_vlm else 'text'}"],
    )


if __name__ == "__main__":
    with_distributed_teardown(main)()
