#!/usr/bin/env python
"""Distributed SMPO training (text or VLM) with Expert, Context, and Tensor Parallelism support.

Smooth Margin Preference Optimization — reference-model-free preference training. One script serves
both text and vision-language models. The model class follows the checkpoint
(``load_model_for_training`` auto-detects it); the data path follows the run (``is_vlm_run``), so a
natively-multimodal checkpoint trained on text-only pairs takes the text path and keeps CP,
padding_free and PP. A VLM run passes its raw rows through: the trainer normalizes and
chat-templates them itself and auto-selects ``DataCollatorForVLMSMPO`` (image processing at
collation, vision tensors threaded through the chosen/rejected concat). A text run templates the
pairs here and runs generation-eval examples.

Supported Parallelism Modes: EP, CP, TP, EP+CP, EP+TP (TP+CP unsupported; a VLM run supports
neither CP nor padding_free).

Usage:
    torchrun --nproc_per_node=8 scripts/training/preference/smpo.py \\
        examples/preference/qwen3_5/smpo-qwen3.5-9b-tulu3-prefmix.yaml --expert_parallel_size=8

Dataset: text → {"prompt", "chosen", "rejected"} message lists; VLM → the same plus an
``images``/``image`` column (single image or list), which is what declares the VLM run.
"""

from trl import ModelConfig

from src.args.distributed_args import DistributedArguments
from src.args.smpo_args import SMPOScriptArguments
from src.configs.smpo_config import SmoothMarginPOConfig
from src.data.vlm import is_vlm_run
from src.distributed.loading.peft_setup import setup_peft_model
from src.distributed.loading.vlm_setup import load_model_for_training
from src.distributed.runtime import barrier
from src.models.loading.model_preparation import log_model_info
from src.trainers.preference.smpo import SmoothMarginPOTrainer
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
    padded_workload_attn_implementation,
    prepare_script_preference_data,
    reject_images_under_text_only_model,
    reject_unsupported_args,
    run_trainer,
)


def main():
    parser = H4ArgumentParser((SMPOScriptArguments, SmoothMarginPOConfig, ModelConfig, DistributedArguments))
    args, smpo_config, model_config, dist_args = parser.parse()

    runtime = init_training_script(
        args, smpo_config, model_config, dist_args, script_prefix="smpo", trainer_cls=SmoothMarginPOTrainer
    )
    parallelism_config = runtime.parallelism_config

    ds, dataset_presharded = load_script_datasets(args, parallelism_config)
    # Ahead of the verdict below, which reads the checkpoint's config: under text_only_model that
    # config still says multimodal while the loaded class has no vision path.
    reject_images_under_text_only_model(args, ds, text_only_model=dist_args.text_only_model)

    # The run's data path, not the checkpoint's modality: a multimodal checkpoint carrying text-only
    # pairs is a text run, and CP / padding_free / PP stay legal for it. Decided here so the VLM
    # guards (the trainer enforces the same ones) raise before the model load, and pinned to the same
    # revision as that load, since hub `main` can name a different modality than the commit this run
    # trains.
    is_vlm = is_vlm_run(args, model_config.model_name_or_path, ds, revision=model_config.model_revision)
    if dist_args.context_parallel_size > 1 and is_vlm:
        raise ValueError("SMPO VLM mode does not support Context Parallelism — drop --context_parallel_size.")
    if is_vlm:
        # The trainer's VLM row render (tokenize_vlm_preference_row) templates the prompt without
        # `tools=`, so a declared tools column would render toolless.
        reject_unsupported_args("SMPO VLM mode", tools_field=args.tools_field)

    # --- Model (class follows the checkpoint); padded preference takes the shared padded-workload
    # backend (SDPA, dropped under live sinks) ---
    model, processing_class, tokenizer, is_vlm_checkpoint = load_model_for_training(
        model_config,
        smpo_config,
        parallelism_config,
        attn_default=padded_workload_attn_implementation(model_config, sinks_reset=dist_args.reset_sinks),
        reset_sinks=dist_args.reset_sinks,
        train_sinks=dist_args.train_sinks,
        weights_source=runtime.model_source,
        text_only_model=dist_args.text_only_model,
    )

    tokenizer = apply_max_length(smpo_config, args, model, tokenizer)
    # The checkpoint verdict: a text run on a multimodal checkpoint still saves through the
    # processor, so every checkpoint it writes carries a processor_config.json.
    processing_class = install_resolved_tokenizer(processing_class, tokenizer, is_vlm_checkpoint)
    enforce_text_path_padding_side(tokenizer, is_vlm)

    peft_config = setup_peft_model(args, model, model_config, "CAUSAL_LM")

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

    callbacks = build_training_callbacks(
        args, smpo_config, model, parallelism_config, generate_callback=generate_callback
    )

    apply_distributed_trainer_config(smpo_config, parallelism_config)

    barrier()

    trainer = SmoothMarginPOTrainer(
        model=model,
        args=smpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processing_class,
        peft_config=peft_config,
        callbacks=callbacks,
        **distributed_trainer_kwargs(args, dist_args, parallelism_config, dataset_presharded=dataset_presharded),
        is_vlm=is_vlm,
    )
    run_trainer(
        trainer,
        runtime,
        method_name="SMPO",
        extra_start_log=[f"modality: {'vlm' if is_vlm else 'text'}"],
    )


if __name__ == "__main__":
    run_training(main)()
