#!/usr/bin/env python
"""Distributed DPO training (text or VLM) with Expert and Tensor Parallelism support.

Direct Preference Optimization on preference pairs (chosen vs rejected). One script serves both
text and vision-language models: ``load_model_for_training`` auto-detects the modality. For a VLM
the processor is the ``processing_class``, which flips TRL 1.6's DPOTrainer into vision mode and
auto-selects ``DataCollatorForVisionPreference`` for an images-bearing dataset; for a text model
the repo tokenizes the pairs and runs generation-eval examples.

CP is not supported (``concatenated_forward`` needs full sequences); use EP and/or TP. Under EP/TP
use PEFT (``ref_model=None``) or ``precompute_ref_log_probs`` — the reference is not parallelized.

Usage:
    # Text or VLM (auto-detected) — EP / TP via torchrun
    torchrun --nproc_per_node=8 scripts/training/preference/dpo.py \\
        examples/preference/qwen3_5/dpo-qwen3.5-9b-tulu3-prefmix.yaml --expert_parallel_size=8

Dataset: text → {"prompt", "chosen", "rejected"} message lists; VLM → the same plus an
``images``/``image`` column (TRL applies the chat template and threads pixel_values), or
``images_field`` naming the column a hub dataset stores them in.
"""

from trl import DPOConfig, ModelConfig

from src.args.distributed_args import DistributedArguments
from src.args.dpo_args import DPOScriptArguments
from src.data.sources.loading import alias_images_column
from src.data.vlm import is_vlm_run
from src.distributed.loading.frozen_models import load_reference_model_for_preference
from src.distributed.loading.peft_setup import setup_peft_model
from src.distributed.loading.vlm_setup import load_model_for_training
from src.distributed.runtime import barrier
from src.models.loading.model_preparation import log_model_info
from src.trainers.preference.dpo import DistributedDPOTrainer
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
    parser = H4ArgumentParser((DPOScriptArguments, DPOConfig, ModelConfig, DistributedArguments))
    args, dpo_config, model_config, dist_args = parser.parse()

    # DPOConfig also declares pad_token; the resolve-conflict parser captures the YAML key there,
    # so sync_tokens mirrors it back onto the script args the tokenizer setup reads.
    runtime = init_training_script(
        args,
        dpo_config,
        model_config,
        dist_args,
        script_prefix="dpo",
        trainer_cls=DistributedDPOTrainer,
        supports_cp=False,
        sync_tokens=("pad_token",),
    )
    parallelism_config = runtime.parallelism_config

    # --- Model (text or VLM, auto-detected); padded preference takes the shared padded-workload
    # backend (SDPA, dropped under live sinks). The reference load uses the same binding: a logratio
    # whose halves came from different kernels is biased.
    attn_default = padded_workload_attn_implementation(model_config, sinks_reset=dist_args.reset_sinks)
    model, processing_class, tokenizer, is_vlm = load_model_for_training(
        model_config,
        dpo_config,
        parallelism_config,
        attn_default=attn_default,
        reset_sinks=dist_args.reset_sinks,
        train_sinks=dist_args.train_sinks,
        weights_source=runtime.model_source,
        text_only_model=dist_args.text_only_model,
    )

    peft_config = setup_peft_model(args, model, model_config, "CAUSAL_LM")

    # Reference model: under PEFT the reference is the adapter-free base; full finetune loads a copy.
    model_ref = load_reference_model_for_preference(
        args,
        model_config,
        dpo_config,
        parallelism_config,
        tokenizer,
        is_vlm=is_vlm,
        method="DPO",
        reset_sinks=dist_args.reset_sinks,
        attn_default=attn_default,
    )

    tokenizer = apply_max_length(dpo_config, args, model, tokenizer)
    processing_class = install_resolved_tokenizer(processing_class, tokenizer, is_vlm)
    log_model_info(model, tokenizer)

    ds, dataset_presharded = load_script_datasets(args, parallelism_config)
    reject_images_under_text_only_model(args, ds, text_only_model=dist_args.text_only_model)
    # Ahead of the dispatch: is_vlm_run reads images_field while TRL's vision probe reads the column
    # name, so a declared column has to carry TRL's spelling before either verdict is taken.
    ds = alias_images_column(ds, args.images_field, str(args.dataset))

    # Vision routing keys on the dataset, not the model: natively-multimodal models train on text
    # preference data through the normal text pipeline, and only an image-carrying dataset takes TRL's
    # vision path. There the rows pass through untouched: TRL tokenizes them, auto-selects
    # DataCollatorForVisionPreference and applies no hub-shape normalization of its own.
    is_vlm_data = is_vlm_run(args, model_config.model_name_or_path, ds, config=model.config)
    if is_vlm_data:
        # TRL's DataCollatorForVisionPreference templates the rows without `tools=`, so a declared
        # tools column would survive the signature filter and render toolless.
        reject_unsupported_args("DPO VLM mode", tools_field=args.tools_field)
    enforce_text_path_padding_side(tokenizer, is_vlm_data)
    train_dataset, eval_dataset, generate_callback = prepare_script_preference_data(
        args,
        dpo_config,
        ds,
        tokenizer,
        is_vlm_data=is_vlm_data,
        generation_max_prompt_length=args.generation_max_prompt_length,
    )

    apply_distributed_trainer_config(dpo_config, parallelism_config)

    callbacks = build_training_callbacks(
        args, dpo_config, model, parallelism_config, generate_callback=generate_callback
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
        **distributed_trainer_kwargs(args, dist_args, parallelism_config, dataset_presharded=dataset_presharded),
    )
    run_trainer(
        trainer,
        runtime,
        method_name="DPO",
        extra_start_log=[f"modality: {'vlm' if is_vlm_data else 'text'}"],
    )


if __name__ == "__main__":
    run_training(main)()
