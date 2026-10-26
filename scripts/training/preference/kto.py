#!/usr/bin/env python
"""Distributed KTO training (text or VLM).

Kahneman-Tversky Optimization on *unpaired* preference data — each row is a single
``{prompt, completion, label}`` triple (``label`` marks the completion desirable or not). One
script serves both text and vision-language models: ``load_policy_model`` auto-detects the
modality. For a VLM the processor is the ``processing_class``, flipping TRL 1.14's KTOTrainer into
vision mode (``DataCollatorForVisionUnpairedPreference``; vision KTO is unpaired-only) — add an
``images``/``image`` column to the dataset, or point ``images_field`` at the column that holds them.

The vision-vs-text data path follows the dataset, and TRL decides it from the row's columns alone;
image parts embedded in the messages of a column-less dataset are refused here, since nothing on
the text path would render them with pixels behind them.

CP is not supported (full-sequence log-prob pooling + KL reference term). Under EP/TP use PEFT
(``ref_model=None``) or ``precompute_ref_log_probs``; PP runs text-only ``apo_zero_unpaired`` full
fine-tuning with ``precompute_ref_log_probs`` only.

Usage:
    torchrun --nproc_per_node=8 scripts/training/preference/kto.py \\
        examples/preference/qwen3_5/kto-qwen3.5-9b-kto-mix-14k.yaml
"""

from trl import KTOConfig

from src.config_validation import reject_unsupported_args
from src.configs.parallelism_arguments import ParallelismArguments
from src.configs.preference import KTOScriptArguments
from src.configs.script_arguments import ModelArguments
from src.data.pipeline.map_cache_key import carry_cache_key
from src.data.render_columns import alias_images_column, reject_missing_render_column
from src.data.vlm_routing import dataset_declares_images
from src.entrypoints.data_setup import load_script_datasets, log_script_dataset_examples, resolve_vlm_run
from src.entrypoints.model_setup import (
    apply_max_length,
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
from src.trainers.preference.kto import DistributedKTOTrainer

_KTO_RENAMED_FIELDS = (("completion_field", "completion"), ("label_field", "label"))


def _reject_embedded_image_parts(dataset, completion_field: str) -> None:
    """Raise on image content parts in a dataset TRL would read as text-only.

    TRL's KTOTrainer decides vision-vs-text from the first row's columns alone, so a dataset that
    embeds ``{"type": "image"}`` parts in its prompt/completion messages while shipping no image
    column takes the text path: the chat template expands each part into vision placeholder tokens
    with no pixels behind them. KTO renders nothing itself (TRL templates the raw
    ``prompt``/``completion`` columns), so the per-row ``reject_image_content`` backstop the other
    preference scripts inherit does not run here.
    """
    if dataset_declares_images(dataset):  # an image column, which TRL's own probe sees
        return
    # TRL hard-codes the prompt column; the completion column is the configured one.
    embedded = [column for column in ("prompt", completion_field) if dataset_declares_images(dataset, column)]
    if not embedded:
        return
    raise ValueError(
        f"KTO dataset column(s) {embedded} embed image content parts, but the dataset carries no "
        f"images/image column, which is all TRL's vision-vs-text probe reads. The run would take the "
        f"TEXT path and render every image part as placeholder tokens with no pixels behind them. "
        f"Either move the images into an 'images' column (one list per row, leaving unfilled "
        f"{{'type': 'image'}} placeholders in the messages; images_field renames any other column to "
        f"it) to take TRL's vision path, or drop the image parts and train the text."
    )


def _require_kto_columns(ds, args) -> None:
    """Raise on a configured KTO column the dataset does not carry.

    Skipping the rename would leave TRL's KTOTrainer to fail downstream on a missing column, or to
    train on a stale ``completion``/``label`` column that happens to exist under the default name.
    Reported through the shared render-column guard, so a mistyped column reads the same here as on
    every other data path.
    """
    for knob, target in _KTO_RENAMED_FIELDS:
        column = getattr(args, knob)
        if column != target:
            reject_missing_render_column(ds, str(args.dataset), knob, column)


def _rename_kto_columns(dataset, args):
    """Normalize dataset columns to KTO's expected ``completion``/``label`` names.

    Renames exactly the pairs :func:`_require_kto_columns` checked, from the same roster. Presence is
    that function's job, checked once over the whole DatasetDict before the first rename.
    """
    for knob, target in _KTO_RENAMED_FIELDS:
        source = getattr(args, knob)
        if source != target:
            dataset = carry_cache_key(dataset, dataset.rename_column(source, target), f"rename:{source}->{target}")
    return dataset


def main():
    args, kto_config, model_config, dist_args = parse_script_args(
        KTOScriptArguments, KTOConfig, ModelArguments, ParallelismArguments
    )

    # TRL's KTOTrainer chat-templates and tokenizes the raw prompt/completion columns itself: it
    # passes no `tools=`, so a dataset tool column would never reach the template, and the rows the
    # script hands it carry no input_ids, so the decoded-sample writer would have nothing to decode.
    reject_unsupported_args("KTO training", tools_field=args.tools_field, log_decoded_samples=args.log_decoded_samples)

    context = init_training_script(
        args,
        kto_config,
        model_config,
        dist_args,
        script_prefix="kto",
        trainer_cls=DistributedKTOTrainer,
    )
    parallelism_config = context.parallelism_config

    ds, dataset_presharded = load_script_datasets(args, parallelism_config)
    # Ahead of the dispatch: is_vlm_run reads images_field while TRL's vision probe reads the column
    # name, so a declared column has to carry TRL's spelling before either verdict is taken.
    ds = alias_images_column(ds, args.images_field, str(args.dataset))
    # Vision routing keys on the dataset, not the checkpoint: a natively-multimodal model trains
    # text-only unpaired data through TRL's text path. Decided before the model load, which requires
    # the checkpoint's processor for a vision run.
    is_vlm_data = resolve_vlm_run(args, model_config, ds, text_only_model=args.text_only_model)
    _reject_embedded_image_parts(ds, args.completion_field)

    # --- Model (text or VLM, auto-detected); padded preference takes the shared padded-workload
    # backend (SDPA, dropped under live sinks). The reference load uses the same binding: a logratio
    # whose halves came from different attention kernels is biased. ---
    attn_default = default_padded_attn_implementation(model_config, sinks_reset=args.reset_sinks)
    policy = load_policy_model(context, kto_config, model_config, args, vlm_run=is_vlm_data, attn_default=attn_default)
    model = policy.model

    peft_config = configure_adapter_training(args, model, model_config, "CAUSAL_LM")

    # Reference model: under PEFT the reference is the adapter-free base; full finetune loads a copy.
    model_ref = load_reference_model_for_preference(
        args,
        model_config,
        kto_config,
        parallelism_config,
        policy.tokenizer,
        is_vlm=policy.is_vlm,
        method="KTO",
        reset_sinks=args.reset_sinks,
        attn_default=attn_default,
    )

    tokenizer = apply_max_length(kto_config, args, model, policy.tokenizer)
    processing_class = install_resolved_tokenizer(policy.processing_class, tokenizer)
    log_model_info(model, tokenizer)

    _require_kto_columns(ds, args)
    train_dataset = _rename_kto_columns(ds["train"], args)
    eval_dataset = _rename_kto_columns(ds["test"], args) if "test" in ds else None

    log_script_dataset_examples({"train": train_dataset, "test": eval_dataset}, tokenizer, args, kto_config)

    callbacks = configure_balancing_and_build_callbacks(args, kto_config, model, context)

    barrier()

    trainer = DistributedKTOTrainer(
        model=model,
        args=kto_config,
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
        method_name="KTO",
        extra_start_log=[f"modality: {'vlm' if is_vlm_data else 'text'}"],
    )


if __name__ == "__main__":
    with_distributed_teardown(main)()
