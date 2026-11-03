#!/usr/bin/env python
"""Distributed KTO training (text or VLM) on *unpaired* ``{prompt, completion, label}`` rows. A VLM's processor flips
TRL's KTOTrainer into vision mode (vision KTO is unpaired-only).

CP is not supported (full-sequence log-prob pooling + KL reference term). Under EP/TP use PEFT (``ref_model=None``) or
``precompute_ref_log_probs``; PP runs text-only ``apo_zero_unpaired`` full fine-tuning with
``precompute_ref_log_probs`` only."""

from trl import KTOConfig

from halo.config_validation import reject_unsupported_args
from halo.configs.parallelism_arguments import ParallelismArguments
from halo.configs.preference import KTOScriptArguments
from halo.configs.script_arguments import ModelArguments
from halo.data.pipeline.map_cache_key import carry_cache_key
from halo.data.render_columns import alias_images_column, reject_missing_render_column
from halo.data.vlm_detection import dataset_declares_images_across_ranks, reject_unsupported_media_across_ranks
from halo.entrypoints.data_setup import (
    apply_vision_options,
    load_script_datasets,
    log_script_dataset_examples,
    resolve_preference_vlm_run,
)
from halo.entrypoints.model_setup import load_preference_models
from halo.entrypoints.script import (
    configure_balancing_and_build_callbacks,
    distributed_trainer_kwargs,
    init_training_script,
    parse_script_args,
    run_trainer,
    with_distributed_teardown,
)
from halo.runtime.collectives import barrier
from halo.trainers.preference.kto import DistributedKTOTrainer

_KTO_RENAMED_FIELDS = (("completion_field", "completion"), ("label_field", "label"))


def _reject_embedded_image_parts(dataset, completion_field: str) -> None:
    """Raise on image parts in a dataset TRL would read as text-only: TRL decides from the first row's columns alone,
    so embedded parts with no image column become placeholder tokens with no pixels. KTO renders nothing itself, so the
    per-row ``reject_image_content`` backstop does not run here."""
    if dataset_declares_images_across_ranks(dataset):  # an image column, which TRL's own probe sees
        return
    # TRL hard-codes the prompt column; the completion column is the configured one.
    embedded = [
        column for column in ("prompt", completion_field) if dataset_declares_images_across_ranks(dataset, column)
    ]
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
    """Skipping the rename would leave TRL to fail on a missing column, or train on a stale ``completion``/``label``
    column under the default name."""
    for knob, target in _KTO_RENAMED_FIELDS:
        column = getattr(args, knob)
        if column != target:
            reject_missing_render_column(ds, str(args.dataset), knob, column)


def _rename_kto_columns(dataset, args):
    """Renames the pairs :func:`_require_kto_columns` checked, over the whole DatasetDict before the first rename."""
    for knob, target in _KTO_RENAMED_FIELDS:
        source = getattr(args, knob)
        if source != target:
            dataset = carry_cache_key(dataset, dataset.rename_column(source, target), f"rename:{source}->{target}")
    return dataset


def main():
    args, kto_config, model_config, dist_args = parse_script_args(
        KTOScriptArguments, KTOConfig, ModelArguments, ParallelismArguments
    )

    # TRL templates and tokenizes the raw columns itself: it passes no `tools=`, and its rows have no input_ids.
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
    # Ahead of the dispatch: is_vlm_run_across_ranks reads images_field while TRL's vision probe reads the column
    # name, so a declared column has to carry TRL's spelling before either verdict is taken.
    ds = alias_images_column(ds, args.images_field, str(args.dataset))
    _require_kto_columns(ds, args)
    # Vision routing keys on the dataset: a natively-multimodal model trains text-only data through TRL's text path.
    # Before the load, which needs the checkpoint's processor for a vision run.
    is_vlm_data = resolve_preference_vlm_run(
        args, model_config, ds, text_only_model=args.text_only_model, method="KTO"
    )
    _reject_embedded_image_parts(ds, args.completion_field)
    reject_unsupported_media_across_ranks(ds, ("prompt", args.completion_field), "KTO")

    models = load_preference_models(context, kto_config, model_config, args, vlm_run=is_vlm_data, method="KTO")
    apply_vision_options(args, models.model, model_config, models.processing_class, is_vlm=is_vlm_data)

    train_dataset = _rename_kto_columns(ds["train"], args)
    eval_dataset = _rename_kto_columns(ds["test"], args) if "test" in ds else None

    log_script_dataset_examples({"train": train_dataset, "test": eval_dataset}, models.tokenizer, args, kto_config)

    callbacks = configure_balancing_and_build_callbacks(args, kto_config, models.model, context)

    barrier()

    trainer = DistributedKTOTrainer(
        model=models.model,
        args=kto_config,
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
        method_name="KTO",
        extra_start_log=[f"modality: {'vlm' if is_vlm_data else 'text'}"],
    )


if __name__ == "__main__":
    with_distributed_teardown(main)()
