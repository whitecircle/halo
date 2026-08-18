#!/usr/bin/env python
"""Distributed KTO training (text or VLM) with EP and TP support.

Kahneman-Tversky Optimization on *unpaired* preference data — each row is a single
``{prompt, completion, label}`` triple (``label`` marks the completion desirable or not). One
script serves both text and vision-language models: ``load_model_for_training`` auto-detects the
modality. For a VLM the processor is the ``processing_class``, flipping TRL 1.6's KTOTrainer into
vision mode (``DataCollatorForVisionUnpairedPreference``; vision KTO is unpaired-only) — add an
``images``/``image`` column to the dataset, or point ``images_field`` at the column that holds them.

The vision-vs-text data path follows the dataset, and TRL decides it from the row's columns alone;
image parts embedded in the messages of a column-less dataset are refused here, since nothing on
the text path would render them with pixels behind them.

CP is not supported (full-sequence log-prob pooling + KL reference term); use EP and/or TP. Under
EP/TP use PEFT (``ref_model=None``) or ``precompute_ref_log_probs``.

Usage:
    torchrun --nproc_per_node=8 scripts/training/preference/kto.py \\
        examples/preference/qwen3_5/kto-qwen3.5-9b-kto-mix-14k.yaml --expert_parallel_size=8
"""

from trl import KTOConfig, ModelConfig

from src.args.distributed_args import DistributedArguments
from src.args.kto_args import KTOScriptArguments
from src.data.pipeline.processing import require_render_column
from src.data.sources.loading import alias_images_column
from src.data.vlm import dataset_declares_images, is_vlm_run
from src.distributed.loading.frozen_models import load_reference_model_for_preference
from src.distributed.loading.peft_setup import setup_peft_model
from src.distributed.loading.vlm_setup import load_model_for_training
from src.distributed.runtime import barrier
from src.models.loading.model_preparation import log_model_info
from src.trainers.preference.kto import DistributedKTOTrainer
from src.training.environment import run_training
from src.training.parser import H4ArgumentParser
from src.training.script_runner import (
    apply_distributed_trainer_config,
    apply_max_length,
    build_training_callbacks,
    distributed_trainer_kwargs,
    init_training_script,
    install_resolved_tokenizer,
    load_script_datasets,
    log_script_dataset_examples,
    padded_workload_attn_implementation,
    reject_images_under_text_only_model,
    reject_unsupported_args,
    run_trainer,
)


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


# The two column knobs KTO renames onto TRL's own spelling, as ``(knob, target)``.
_KTO_RENAMED_FIELDS = (("completion_field", "completion"), ("label_field", "label"))


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
            require_render_column(ds, str(args.dataset), knob, column)


def _rename_kto_columns(dataset, args):
    """Normalize dataset columns to KTO's expected ``completion``/``label`` names.

    Renames exactly the pairs :func:`_require_kto_columns` checked, from the same roster. Presence is
    that function's job, checked once over the whole DatasetDict before the first rename.
    """
    for knob, target in _KTO_RENAMED_FIELDS:
        source = getattr(args, knob)
        if source != target:
            dataset = dataset.rename_column(source, target)
    return dataset


def main():
    parser = H4ArgumentParser((KTOScriptArguments, KTOConfig, ModelConfig, DistributedArguments))
    args, kto_config, model_config, dist_args = parser.parse()

    # TRL's KTOTrainer chat-templates and tokenizes the raw prompt/completion columns itself: it
    # passes no `tools=`, so a dataset tool column would never reach the template, and the rows the
    # script hands it carry no input_ids, so the decoded-sample writer would have nothing to decode.
    reject_unsupported_args("KTO training", tools_field=args.tools_field, log_decoded_samples=args.log_decoded_samples)

    runtime = init_training_script(
        args,
        kto_config,
        model_config,
        dist_args,
        script_prefix="kto",
        trainer_cls=DistributedKTOTrainer,
        supports_cp=False,
    )
    parallelism_config = runtime.parallelism_config

    # --- Model (text or VLM, auto-detected); padded preference takes the shared padded-workload
    # backend (SDPA, dropped under live sinks). The reference load uses the same binding: a logratio
    # whose halves came from different attention kernels is biased. ---
    attn_default = padded_workload_attn_implementation(model_config, sinks_reset=dist_args.reset_sinks)
    model, processing_class, tokenizer, is_vlm = load_model_for_training(
        model_config,
        kto_config,
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
        kto_config,
        parallelism_config,
        tokenizer,
        is_vlm=is_vlm,
        method="KTO",
        reset_sinks=dist_args.reset_sinks,
        attn_default=attn_default,
    )

    tokenizer = apply_max_length(kto_config, args, model, tokenizer)
    processing_class = install_resolved_tokenizer(processing_class, tokenizer, is_vlm)
    log_model_info(model, tokenizer)

    ds, dataset_presharded = load_script_datasets(args, parallelism_config)
    reject_images_under_text_only_model(args, ds, text_only_model=dist_args.text_only_model)
    # Ahead of the dispatch: is_vlm_run reads images_field while TRL's vision probe reads the column
    # name, so a declared column has to carry TRL's spelling before either verdict is taken.
    ds = alias_images_column(ds, args.images_field, str(args.dataset))
    _reject_embedded_image_parts(ds, args.completion_field)
    # Vision routing keys on the dataset, not the checkpoint: a natively-multimodal model trains
    # text-only unpaired data through TRL's text path.
    is_vlm_data = is_vlm_run(args, model_config.model_name_or_path, ds, config=model.config)

    _require_kto_columns(ds, args)
    train_dataset = _rename_kto_columns(ds["train"], args)
    eval_dataset = _rename_kto_columns(ds["test"], args) if ds.get("test") else None

    log_script_dataset_examples({"train": train_dataset, "test": eval_dataset}, tokenizer, args, kto_config)

    apply_distributed_trainer_config(kto_config, parallelism_config)

    callbacks = build_training_callbacks(args, kto_config, model, parallelism_config)

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
        **distributed_trainer_kwargs(args, dist_args, parallelism_config, dataset_presharded=dataset_presharded),
    )
    run_trainer(
        trainer,
        runtime,
        method_name="KTO",
        extra_start_log=[f"modality: {'vlm' if is_vlm_data else 'text'}"],
    )


if __name__ == "__main__":
    run_training(main)()
