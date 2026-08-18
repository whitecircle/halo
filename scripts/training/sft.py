#!/usr/bin/env python
"""Distributed SFT training (text or VLM) with Expert, Context, and Tensor Parallelism support.

Supervised fine-tuning for both language and vision-language models. The model class follows the
checkpoint; the DATA path follows the run (``is_vlm_run``), so a natively-multimodal checkpoint
trained on a text-only dataset takes the text path. That path supports packing, padding-free,
pre-processed datasets, QLoRA, and ``init_from_scratch``; the VLM path loads an
``AutoModelForImageTextToText`` + processor and uses the VLM collators (packing/padding-free do not
apply to images).

Supported Parallelism Modes: EP, CP, TP, EP+CP, EP+TP (TP+CP unsupported; CP incompatible with
padding-free, and CP patches only the text-decoder attention on VLMs).

Usage:
    torchrun --nproc_per_node=8 scripts/training/sft.py \\
        examples/sft/gptoss/gptoss-20b-multinode-ep.yaml --expert_parallel_size=8
"""

from accelerate.logging import get_logger
from trl import ModelConfig, SFTConfig

from src.args.distributed_args import DistributedArguments
from src.args.sft_args import SFTScriptArguments
from src.callbacks.generate_examples import GenerateExamplesCallback
from src.data.collators.factory import select_data_collator
from src.data.collators.vlm import PreprocessedVLMDataCollator, VLMDataCollator
from src.data.pipeline.preprocessed_metadata import load_preprocessed_metadata, validate_preprocessing_compatibility
from src.data.pipeline.processing import (
    pack_dataset_coordinated,
    process_dataset_with_map_and_filter,
    resolve_map_num_proc,
)
from src.data.pipeline.row_processors import create_llm_processor
from src.data.pipeline.vlm_dataset import prepare_vlm_dataset, vlm_map_features
from src.data.probe_consensus import agree_probe_across_ranks
from src.data.sources.loading import load_datasets_auto
from src.data.vlm import is_vlm_run
from src.distributed.loading.peft_setup import setup_peft_model
from src.distributed.loading.vlm_setup import load_model_for_training
from src.distributed.runtime import barrier, init_distributed, is_global_main_process
from src.models.loading.model_preparation import log_model_info
from src.models.modality import is_vlm_model
from src.trainers.sft import DistributedSFTTrainer
from src.training.environment import run_training
from src.training.parser import H4ArgumentParser
from src.training.script_runner import (
    apply_distributed_trainer_config,
    apply_max_length,
    build_training_callbacks,
    disable_trl_dataset_prep,
    distributed_trainer_kwargs,
    enforce_text_path_padding_side,
    init_training_script,
    install_resolved_tokenizer,
    load_script_datasets,
    log_script_dataset_examples,
    reject_images_under_text_only_model,
    reject_unsupported_args,
    run_trainer,
)

logger = get_logger(__name__, log_level="INFO")


def _reject_empty_split(dataset, split: str, max_length: int) -> None:
    """Fail loud on a split the length filter emptied, before packing turns it into a cryptic error.

    The chat processor drops every row whose tokenized length exceeds ``max_length`` (it does not
    truncate — truncating a conversation mid-turn corrupts it). Packing an empty split yields a
    dataset with no ``seq_lengths`` column and fails far from the cause.

    The length is a per-rank fact on a presharded corpus (each data-parallel rank holds a disjoint
    slice), and the packing that follows is coordinated — one barrier plus two store phases — so the
    verdict is agreed on the world and the raise lands on every rank at once.
    """
    if agree_probe_across_ranks(len(dataset) == 0, f"the {split} split", "split emptied by tokenization"):
        raise ValueError(
            f"All {split} rows were dropped after tokenization on at least one data-parallel rank: "
            f"every conversation exceeded max_length={max_length}. Increase max_length (e.g. for long "
            f"agent/tool traces) or use a dataset with shorter conversations."
        )


def _prepare_text_data(ds, is_preprocessed, args, sft_config, model_config, tokenizer, parallelism_config, hf_config):
    """Text dataset path: preprocessed passthrough, or raw tokenize (+ packing); returns
    (train, eval, generate, collator)."""
    if is_preprocessed:
        metadata = load_preprocessed_metadata(args.dataset)
        validate_preprocessing_compatibility(
            metadata,
            required_max_length=sft_config.max_length,
            required_model=model_config.model_name_or_path,
            required_train_on_completions_only=args.train_on_completions_only,
            render_args=args,
            required_packing=sft_config.packing,
        )
        # Offline-packed data still needs the packing collator (per-document position_ids from seq_lengths, else
        # documents attend across pack boundaries); the collator follows the DATA (metadata.packed), not the flag.
        collator_packing = metadata.packed
        train_dataset, eval_dataset, generate_dataset = ds["train"], ds["test"], None
    else:
        collator_packing = sft_config.packing
        use_padding = not sft_config.padding_free
        common = {
            "tokenizer": tokenizer,
            "max_length": sft_config.max_length,
            "conversation_field": args.conversation_field,
            "system_prompt": args.system_prompt,
            "model_supports_system_role": args.model_supports_system_role,
            "interleaved_thinking": args.interleaved_thinking,
            "tools_field": args.tools_field,
        }
        train_processor = create_llm_processor(**common, add_generation_prompt=False, use_padding=use_padding)
        generate_processor = create_llm_processor(**common, add_generation_prompt=True, use_padding=True)
        # sorted: this list feeds the coordinated-map cache key — set order is hash-randomized per process.
        extra_columns = sorted(set(ds["train"].column_names))
        map_kwargs = {"num_proc": resolve_map_num_proc(sft_config.dataset_num_proc)}
        cache_extras = {
            "conversation_field": args.conversation_field,
            "system_prompt": args.system_prompt,
            "model_supports_system_role": args.model_supports_system_role,
            "interleaved_thinking": args.interleaved_thinking,
            "tools_field": args.tools_field,
        }
        # Build the generation set from the raw test split before the train map remaps `ds`.
        generate_dataset = process_dataset_with_map_and_filter(
            ds["test"],
            generate_processor,
            desc="Processing generate dataset",
            cache_key_extras={**cache_extras, "add_generation_prompt": True},
            **map_kwargs,
        )
        ds = process_dataset_with_map_and_filter(
            ds,
            train_processor,
            remove_columns=extra_columns,
            desc="Processing dataset",
            cache_key_extras={**cache_extras, "add_generation_prompt": False},
            **map_kwargs,
        )
        train_dataset, eval_dataset = ds["train"], ds["test"]
        _reject_empty_split(train_dataset, "training", sft_config.max_length)
        if sft_config.packing:
            eval_packing = sft_config.eval_packing if sft_config.eval_packing is not None else sft_config.packing
            train_dataset = pack_dataset_coordinated(
                train_dataset, seq_length=sft_config.max_length, strategy=sft_config.packing_strategy, split="train"
            )
            if eval_packing:
                _reject_empty_split(eval_dataset, "evaluation", sft_config.max_length)
                eval_dataset = pack_dataset_coordinated(
                    eval_dataset, seq_length=sft_config.max_length, strategy=sft_config.packing_strategy, split="eval"
                )

    collator = select_data_collator(
        tokenizer=tokenizer,
        padding_free=sft_config.padding_free,
        packing=collator_packing,
        # Preprocessed labels are baked (completion masking included); runtime re-masking would overwrite them —
        # re-masking a packed chunk whose response marker landed in the previous chunk drops its trained tokens.
        train_on_completions_only=args.train_on_completions_only and not is_preprocessed,
        assistant_message_template=args.assistant_message_template,
        pad_to_multiple_of=parallelism_config.cp_size if parallelism_config.is_cp_mode else None,
        use_context_parallel=parallelism_config.is_cp_mode,
        train_on_last_assistant_only=args.train_on_last_assistant_only,
        model_config=hf_config,
        per_device_train_batch_size=sft_config.per_device_train_batch_size,
        keeps_packed_rows=parallelism_config.pp_size > 1,
    )
    return train_dataset, eval_dataset, generate_dataset, collator


def _prepare_vlm_data(ds, is_preprocessed, args, sft_config, model_config, processor, tokenizer, hf_config):
    """VLM dataset path: preprocessed passthrough, or raw VLM map; returns
    (train, eval, generate, collator)."""
    if is_preprocessed:
        metadata = load_preprocessed_metadata(args.dataset)
        validate_preprocessing_compatibility(
            metadata,
            required_max_length=sft_config.max_length,
            required_model=model_config.model_name_or_path,
            required_train_on_completions_only=args.train_on_completions_only,
            render_args=args,
            required_packing=sft_config.packing,
        )
        if not metadata.is_vlm:
            raise ValueError(f"Dataset at '{args.dataset}' is not a VLM preprocessed dataset.")
        collator = PreprocessedVLMDataCollator(tokenizer=tokenizer, max_length=sft_config.max_length)
        return ds["train"], ds["test"], None, collator

    # Fail loud on options the VLM path does not implement — silently ignoring them trains something
    # other than what the config states.
    if sft_config.packing or sft_config.padding_free:
        raise ValueError("packing / padding_free are not supported for VLM training (images cannot be packed).")
    if args.train_on_last_assistant_only:
        raise ValueError(
            "train_on_last_assistant_only is not implemented on the VLM path (all assistant turns train)."
        )

    num_proc = resolve_map_num_proc(sft_config.dataset_num_proc)
    ds = prepare_vlm_dataset(
        ds,
        args,
        processor,
        tokenizer,
        sft_config.max_length,
        num_proc,
        features=vlm_map_features(),  # pinned schema — shard-wise inference diverges on mixed datasets
        desc="Processing train/eval dataset",
    )
    # The mapped columns (history/images/…) are not model-forward kwargs — HF's default
    # remove_unused_columns=True would strip them before they reach the collator.
    sft_config.remove_unused_columns = False
    collator = VLMDataCollator(
        processor,
        tokenizer,
        sft_config.max_length,
        response_prompt_template=args.assistant_message_template if args.train_on_completions_only else None,
        train_on_completions_only=args.train_on_completions_only,
        model_config=hf_config,
    )
    # No generate dataset: GenerateExamplesCallback needs tokenized input_ids and is skipped for VLM,
    # so mapping the test split with the generation processor was pure wasted work.
    return ds["train"], ds["test"], None, collator


def main():
    parser = H4ArgumentParser((SFTScriptArguments, SFTConfig, ModelConfig, DistributedArguments))
    args, sft_config, model_config, dist_args = parser.parse()

    if dist_args.context_parallel_size > 1 and sft_config.padding_free:
        raise ValueError("Context Parallelism (CP) is NOT compatible with padding_free mode. Disable padding_free.")
    if dist_args.pipeline_parallel_size > 1 and sft_config.padding_free:
        # The trainer mixin rejects this too, keyed on the live collator — but only after the model
        # load and dataset prep. Same message shape as the CP gate: fail before anything expensive.
        raise ValueError(
            "Pipeline Parallelism (PP) is NOT compatible with padding_free mode (the flattened "
            "width varies every step while the pipeline's P2P buffers freeze on the first). Use "
            "packing instead."
        )
    if sft_config.packing and sft_config.padding_free:
        raise ValueError("Cannot use both 'packing' and 'padding_free' simultaneously.")
    # TRL applies these inside its own dataset prep + default collator, both replaced here, so they would
    # parse and mask nothing. Completion masking is train_on_completions_only + assistant_message_template.
    reject_unsupported_args(
        "Halo SFT",
        # Tri-state: an explicit False ("train on the full sequence") is as silently-ignored as True.
        completion_only_loss=sft_config.completion_only_loss is not None,
        assistant_only_loss=sft_config.assistant_only_loss,
    )

    # The CHECKPOINT's modality: it picks the processing class and the run label, both needed before
    # the dataset exists. Same pin the model load uses — an unpinned probe reads hub `main`, whose
    # config can name a different modality than the commit this run trains. The process group comes
    # first (init_training_script's own call is then a no-op): this is the run's first hub read, and
    # its main-rank-first ordering — the guard on transformers' unlocked remote-code module cache —
    # exists only under a live group; before it every rank of every node fetches at once.
    init_distributed()
    is_vlm_checkpoint = is_vlm_model(model_config.model_name_or_path, revision=model_config.model_revision)
    runtime = init_training_script(
        args,
        sft_config,
        model_config,
        dist_args,
        script_prefix=f"sft{'-vlm' if is_vlm_checkpoint and not dist_args.text_only_model else ''}",
        sync_tokens=("eos_token", "pad_token"),
        allow_low_precision=True,
        supports_init_from_scratch=True,
    )
    parallelism_config = runtime.parallelism_config

    model, processing_class, tokenizer, is_vlm_checkpoint = load_model_for_training(
        model_config,
        sft_config,
        parallelism_config,
        reset_sinks=dist_args.reset_sinks,
        train_sinks=dist_args.train_sinks,
        init_from_scratch=dist_args.init_from_scratch,
        weights_source=runtime.model_source,
        text_only_model=dist_args.text_only_model,
    )

    # Resolve max_length: null/non-positive → the model's context window. Packing uses it as the fixed pack
    # size that bounds memory, so require it explicitly there instead of silently defaulting to full context.
    if (sft_config.max_length is None or sft_config.max_length <= 0) and sft_config.packing:
        raise ValueError(
            "packing=True requires an explicit max_length (the fixed pack/sequence size that bounds "
            "memory); it cannot default to the model context window. Set max_length in the config."
        )
    tokenizer = apply_max_length(sft_config, args, model, tokenizer)
    processing_class = install_resolved_tokenizer(processing_class, tokenizer, is_vlm_checkpoint)

    peft_config = setup_peft_model(args, model, model_config, "CAUSAL_LM")
    log_model_info(model, tokenizer)

    (ds, is_preprocessed), dataset_presharded = load_script_datasets(
        args,
        parallelism_config,
        loader=load_datasets_auto,
        conversation_field=args.conversation_field,
    )

    # The RUN's data path, decided now that the dataset is known: a multimodal checkpoint carrying
    # text-only rows is a text run, and packing / padding_free / train_on_last_assistant_only stay
    # legal for it. The model class is unaffected — it was resolved from the checkpoint above.
    reject_images_under_text_only_model(args, ds, text_only_model=dist_args.text_only_model)
    is_vlm = is_vlm_run(args, model_config.model_name_or_path, ds, config=model.config)
    enforce_text_path_padding_side(tokenizer, is_vlm)

    if is_vlm:
        train_dataset, eval_dataset, generate_dataset, collator = _prepare_vlm_data(
            ds, is_preprocessed, args, sft_config, model_config, processing_class, tokenizer, model.config
        )
    else:
        train_dataset, eval_dataset, generate_dataset, collator = _prepare_text_data(
            ds, is_preprocessed, args, sft_config, model_config, tokenizer, parallelism_config, model.config
        )

    log_script_dataset_examples(
        {"train": train_dataset, "test": eval_dataset, "generate": generate_dataset}, tokenizer, args, sft_config
    )

    generate_callback = GenerateExamplesCallback.from_config(args, sft_config, generate_dataset, tokenizer)
    if generate_callback is None and args.generate_eval_examples and is_global_main_process():
        # Both paths that build no generate split: a VLM run (no tokenized input_ids) and a
        # pre-processed dataset (baked input_ids/labels, no raw conversations to render a prompt from).
        logger.warning(
            "generate_eval_examples skipped: %s.",
            "not supported for VLM datasets" if is_vlm else "a pre-processed dataset carries no raw prompts",
        )
    callbacks = build_training_callbacks(
        args, sft_config, model, parallelism_config, generate_callback=generate_callback
    )

    disable_trl_dataset_prep(sft_config)
    apply_distributed_trainer_config(sft_config, parallelism_config)

    barrier()

    trainer = DistributedSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processing_class,
        peft_config=peft_config,
        data_collator=collator,
        callbacks=callbacks,
        **distributed_trainer_kwargs(args, dist_args, parallelism_config, dataset_presharded=dataset_presharded),
    )
    run_trainer(
        trainer,
        runtime,
        method_name="SFT",
        extra_start_log=[f"modality: {'vlm' if is_vlm else 'text'}"],
    )


if __name__ == "__main__":
    run_training(main)()
