#!/usr/bin/env python
"""Distributed SFT (text or VLM): the model class follows the checkpoint, the data path the run (``resolve_vlm_run``).

Usage: torchrun --nproc_per_node=8 scripts/training/sft.py examples/sft/qwen3_5/qwen3.5-35b-a3b-ultrachat-ep.yaml
"""

import logging

from trl import SFTConfig

from src.callbacks.generate_examples import GenerateExamplesCallback
from src.config_validation import reject_non_default_args
from src.configs.parallelism_arguments import ParallelismArguments
from src.configs.script_arguments import ModelArguments
from src.configs.sft import TREE_PACKING_FIELDS, SFTScriptArguments
from src.data.collators.factory import select_data_collator
from src.data.collators.tree_packing import DataCollatorWithTreePacking
from src.data.collators.vlm import PreprocessedVLMDataCollator
from src.data.pipeline.coordinated_map import coordinated_map_and_filter, coordinated_pack, resolve_map_num_proc
from src.data.pipeline.text_rows import build_chat_row_processor, text_render_kwargs
from src.data.pipeline.tree_rows import prepare_tree_dataset
from src.data.preprocessed.metadata import load_preprocessed_metadata, verify_preprocessed_dataset
from src.data.sources.loading import load_datasets_auto
from src.data.turn_markers import reject_unusable_response_marker, resolve_eos_token_ids
from src.entrypoints.data_setup import (
    disable_trl_dataset_prep,
    load_script_datasets,
    log_script_dataset_examples,
    prepare_chat_text_split,
    prepare_raw_vlm_split,
    reject_trl_dataset_prep_args,
    resolve_vlm_run,
)
from src.entrypoints.model_setup import (
    apply_max_length,
    apply_text_path_padding_side,
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
from src.model_loading.dtype import resolve_training_dtype
from src.model_loading.parallel.loader import resolve_policy_attn_implementation
from src.model_loading.parallel.model_source import probe_checkpoint
from src.model_loading.parallel.peft_setup import configure_adapter_training
from src.model_loading.parallel.whole_model import warm_tree_attention_kernels
from src.models.attention.gpt_oss_sinks import SinksPolicy
from src.models.attention.tree import (
    install_tree_attention,
    reject_unsupported_tree_attention,
    resolve_tree_plan_options,
)
from src.models.segment_markers import reject_compressed_kv_rows
from src.runtime.collectives import barrier
from src.runtime.consensus import agree_probe_across_ranks
from src.runtime.ranks import is_global_main_process
from src.trainers.sft import DistributedSFTTrainer
from src.trainers.sft_rules import (
    reject_unsupported_sft_data_options,
    reject_unsupported_tree_packing,
    reject_unsupported_vlm_sft_options,
)

logger = logging.getLogger(__name__)


def _reject_empty_split(dataset, split: str, max_length: int) -> None:
    """COLLECTIVE: raise on a split the length filter emptied, before packing fails further from the cause; on a
    presharded corpus the length is per-rank, so the verdict is agreed and the raise lands on every rank at once."""
    if agree_probe_across_ranks(len(dataset) == 0, f"the {split} split", "split emptied by tokenization"):
        raise ValueError(
            f"All {split} rows were dropped after tokenization on at least one data-parallel rank: "
            f"every conversation exceeded max_length={max_length}. Increase max_length (e.g. for long "
            f"agent/tool traces) or use a dataset with shorter conversations."
        )


def _prepare_tree_rows(ds, args, sft_config, tokenizer, hf_config, **processor_kwargs):
    """Raw splits to tree rows, as ``(train, eval)``; ``processor_kwargs`` are the render and map arguments of
    the flat path."""
    reject_unusable_response_marker(
        tokenizer, args.assistant_message_template, args.train_on_completions_only, "tree packing"
    )
    ds = prepare_tree_dataset(
        ds,
        tokenizer,
        sft_config.max_length,
        share_markers=tuple(args.tree_share_markers),
        share_requires=args.tree_share_requires,
        min_shared_tokens=args.tree_min_shared_tokens,
        eos_token_ids=resolve_eos_token_ids(tokenizer, hf_config),
        assistant_message_template=args.assistant_message_template,
        train_on_last_assistant_only=args.train_on_last_assistant_only,
        **processor_kwargs,
    )
    return ds["train"], ds["test"]


def _prepare_raw_text_splits(ds, args, sft_config, tokenizer, hf_config):
    """COLLECTIVE: raw text splits to ``(train, eval, generate)``: one chat-template tokenize, as flat rows or
    trees, then the packing."""
    render_kwargs = text_render_kwargs(args)
    num_proc = resolve_map_num_proc(sft_config.dataset_num_proc)
    common = {"tokenizer": tokenizer, "max_length": sft_config.max_length, **render_kwargs}
    # From the raw test split, before the train map consumes it.
    generate_dataset = coordinated_map_and_filter(
        ds["test"],
        build_chat_row_processor(**common, add_generation_prompt=True, use_padding=True),
        desc="Processing generate dataset",
        cache_key_extras={**render_kwargs, "add_generation_prompt": True},
        num_proc=num_proc,
    )
    if args.tree_packing:
        train_dataset, eval_dataset = _prepare_tree_rows(
            ds, args, sft_config, tokenizer, hf_config, **render_kwargs, num_proc=num_proc
        )
    else:
        train_dataset, eval_dataset = prepare_chat_text_split(
            ds, tokenizer, args, sft_config, desc="Processing dataset", use_padding=not sft_config.padding_free
        )
    _reject_empty_split(train_dataset, "training", sft_config.max_length)
    if sft_config.packing:
        pack = {"seq_length": sft_config.max_length, "strategy": sft_config.packing_strategy}
        train_dataset = coordinated_pack(train_dataset, **pack, split="train")
        if sft_config.eval_packing is not False:
            _reject_empty_split(eval_dataset, "evaluation", sft_config.max_length)
            eval_dataset = coordinated_pack(eval_dataset, **pack, split="eval")
    return train_dataset, eval_dataset, generate_dataset


def _load_verified_metadata(args, sft_config, model_config):
    """The pre-processed dataset's metadata, verified against this run's length, model, masking, render and
    packing."""
    metadata = load_preprocessed_metadata(args.dataset)
    verify_preprocessed_dataset(
        metadata,
        required_max_length=sft_config.max_length,
        required_model=model_config.model_name_or_path,
        required_train_on_completions_only=args.train_on_completions_only,
        render_args=args,
        required_packing=sft_config.packing,
    )
    return metadata


def _prepare_text_data(ds, is_preprocessed, args, sft_config, model_config, tokenizer, parallelism_config, hf_config):
    """Text dataset path: the pre-processed splits as baked, or the raw ones tokenized and packed; returns
    ``(train, eval, generate, collator)``."""
    if is_preprocessed:
        # Offline-packed data still needs the packing collator: without per-document position_ids from
        # seq_lengths documents attend across pack boundaries. Keyed on metadata.packed, not the flag.
        collator_packing = _load_verified_metadata(args, sft_config, model_config).packed
        train_dataset, eval_dataset, generate_dataset = ds["train"], ds["test"], None
    else:
        collator_packing = sft_config.packing
        train_dataset, eval_dataset, generate_dataset = _prepare_raw_text_splits(
            ds, args, sft_config, tokenizer, hf_config
        )

    if args.tree_packing:
        if is_global_main_process():
            logger.info(f"Using DataCollatorWithTreePacking (full layers: {args.tree_full_attention})")
        collator = DataCollatorWithTreePacking(
            tokenizer, full_form=args.tree_full_attention, **resolve_tree_plan_options(hf_config)
        )
        return train_dataset, eval_dataset, generate_dataset, collator
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
    """VLM dataset path: the pre-processed splits as baked, or the raw VLM map; returns
    ``(train, eval, None, collator)``, since GenerateExamplesCallback needs tokenized input_ids."""
    if not is_preprocessed:
        train_dataset, eval_dataset, collator = prepare_raw_vlm_split(
            ds, args, sft_config, processor, tokenizer, hf_config, desc="Processing train/eval dataset"
        )
        return train_dataset, eval_dataset, None, collator
    if not _load_verified_metadata(args, sft_config, model_config).is_vlm:
        raise ValueError(f"Dataset at '{args.dataset}' is not a VLM preprocessed dataset.")
    collator = PreprocessedVLMDataCollator(tokenizer=tokenizer, max_length=sft_config.max_length)
    return ds["train"], ds["test"], None, collator


def main():
    args, sft_config, model_config, dist_args = parse_script_args(
        SFTScriptArguments, SFTConfig, ModelArguments, ParallelismArguments
    )

    reject_unsupported_sft_data_options(sft_config, dist_args)
    if args.tree_packing:
        reject_unsupported_tree_packing(args, sft_config, dist_args)
    else:
        reject_non_default_args("Halo SFT with tree_packing off", args, *TREE_PACKING_FIELDS)
    reject_trl_dataset_prep_args(
        "Halo SFT",
        sft_config,
        render_remedy="it renders conversation_field itself; tokenize a raw-text column offline with "
        "scripts/before_training/prepare_dataset.py --mode text",
    )

    # The modality names the run before the dataset exists. Pinned like the model load: hub `main`'s config can name
    # another modality than the commit this run trains.
    checkpoint_config, is_vlm_checkpoint = probe_checkpoint(
        model_config.model_name_or_path,
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
    )
    # The collator factory refuses the same rows, but only after the model load and the packing pass.
    if sft_config.packing or sft_config.padding_free:
        reject_compressed_kv_rows(checkpoint_config, "packing" if sft_config.packing else "padding_free")
    context = init_training_script(
        args,
        sft_config,
        model_config,
        dist_args,
        script_prefix=f"sft{'-vlm' if is_vlm_checkpoint and not args.text_only_model else ''}",
        trainer_cls=DistributedSFTTrainer,
        supports_init_from_scratch=True,
    )
    parallelism_config = context.parallelism_config
    if args.tree_packing:
        # Before any weight loads: the backend the loader will resolve for this config, dtype and GPU.
        reject_unsupported_tree_attention(
            checkpoint_config,
            resolve_policy_attn_implementation(
                checkpoint_config,
                model_config.attn_implementation,
                resolve_training_dtype(sft_config),
                sinks_policy=SinksPolicy.from_flags(reset_sinks=args.reset_sinks, train_sinks=args.train_sinks),
                init_from_scratch=dist_args.init_from_scratch,
            ),
            args.tree_full_attention,
        )

    (ds, is_preprocessed), dataset_presharded = load_script_datasets(
        args, parallelism_config, loader=load_datasets_auto, conversation_field=args.conversation_field
    )
    if is_preprocessed:
        # Both splits are used as baked, so eval_packing has nothing to decide.
        reject_non_default_args("Halo SFT on a pre-processed dataset", sft_config, "eval_packing")
        reject_non_default_args("Halo SFT on a pre-processed dataset", args, "tree_packing")

    # Before the model load, which needs the checkpoint's processor only for an image run: a multimodal checkpoint
    # carrying text-only rows is a text run, its model class unchanged.
    is_vlm = resolve_vlm_run(
        args, model_config, ds, text_only_model=args.text_only_model, vlm_checkpoint=is_vlm_checkpoint
    )
    if is_vlm:
        reject_unsupported_vlm_sft_options(args, sft_config, preprocessed=is_preprocessed)

    policy = load_policy_model(
        context, sft_config, model_config, args, vlm_run=is_vlm, init_from_scratch=dist_args.init_from_scratch
    )
    model = policy.model
    tokenizer = apply_max_length(sft_config, args, model, policy.tokenizer)
    processing_class = install_resolved_tokenizer(policy.processing_class, tokenizer)
    apply_text_path_padding_side(tokenizer, is_vlm)

    peft_config = configure_adapter_training(args, model, model_config, "CAUSAL_LM")
    log_model_info(model, tokenizer)
    if args.tree_packing:
        install_tree_attention(model, args.tree_full_attention)
        warm_tree_attention_kernels(
            model, dtype=resolve_training_dtype(sft_config), full_form=args.tree_full_attention
        )

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
        logger.warning(
            "generate_eval_examples skipped: %s.",
            "not supported for VLM datasets" if is_vlm else "a pre-processed dataset carries no raw prompts",
        )
    callbacks = configure_balancing_and_build_callbacks(
        args, sft_config, model, context, generate_callback=generate_callback
    )

    disable_trl_dataset_prep(sft_config)

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
        **distributed_trainer_kwargs(args, parallelism_config, dataset_presharded=dataset_presharded),
    )
    run_trainer(trainer, context, method_name="SFT", extra_start_log=[f"modality: {'vlm' if is_vlm else 'text'}"])


if __name__ == "__main__":
    with_distributed_teardown(main)()
