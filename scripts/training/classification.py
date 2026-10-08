#!/usr/bin/env python
"""Distributed sequence-classification training.

Modality: text-only. VLM sequence classification needs a multimodal classification head, which
transformers ships for only a few families (Gemma3, dense Qwen3.5, T5Gemma2, ModernVBert); the
toolkit registers two more itself in src/models/seq_cls_heads.py — Gemma 4 and MoE Qwen3.5/3.6,
each in both spellings a checkpoint can carry (composite and text tower). Other VLM families have
no such head, so general VLM classification is not supported upstream.

Supported Parallelism Modes: EP, TP, ETP (PP is declared but not yet available in this release).
CP is not supported — the trainer pools over the complete sequence, which no CP shard holds.

Usage:
    torchrun --nproc_per_node=8 scripts/training/classification.py \\
        examples/classification/gptoss/clf-gptoss-20b-mage-ep.yaml
"""

import torch.distributed as dist
from accelerate.logging import get_logger
from datasets import DatasetDict
from transformers import AutoModelForSequenceClassification
from trl import ModelConfig

from src.args.classification_args import CLFScriptArguments
from src.args.distributed_args import DistributedArguments
from src.configs.classification_config import ClassificationConfig
from src.data.pipeline.conversation import chat_template_kwargs, maybe_parse_json, reject_image_content
from src.data.pipeline.processing import coordinated_map, resolve_map_num_proc
from src.data.pipeline.rendered import tokenize_rendered
from src.data.sources.loading import reject_image_columns
from src.data.sources.paths import EVAL_SPLIT_NAMES
from src.distributed.loading.peft_setup import setup_peft_model
from src.distributed.loading.vlm_setup import require_multimodal_sequence_classification_head
from src.distributed.runtime import barrier, get_global_world_size, is_global_main_process, reject_across_ranks
from src.models.loading.model_preparation import log_model_info
from src.trainers.mixins.validation import evaluation_runs
from src.trainers.reward.classification import ClassificationTrainer
from src.training.environment import run_training
from src.training.parser import H4ArgumentParser
from src.training.script_runner import (
    apply_distributed_trainer_config,
    apply_max_length,
    build_training_callbacks,
    distributed_trainer_kwargs,
    init_training_script,
    load_script_datasets,
    load_script_model,
    log_script_dataset_examples,
    padded_workload_attn_implementation,
    reject_unsupported_args,
    run_trainer,
)

logger = get_logger(__name__, log_level="INFO")

# A "no label" row, in the stringified form get_label_list gives every label.
NO_LABEL_SENTINEL = "-1"

# Every split whose labels join the class list: an eval split may hold a class train lacks, and a
# split the run never reads still counts, so the head does not change shape with the eval setting.
LABEL_SPLITS = ("train", *EVAL_SPLIT_NAMES)


def tokenize_classification_row(
    example: dict,
    *,
    tokenizer,
    max_length: int,
    label_to_id: dict[str, int],
    is_multi_label: bool,
    text_field: str | None = None,
    tools_field: str | None = None,
) -> dict:
    """Chat-template and tokenize one classification row, mapping its label to an id.

    Accepts either a pre-built ``prompt`` conversation or a raw ``text_field`` column, so text/label
    datasets (e.g. imdb) train without pre-wrapping each row as a conversation. Kept module-level
    with every tunable in the signature so the map's state rides in ``fn_kwargs``, which the cache
    key fingerprints, rather than in closure cells it cannot.

    Rows over ``max_length`` are truncated rather than dropped as on the SFT path: a classification
    label describes the whole document, so a shortened document keeps a valid label, while dropping
    the row would shrink the training set and skew the class balance.
    """
    prompt = maybe_parse_json(example.get("prompt"))
    if prompt is None and text_field:
        prompt = [{"role": "user", "content": example[text_field]}]

    # Text renderer: an image part would become placeholder tokens with no pixels behind them.
    reject_image_content(prompt, "classification field 'prompt'")
    text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=False,
        **chat_template_kwargs(example, interleaved_thinking=False, tools_field=tools_field),
    )
    # tokenize_rendered handles the special tokens: a bare tokenizer() call re-adds the
    # post-processor's BOS on top of the template's (gemma-3 / Zaya / LFM2 all double it).
    tokenized = tokenize_rendered(tokenizer, text, truncation=True, max_length=max_length)

    if "label" in example:
        if is_multi_label:
            ids = [0.0] * len(label_to_id)
            for label in example["label"]:
                # The label set is stringified and carries no "-1" sentinel: in a multi-hot row it is absence.
                if str(label) == NO_LABEL_SENTINEL:
                    continue
                ids[label_to_id[str(label)]] = 1.0
            tokenized["label"] = ids
        else:
            tokenized["label"] = label_to_id[str(example["label"])]

    return tokenized


def require_prompt_or_text_column(train_columns: list[str], text_field: str | None) -> None:
    """Config-time check that tokenization has an input column, before the model is loaded.

    ``tokenize_classification_row`` reads a pre-built ``prompt`` conversation, falling back to
    ``text_field`` — with neither present the failure would otherwise surface as a per-row
    ``KeyError`` deep in the dataset map, after the full (possibly multi-node) model load.
    """
    if "prompt" in train_columns or (text_field and text_field in train_columns):
        return
    text_field_state = f"text_field='{text_field}' names no existing column" if text_field else "text_field is not set"
    raise ValueError(
        f"Classification dataset has no 'prompt' column and {text_field_state} "
        f"(columns: {sorted(train_columns)}). Provide a 'prompt' conversation column or set "
        f"text_field to a raw-text column."
    )


def run_splits(training_config) -> tuple[str, ...]:
    """The splits the run reads: train, and the ``test`` eval split when the run evaluates."""
    return ("train", "test") if evaluation_runs(training_config) else ("train",)


def split_label_sets(ds, dataset_presharded: bool) -> dict[str, set[str]]:
    """Each label-bearing split's stringified label set, unioned across the world on a pre-sharded load.

    A pre-sharded dataset leaves each rank part of a split's labels, giving a rank-divergent
    num_labels and so a different [num_labels, hidden] score head per rank, which breaks the
    FSDP2/DDP all-reduce over it. Every rank needs a train row, which the label type is read from.
    """
    if dataset_presharded:
        reject_across_ranks(
            None if len(ds["train"]) else "its train shard holds no rows; shard so every rank gets one",
            "Loading the pre-sharded dataset",
            ValueError,
        )
    labels = {split: set(get_label_list(ds, split)) for split in LABEL_SPLITS if split in ds}
    if dataset_presharded and get_global_world_size() > 1:
        gathered = [None] * get_global_world_size()
        dist.all_gather_object(gathered, labels)
        labels = {split: set().union(*(part[split] for part in gathered)) for split in labels}
    return labels


def refuse_unlabeled_rows(split_labels: dict[str, set[str]], used_splits, is_multi_label: bool) -> None:
    """Refuse the ``-1`` "no label" sentinel in a single-label split the run reads.

    A multi-hot row reads the sentinel as absence. A single-label row has no class to put it in:
    ``-1`` would reach the cross-entropy as an out-of-range class index. A split the run never reads
    is dropped before tokenization, so an unlabeled GLUE-style test split does not block a run that
    does not evaluate. Runs on the world-agreed label sets, before the model load, so every rank
    raises together.
    """
    if is_multi_label:
        return
    for split in used_splits:
        if NO_LABEL_SENTINEL not in split_labels.get(split, ()):
            continue
        if split == "train":
            remedy = (
                f"Filter those rows out, e.g. dataset.filter(lambda row: str(row['label']) != {NO_LABEL_SENTINEL!r})."
            )
        else:
            remedy = (
                "Evaluate on labeled rows instead: point dataset at the train split alone (e.g. "
                "dataset: <hub id>@train) and cut the eval split from it with test_size, or set "
                "eval_strategy: no."
            )
        raise ValueError(
            f"The '{split}' split of this single-label dataset carries the {NO_LABEL_SENTINEL!r} 'no "
            f"label' sentinel (unlabeled rows, as in GLUE-style test splits), and the run reads that "
            f"split. A single-label row has no class to put it in, and the loss would read it as an "
            f"out-of-range class index. {remedy}"
        )


def build_label_list(split_labels: dict[str, set[str]]) -> list[str]:
    """The sorted class list: every split's labels, less the ``-1`` "no label" sentinel."""
    for split, labels in split_labels.items():
        eval_only = sorted(labels - split_labels["train"] - {NO_LABEL_SENTINEL})
        if eval_only and is_global_main_process():
            logger.warning(f"Labels {eval_only} in {split} set but not in training set, adding them")
    union = set().union(*split_labels.values())
    if NO_LABEL_SENTINEL in union and is_global_main_process():
        logger.warning(f"Label {NO_LABEL_SENTINEL} found in label list, removing it.")
    return sorted(union - {NO_LABEL_SENTINEL})


def get_label_list(raw_dataset, split="train") -> list[str]:
    """The stringified labels of ``split``, flattened for a multi-label one.

    Empty for an empty split: a pre-sharded eval split can leave a rank without rows, and that rank
    must still reach the label-set gather its peers enter.
    """
    if len(raw_dataset[split]) == 0:
        return []
    if isinstance(raw_dataset[split]["label"][0], list):
        label_list = [label for sample in raw_dataset[split]["label"] for label in sample]
        label_list = list(set(label_list))
    else:
        label_list = raw_dataset[split].unique("label")
    label_list = [str(label) for label in label_list]
    return label_list


def main():
    parser = H4ArgumentParser((CLFScriptArguments, ClassificationConfig, ModelConfig, DistributedArguments))
    args, classification_config, model_config, dist_args = parser.parse()

    # The score head is pinned to AutoModelForSequenceClassification, so the loader would only warn
    # and build the wrapper anyway; reject the flag rather than accept one that cannot take effect.
    reject_unsupported_args("Classification", text_only_model=dist_args.text_only_model)

    runtime = init_training_script(
        args,
        classification_config,
        model_config,
        dist_args,
        script_prefix="classification",
        trainer_cls=ClassificationTrainer,
    )
    parallelism_config = runtime.parallelism_config

    # Dataset first: num_labels comes from it.
    ds, dataset_presharded = load_script_datasets(args, parallelism_config)
    reject_image_columns(ds, "Classification")

    require_prompt_or_text_column(ds["train"].column_names, args.text_field)
    split_labels = split_label_sets(ds, dataset_presharded)

    is_multi_label = False
    if isinstance(ds["train"][0]["label"], list):
        is_multi_label = True
        if is_global_main_process():
            logger.info("Label type is list, doing multi-label classification")

    used_splits = run_splits(classification_config)
    refuse_unlabeled_rows(split_labels, used_splits, is_multi_label)
    label_list = build_label_list(split_labels)
    num_labels = len(label_list)
    if num_labels <= 1:
        raise ValueError("You need more than one label to do classification.")

    label_to_id = {v: i for i, v in enumerate(label_list)}

    # Before the model load: a multimodal family with no score head otherwise fails inside Auto*
    # resolution on every rank, naming no alternative. After the distributed init, so the hub read it
    # makes is ordered main-rank-first.
    require_multimodal_sequence_classification_head(model_config)

    # ClassificationTrainer's DataCollatorWithPadding right-pads every batch — keep those shapes off
    # the auto-selected FA4's slow varlen path.
    model, tokenizer = load_script_model(
        runtime,
        classification_config,
        model_config,
        dist_args,
        model_class=AutoModelForSequenceClassification,
        model_config_overrides={"num_labels": num_labels},
        attn_implementation=padded_workload_attn_implementation(model_config, sinks_reset=dist_args.reset_sinks),
    )

    tokenizer = apply_max_length(classification_config, args, model, tokenizer)

    if is_multi_label:
        model.config.problem_type = "multi_label_classification"
    else:
        model.config.problem_type = "single_label_classification"

    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in label_to_id.items()}

    peft_config = setup_peft_model(args, model, model_config, "SEQ_CLS")

    log_model_info(model, tokenizer)

    # Only the splits the run reads are tokenized; an unread one may hold rows no class id covers.
    ds = coordinated_map(
        DatasetDict({split: ds[split] for split in used_splits}),
        tokenize_classification_row,
        num_proc=resolve_map_num_proc(classification_config.dataset_num_proc),
        batched=False,
        desc="Preprocessing dataset for classification",
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": classification_config.max_length,
            "label_to_id": label_to_id,
            "is_multi_label": is_multi_label,
            "text_field": args.text_field,
            "tools_field": args.tools_field,
        },
        # The baked label_to_id keys the map cache: with presharded data the ids change with world size,
        # and a stale cache would keep the old ones.
        cache_key_extras={"label_list": label_list},
    )

    train_dataset = ds["train"]
    eval_dataset = ds.get("test")

    log_script_dataset_examples({"train": train_dataset, "test": eval_dataset}, tokenizer, args, classification_config)

    apply_distributed_trainer_config(classification_config, parallelism_config)

    barrier()

    callbacks = build_training_callbacks(args, classification_config, model, parallelism_config)

    trainer = ClassificationTrainer(
        model=model,
        processing_class=tokenizer,
        args=classification_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        callbacks=callbacks,
        is_binary=len(label_to_id) == 2 and not is_multi_label,
        is_multi_label=is_multi_label,
        label_names_list=label_list,
        **distributed_trainer_kwargs(args, dist_args, parallelism_config, dataset_presharded=dataset_presharded),
    )
    run_trainer(trainer, runtime, method_name="Classification")


if __name__ == "__main__":
    run_training(main)()
