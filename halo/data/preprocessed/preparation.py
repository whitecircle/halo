"""Offline dataset preparation: tokenize, optionally pack, then save or shard the artifact beside its stamp."""

import logging
import os
from dataclasses import asdict
from typing import Any

from datasets import Dataset, DatasetDict

from halo.data.pipeline.coordinated_map import coordinated_pack, resolve_map_num_proc
from halo.data.preprocessed.metadata import PreprocessedDatasetMetadata, PreprocessingConfig
from halo.data.preprocessed.shard_index import SHARD_INDEX_FILE, ShardIndex, ShardInfo, write_stamped_sidecar
from halo.data.preprocessed.tokenization import resolve_label_eos_token_ids, tokenize_dataset, tokenize_vlm_dataset
from halo.data.sources.paths import METADATA_FILE, eval_split_name
from halo.data.vlm_detection import carried_image_columns
from halo.model_loading.tokenizer_setup import resolve_tokenizer
from halo.runtime.ranks import is_local_main_process

logger = logging.getLogger(__name__)


def shard_dataset(
    dataset: Dataset,
    num_shards: int,
    output_dir: str,
    split_name: str = "train",
) -> ShardIndex:
    """Arrow files, the remainder distributed to the first shards; empty shards skipped."""
    total_examples = len(dataset)
    base_shard_size = total_examples // num_shards
    remainder = total_examples % num_shards

    shards = []
    start_idx = 0

    shard_dir = os.path.join(output_dir, split_name)
    os.makedirs(shard_dir, exist_ok=True)

    logger.info(f"Sharding {split_name} into {num_shards} shards...")

    for shard_id in range(num_shards):
        shard_size = base_shard_size + (1 if shard_id < remainder else 0)
        end_idx = start_idx + shard_size

        if shard_size == 0:
            continue

        shard_folder = os.path.join(shard_dir, f"shard_{shard_id:04d}")
        dataset.select(range(start_idx, end_idx)).save_to_disk(shard_folder)
        byte_size = sum(os.path.getsize(os.path.join(shard_folder, f)) for f in os.listdir(shard_folder))

        shards.append(
            ShardInfo(
                id=shard_id,
                path=f"{split_name}/shard_{shard_id:04d}",
                num_examples=shard_size,
                byte_size=byte_size,
            )
        )

        start_idx = end_idx

        if is_local_main_process() and (shard_id + 1) % 10 == 0:
            logger.info(f"  Created {shard_id + 1}/{num_shards} shards")

    logger.info(f"Created {len(shards)} shards for {split_name}")

    return ShardIndex(
        split=split_name,
        num_shards=len(shards),
        total_examples=total_examples,
        shards=shards,
    )


def _warn_on_shard_count_ceiling(shard_indices: dict[str, ShardIndex], requested_shards: int) -> None:
    """Warn of the data-parallel ceiling a split with fewer rows than ``requested_shards`` sets: empty shards are
    skipped, and a rank with no shard fails its train split or has its eval gather refused."""
    ceiling = min(index.num_shards for index in shard_indices.values())
    if ceiling >= requested_shards:
        return
    per_split = ", ".join(f"{split}={index.num_shards}" for split, index in sorted(shard_indices.items()))
    logger.warning(
        "Requested --num_shards %d but a split had too few examples to fill them (%s): empty shards are "
        "skipped. This dataset can only train at data_parallel_size <= %d — above that, ranks without a "
        "shard get zero examples (train: hard failure; eval: rejected to avoid a metrics-gather hang). "
        "Lower --num_shards or raise --test_size so every split has >= the intended DP degree.",
        requested_shards,
        per_split,
        ceiling,
    )


def _reject_unconsumed_image_columns(dataset: Dataset | DatasetDict, config: PreprocessingConfig) -> None:
    """Refuse a source image column (the spellings the runtime dispatch routes on) no ``images_field`` names:
    tokenization drops every source column, so its images would vanish without a diagnostic."""
    unconsumed = sorted(carried_image_columns(dataset) - {config.images_field})
    if not unconsumed:
        return
    raise ValueError(
        f"The source dataset carries the image column(s) {unconsumed}, which this run consumes "
        f"nowhere (images_field={config.images_field!r}, is_vlm={config.is_vlm}): tokenization drops "
        f"every source column, so the images would be silently discarded and the artifact would bake "
        f"the rows' text alone. Prepare it with --is_vlm --images_field <column>, or drop the column(s)."
    )


def _resolve_baked_splits(dataset: Dataset | DatasetDict) -> tuple[Dataset | None, Dataset | None]:
    """``(train, test)``, a lone ``validation`` split baked as ``test``; any other split is left out with a warning."""
    if isinstance(dataset, Dataset):
        return dataset, None
    test_source = eval_split_name(dataset)
    unbaked = sorted(set(dataset) - {"train", test_source})
    if unbaked:
        logger.warning(f"The input's split(s) {unbaked} are not baked: the prepared dataset holds train and test only")
    if test_source not in (None, "test"):
        logger.info(f"Baking the input's {test_source!r} split as the prepared dataset's 'test' split")
    return dataset.get("train"), dataset[test_source] if test_source else None


def _tokenize_splits(
    splits: dict[str, Dataset | None], tokenizer_or_processor: Any, tokenizer: Any, config: PreprocessingConfig
) -> dict[str, Dataset]:
    tokenize = tokenize_vlm_dataset if config.is_vlm else tokenize_dataset
    # Before any tokenization pass: the set reads the model config, and an unreadable one would surface only later.
    eos_token_ids = resolve_label_eos_token_ids(config, tokenizer) if config.train_on_completions_only else None
    tokenized = {}
    for split, split_data in splits.items():
        if split_data is None:
            continue
        rows = tokenize(split_data, tokenizer_or_processor, config, split, eos_token_ids=eos_token_ids)
        if config.pack_sequences:
            rows = coordinated_pack(
                rows,
                seq_length=config.max_length,
                strategy=config.packing_strategy,
                split=split,
                num_proc=resolve_map_num_proc(config.num_proc),
            )
        tokenized[split] = rows
    return tokenized


def _has_pixel_values(tokenized: dict[str, Dataset], config: PreprocessingConfig) -> bool:
    """Warns when a VLM bake stored no pixels."""
    train = tokenized.get("train")
    if not config.is_vlm or train is None:
        return False
    # The VLM schema is uniform (vision columns exist as nulls), so presence proves nothing; count non-nulls via Arrow.
    has_pixels = "pixel_values" in train.column_names and train.data.column("pixel_values").null_count < len(train)
    if not has_pixels:
        # Legitimate for a text-only corpus, but also what a dropped images column looks like; training reads is_vlm.
        logger.warning(
            "VLM preprocessing produced NO pixel_values: every row baked text only, while the "
            "artifact is stamped is_vlm=True. If the images live in a separate column, name it "
            "with --images_field; if they are embedded in the conversation, check "
            "conversation_field."
        )
    return has_pixels


def _write_artifact(
    splits: dict[str, Dataset], metadata: PreprocessedDatasetMetadata, config: PreprocessingConfig, output_dir: str
) -> dict[str, ShardIndex] | None:
    """Save the splits, sharded at ``num_shards > 1``, then the stamp; returns a sharded artifact's shard indices."""
    os.makedirs(output_dir, exist_ok=True)
    shard_indices = None
    if config.num_shards > 1:
        # Test shards like train: a rank with no eval shard would hang gather_for_metrics.
        shard_indices = {}
        for split, split_data in splits.items():
            index = shard_dataset(split_data, config.num_shards, output_dir, split)
            shard_indices[split] = index
            write_stamped_sidecar(os.path.join(output_dir, split, SHARD_INDEX_FILE), index)
        _warn_on_shard_count_ceiling(shard_indices, config.num_shards)
    else:
        # No shard index: every rank loads the whole dataset and its DataLoader splits by DP rank, at any DP size.
        DatasetDict(splits).save_to_disk(output_dir)

    write_stamped_sidecar(os.path.join(output_dir, METADATA_FILE), metadata)
    logger.info(f"Saved metadata to {os.path.join(output_dir, METADATA_FILE)}")
    return shard_indices


def preprocess_dataset(
    dataset: Dataset | DatasetDict,
    tokenizer_or_processor: Any,
    config: PreprocessingConfig,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Returns the baked ``train``/``test`` splits, ``metadata`` and, when sharded, ``shard_indices``."""
    _reject_unconsumed_image_columns(dataset, config)
    train_data, test_data = _resolve_baked_splits(dataset)

    # Before tokenizing: training rejects a sharded dataset without a test split; an unsharded one gets a placeholder.
    if output_dir is not None and config.num_shards > 1 and test_data is None:
        raise ValueError(
            f"num_shards={config.num_shards} writes a sharded dataset, which training refuses without a "
            f"'test' split, and this input has none. Cut one with --test_size, or keep --num_shards at 1 "
            f"for an unsharded dataset, which trains with a placeholder test split taken from train."
        )

    # The VLM tokenization takes the processor itself; the metadata needs the tokenizer it wraps.
    tokenizer = resolve_tokenizer(tokenizer_or_processor) if config.is_vlm else tokenizer_or_processor
    tokenized = _tokenize_splits({"train": train_data, "test": test_data}, tokenizer_or_processor, tokenizer, config)
    metadata = PreprocessedDatasetMetadata(
        model_name=config.model_name_or_path,
        tokenizer_vocab_size=len(tokenizer),
        # The processor's for a VLM bake: it renders with a template of its own.
        rendered_chat_template=tokenizer_or_processor.chat_template if config.mode == "chat" else None,
        max_length=config.max_length,
        packed=config.pack_sequences,
        packing_strategy=config.packing_strategy if config.pack_sequences else None,
        train_on_completions_only=config.train_on_completions_only,
        is_vlm=config.is_vlm,
        has_pixel_values=_has_pixel_values(tokenized, config),
        num_shards=config.num_shards,
        total_train_examples=len(tokenized.get("train", [])),
        total_test_examples=len(tokenized.get("test", [])),
        config=asdict(config),
    )
    result = {**tokenized, "metadata": metadata}
    if output_dir is not None:
        shard_indices = _write_artifact(tokenized, metadata, config, output_dir)
        if shard_indices is not None:
            result["shard_indices"] = shard_indices
    return result
