"""Offline dataset preparation: pre-tokenization, packing and Megatron-LM-style sharding.

Covers text-only SFT, raw-text pretraining and VLM corpora; the artifact's ``metadata.json``
contract lives in :mod:`src.data.pipeline.preprocessed_metadata`. The other methods (DPO, SMPO,
GRPO) tokenize at train time through their own pipelines.
"""

import logging
import os
from dataclasses import asdict
from typing import Any

import torch
from datasets import Dataset, DatasetDict
from jinja2 import TemplateError
from PIL import UnidentifiedImageError
from PIL.Image import DecompressionBombError
from transformers import AutoConfig, PreTrainedTokenizer

from src.data.pipeline.conversation import maybe_parse_json
from src.data.pipeline.preprocessed_metadata import PreprocessedDatasetMetadata, PreprocessingConfig
from src.data.pipeline.processing import (
    coordinated_filter,
    coordinated_map,
    pack_dataset_coordinated,
    report_rejected_rows,
    resolve_map_num_proc,
)
from src.data.pipeline.row_processors import (
    build_vlm_history,
    create_llm_processor,
    create_text_processor,
    is_valid_example,
)
from src.data.pipeline.tokenizer_backend import resolve_processor_backend, resolve_tokenizer_backend
from src.data.shard_index import SHARD_INDEX_FILE, ShardIndex, ShardInfo
from src.data.sources.paths import METADATA_FILE
from src.data.spans import (
    COLLATOR_SPAN_POLICY,
    LABEL_IGNORE_INDEX,
    PACKED_SPAN_POLICY,
    build_completion_only_labels,
    mask_batch_to_completion_spans,
    resolve_eos_token_ids,
    tokenize_response_template,
)
from src.data.vlm import (
    VLM_OUTPUT_COLUMNS,
    VLM_OUTPUT_FEATURES,
    carried_image_columns,
    get_image_token_ids,
    render_vlm_text,
    run_vlm_processor,
)
from src.distributed.runtime import is_local_main_process
from src.models.structure import resolve_tokenizer

logger = logging.getLogger(__name__)

# Per-row VLM failures that are a property of the ROW — a bad image, a content shape the chat
# template refuses, a placeholder-vs-image count mismatch — are dropped with a warning; every other
# exception aborts the map, since a blanket catch thins the corpus silently. The two image errors are
# named rather than their shared ``OSError`` base so a full disk aborts instead of counting as a row.
_VLM_ROW_DATA_ERRORS = (
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    FileNotFoundError,
    UnidentifiedImageError,
    DecompressionBombError,
    TemplateError,
)


def _completion_only_labels(
    input_ids: list[int],
    tokenizer: PreTrainedTokenizer,
    assistant_template: str,
    response_token_ids: list[int],
    extra_ignore_token_ids: tuple[int, ...] = (),
    eos_token_ids: frozenset[int] | None = None,
    span_policy: dict[str, bool] | None = None,
) -> list[int]:
    """Completion-only loss labels for one tokenized example, baked with a named span policy.

    Both this and the runtime collators call :func:`mask_batch_to_completion_spans`, so a preprocessed
    row and an untouched one see identical loss masks. ``span_policy`` follows the artifact: an
    unpacked one is collated by the padded collator (:data:`~src.data.spans.COLLATOR_SPAN_POLICY`,
    the default), a packed one by :data:`~src.data.spans.PACKED_SPAN_POLICY`, and the two differ
    on a turn whose terminator is missing. ``response_token_ids`` is the pre-tokenized
    ``assistant_template``, hoisted by callers out of the per-row map like ``eos_token_ids``;
    ``extra_ignore_token_ids`` (e.g. image tokens) are masked on top.
    """
    ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    batch = {"input_ids": ids, "labels": ids.clone()}
    batch = mask_batch_to_completion_spans(
        batch,
        response_token_ids,
        eos_token_ids if eos_token_ids is not None else resolve_eos_token_ids(tokenizer),
        ignore_index=LABEL_IGNORE_INDEX,
        train_on_last_assistant_only=False,
        response_prompt_template=assistant_template,
        tokenizer=tokenizer,
        span_policy=span_policy,
        extra_ignore_token_ids=extra_ignore_token_ids,
    )
    return batch["labels"][0].tolist()


def _resolve_config_eos_token_ids(config: PreprocessingConfig, tokenizer: PreTrainedTokenizer) -> frozenset[int]:
    """Assistant-turn terminator ids for preprocessing — load the model's HF config (for its
    ``eos_token_id`` list) and fold in the tokenizer's eos/pad. Falls back to tokenizer-only on a
    config-load failure so preprocessing never hard-fails on a metadata read.
    """
    try:
        hf_config = AutoConfig.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    except Exception as exc:  # config read is best-effort; tokenizer eos/pad still apply
        logger.warning(f"Could not load model config for eos_token_id resolution ({exc}); using tokenizer eos/pad.")
        hf_config = None
    return resolve_eos_token_ids(tokenizer, hf_config)


def _drop_rejected_rows(
    tokenized: Dataset,
    config: PreprocessingConfig,
    split_name: str,
    *,
    stage: str,
    cause: str,
    context: str,
) -> Dataset:
    """Drop the rejection sentinels a tokenizing map emitted, refusing a split that lost every row.

    Shared by the text and VLM bakes: an empty split is a config bug rather than a length budget, and
    ``cause`` names the ones that apply to ``stage``.
    """
    original_len = len(tokenized)
    tokenized = coordinated_filter(
        tokenized,
        is_valid_example,
        desc=f"filtering {split_name}",
        num_proc=resolve_map_num_proc(config.num_proc),
    )
    filtered_len = len(tokenized)

    if original_len > 0 and filtered_len == 0:
        raise ValueError(
            f"Every example in the '{split_name}' split was dropped during {stage} "
            f"({original_len} → 0). The usual cause is {cause}, not max_length={config.max_length}. "
            f"Inspect one raw example against the config."
        )

    report_rejected_rows(original_len, filtered_len, context)
    return tokenized


def _reject_fully_masked_labels(tokenized: Dataset, split_name: str) -> None:
    """Refuse a baked split whose probed rows are ALL ignore-index — it would train zero tokens.

    Probed AFTER the bake (labels do not exist before it); head AND tail, because an ordered
    pre-labeled dataset may legitimately open with fully-masked rows. Shared by the text and VLM
    bakes: both write a ``labels`` column the same way, and a mistyped assistant-message template
    masks everything on either path — silently, since the row filters key on ``input_ids``.
    """
    total = len(tokenized)
    if total == 0 or "labels" not in tokenized.column_names:
        return
    indices = list(range(min(total, 64)))
    if total > 64:
        indices += list(range(max(total - 64, 64), total))
    probe = tokenized.select(indices)
    if all(all(v == LABEL_IGNORE_INDEX for v in row) for row in probe["labels"]):
        raise ValueError(
            f"Every probed label in '{split_name}' ({len(probe)} rows from head and tail) is "
            f"the ignore index ({LABEL_IGNORE_INDEX}): the assistant_message_template never "
            f"matched a rendered assistant turn, so this dataset would train ZERO tokens. "
            f"Check the template against the tokenizer's chat template before re-preprocessing."
        )


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    config: PreprocessingConfig,
    split_name: str = "train",
) -> Dataset:
    """Tokenize a dataset (raw-text or chat mode), returning input_ids/attention_mask/labels."""
    tokenizer = resolve_tokenizer_backend(tokenizer, config.tokenizer_backend)

    if config.mode == "text":
        # Don't truncate when packing: the packing strategy owns everything past max_length.
        processor = create_text_processor(
            tokenizer=tokenizer,
            max_length=config.max_length,
            text_field=config.text_field,
            append_eos=config.append_eos,
            truncate=not config.pack_sequences,
        )
    else:
        processor = create_llm_processor(
            tokenizer=tokenizer,
            max_length=config.max_length,
            conversation_field=config.conversation_field,
            system_prompt=config.system_prompt,
            model_supports_system_role=config.model_supports_system_role,
            add_generation_prompt=False,
            use_padding=False,
            interleaved_thinking=config.interleaved_thinking,
            tools_field=config.tools_field,
        )

    # Every source column goes, exactly as the runtime path drops them: a source column named
    # `labels` (classification/reward corpora carry one) would otherwise survive the map and be
    # baked as this dataset's loss targets, since the labels step below only fills in a MISSING
    # column. input_ids/attention_mask/labels below are the processor's output, added after removal.
    columns_to_remove = list(dataset.column_names)

    logger.info(f"Tokenizing {split_name} split ({len(dataset)} examples)...")
    tokenized = coordinated_map(
        dataset,
        processor,
        desc=f"tokenizing {split_name}",
        num_proc=resolve_map_num_proc(config.num_proc),
        remove_columns=columns_to_remove,
    )

    tokenized = _drop_rejected_rows(
        tokenized,
        config,
        split_name,
        stage="tokenization",
        cause=(
            "a wrong field name (check conversation_field / text_field) or an "
            "assistant_message_template that does not match the rendered assistant turn"
        ),
        context=f"tokenization of '{split_name}' (rows over max_length={config.max_length})",
    )

    # Labels are baked here — the preprocessed dataset's collators only pad, never re-mask.
    if config.train_on_completions_only:
        marker = config.assistant_message_template
        eos_token_ids = _resolve_config_eos_token_ids(config, tokenizer)
        response_token_ids = tokenize_response_template(marker, tokenizer)
        # The policy follows the artifact, because the collator that would have masked these rows at
        # runtime does: a packed artifact is collated by the packing collator, which ends a
        # terminator-less turn at the sequence end instead of dropping it from the loss.
        span_policy = PACKED_SPAN_POLICY if config.pack_sequences else COLLATOR_SPAN_POLICY
        tokenized = coordinated_map(
            tokenized,
            lambda x: {
                "labels": _completion_only_labels(
                    x["input_ids"],
                    tokenizer,
                    marker,
                    response_token_ids,
                    eos_token_ids=eos_token_ids,
                    span_policy=span_policy,
                )
            },
            desc=f"masking completions in {split_name}",
            num_proc=resolve_map_num_proc(config.num_proc),
            cache_key_extras={
                "completion_only": True,
                "assistant_template": marker,
                "eos_token_ids": sorted(eos_token_ids),
                # Two artifacts of the same corpus differ only by this policy; without it in the
                # key the second bake would load the first's labels out of the map cache.
                "span_policy": span_policy,
            },
        )
    elif "labels" not in tokenized.column_names:
        tokenized = coordinated_map(
            tokenized,
            lambda x: {"labels": x["input_ids"].copy()},
            desc=f"adding labels to {split_name}",
            num_proc=resolve_map_num_proc(config.num_proc),
            cache_key_extras={"labels": "full_copy"},
        )

    _reject_fully_masked_labels(tokenized, split_name)

    return tokenized


def _vlm_none_row() -> dict[str, Any]:
    """Schema-uniform dropped/failed VLM row — all-None is Arrow-safe here because the map pins
    ``VLM_OUTPUT_FEATURES``; dropped downstream by :func:`is_valid_example`."""
    return dict.fromkeys(VLM_OUTPUT_COLUMNS)


def tokenize_vlm_dataset(
    dataset: Dataset,
    processor: Any,
    config: PreprocessingConfig,
    split_name: str = "train",
) -> Dataset:
    """FULL VLM tokenization including vision tokens.

    Produces input_ids (vision placeholders expanded), attention_mask, labels, pixel_values (float16
    bytes), and image_grid_thw (Qwen-VL specific).
    """
    if config.tools_field or config.interleaved_thinking:
        raise NotImplementedError(
            "VLM offline preprocessing does not render tools_field / interleaved_thinking — the "
            "preprocessed tokenization would silently diverge from the runtime render. Train these "
            "datasets through the runtime VLM path instead."
        )
    logger.info(f"Tokenizing VLM {split_name} split ({len(dataset)} examples)...")

    processor = resolve_processor_backend(processor, config.tokenizer_backend)
    tokenizer = resolve_tokenizer(processor)

    # process_vlm_example swallows exceptions (dropping the row), so a missing marker must raise here.
    if config.train_on_completions_only and not config.assistant_message_template:
        raise ValueError(
            "train_on_completions_only=True requires assistant_message_template (the assistant "
            "response marker) for VLM preprocessing so completion-only labels can be built."
        )

    eos_token_ids = _resolve_config_eos_token_ids(config, tokenizer) if config.train_on_completions_only else None
    response_token_ids = (
        tokenize_response_template(config.assistant_message_template, tokenizer)
        if config.train_on_completions_only
        else None
    )
    # Construction-time, like the eos set: the union reads the tokenizer's full vocab dict, and the
    # verdict is fixed across the corpus. Sorted so the closure fingerprint is order-independent.
    image_token_ids = tuple(sorted(get_image_token_ids(tokenizer, processor)))

    def process_vlm_example(example: dict[str, Any]) -> dict[str, Any]:
        try:
            raw_conversation = maybe_parse_json(example.get(config.conversation_field))
            if not raw_conversation:
                return _vlm_none_row()

            # The runtime VLM path's own builder and renderer (create_vlm_processor →
            # VLMDataCollator): a row baked here must tokenize exactly as an untouched run renders
            # it, which only one shared render can guarantee.
            history, images = build_vlm_history(
                raw_conversation,
                example.get(config.images_field) if config.images_field else None,
                system_prompt=config.system_prompt,
                model_supports_system_role=config.model_supports_system_role,
            )
            text = render_vlm_text(processor, history)

            # NEVER truncate a VLM row: cutting expanded image placeholders desyncs text from vision.
            result = run_vlm_processor(processor, [text], images, padding=False, truncation=False)

            # Vision tensors the stored schema cannot hold must refuse, not train on wrong inputs;
            # text-only rows may drop such extras (derivable from input_ids without vision content).
            unsupported = set(result) - set(VLM_OUTPUT_COLUMNS)
            if unsupported and images:
                raise NotImplementedError(
                    f"Processor emitted vision keys {sorted(unsupported)} the preprocessed VLM schema "
                    "cannot store — train this model family through the runtime VLM path instead."
                )

            input_ids = result["input_ids"]
            attention_mask = result["attention_mask"]

            if input_ids.dim() == 2:
                input_ids = input_ids[0]
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[0]

            if input_ids.shape[0] > config.max_length:
                return _vlm_none_row()

            output = _vlm_none_row()
            output["input_ids"] = input_ids.tolist()
            output["attention_mask"] = attention_mask.tolist()

            if "pixel_values" in result:
                pixel_values = result["pixel_values"]
                if hasattr(pixel_values, "numpy"):
                    pixel_values = pixel_values.numpy()
                output["pixel_values"] = pixel_values.astype("float16").tobytes()
                output["pixel_values_shape"] = list(pixel_values.shape)

            if "image_grid_thw" in result:
                grid = result["image_grid_thw"]
                if hasattr(grid, "numpy"):
                    grid = grid.numpy()
                if hasattr(grid, "tolist"):
                    grid = grid.tolist()
                output["image_grid_thw"] = grid

            # Image tokens are always masked out of the loss; completions-only masking goes on top.
            if config.train_on_completions_only:
                output["labels"] = _completion_only_labels(
                    output["input_ids"],
                    tokenizer,
                    config.assistant_message_template,
                    response_token_ids,
                    extra_ignore_token_ids=image_token_ids,
                    eos_token_ids=eos_token_ids,
                )
            else:
                # The runtime collator's own builder, with the row's (all-ones, unpadded) mask passed
                # explicitly so its value-based pad fallback — which erases real EOS where
                # pad_token_id == eos_token_id — cannot run here.
                output["labels"] = build_completion_only_labels(
                    torch.tensor(output["input_ids"], dtype=torch.long),
                    tokenizer,
                    None,
                    False,
                    extra_ignore_token_ids=image_token_ids,
                    attention_mask=torch.tensor(output["attention_mask"], dtype=torch.long),
                ).tolist()

            return output

        except NotImplementedError:
            # Dataset-wide capability refusal, not a bad row: swallowing it trains text-only silently.
            raise
        except _VLM_ROW_DATA_ERRORS as e:
            logger.warning(f"Dropping VLM example ({type(e).__name__}: {e})")
            return _vlm_none_row()

    columns_to_remove = [col for col in dataset.column_names if col not in VLM_OUTPUT_COLUMNS]

    # process_vlm_example closes over the whole `config` dataclass, invisible to the closure fingerprint:
    # output-affecting fields must go through cache_key_extras or a re-run reuses a stale cache.
    tokenized = coordinated_map(
        dataset,
        process_vlm_example,
        desc=f"tokenizing VLM {split_name}",
        num_proc=resolve_map_num_proc(config.num_proc),
        remove_columns=columns_to_remove,
        features=VLM_OUTPUT_FEATURES,
        cache_key_extras={
            "max_length": config.max_length,
            "conversation_field": config.conversation_field,
            "system_prompt": config.system_prompt,
            "model_supports_system_role": config.model_supports_system_role,
            "images_field": config.images_field,
            "train_on_completions_only": config.train_on_completions_only,
            "assistant_template": config.assistant_message_template,
            "eos_token_ids": sorted(eos_token_ids) if eos_token_ids else None,
            # The resolution budget lives on the image_processor: without it, re-preprocessing at a
            # new resolution reuses stale pixels.
            "min_pixels": config.min_pixels,
            "max_pixels": config.max_pixels,
        },
    )

    tokenized = _drop_rejected_rows(
        tokenized,
        config,
        split_name,
        stage="VLM tokenization",
        cause=(
            "a wrong field name (check conversation_field), a processor that cannot render the "
            "conversations, or an assistant_message_template that does not match the rendered "
            "assistant turn"
        ),
        context=f"VLM tokenization of '{split_name}'",
    )
    _reject_fully_masked_labels(tokenized, split_name)

    return tokenized


def shard_dataset(
    dataset: Dataset,
    num_shards: int,
    output_dir: str,
    split_name: str = "train",
) -> ShardIndex:
    """Shard a dataset into Arrow files (remainder distributed to the first shards)."""
    if num_shards <= 1:
        shard_dir = os.path.join(output_dir, split_name)
        os.makedirs(shard_dir, exist_ok=True)

        shard_save_path = os.path.join(shard_dir, "shard_0000")
        dataset.save_to_disk(shard_save_path)

        return ShardIndex(
            split=split_name,
            num_shards=1,
            total_examples=len(dataset),
            shards=[
                ShardInfo(
                    id=0,
                    path=f"{split_name}/shard_0000",
                    num_examples=len(dataset),
                )
            ],
        )

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

        shard_data = dataset.select(range(start_idx, end_idx))
        shard_data.save_to_disk(os.path.join(shard_dir, f"shard_{shard_id:04d}"))

        byte_size = 0
        shard_folder = os.path.join(shard_dir, f"shard_{shard_id:04d}")
        if os.path.exists(shard_folder):
            for f in os.listdir(shard_folder):
                byte_size += os.path.getsize(os.path.join(shard_folder, f))

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
    """Report the maximum data-parallel degree this sharded dataset can be trained at.

    :func:`shard_dataset` skips empty shards, so a split with fewer examples than ``--num-shards``
    yields fewer shards than asked for — and a rank with no shard gets zero examples, which the
    trainer's pre-sharded equalizer turns into a hard failure (train) or an eval-gather hang it has to
    reject (eval). Surfacing the ceiling here, while re-preprocessing is cheap, beats discovering it
    after a multi-hour training startup.
    """
    if not shard_indices:
        return
    ceiling = min(index.num_shards for index in shard_indices.values())
    if ceiling >= requested_shards:
        return
    per_split = ", ".join(f"{split}={index.num_shards}" for split, index in sorted(shard_indices.items()))
    logger.warning(
        "Requested --num-shards %d but a split had too few examples to fill them (%s): empty shards are "
        "skipped. This dataset can only train at data_parallel_size <= %d — above that, ranks without a "
        "shard get zero examples (train: hard failure; eval: rejected to avoid a metrics-gather hang). "
        "Lower --num-shards or raise --test-size so every split has >= the intended DP degree.",
        requested_shards,
        per_split,
        ceiling,
    )


def _reject_unconsumed_image_columns(dataset: Dataset | DatasetDict, config: PreprocessingConfig) -> None:
    """Refuse a source carrying an image column this run does not consume.

    Tokenization drops every source column — the text path removes all of them, the VLM path keeps
    only :data:`VLM_OUTPUT_COLUMNS` — so an image column no ``images_field`` names is deleted without
    a word. The result is the worst artifact of the two: a ``--vlm`` run stamps ``is_vlm=True`` on
    rows holding no pixels, and the training side, which reads that stamp, trains a multimodal
    recipe on text. Read off the shared :data:`~src.data.vlm.VLM_IMAGE_COLUMNS`
    spellings, so what the runtime dispatch routes on is what preparation refuses to ignore.
    """
    unconsumed = sorted(carried_image_columns(dataset) - {config.images_field})
    if not unconsumed:
        return
    raise ValueError(
        f"The source dataset carries the image column(s) {unconsumed}, which this run consumes "
        f"nowhere (images_field={config.images_field!r}, is_vlm={config.is_vlm}): tokenization drops "
        f"every source column, so the images would be silently discarded and the artifact would bake "
        f"the rows' text alone. Prepare it with --vlm --images-field <column>, or drop the column(s)."
    )


def preprocess_dataset(
    dataset: Dataset | DatasetDict,
    tokenizer_or_processor: Any,
    config: PreprocessingConfig,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Preprocess a dataset: tokenize, optionally pack, and optionally shard/save.

    For VLM, pass the processor instead of a tokenizer and set config.is_vlm=True. Returns a dict with
    "train"/"test" datasets, "metadata", and (if output_dir given) "shard_indices".
    """
    result = {}

    if config.is_vlm and config.pack_sequences:
        raise ValueError("Packing is not supported for VLM datasets. Set pack_sequences=False when using is_vlm=True.")

    # Fail fast on invalid packing strategy instead of crashing deep inside trl.pack_dataset.
    if config.pack_sequences and config.packing_strategy not in {"bfd", "bfd_split", "wrapped"}:
        raise ValueError(
            f"Invalid packing_strategy '{config.packing_strategy}'. "
            "TRL pack_dataset only accepts 'bfd', 'bfd_split' or 'wrapped'."
        )

    _reject_unconsumed_image_columns(dataset, config)

    if isinstance(dataset, DatasetDict):
        train_data = dataset.get("train")
        test_data = dataset.get("test")
    else:
        train_data = dataset
        test_data = None

    if config.is_vlm:
        tokenize_fn = tokenize_vlm_dataset
        # tokenize_fn takes the processor itself; the metadata below needs the tokenizer it wraps.
        tokenizer = resolve_tokenizer(tokenizer_or_processor)
    else:
        tokenize_fn = tokenize_dataset
        tokenizer = tokenizer_or_processor

    for split, split_data in (("train", train_data), ("test", test_data)):
        if split_data is None:
            continue

        tokenized = tokenize_fn(split_data, tokenizer_or_processor, config, split)

        if config.pack_sequences:
            tokenized = pack_dataset_coordinated(
                tokenized,
                seq_length=config.max_length,
                strategy=config.packing_strategy,
                split=split,
            )

        result[split] = tokenized

    has_pixel_values = False
    if config.is_vlm and "train" in result:
        train_tokenized = result["train"]
        # The VLM schema is uniform (vision columns exist as nulls on text-only rows), so column
        # presence proves nothing — count non-nulls via Arrow metadata.
        if "pixel_values" in train_tokenized.column_names:
            has_pixel_values = train_tokenized.data.column("pixel_values").null_count < len(train_tokenized)
        if not has_pixel_values:
            # Legitimate for a text-only corpus rendered with a VLM processor, but it is also what a
            # dropped images column looks like — and the training side reads is_vlm, not this flag.
            logger.warning(
                "VLM preprocessing produced NO pixel_values: every row baked text only, while the "
                "artifact is stamped is_vlm=True. If the images live in a separate column, name it "
                "with --images-field; if they are embedded in the conversation, check "
                "conversation_field."
            )

    metadata = PreprocessedDatasetMetadata(
        model_name=config.model_name_or_path,
        tokenizer_vocab_size=len(tokenizer),
        max_length=config.max_length,
        packed=config.pack_sequences,
        packing_strategy=config.packing_strategy if config.pack_sequences else None,
        train_on_completions_only=config.train_on_completions_only,
        is_vlm=config.is_vlm,
        has_pixel_values=has_pixel_values,
        num_shards=config.num_shards,
        total_train_examples=len(result.get("train", [])),
        total_test_examples=len(result.get("test", [])),
        config=asdict(config),
    )
    result["metadata"] = metadata

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        shard_indices = {}

        # Test is sharded the same way as train: a split whose shards don't reach every DP rank leaves
        # those ranks with an empty eval set → mismatched gather_for_metrics counts → eval hang.
        for split in ("train", "test"):
            if split not in result:
                continue
            index = shard_dataset(
                result[split],
                config.num_shards,
                output_dir,
                split,
            )
            shard_indices[split] = index
            index.save(os.path.join(output_dir, split, SHARD_INDEX_FILE))

        result["shard_indices"] = shard_indices
        _warn_on_shard_count_ceiling(shard_indices, config.num_shards)

        metadata.save(os.path.join(output_dir, METADATA_FILE))
        logger.info(f"Saved metadata to {os.path.join(output_dir, METADATA_FILE)}")

    return result
