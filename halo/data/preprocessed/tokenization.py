"""Offline tokenization of a corpus: text and VLM rows to input_ids, attention_mask and baked labels."""

import functools
import logging
from collections.abc import Callable
from typing import Any

import torch
from datasets import Dataset, Features
from jinja2 import TemplateError
from PIL import UnidentifiedImageError
from PIL.Image import DecompressionBombError
from transformers import AutoConfig, PreTrainedTokenizer

from halo.data.completion_spans import (
    COLLATOR_SPAN_POLICY,
    PACKED_SPAN_POLICY,
    build_completion_only_labels,
    mask_batch_to_completion_spans,
)
from halo.data.conversation import resolve_json_value
from halo.data.pipeline.coordinated_map import coordinated_filter, coordinated_map, resolve_map_num_proc
from halo.data.pipeline.rejected_rows import is_valid_example, report_rejected_rows
from halo.data.pipeline.text_rows import build_chat_row_processor, build_text_row_processor
from halo.data.pipeline.vlm_rows import build_vlm_history
from halo.data.preprocessed.metadata import PreprocessingConfig
from halo.data.turn_markers import resolve_eos_token_ids, tokenize_response_template
from halo.data.vlm_render import VLM_OUTPUT_FEATURES, render_vlm_text, resolve_image_token_ids, run_vlm_processor
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.model_loading.tokenizer_backend import resolve_processor_backend, resolve_tokenizer_backend
from halo.model_loading.tokenizer_setup import resolve_tokenizer

logger = logging.getLogger(__name__)

# Per-row data failures, dropped with a warning. Image errors are named rather than ``OSError``, so a full disk aborts.
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

# Rows the fully-masked-labels refusal probes at each end of a baked split.
_LABEL_PROBE_ROWS = 64


def _completion_only_labels(
    input_ids: list[int],
    tokenizer: PreTrainedTokenizer,
    assistant_template: str,
    response_token_ids: list[int],
    eos_token_ids: frozenset[int],
    extra_ignore_token_ids: tuple[int, ...] = (),
    span_policy: dict[str, bool] | None = None,
) -> list[int]:
    """By the span rule every runtime collator labels through, so a baked row masks exactly like an untouched one;
    ``span_policy`` is that collator's, ``extra_ignore_token_ids`` (image tokens) are masked on top."""
    ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    batch = {"input_ids": ids, "labels": ids.clone()}
    batch = mask_batch_to_completion_spans(
        batch,
        response_token_ids,
        eos_token_ids,
        train_on_last_assistant_only=False,
        response_prompt_template=assistant_template,
        tokenizer=tokenizer,
        span_policy=span_policy,
        extra_ignore_token_ids=extra_ignore_token_ids,
    )
    return batch["labels"][0].tolist()


def resolve_label_eos_token_ids(config: PreprocessingConfig, tokenizer: PreTrainedTokenizer) -> frozenset[int]:
    """The label bake's turn terminators: the model config's ``eos_token_id`` list folded with the tokenizer's, as
    the runtime collator masks. An unreadable config raises: GLM-4's turn terminators are listed only there."""
    hf_config = AutoConfig.from_pretrained(config.model_name_or_path, trust_remote_code=config.trust_remote_code)
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
    """Refuses a split that lost every row: a config error, and ``cause`` names the ones that apply to ``stage``."""
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
    """A mistyped assistant template masks everything without tripping the ``input_ids`` filters. Head and tail are
    probed, since an ordered dataset may open with legitimately masked rows."""
    total = len(tokenized)
    if total == 0 or "labels" not in tokenized.column_names:
        return
    indices = list(range(min(total, _LABEL_PROBE_ROWS)))
    if total > _LABEL_PROBE_ROWS:
        indices += list(range(max(total - _LABEL_PROBE_ROWS, _LABEL_PROBE_ROWS), total))
    probe = tokenized.select(indices)
    if all(all(v == LABEL_IGNORE_INDEX for v in row) for row in probe["labels"]):
        raise ValueError(
            f"Every probed label in '{split_name}' ({len(probe)} rows from head and tail) is "
            f"the ignore index ({LABEL_IGNORE_INDEX}): the assistant_message_template never "
            f"matched a rendered assistant turn, so this dataset would train ZERO tokens. "
            f"Check the template against the tokenizer's chat template before re-preprocessing."
        )


def _row_processor(config: PreprocessingConfig, tokenizer: PreTrainedTokenizer) -> Callable:
    """A packed raw-text artifact leaves the overflow to the strategy."""
    if config.mode == "text":
        return build_text_row_processor(
            tokenizer=tokenizer,
            max_length=config.max_length,
            text_field=config.text_field,
            append_eos=config.append_eos,
            truncate=not config.pack_sequences,
        )
    return build_chat_row_processor(
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


def _bake_labels(
    tokenized: Dataset,
    tokenizer: PreTrainedTokenizer,
    config: PreprocessingConfig,
    split_name: str,
    eos_token_ids: frozenset[int] | None,
) -> Dataset:
    """Baked for good (the artifact's collators only pad): completion-only under the span policy of the collator that
    would have masked them, else none. Stored at ``input_ids``' integer width, which inference would widen."""
    if not config.train_on_completions_only:
        return tokenized
    marker = config.assistant_message_template
    response_token_ids = tokenize_response_template(marker, tokenizer)
    span_policy = PACKED_SPAN_POLICY if config.pack_sequences else COLLATOR_SPAN_POLICY
    return coordinated_map(
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
        features=Features({**tokenized.features, "labels": tokenized.features["input_ids"]}),
        # Two artifacts of one corpus can differ by the span policy alone.
        cache_key_extras={
            "completion_only": True,
            "assistant_template": marker,
            "eos_token_ids": sorted(eos_token_ids),
            "span_policy": span_policy,
        },
    )


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    config: PreprocessingConfig,
    split_name: str = "train",
    *,
    eos_token_ids: frozenset[int] | None = None,
) -> Dataset:
    """Input_ids and attention_mask, plus labels under ``train_on_completions_only``; resolves unset terminators."""
    tokenizer = resolve_tokenizer_backend(tokenizer, config.tokenizer_backend)
    if config.train_on_completions_only and eos_token_ids is None:
        eos_token_ids = resolve_label_eos_token_ids(config, tokenizer)

    logger.info(f"Tokenizing {split_name} split ({len(dataset)} examples)...")
    tokenized = coordinated_map(
        dataset,
        _row_processor(config, tokenizer),
        desc=f"tokenizing {split_name}",
        num_proc=resolve_map_num_proc(config.num_proc),
        # Every source column goes, as at runtime: a source `labels` column would otherwise be baked as targets.
        remove_columns=list(dataset.column_names),
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
    tokenized = _bake_labels(tokenized, tokenizer, config, split_name, eos_token_ids)
    _reject_fully_masked_labels(tokenized, split_name)
    return tokenized


def _vision_columns(result: Any) -> dict[str, Any]:
    """Stored form: float16 ``pixel_values`` bytes with their shape, ``image_grid_thw`` as a list."""
    columns = {}
    if "pixel_values" in result:
        pixel_values = result["pixel_values"]
        if hasattr(pixel_values, "numpy"):
            pixel_values = pixel_values.numpy()
        columns["pixel_values"] = pixel_values.astype("float16").tobytes()
        columns["pixel_values_shape"] = list(pixel_values.shape)
    if "image_grid_thw" in result:
        grid = result["image_grid_thw"]
        if hasattr(grid, "numpy"):
            grid = grid.numpy()
        columns["image_grid_thw"] = grid.tolist() if hasattr(grid, "tolist") else grid
    return columns


def _vlm_labels(
    input_ids: list[int],
    attention_mask: list[int],
    *,
    tokenizer: PreTrainedTokenizer,
    config: PreprocessingConfig,
    response_token_ids: list[int] | None,
    image_token_ids: tuple[int, ...],
    eos_token_ids: frozenset[int] | None,
) -> list[int]:
    if config.train_on_completions_only:
        return _completion_only_labels(
            input_ids,
            tokenizer,
            config.assistant_message_template,
            response_token_ids,
            extra_ignore_token_ids=image_token_ids,
            eos_token_ids=eos_token_ids,
        )
    # The runtime builder, given the row's all-ones mask so its pad-value fallback cannot erase a real EOS at pad==eos.
    return build_completion_only_labels(
        torch.tensor(input_ids, dtype=torch.long),
        tokenizer,
        None,
        False,
        extra_ignore_token_ids=image_token_ids,
        attention_mask=torch.tensor(attention_mask, dtype=torch.long),
    ).tolist()


def tokenize_vlm_row(
    example: dict[str, Any],
    *,
    config: PreprocessingConfig,
    processor: Any,
    tokenizer: PreTrainedTokenizer,
    response_token_ids: list[int] | None,
    image_token_ids: tuple[int, ...],
    eos_token_ids: frozenset[int] | None,
) -> dict[str, Any]:
    """Through the runtime VLM path's own builder and renderer, so it tokenizes as an untouched run renders it. An
    empty, over-length or bad-data row becomes the all-None row, Arrow-safe because the map pins
    ``VLM_OUTPUT_FEATURES``."""
    try:
        raw_conversation = resolve_json_value(example.get(config.conversation_field))
        if not raw_conversation:
            return dict.fromkeys(VLM_OUTPUT_FEATURES)
        history, images = build_vlm_history(
            raw_conversation,
            example.get(config.images_field) if config.images_field else None,
            system_prompt=config.system_prompt,
            model_supports_system_role=config.model_supports_system_role,
        )
        # No truncation: cutting expanded image placeholders desyncs text from vision.
        result = run_vlm_processor(
            processor, [render_vlm_text(processor, history)], [images], padding=False, truncation=False
        )
        # Text-only rows may drop such extras, being derivable from input_ids alone.
        unsupported = set(result) - set(VLM_OUTPUT_FEATURES)
        if unsupported and images:
            raise NotImplementedError(
                f"Processor emitted vision keys {sorted(unsupported)} the preprocessed VLM schema "
                "cannot store — train this model family through the runtime VLM path instead."
            )
        input_ids, attention_mask = result["input_ids"], result["attention_mask"]
        input_ids = input_ids[0] if input_ids.dim() == 2 else input_ids
        attention_mask = attention_mask[0] if attention_mask.dim() == 2 else attention_mask
        if input_ids.shape[0] > config.max_length:
            return dict.fromkeys(VLM_OUTPUT_FEATURES)
        output = {
            **dict.fromkeys(VLM_OUTPUT_FEATURES),
            **_vision_columns(result),
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.tolist(),
        }
        output["labels"] = _vlm_labels(
            output["input_ids"],
            output["attention_mask"],
            tokenizer=tokenizer,
            config=config,
            response_token_ids=response_token_ids,
            image_token_ids=image_token_ids,
            eos_token_ids=eos_token_ids,
        )
        return output
    except _VLM_ROW_DATA_ERRORS as e:
        logger.warning(f"Dropping VLM example ({type(e).__name__}: {e})")
        return dict.fromkeys(VLM_OUTPUT_FEATURES)


def tokenize_vlm_dataset(
    dataset: Dataset,
    processor: Any,
    config: PreprocessingConfig,
    split_name: str = "train",
    *,
    eos_token_ids: frozenset[int] | None = None,
) -> Dataset:
    """Input_ids with expanded placeholders, attention_mask, labels, fp16 ``pixel_values`` and ``image_grid_thw``."""
    if config.tools_field or config.interleaved_thinking:
        raise NotImplementedError(
            "VLM offline preprocessing does not render tools_field / interleaved_thinking — the "
            "preprocessed tokenization would silently diverge from the runtime render. Train these "
            "datasets through the runtime VLM path instead."
        )
    logger.info(f"Tokenizing VLM {split_name} split ({len(dataset)} examples)...")

    processor = resolve_processor_backend(processor, config.tokenizer_backend)
    tokenizer = resolve_tokenizer(processor)
    if config.train_on_completions_only and eos_token_ids is None:
        eos_token_ids = resolve_label_eos_token_ids(config, tokenizer)
    tokenize_row = functools.partial(
        tokenize_vlm_row,
        config=config,
        processor=processor,
        tokenizer=tokenizer,
        response_token_ids=(
            tokenize_response_template(config.assistant_message_template, tokenizer)
            if config.train_on_completions_only
            else None
        ),
        # Read once from the full vocab; sorted so the key does not depend on set order.
        image_token_ids=tuple(sorted(resolve_image_token_ids(tokenizer, processor))),
        eos_token_ids=eos_token_ids,
    )
    tokenized = coordinated_map(
        dataset,
        tokenize_row,
        desc=f"tokenizing VLM {split_name}",
        num_proc=resolve_map_num_proc(config.num_proc),
        remove_columns=[col for col in dataset.column_names if col not in VLM_OUTPUT_FEATURES],
        features=VLM_OUTPUT_FEATURES,
        # The key reads ``config`` by type alone, so every field the row reads rides here; the pixel budget lives on
        # the image processor.
        cache_key_extras={
            "max_length": config.max_length,
            "conversation_field": config.conversation_field,
            "system_prompt": config.system_prompt,
            "model_supports_system_role": config.model_supports_system_role,
            "images_field": config.images_field,
            "train_on_completions_only": config.train_on_completions_only,
            "assistant_template": config.assistant_message_template,
            "eos_token_ids": sorted(eos_token_ids) if eos_token_ids else None,
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
