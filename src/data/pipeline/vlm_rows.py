"""VLM row processors: hub-shape normalization, the placeholder history and images every VLM render takes, and
the raw-VLM training map with its pinned Arrow schema and over-length filter."""

import json
from collections.abc import Callable
from typing import Any

import datasets
from datasets.features import Json
from PIL import Image

from src.data.pipeline.conversation import IMAGE_PART_PAYLOAD_KEY, IMAGE_PART_TYPE, resolve_json_value
from src.data.pipeline.coordinated_map import coordinated_filter, coordinated_map
from src.data.pipeline.rejected_rows import dataset_total_size, report_rejected_rows
from src.data.render_columns import reject_missing_render_column
from src.data.turn_markers import reject_missing_response_marker
from src.data.vlm_render import extract_conversation_images, render_vlm_text, vlm_row_tools
from src.runtime.consensus import agree_probe_across_ranks

# A row's history to the longest conversation its collator renders from it, for the over-length filter.
_MeasuredHistory = Callable[[list[dict[str, Any]], dict[str, Any]], list[dict[str, Any]]]


def normalize_vlm_conversation(conversation: list[dict[str, Any]], images: Any = None) -> list[dict[str, Any]]:
    """New role/content messages for a hub conversation, native or in the HuggingFaceM4 ``{"user", "assistant"}``
    pair shape (FineVision, the_cauldron, Docmatix), with a top-level ``images`` value merged in: filling the
    image placeholders in order, else ahead of the first user turn's text."""
    messages: list[dict[str, Any]] = []
    for turn in conversation:
        if "role" in turn:
            content = turn["content"]
            if isinstance(content, list):  # copy part dicts so image fill-in never mutates the source row
                content = [dict(part) for part in content]
            messages.append({"role": turn["role"], "content": content})
        elif "user" in turn:
            messages.append({"role": "user", "content": turn["user"]})
            if turn.get("assistant") is not None:
                messages.append({"role": "assistant", "content": turn["assistant"]})
        else:
            raise ValueError(f"Unrecognized conversation turn shape: {sorted(turn)}")

    if images is None:
        return messages
    image_list = images if isinstance(images, (list, tuple)) else [images]
    if not image_list:
        return messages

    # Placeholders pair with the images column by order (TRL): fill in place, never inject on top.
    placeholders = [
        part
        for message in messages
        if isinstance(message["content"], list)
        for part in message["content"]
        if part.get("type") == IMAGE_PART_TYPE and not part.get(IMAGE_PART_PAYLOAD_KEY)
    ]
    if placeholders:
        if len(placeholders) != len(image_list):
            raise ValueError(
                f"Conversation carries {len(placeholders)} image placeholders but the images column "
                f"holds {len(image_list)} images — counts must match to pair them by order."
            )
        for part, img in zip(placeholders, image_list, strict=True):
            part[IMAGE_PART_PAYLOAD_KEY] = img
        return messages

    first_user = next((m for m in messages if m["role"] == "user"), None)
    if first_user is None:
        raise ValueError("images_field set but the conversation has no user turn to attach images to")
    image_parts = [{"type": IMAGE_PART_TYPE, IMAGE_PART_PAYLOAD_KEY: img} for img in image_list]
    content = first_user["content"]
    text_parts = [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
    first_user["content"] = image_parts + text_parts
    return messages


def build_vlm_history(
    raw_conversation: Any,
    images: Any = None,
    *,
    system_prompt: str | None = None,
    model_supports_system_role: bool = True,
) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    """One raw row's conversation to the placeholder history and PIL images every VLM render takes, shared by the
    runtime row map and the offline tokenizer. An empty conversation passes through; ``None`` or a string that is
    not JSON raises as a malformed column."""
    conversation = resolve_json_value(raw_conversation)
    if conversation is None or isinstance(conversation, str):
        raise ValueError(
            f"The conversation column holds a string that is not a valid JSON conversation: {conversation}"
        )

    return extract_conversation_images(
        normalize_vlm_conversation(conversation, images),
        system_prompt=system_prompt,
        model_supports_system_role=model_supports_system_role,
    )


def build_vlm_row_processor(
    conversation_field: str = "messages",
    system_prompt: str | None = None,
    model_supports_system_role: bool = True,
    tools_field: str | None = None,
    images_field: str | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build the VLM row map: images extracted, the conversation reshaped into history. It takes no tokenizer,
    since the render runs later in ``VLMDataCollator``, so its output and cache key are tokenizer-independent."""

    def process_row(row: dict[str, Any]) -> dict[str, Any]:
        history, images = build_vlm_history(
            row[conversation_field],
            row.get(images_field) if images_field else None,
            system_prompt=system_prompt,
            model_supports_system_role=model_supports_system_role,
        )

        # JSON keeps the column str|None; native list-of-dicts risks Arrow schema drift across rows.
        tools_json = None
        if tools_field and tools_field in row and row[tools_field] is not None:
            raw = row[tools_field]
            tools_json = raw if isinstance(raw, str) else json.dumps(raw)

        return {"history": history, "images": images, "tools_json": tools_json}

    return process_row


def vlm_map_features() -> datasets.Features:
    """The Arrow schema of :func:`build_vlm_row_processor` rows, pinned because shard-wise inference disagrees on a
    mixed dataset (text-only ``content`` infers ``List(struct)``, image parts ``Json``, no images ``List(null)``)
    and a multiprocess map then fails to align its shards."""
    return datasets.Features(
        {
            "history": [{"role": datasets.Value("string"), "content": Json()}],
            "images": datasets.Sequence(datasets.Image()),
            "tools_json": datasets.Value("string"),
        }
    )


# The columns a mapped VLM dataset carries into collation: the map's output plus a preprocessed artifact's.
_VLM_SIGNATURE_COLUMNS = ("input_ids", "labels", "attention_mask", *vlm_map_features())


def _filter_vlm_over_length(
    ds, processor, tokenizer, max_length: int, num_proc: int, measured_history: _MeasuredHistory | None
):
    """Drop the mapped VLM rows whose rendered text alone exceeds ``max_length``, ahead of the collators that fail
    loud on them; vision tokens stay uncounted, the collator being the exact backstop. The all-dropped refusal is
    world-agreed, since a presharded corpus counts per rank."""

    def fits(row) -> bool:
        # The collator's own render, tools included, so the filter measures the sequence it collates.
        history = row["history"] if measured_history is None else measured_history(row["history"], row)
        text = render_vlm_text(processor, history, tools=vlm_row_tools(row))
        return len(tokenizer(text, add_special_tokens=False)["input_ids"]) <= max_length

    before = dataset_total_size(ds)
    ds = coordinated_filter(
        ds,
        fits,
        desc="Dropping over-length VLM rows",
        num_proc=num_proc,
    )
    after = dataset_total_size(ds)
    if agree_probe_across_ranks(after == 0, "the VLM dataset", "every row over max_length"):
        raise ValueError(
            f"All {before} rows exceed max_length={max_length} by text alone on at least one "
            f"data-parallel rank — raise the budget."
        )
    report_rejected_rows(before, after, f"VLM over-length filtering (text tokens alone over max_length={max_length})")
    return ds


def prepare_vlm_dataset(
    ds,
    args,
    processor,
    tokenizer,
    max_length: int,
    num_proc: int,
    *,
    keep: tuple[str, ...] = (),
    features: datasets.Features | None = None,
    desc: str = "Processing VLM dataset",
    measured_history: _MeasuredHistory | None = None,
):
    """Map every split through :func:`build_vlm_row_processor`, dropping the columns outside the collator signature
    but ``keep``, then drop the over-length rows. ``features`` pins the map's schema (:func:`vlm_map_features`
    unless ``keep`` is set); ``measured_history`` must be a module-level function or a partial of one, which the
    filter's cache key fingerprints."""
    # Before the map: the collator is built after it, so a missing marker would cost the whole tokenization.
    reject_missing_response_marker(args.assistant_message_template, args.train_on_completions_only, "VLM training")
    if args.interleaved_thinking:
        raise ValueError(
            "interleaved_thinking is text-only: no supported VLM family's chat template renders a "
            "clear_thinking kwarg, so on the VLM path the flag would silently do nothing."
        )
    if features is not None and keep:
        raise ValueError(
            f"prepare_vlm_dataset got both a pinned Arrow schema and keep={list(keep)}: the schema "
            f"declares exactly the mapped columns, so the kept ones have no slot in it and the map "
            f"would die on an Arrow cast inside a worker. Pass one or the other."
        )
    if features is None and not keep:
        features = vlm_map_features()
    # Here, not per script: every consumer would otherwise train text-only on a mistyped column in silence.
    if args.images_field:
        reject_missing_render_column(ds, str(args.dataset), "images_field", args.images_field)
    cache_key_extras = {
        "conversation_field": args.conversation_field,
        "system_prompt": args.system_prompt,
        "model_supports_system_role": args.model_supports_system_role,
        "tools_field": args.tools_field,
        "images_field": args.images_field,
    }
    ds = coordinated_map(
        ds,
        build_vlm_row_processor(**cache_key_extras),
        num_proc=num_proc,
        # Sorted: the list feeds the cache key, and set order is hash-randomized per process.
        remove_columns=sorted(set(ds["train"].column_names) - set(_VLM_SIGNATURE_COLUMNS) - set(keep)),
        desc=desc,
        cache_key_extras=cache_key_extras,
        **({"features": features} if features is not None else {}),
    )
    return _filter_vlm_over_length(ds, processor, tokenizer, max_length, num_proc, measured_history)
