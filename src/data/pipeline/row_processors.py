"""Row-processor factories for the dataset maps: the per-row chat render, tokenization and VLM reshape.

Each ``create_*`` returns the closure a coordinated map in
:mod:`src.data.pipeline.processing` runs over the corpus. Also the home of the Arrow-type-stable
rejection sentinel those closures emit and of the predicate that drops it, so the row shape and the
map machinery stay separable: nothing here imports the coordinated-map module.
"""

import json
from collections.abc import Callable
from typing import Any

from PIL import Image
from transformers import PreTrainedTokenizer

from src.data.pipeline.conversation import (
    IMAGE_PART_PAYLOAD_KEY,
    IMAGE_PART_TYPE,
    chat_template_kwargs,
    maybe_parse_json,
    reject_image_content,
)
from src.data.pipeline.rendered import render_conversation, tokenize_rendered
from src.data.vlm import process_vlm_conversation
from src.models.loading.tokenizer_setup import is_bounded_length


def _rejection_sentinel(sample_output: dict[str, Any] | None = None) -> dict[str, list]:
    """An Arrow-type-stable rejection sentinel row for tokenization maps.

    Every key holds a one-element list of the real element type, never ``None`` or ``[]``: a worker
    whose first writer batch is entirely rejected would infer a ``null`` column and crash casting the
    real ``list<int64>`` batches. ``attention_mask`` is ``[0]`` — a real row always attends to at least
    one token — which is what :func:`is_valid_example` drops these on. ``sample_output`` supplies the
    key set and element types; ``None`` yields the minimal shape.
    """
    if sample_output is None:
        return {"input_ids": [0], "attention_mask": [0]}
    return {
        key: [0] if key == "attention_mask" else (list(values[:1]) or [0]) for key, values in sample_output.items()
    }


def create_tokenizer_none_example(tokenizer, **tokenizer_kwargs) -> dict[str, list]:
    """The :func:`_rejection_sentinel` shaped to this tokenizer's output keys (probed once)."""
    return _rejection_sentinel(tokenizer("sample", **tokenizer_kwargs))


def _is_content_bearing(value: Any) -> bool:
    """Whether a filter-field value carries real content.

    ``None``, blank strings, and empty lists do not (an empty or whitespace-only prompt is genuinely
    invalid); a message list does only when at least one message content is non-blank — the
    type-stable rejection shape for conversational prompt columns. Message-shaped dicts are
    recognized by a ``role``/``content`` key, so multimodal content-part lists (``type``/``image``
    dicts) and every other value (token-id lists, scalars, …) count as content-bearing.
    """
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        if not value:
            return False
        if all(isinstance(item, dict) and ("role" in item or "content" in item) for item in value):
            return any(_is_content_bearing(item.get("content")) for item in value)
    return True


def is_valid_example(row: dict[str, Any], filter_field: str = "input_ids") -> bool:
    """Keep-predicate shared by every post-processing filter: True iff ``row`` is a real example,
    not a rejection sentinel.

    Sentinels come in two shapes — a content-free ``filter_field`` (the rejection rows the GRPO
    scripts emit; blank rather than ``None``, so an all-rejected first writer batch cannot infer a
    ``null`` column), or the typed sentinel from :func:`create_tokenizer_none_example`, recognized by
    an all-zero ``attention_mask``. Rows without that column are judged by ``filter_field`` alone.
    """
    if not _is_content_bearing(row.get(filter_field)):
        return False
    mask = row.get("attention_mask")
    return mask is None or any(mask)


def apply_chat_template_to_conversations(
    row: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    conversation_field: str = "messages",
    system_prompt: str | None = None,
    model_supports_system_role: bool = True,
    add_generation_prompt: bool = False,
    interleaved_thinking: bool = False,
    tools_field: str | None = None,
    drop_last_turn_on_generation: bool = False,
) -> str:
    """Chat-template the conversation at ``row[conversation_field]``, returning the formatted string.

    The row-extraction half of :func:`~src.data.pipeline.rendered.render_conversation`, which
    owns the render itself. ``drop_last_turn_on_generation`` renders the conversation WITHOUT its
    final message when ``add_generation_prompt`` is set — the generation-eval path, where the row's
    trailing assistant turn is the reference answer to be regenerated.
    """
    conversation = maybe_parse_json(row[conversation_field])
    if add_generation_prompt and drop_last_turn_on_generation:
        conversation = conversation[:-1]

    return render_conversation(
        tokenizer,
        conversation,
        row,
        conversation_field=conversation_field,
        system_prompt=system_prompt,
        model_supports_system_role=model_supports_system_role,
        add_generation_prompt=add_generation_prompt,
        interleaved_thinking=interleaved_thinking,
        tools_field=tools_field,
    )


def prepare_generative_row(row, tokenizer, max_length, tools_field=None):
    # Text renderer: an image part would become placeholder tokens with no pixels behind them.
    reject_image_content(row["prompt"], "prompt field 'prompt'")
    constructed_prompt = tokenizer.apply_chat_template(
        row["prompt"],
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs(row, interleaved_thinking=False, tools_field=tools_field),
    )
    # for_generation: specials apply exactly once and no trailing terminator ends the prompt early.
    # truncation follows the cap: an unset one means "no cap", not "cap at tokenizer.model_max_length"
    # (which is what HF resolves `truncation=True, max_length=None` to).
    return tokenize_rendered(
        tokenizer,
        constructed_prompt,
        for_generation=True,
        truncation=is_bounded_length(max_length),
        padding=True,
        max_length=max_length,
    )


def create_llm_processor(
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    conversation_field: str = "messages",
    system_prompt: str | None = None,
    model_supports_system_role: bool = True,
    add_generation_prompt: bool = False,
    use_padding: bool = True,
    interleaved_thinking: bool = False,
    tools_field: str | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build a row processor that chat-templates and tokenizes a conversation.

    Rows exceeding max_length yield the typed rejection sentinel
    (:func:`create_tokenizer_none_example`), dropped downstream by :func:`is_valid_example`.
    use_padding=False for padding-free mode.
    """
    none_example = create_tokenizer_none_example(tokenizer, truncation=True, padding=False, max_length=max_length)

    def process_row(row: dict[str, Any]) -> dict[str, Any]:
        constructed_prompt = apply_chat_template_to_conversations(
            row,
            tokenizer,
            conversation_field=conversation_field,
            system_prompt=system_prompt,
            model_supports_system_role=model_supports_system_role,
            add_generation_prompt=add_generation_prompt,
            interleaved_thinking=interleaved_thinking,
            tools_field=tools_field,
            drop_last_turn_on_generation=True,
        )

        # Generation prompts must not end with a tokenizer-appended turn terminator (Zaya-style).
        tokenized = tokenize_rendered(
            tokenizer, constructed_prompt, for_generation=add_generation_prompt, truncation=False, padding=False
        )
        if len(tokenized["input_ids"]) > max_length:
            return none_example.copy()

        if not use_padding:
            # Already the final form: the row fits the budget, so `truncation=True` would cut nothing
            # and padding is off — re-tokenizing would only repeat the work.
            return tokenized

        return tokenize_rendered(
            tokenizer,
            constructed_prompt,
            for_generation=add_generation_prompt,
            truncation=True,
            padding=use_padding,
            max_length=max_length,
        )

    return process_row


def create_text_processor(
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    text_field: str = "text",
    append_eos: bool = True,
    truncate: bool = True,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build a row processor for RAW-TEXT (pretraining) data — no chat template.

    Each document gets a trailing EOS (``append_eos``) so packing preserves boundaries. Set
    ``truncate=False`` when packing so the packing strategy owns the overflow past max_length —
    ``bfd_split`` keeps it, ``bfd`` discards it. Empty/missing text yields the typed rejection
    sentinel (dropped downstream by :func:`is_valid_example`).
    """
    eos_id = tokenizer.eos_token_id

    def process_row(row: dict[str, Any]) -> dict[str, Any]:
        text = row.get(text_field)
        if not isinstance(text, str) or not text:
            return _rejection_sentinel()
        ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        add_eos = append_eos and eos_id is not None
        # Reserve the last slot for EOS before appending, so an at/over-max_length doc keeps its EOS.
        if truncate:
            cap = max_length - 1 if add_eos else max_length
            if len(ids) > cap:
                ids = ids[:cap]
        if add_eos:
            ids = ids + [eos_id]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    return process_row


def normalize_vlm_conversation(conversation: list[dict[str, Any]], images: Any = None) -> list[dict[str, Any]]:
    """Normalize a hub conversation to role/content messages and embed top-level images.

    Accepts either native ``{"role", "content"}`` messages (passed through) or the paired
    ``{"user", "assistant"}`` turn shape shared by the HuggingFaceM4 datasets
    (FineVision / the_cauldron / Docmatix ``texts`` column), each pair expanding to two messages.
    ``images`` (a single image or list, any form :func:`process_image` accepts) is injected into the
    first user turn ahead of its text — the only image placement those datasets define. Returns new
    message dicts; the input rows are never mutated.
    """
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
    drop_last_turn: bool = False,
) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    """One raw dataset row → the placeholder history and PIL images every VLM render takes.

    The pre-render half the runtime row map and the offline tokenizer share: JSON-string parse,
    hub-shape normalization plus ``images`` column merge, the optional generation-prompt slice — taken
    AFTER normalization, else a paired ``{user, assistant}`` turn drops whole — and image extraction.
    Emptiness is the caller's policy, so an empty conversation passes through as an empty history; a
    ``None`` or unparseable JSON string still raises, being a malformed column rather than an empty row.
    """
    conversation = maybe_parse_json(raw_conversation)
    if conversation is None or isinstance(conversation, str):
        raise ValueError(
            f"The conversation column holds a string that is not a valid JSON conversation: {conversation}"
        )

    conversation = normalize_vlm_conversation(conversation, images)
    if drop_last_turn:
        conversation = conversation[:-1]

    return process_vlm_conversation(
        conversation,
        system_prompt=system_prompt,
        model_supports_system_role=model_supports_system_role,
    )


def create_vlm_processor(
    conversation_field: str = "messages",
    system_prompt: str | None = None,
    model_supports_system_role: bool = True,
    add_generation_prompt: bool = False,
    tools_field: str | None = None,
    images_field: str | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build a VLM row processor: extract images and reshape the conversation into history.

    Takes no tokenizer: this map only reshapes the row, and the render that needs one runs later in
    ``VLMDataCollator`` — so the output is tokenizer-independent and the map's cache key must not
    pretend otherwise. No truncation anywhere downstream: over-length rows fail loud at collation
    (pre-filter via ``prepare_vlm_dataset`` or preprocess offline). ``images_field`` names a
    top-level image column merged into the conversation via :func:`normalize_vlm_conversation`.
    """

    def process_row(row: dict[str, Any]) -> dict[str, Any]:
        history, images = build_vlm_history(
            row[conversation_field],
            row.get(images_field) if images_field else None,
            system_prompt=system_prompt,
            model_supports_system_role=model_supports_system_role,
            drop_last_turn=add_generation_prompt,
        )

        # JSON keeps the column str|None; native list-of-dicts risks Arrow schema drift across rows.
        tools_json = None
        if tools_field and tools_field in row and row[tools_field] is not None:
            raw = row[tools_field]
            tools_json = raw if isinstance(raw, str) else json.dumps(raw)

        return {"history": history, "images": images, "tools_json": tools_json}

    return process_row
