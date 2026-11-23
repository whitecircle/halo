"""VLM render primitives: the chat render, the processor call, the over-length refusal, image decoding and
the placeholder token ids, shared by every VLM path so a runtime batch and an offline bake tokenize alike."""

import base64
import dataclasses
import io
import json
from typing import Any, get_type_hints

from datasets import Features, Sequence, Value
from PIL import Image

from halo.data.chat_render import has_open_final_assistant_turn, render_chat_template
from halo.data.conversation import (
    IMAGE_PART_PAYLOAD_KEY,
    IMAGE_PART_SOURCE_KEYS,
    IMAGE_PART_TYPE,
    fold_system_into_conversation,
    reject_unsupported_media_parts,
)

# The per-image patch grid a processor emits beside ``pixel_values``: Qwen's (temporal, height, width), Nemotron's
# (height, width).
VLM_GRID_COLUMNS = ("image_grid_thw", "image_grid_hw")
# A preprocessed VLM dataset's schema, pinned on the bake (an all-None batch infers null columns).
VLM_OUTPUT_FEATURES = Features(
    {
        "input_ids": Sequence(Value("int64")),
        "attention_mask": Sequence(Value("int64")),
        "labels": Sequence(Value("int64")),
        "pixel_values": Value("binary"),
        "pixel_values_shape": Sequence(Value("int64")),
        **dict.fromkeys(VLM_GRID_COLUMNS, Sequence(Sequence(Value("int64")))),
    }
)

# Per-token processor tensors (M-RoPE): padded and swapped with ``input_ids``, never row-major concatenated.
SEQUENCE_ALIGNED_VISION_KEYS = frozenset({"mm_token_type_ids", "token_type_ids"})
# The history message key holding, as JSON text, every key the message carries beside its role and content (tool calls,
# reasoning, a tool reply's call id and name): the mapped schema pins one struct for every message, which would drop a
# key it does not name, and JSON text keeps a call's arguments exact.
MESSAGE_FIELDS_KEY = "fields_json"


def vlm_row_template_kwargs(row: dict[str, Any]) -> dict[str, Any]:
    """The template variables a mapped VLM row carries, each only when it has one."""
    template_kwargs = {}
    tools = json.loads(row["tools_json"]) if row.get("tools_json") else None
    # ``is None``, not truthiness: a row declaring ``tools: []`` renders under ``tools is defined``.
    if tools is not None:
        template_kwargs["tools"] = tools
    if row.get("reasoning_effort") is not None:
        template_kwargs["reasoning_effort"] = row["reasoning_effort"]
    return template_kwargs


def _decode_message_fields(message: dict[str, Any]) -> dict[str, Any]:
    """The message with the keys :func:`_encode_message_fields` folded restored; a mapped row's null slot drops."""
    fields = message.get(MESSAGE_FIELDS_KEY)
    restored = {key: value for key, value in message.items() if key != MESSAGE_FIELDS_KEY}
    return {**restored, **json.loads(fields)} if fields else restored


def render_vlm_text(processor, history: list[dict[str, Any]], **template_kwargs) -> str:
    """Every VLM path renders here, so an artifact, a runtime batch and a preference pair tokenize alike: each message
    whole again, its tool calls' arguments as the processor's template takes them (:func:`render_chat_template`)."""
    return render_chat_template(processor, [_decode_message_fields(message) for message in history], **template_kwargs)


def render_vlm_training_text(
    processor, tokenizer, history: list[dict[str, Any]], eos_token_ids: frozenset[int], **template_kwargs
) -> str:
    """A training row's render, the tokenizer's eos token appended where the row ends on an assistant turn the template
    leaves open (:func:`~halo.data.chat_render.has_open_final_assistant_turn`). The text takes it, not the ids: a
    processor tokenizes and pads in one call, so only there does the eos reach every per-token column it builds."""
    text = render_vlm_text(processor, history, **template_kwargs)
    # The verdict the processor's ids would give: expanding a placeholder changes no token after it.
    input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if has_open_final_assistant_turn(input_ids, history, tokenizer, eos_token_ids):
        text += tokenizer.eos_token
    return text


def apply_image_pixel_bounds(processor, min_pixels: int | None, max_pixels: int | None) -> None:
    """On the loaded processor, so every collator and the exported processor config read one geometry. A smart-resize
    image processor keeps them as ``size``'s edges; anything else would ignore them in silence."""
    if min_pixels is None and max_pixels is None:
        return
    image_processor = getattr(processor, "image_processor", None)
    declared = get_type_hints(image_processor.valid_kwargs) if hasattr(image_processor, "valid_kwargs") else {}
    size = getattr(image_processor, "size", None)
    if not ({"min_pixels", "max_pixels"} <= declared.keys() and getattr(size, "longest_edge", None)):
        raise ValueError(
            f"min_pixels/max_pixels bound a smart-resize image processor's pixel count, and "
            f"{type(image_processor).__name__} resizes no such way, so it would ignore them. Drop them for this model."
        )
    image_processor.size = dataclasses.replace(
        size,
        shortest_edge=size.shortest_edge if min_pixels is None else min_pixels,
        longest_edge=size.longest_edge if max_pixels is None else max_pixels,
    )


def run_vlm_processor(processor, texts, row_images, **overrides):
    """``add_special_tokens=False``, as the render carries them. ``row_images`` holds one image list per text: Gemma
    3/4 read a flat list as one text's images and refuse a batch. A ``StopIteration`` (placeholder vs image count
    mismatch) re-raises as ``ValueError``, since one leaked from a collator ends the dataloader silently."""
    kwargs = {"text": texts, "return_tensors": "pt", "add_special_tokens": False, **overrides}
    if any(row_images):
        kwargs["images"] = [list(images) for images in row_images]
    try:
        return processor(**kwargs)
    except StopIteration as exc:
        raise ValueError(
            "Processor raised StopIteration — the text's image placeholder count does not match the provided images."
        ) from exc


def reject_over_length_vlm_batch(encoded, max_length: int | None, subject: str) -> None:
    """The only point that sees the length once image placeholders expand; ``None`` disables."""
    if max_length is None:
        return
    attention_mask = encoded.get("attention_mask")
    longest = int(attention_mask.sum(dim=-1).max()) if attention_mask is not None else encoded["input_ids"].shape[-1]
    if longest <= max_length:
        return
    raise ValueError(
        f"VLM {subject} contains a {longest}-token sequence, {longest - max_length} tokens over "
        f"max_length={max_length}, and image expansion happens at collation so a text-only pre-filter "
        f"could not see it. VLM sequences cannot be truncated at collation (cutting expanded image "
        f"placeholder tokens desyncs them from pixel_values and their grid), and a runtime batch cannot "
        f"drop rows without desyncing DP ranks. Raise max_length, pre-filter over-length rows (e.g. via "
        f"dataset preprocessing), or use smaller images."
    )


def decode_image(image_data) -> Image.Image:
    if isinstance(image_data, str):
        if image_data.startswith("data:image"):
            image_data = image_data.split(",")[1]
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        else:
            image = Image.open(image_data).convert("RGB")
    elif isinstance(image_data, Image.Image):
        image = image_data
    elif isinstance(image_data, bytes):
        image = Image.open(io.BytesIO(image_data))
    else:
        raise ValueError(f"Unsupported image format: {type(image_data)}")

    return image


def _decode_chat_format_source(key: str, source: str) -> Image.Image:
    """The transformers chat format's ``path``, ``base64`` and ``url``. No fetch: a map worker downloading per row would
    make a run depend on a remote host, and ranks on what each one got."""
    if key == "base64":
        return decode_image(source if source.startswith("data:image") else base64.b64decode(source))
    if key == "url" and source.startswith(("http://", "https://")):
        raise ValueError(
            f"An image part points at the remote URL {source!r}, and training fetches nothing. Download the images "
            f"and point 'path' at them, or embed them in '{IMAGE_PART_PAYLOAD_KEY}' or 'base64'."
        )
    return decode_image(source.removeprefix("file://"))


def decode_image_part(part: dict[str, Any]) -> Image.Image:
    """Presence, not truthiness, so an empty payload reaches the decoder and raises. By now the images column has filled
    every placeholder, so a part with no source has no pixels anywhere."""
    if part.get(IMAGE_PART_PAYLOAD_KEY) is not None:
        return decode_image(part[IMAGE_PART_PAYLOAD_KEY])
    for key in IMAGE_PART_SOURCE_KEYS:
        if part.get(key) is not None:
            return _decode_chat_format_source(key, part[key])
    raise ValueError(
        f"An image part carries no source ('{IMAGE_PART_PAYLOAD_KEY}' or one of {list(IMAGE_PART_SOURCE_KEYS)}) and no "
        f"images column filled it, so its placeholder would train with no pixels. Name the column with images_field."
    )


def resolve_image_token_ids(tokenizer, processor=None) -> set:
    """Image/video placeholder token ids to mask out of labels, unioned over every source: a missed one
    raises nothing, its tokens just train as text. A processor that looks up a token its tokenizer lacks holds the
    unknown token's id, which no placeholder is."""
    image_token_ids = set()
    unknown_token_id = getattr(tokenizer, "unk_token_id", None)

    for obj in (processor, getattr(processor, "tokenizer", None), tokenizer):
        for attr in ("image_token_id", "video_token_id"):
            token_id = getattr(obj, attr, None)
            if token_id is not None and token_id != unknown_token_id:
                image_token_ids.add(token_id)

    vocab = tokenizer.get_vocab()
    common_image_tokens = [
        "<image>",
        "<img>",
        "</img>",
        "<|image|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<image_soft_token>",
    ]
    for token in common_image_tokens:
        if token in vocab:
            image_token_ids.add(vocab[token])

    for obj in (processor, tokenizer):
        token = getattr(obj, "image_token", None)
        if isinstance(token, str) and token in vocab:
            image_token_ids.add(vocab[token])

    return image_token_ids


def _encode_message_fields(message: dict[str, Any]) -> dict[str, str]:
    """A null key is Arrow's padding of one a sibling message carries, so it is no field of this message."""
    fields = {key: value for key, value in message.items() if key not in ("role", "content") and value is not None}
    if not fields:
        return {}
    try:
        return {MESSAGE_FIELDS_KEY: json.dumps(fields)}
    except TypeError as exc:
        raise ValueError(
            f"A '{message.get('role')}' message carries {sorted(fields)} beside its role and content, which the VLM "
            f"row map hands the chat template as JSON, and one holds a value JSON cannot write ({exc}). Store that "
            f"value as JSON text or drop the key from the conversation column."
        ) from exc


def extract_conversation_images(
    conversation: list[dict[str, Any]],
    system_prompt: str | None = None,
    model_supports_system_role: bool = True,
) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    """The conversation with each image replaced by a bare placeholder, and the decoded images. Each message keeps every
    key it carries, those beside its role and content under :data:`MESSAGE_FIELDS_KEY`."""
    reject_unsupported_media_parts(conversation, "This row")
    # Without a system role the prompt is folded into the first turn, not emitted as one the template drops.
    conversation = fold_system_into_conversation(
        conversation, system_prompt, model_supports_system_role, demote_existing_system=False
    )

    history = []
    images = []

    for msg in conversation:
        role = msg["role"]
        content = msg.get("content")
        fields = _encode_message_fields(msg)

        if content is None or content == "":
            # As the text path hands it over: a template tests an empty content (Qwen3-VL's newline before a call,
            # Inkling's empty turn), where an empty text part reads as text and no part as no turn.
            history.append({"role": role, "content": content, **fields})
        elif isinstance(content, str):
            # Parts-form: the mapped schema's Json content would read a string holding JSON ("42") back as its value.
            # Processors render parts-form text byte-identically.
            history.append({"role": role, "content": [{"type": "text", "text": content}], **fields})
        elif isinstance(content, list):
            new_content = []
            for item in content:
                if item.get("type") == IMAGE_PART_TYPE:
                    images.append(decode_image_part(item))
                    new_content.append({"type": IMAGE_PART_TYPE})  # processor expects this placeholder
                else:
                    # Arrow pads a text part with a null image key, which templates test for (Qwen-VL).
                    new_content.append({key: value for key, value in item.items() if value is not None})

            history.append({"role": role, "content": new_content, **fields})
        else:
            raise ValueError(f"Unsupported message content type {type(content).__name__} for role '{role}'")

    return history, images
