"""VLM render primitives: the chat render, the processor call, the over-length refusal, image decoding and
the placeholder token ids, shared by every VLM path so a runtime batch and an offline bake tokenize alike."""

import base64
import io
import json
from typing import Any

from datasets import Features, Sequence, Value
from PIL import Image

from halo.data.conversation import IMAGE_PART_PAYLOAD_KEY, IMAGE_PART_TYPE, fold_system_into_conversation

# A preprocessed VLM dataset's schema, pinned on the bake (an all-None batch infers null columns).
VLM_OUTPUT_FEATURES = Features(
    {
        "input_ids": Sequence(Value("int64")),
        "attention_mask": Sequence(Value("int64")),
        "labels": Sequence(Value("int64")),
        "pixel_values": Value("binary"),
        "pixel_values_shape": Sequence(Value("int64")),
        "image_grid_thw": Sequence(Sequence(Value("int64"))),
    }
)

# Per-token processor tensors (M-RoPE): padded and swapped with ``input_ids``, never row-major concatenated.
SEQUENCE_ALIGNED_VISION_KEYS = frozenset({"mm_token_type_ids", "token_type_ids"})


def vlm_row_tools(row: dict[str, Any]) -> list | None:
    tools_json = row.get("tools_json")
    return json.loads(tools_json) if tools_json else None


def render_vlm_text(processor, history: list[dict[str, Any]], *, tools: list | None = None) -> str:
    """Every VLM path renders here, so an artifact, a runtime batch and a preference pair tokenize alike."""
    # ``is None``, not truthiness: a row declaring ``tools: []`` renders under ``tools is defined``.
    template_kwargs = {} if tools is None else {"tools": tools}
    return processor.apply_chat_template(history, tokenize=False, add_generation_prompt=False, **template_kwargs)


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
        f"placeholder tokens desyncs them from pixel_values/image_grid_thw), and a runtime batch cannot "
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


def resolve_image_token_ids(tokenizer, processor=None) -> set:
    """Image/video placeholder token ids to mask out of labels, unioned over every source: a missed one
    raises nothing, its tokens just train as text."""
    image_token_ids = set()

    for obj in (processor, getattr(processor, "tokenizer", None), tokenizer):
        for attr in ("image_token_id", "video_token_id"):
            token_id = getattr(obj, attr, None)
            if token_id is not None:
                image_token_ids.add(token_id)

    vocab = tokenizer.get_vocab()
    common_image_tokens = [
        "<image>",
        "<img>",
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


def extract_conversation_images(
    conversation: list[dict[str, Any]],
    system_prompt: str | None = None,
    model_supports_system_role: bool = True,
) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    """The conversation with each image replaced by a bare placeholder, and the decoded images."""
    # Without a system role the prompt is folded into the first turn, not emitted as one the template drops.
    conversation = fold_system_into_conversation(
        conversation, system_prompt, model_supports_system_role, demote_existing_system=False
    )

    history = []
    images = []

    for msg in conversation:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            # Uniformly parts-form: mixed string/list content makes Arrow type inference diverge across
            # .map workers (struct vs Json vs null). Processors render parts-form text byte-identically.
            history.append({"role": role, "content": [{"type": "text", "text": content}]})
        elif isinstance(content, list):
            new_content = []
            for item in content:
                if item.get("type") == IMAGE_PART_TYPE:
                    image_data = item.get(IMAGE_PART_PAYLOAD_KEY)
                    # Presence, not truthiness: an empty payload must reach decode_image and raise.
                    if image_data is not None:
                        image = decode_image(image_data)
                        images.append(image)
                    new_content.append({"type": IMAGE_PART_TYPE})  # processor expects this placeholder
                else:
                    # Arrow pads a text part with a null image key, which templates test for (Qwen-VL).
                    new_content.append({key: value for key, value in item.items() if value is not None})

            history.append({"role": role, "content": new_content})
        else:
            raise ValueError(f"Unsupported message content type {type(content).__name__} for role '{role}'")

    return history, images
