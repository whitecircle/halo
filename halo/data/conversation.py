"""Conversation shaping shared by the text and VLM pipelines."""

import json
from collections.abc import Iterator
from typing import Any

# The image content part ``{"type": IMAGE_PART_TYPE, IMAGE_PART_PAYLOAD_KEY: <image>}`` every reader matches, or the
# transformers chat format's spelling of its source under one of ``IMAGE_PART_SOURCE_KEYS``.
IMAGE_PART_TYPE = "image"
IMAGE_PART_PAYLOAD_KEY = "image"
IMAGE_PART_SOURCE_KEYS = ("path", "base64", "url")
# No collator hands these to a processor, while chat templates render their placeholders.
UNSUPPORTED_MEDIA_PART_TYPES = frozenset({"video", "audio", "image_url", "video_url", "audio_url", "input_audio"})


def resolve_json_value(value):
    """Parse a JSON string; any other value (invalid JSON included) passes, so a mis-specified column fails in the chat
    template, not as an opaque ``JSONDecodeError`` in a map worker."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def as_conversation(prompt: Any) -> Any:
    """A plain string as a single user turn; anything else passes through for the caller to refuse."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return prompt


def _content_parts(conversation: Any) -> Iterator[dict]:
    """Anything that is not a message list yields nothing, left for the renderer to refuse."""
    conversation = resolve_json_value(conversation)
    if not isinstance(conversation, list):
        return
    for message in conversation:
        if isinstance(message, dict) and isinstance(message.get("content"), list):
            yield from (part for part in message["content"] if isinstance(part, dict))


def conversation_carries_images(conversation: Any) -> bool:
    return any(part.get("type") == IMAGE_PART_TYPE for part in _content_parts(conversation))


def is_image_placeholder(part: dict) -> bool:
    """An image part with no source: a falsy ``image`` payload counts, as the images column fills those; Arrow pads the
    keys a part lacks with ``None``."""
    return (
        part.get("type") == IMAGE_PART_TYPE
        and not part.get(IMAGE_PART_PAYLOAD_KEY)
        and all(part.get(key) is None for key in IMAGE_PART_SOURCE_KEYS)
    )


def reject_unsupported_media_parts(conversation: Any, source: str) -> None:
    """Refused on the text and VLM pipelines alike: their placeholders would train with nothing behind them."""
    found = {part.get("type") for part in _content_parts(conversation)} & UNSUPPORTED_MEDIA_PART_TYPES
    if found:
        raise ValueError(
            f"{source} carries {sorted(found)} content parts, which no Halo collator feeds a processor, so their "
            f"placeholders would train with nothing behind them. Images train from a '{IMAGE_PART_TYPE}' part whose "
            f"source is '{IMAGE_PART_PAYLOAD_KEY}' or one of {list(IMAGE_PART_SOURCE_KEYS)}, or from an images column."
        )


def reject_image_content(conversation: Any, source: str) -> None:
    """Refuse image parts on a text renderer, which would train placeholder tokens with no pixels: the per-row backstop
    for what the run-intent dispatch cannot see."""
    reject_unsupported_media_parts(conversation, source)
    if not conversation_carries_images(conversation):
        return
    raise ValueError(
        f"{source} carries an image content part, but this row is on the TEXT pipeline, which "
        f"renders images as placeholder tokens with no pixels behind them. Train it on the VLM "
        f"path: a multimodal checkpoint plus image data (an 'images'/'image' column, or "
        f"images_field naming one) routes there, and offline VLM preprocessing takes --is_vlm."
    )


def chat_template_kwargs(row: dict[str, Any], interleaved_thinking: bool, tools_field: str | None) -> dict:
    template_kwargs = {"clear_thinking": False} if interleaved_thinking else {}
    if tools_field and tools_field in row:
        tools = resolve_json_value(row[tools_field])
        if tools is not None:
            template_kwargs["tools"] = tools
    return template_kwargs


def _content_as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if part.get("type") == "text")
    raise ValueError(f"Unsupported message content type for system folding: {type(content).__name__}")


def _prefix_text_to_content(text: str, content: Any) -> Any:
    if isinstance(content, list):
        return [{"type": "text", "text": text + "\n"}, *content]
    return text + "\n" + content


def fold_system_into_conversation(
    messages: list[dict[str, Any]],
    system_prompt: str | None,
    model_supports_system_role: bool,
    demote_existing_system: bool = True,
) -> list[dict[str, Any]]:
    """Apply ``system_prompt``, folding it in when the model has no system role; folded messages are mutated in place
    (rows here are per-map copies). Text (``demote_existing_system``) prepends a system turn unless one leads, and
    without system-role support merges it into the next user turn or demotes it to ``user``. VLM folds only the
    configured prompt, onto a leading system or user turn, and leaves a dataset-borne system turn alone; where the
    model has the role, such a turn replaces the prompt, as templates raise on or drop a second system turn."""
    if demote_existing_system:
        if system_prompt and (not messages or messages[0]["role"] != "system"):
            messages = [{"role": "system", "content": system_prompt}, *messages]
        if not model_supports_system_role and messages and messages[0]["role"] == "system":
            if len(messages) > 1 and messages[1]["role"] == "user":
                system_text = _content_as_text(messages[0]["content"])
                messages[1]["content"] = _prefix_text_to_content(system_text, messages[1]["content"])
                return messages[1:]
            messages[0]["role"] = "user"
        return messages

    if not system_prompt:
        return messages
    if not model_supports_system_role and messages:
        first = messages[0]
        if first["role"] in ("system", "user"):
            first["content"] = _prefix_text_to_content(system_prompt, first["content"])
            return messages
    if messages and messages[0]["role"] == "system":
        return messages
    return [{"role": "system", "content": system_prompt}, *messages]


def _is_missing(value: Any) -> bool:
    """``None``, ``NaN`` and ``pd.NaT`` are unequal to themselves, and ``pd.NA``'s comparison raises on ``bool()``."""
    if value is None:
        return True
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return True


def resolve_system_prompt(row, local_field: str, global_prompt: str | None) -> str | None:
    """``row`` (a Series or a dict)'s own non-null ``local_field`` value, else the run-wide one."""
    value = row.get(local_field)
    if not _is_missing(value):
        return value
    return global_prompt


def build_base_prompt(row, prompt_field: str, system_prompt: str | None) -> list[dict]:
    messages = list(row[prompt_field])
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}, *messages]
    return messages
