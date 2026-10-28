"""Conversation shaping shared by the text and VLM pipelines: JSON columns, image parts, system-prompt folding
and template kwargs."""

import json
from typing import Any

# The image content part ``{"type": IMAGE_PART_TYPE, IMAGE_PART_PAYLOAD_KEY: <image>}`` every reader matches.
IMAGE_PART_TYPE = "image"
IMAGE_PART_PAYLOAD_KEY = "image"


def resolve_json_value(value):
    """A JSON-encoded string parsed, any other value (an invalid JSON string included) unchanged: a mis-specified
    column then fails in the chat template, not as an opaque ``JSONDecodeError`` in a map worker."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def as_conversation(prompt: Any) -> Any:
    """A plain-string prompt as a single user turn; any other value (a message list, or a malformed
    prompt the caller refuses) passes through unchanged."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return prompt


def conversation_carries_images(conversation: Any) -> bool:
    """Whether a conversation (or its JSON string) embeds image parts in its message content; anything that is not
    a message list reads as text, left for the renderer to refuse."""
    conversation = resolve_json_value(conversation)
    if not isinstance(conversation, list):
        return False
    return any(
        isinstance(part, dict) and part.get("type") == IMAGE_PART_TYPE
        for message in conversation
        if isinstance(message, dict) and isinstance(message.get("content"), list)
        for part in message["content"]
    )


def reject_image_content(conversation: Any, source: str) -> None:
    """Refuse a conversation carrying image parts on a text renderer, which would train their placeholder tokens
    with no pixels behind them: the per-row backstop for what the run-intent dispatch cannot see."""
    if not conversation_carries_images(conversation):
        return
    raise ValueError(
        f"{source} carries an image content part, but this row is on the TEXT pipeline, which "
        f"renders images as placeholder tokens with no pixels behind them. Train it on the VLM "
        f"path: a multimodal checkpoint plus image data (an 'images'/'image' column, or "
        f"images_field naming one) routes there, and offline VLM preprocessing takes --is_vlm."
    )


def chat_template_kwargs(row: dict[str, Any], interleaved_thinking: bool, tools_field: str | None) -> dict:
    """The ``apply_chat_template`` kwargs every text render shares: the interleaved-thinking flag and parsed tools."""
    template_kwargs = {"clear_thinking": False} if interleaved_thinking else {}
    if tools_field and tools_field in row:
        tools = resolve_json_value(row[tools_field])
        if tools is not None:
            template_kwargs["tools"] = tools
    return template_kwargs


def _content_as_text(content: Any) -> str:
    """Plain text of message content: pass a str through, join the text parts of a parts-list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if part.get("type") == "text")
    raise ValueError(f"Unsupported message content type for system folding: {type(content).__name__}")


def _prefix_text_to_content(text: str, content: Any) -> Any:
    """Prefix ``text`` + newline onto message content, handling str and parts-list forms."""
    if isinstance(content, list):
        return [{"type": "text", "text": text + "\n"}, *content]
    return text + "\n" + content


def fold_system_into_conversation(
    messages: list[dict[str, Any]],
    system_prompt: str | None,
    model_supports_system_role: bool,
    demote_existing_system: bool = True,
) -> list[dict[str, Any]]:
    """Apply ``system_prompt``, folding it into the conversation when the model has no system role; folded-into
    messages are mutated in place (rows here are per-map copies).

    Text (``demote_existing_system``) prepends a system turn unless one leads, and without system-role support
    merges the leading one into the next user turn or demotes it to ``user``. VLM folds only the configured
    prompt, onto a leading system or user turn, and leaves a dataset-borne system turn alone.
    """
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
    return [{"role": "system", "content": system_prompt}, *messages]


def _is_missing(value: Any) -> bool:
    """Whether ``value`` is a null sentinel: ``None``, ``NaN`` and ``pd.NaT`` are unequal to themselves, and
    ``pd.NA``'s comparison raises on ``bool()``."""
    if value is None:
        return True
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return True


def resolve_system_prompt(row, local_field: str, global_prompt: str | None) -> str | None:
    """A row's system prompt (``row`` a pandas Series or a dict): its own non-null ``local_field`` value, else
    the run-wide one."""
    value = row.get(local_field)
    if not _is_missing(value):
        return value
    return global_prompt


def build_base_prompt(row, prompt_field: str, system_prompt: str | None) -> list[dict]:
    """The row's prompt messages (``row`` a pandas Series or a dict) with ``system_prompt`` prepended when set."""
    messages = list(row[prompt_field])
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}, *messages]
    return messages
