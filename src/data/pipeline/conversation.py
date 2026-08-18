"""Conversation shaping shared by the text and VLM pipelines (system-prompt folding, template kwargs).

Torch-free leaf module: importable from both the text renderers
(:mod:`src.data.pipeline.rendered`) and :mod:`src.data.vlm` without creating an
import cycle.
"""

import json
from typing import Any

# The image content part, spelled once for the whole pipeline:
# ``{"type": IMAGE_PART_TYPE, IMAGE_PART_PAYLOAD_KEY: <image>}``. A re-inlined spelling lets the
# detector and the extractor drift apart, which reads as "this row has no images".
IMAGE_PART_TYPE = "image"
IMAGE_PART_PAYLOAD_KEY = "image"


def maybe_parse_json(value):
    """Parse a JSON-encoded string into Python objects; pass any other value through unchanged.

    A string that isn't valid JSON is returned as-is, so a mis-specified ``conversation_field`` surfaces
    as a clear chat-template error downstream, not an opaque ``JSONDecodeError`` in a ``map`` worker.
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def conversation_carries_images(conversation: Any) -> bool:
    """Whether a conversation embeds images in its message content.

    Reads the exact part shape the VLM pipeline extracts images from, so a conversation this reports
    on is image data whatever column it arrived in; the JSON-string form is accepted for parity with
    the renderers. Anything that is not a message list reads as text — a mis-specified conversation
    field must surface at the renderer, not here.
    """
    conversation = maybe_parse_json(conversation)
    if not isinstance(conversation, list):
        return False
    return any(
        isinstance(part, dict) and part.get("type") == IMAGE_PART_TYPE
        for message in conversation
        if isinstance(message, dict) and isinstance(message.get("content"), list)
        for part in message["content"]
    )


def reject_image_content(conversation: Any, source: str) -> None:
    """Fail loud when a TEXT renderer is handed a conversation carrying image parts.

    Chat templates expand an image part into placeholder tokens the text pipeline never backs with
    pixels, so the row would train them as text in silence. This is the per-row backstop for what the
    run-intent dispatch (:func:`src.data.vlm.is_vlm_run`) cannot see, so every text renderer
    goes through it.
    """
    if not conversation_carries_images(conversation):
        return
    raise ValueError(
        f"{source} carries an image content part, but this row is on the TEXT pipeline, which "
        f"renders images as placeholder tokens with no pixels behind them. Train it on the VLM "
        f"path: a multimodal checkpoint plus image data (an 'images'/'image' column, or "
        f"images_field naming one) routes there, and offline VLM preprocessing takes --vlm."
    )


def chat_template_kwargs(row: dict[str, Any], interleaved_thinking: bool, tools_field: str | None) -> dict:
    """Build the shared ``apply_chat_template`` kwargs (interleaved-thinking flag + parsed tools)."""
    template_kwargs = {"clear_thinking": False} if interleaved_thinking else {}
    if tools_field and tools_field in row:
        tools = maybe_parse_json(row[tools_field])
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
    """Apply ``system_prompt`` to a conversation, folding it away when the model has no system role.

    Content is handled in both str and parts-list form; folded-into messages are mutated in place
    (rows reaching here are per-map copies). The two policies are per caller:

    - ``demote_existing_system=True`` (text): prepend the system turn unless one is already leading;
      without system-role support the leading system turn is merged into the following user turn
      (system text first), or demoted to ``user`` when no user turn follows.
    - ``demote_existing_system=False`` (VLM): only the configured ``system_prompt`` folds, prefixed
      onto a leading system/user turn; a dataset-borne system turn is left untouched.
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


def resolve_system_prompt(row, local_field: str, global_prompt: str | None) -> str | None:
    """Resolve a row's system prompt: a row-local field wins over the run-wide one.

    ``row`` may be a pandas Series or a plain dict (both support ``.get``); a missing value in the
    local column reads as absent, which is what an unset cell deserializes to.
    """
    value = row.get(local_field)
    if not _is_missing(value):
        return value
    return global_prompt


def _is_missing(value: Any) -> bool:
    """Whether ``value`` is a null/NA sentinel, without importing pandas (leaf-module rule).

    Covers ``None``, float/numpy ``NaN``, ``pd.NaT`` (all unequal to themselves) and ``pd.NA``
    (whose comparison yields ``pd.NA``, which raises on ``bool()``) — a nullable/pyarrow-backed
    column otherwise renders its sentinel into a chat message as if it were a real prompt.
    """
    if value is None:
        return True
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return True


def build_base_prompt(row, prompt_field: str, system_prompt: str | None) -> list[dict]:
    """The row's prompt messages with ``system_prompt`` prepended when set.

    ``row`` may be a pandas Series or a plain dict.
    """
    messages = list(row[prompt_field])
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}, *messages]
    return messages
