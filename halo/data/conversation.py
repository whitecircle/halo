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
UNSUPPORTED_MEDIA_PART_TYPES = frozenset(
    {"video", "audio", "image_url", "input_image", "video_url", "audio_url", "input_audio"}
)
# What a template can write as text. ``thinking`` is a real part of the assistant turn on Mistral Small 4 and Command A+,
# whose templates render it as their reasoning block; ``None`` is a part with no type.
_TEXT_PART_TYPES = (None, "text", "thinking")
# The config knobs naming a dataset column whose per-row cell a chat template reads.
TEMPLATE_COLUMN_KNOBS = ("tools_field", "reasoning_effort_field")


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


def _call_function(call: Any) -> Any:
    """What holds a call's name and arguments: its ``function``, or the call itself in the flat form."""
    function = call.get("function") if isinstance(call, dict) else None
    return function if isinstance(function, dict) else call


def _message_tool_calls(message: Any) -> list:
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    return calls if isinstance(calls, list) else []


def has_wire_form_arguments(conversation: Any) -> bool:
    """Whether a tool call carries its ``arguments`` as text, the OpenAI wire form, where a chat template that writes
    one block per argument takes a mapping."""
    if not isinstance(conversation, list):
        return False
    functions = (_call_function(call) for message in conversation for call in _message_tool_calls(message))
    return any(isinstance(function, dict) and isinstance(function.get("arguments"), str) for function in functions)


def _parse_call_arguments(call: Any) -> Any:
    function = _call_function(call)
    if not isinstance(function, dict) or not isinstance(function.get("arguments"), str):
        return call
    text = function["arguments"]
    try:
        arguments = json.loads(text) if text.strip() else {}
    except json.JSONDecodeError:
        arguments = None
    if not isinstance(arguments, dict):
        shown = text if len(text) <= 200 else f"{text[:200]}… (+{len(text) - 200} more characters)"
        raise ValueError(
            f"Tool call '{function.get('name')}' carries arguments that are not a JSON object ({shown!r}). The chat "
            f"template does not write a call's argument text as stored, so it is handed the mapping that text parses "
            f"into, and only a JSON object's text parses into one. Store the arguments as a JSON object or its text, or "
            f"drop the row."
        )
    parsed = {**function, "arguments": arguments}
    return parsed if function is call else {**call, "function": parsed}


def parse_tool_call_arguments(conversation: Any) -> Any:
    """``conversation`` with every tool call's wire-form ``arguments`` (JSON text) parsed into a mapping, an empty
    string as no arguments, the way a serving engine parses a request's before it renders; the input is left as it
    was. Raises on text that is not a JSON object."""
    if not isinstance(conversation, list):
        return conversation
    return [
        {**message, "tool_calls": [_parse_call_arguments(call) for call in _message_tool_calls(message)]}
        if _message_tool_calls(message)
        else message
        for message in conversation
    ]


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
    """Refuse media parts on a text renderer, which would train placeholder tokens with no data behind them or drop the
    part with no error: the per-row backstop for what the run-intent dispatch cannot see. Run routing keeps reading
    :func:`conversation_carries_images`, as the VLM path extracts only ``image`` parts."""
    reject_unsupported_media_parts(conversation, source)
    if conversation_carries_images(conversation):
        raise ValueError(
            f"{source} carries an image content part, but this row is on the TEXT pipeline, which "
            f"renders images as placeholder tokens with no pixels behind them. Train it on the VLM "
            f"path: a multimodal checkpoint plus image data (an 'images'/'image' column, or "
            f"images_field naming one) routes there, and offline VLM preprocessing takes --is_vlm."
        )
    # Each kind once; one with no type is the template's to refuse.
    other = sorted(
        {str(part["type"]) for part in _content_parts(conversation) if part.get("type") not in _TEXT_PART_TYPES}
    )
    if other:
        raise ValueError(
            f"{source} carries {other} content parts, but this row is on the TEXT pipeline: by template they render as "
            f"placeholder tokens with no data behind them or vanish with no error, so the run would train blind to "
            f"them. Write text parts or plain strings, or for images use {{'type': 'image'}} parts or an images column "
            f"on the VLM path."
        )


def template_column_kwargs(config: Any) -> dict[str, str | None]:
    """``config``'s template columns by knob, one dict for a renderer and its map-cache key."""
    return {knob: getattr(config, knob) for knob in TEMPLATE_COLUMN_KNOBS}


def resolve_row_reasoning_effort(row: dict[str, Any], reasoning_effort_field: str | None) -> str | None:
    """The row's level, or ``None`` for a null cell or an unset field; raises on a cell that is not a non-blank
    string, which a template would render as a level (gpt-oss writes ``Reasoning: `` for an empty one)."""
    if not reasoning_effort_field or row.get(reasoning_effort_field) is None:
        return None
    level = row[reasoning_effort_field]
    if not isinstance(level, str) or not level.strip():
        raise ValueError(
            f"reasoning_effort_field='{reasoning_effort_field}' holds the cell {level!r}, while the chat template takes "
            f"reasoning_effort as a level string. Store each level as a string such as 'low', 'medium' or 'high' (a "
            f"null cell passes no level, so the template's own default applies), or point reasoning_effort_field at "
            f"the column holding them."
        )
    return level


def resolve_row_tools(row: dict[str, Any], tools_field: str | None) -> Any:
    """The tool declarations ``row`` holds under ``tools_field``, ``None`` where it holds none."""
    if not tools_field or tools_field not in row:
        return None
    return resolve_json_value(row[tools_field])


def chat_template_kwargs(
    row: dict[str, Any], interleaved_thinking: bool, tools_field: str | None, reasoning_effort_field: str | None
) -> dict:
    """The template variables for ``row``: ``clear_thinking`` under interleaved thinking, then the row's tools and
    reasoning-effort level, each only when it has one."""
    template_kwargs = {"clear_thinking": False} if interleaved_thinking else {}
    tools = resolve_row_tools(row, tools_field)
    if tools is not None:
        template_kwargs["tools"] = tools
    level = resolve_row_reasoning_effort(row, reasoning_effort_field)
    if level is not None:
        template_kwargs["reasoning_effort"] = level
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
