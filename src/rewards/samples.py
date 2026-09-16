"""What a scorer reads: the prompt a policy was given, the turns it produced and the row's reference."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

Message = dict[str, Any]


@dataclass(frozen=True)
class ScoringSample:
    """One completion to score. ``prompt`` is the conversation the policy was prompted with (system
    and task turns), ``completion`` its turns after that (assistant turns, tool results), and
    ``reference`` the row's reference answer when it carries one."""

    prompt: list[Message]
    completion: list[Message]
    reference: Any = None


def final_assistant_text(messages: Sequence[Message]) -> str:
    """The visible text of the last assistant message, or ``""`` when there is none."""
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return _text(message.get("content"))
    return ""


def task_text(prompt: Sequence[Message]) -> str:
    """The user turns of the prompt, the task as the policy read it, without the system prompt."""
    return "\n\n".join(_text(message.get("content")) for message in prompt if message.get("role") == "user")


def scored_messages(sample: ScoringSample, transcript: str) -> list[Message]:
    """The conversation a scorer sees: the prompt plus the whole completion (``full``) or plus one
    assistant message holding the final text (``final``)."""
    if transcript == "full":
        return [*sample.prompt, *sample.completion]
    return [*sample.prompt, {"role": "assistant", "content": final_assistant_text(sample.completion)}]


def render_transcript(messages: Sequence[Message], *, max_chars: int) -> str:
    """The turns as plain text, one block per message with its tool calls, cut at ``max_chars``."""
    blocks = []
    for message in messages:
        role = message.get("role", "?")
        name = message.get("name")
        header = f"[{role}:{name}]" if role == "tool" and name else f"[{role}]"
        lines = [header]
        content = _text(message.get("content"))
        if content:
            lines.append(content)
        for call in message.get("tool_calls") or []:
            function = call.get("function", call) if isinstance(call, dict) else {}
            arguments = function.get("arguments", "")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            lines.append(f"-> {function.get('name', '?')}({arguments})")
        blocks.append("\n".join(lines))
    return truncate_text("\n\n".join(blocks), max_chars)


def truncate_text(text: str, max_chars: int) -> str:
    """``text`` cut to ``max_chars`` with the cut marked, so a scorer knows it read a prefix."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n…[truncated {len(text) - max_chars} chars]"


def samples_from_completions(
    prompts: Sequence[Any], completions: Sequence[Any], references: Sequence[Any] | None = None
) -> list[ScoringSample]:
    """Samples from TRL's shapes: each prompt or completion is a message list (conversational) or a
    plain string, which becomes one user or assistant message."""
    if references is None:
        references = [None] * len(prompts)
    return [
        ScoringSample(
            prompt=_as_messages(prompt, "user"), completion=_as_messages(completion, "assistant"), reference=reference
        )
        for prompt, completion, reference in zip(prompts, completions, references, strict=True)
    ]


def _as_messages(value: Any, role: str) -> list[Message]:
    if isinstance(value, str):
        return [{"role": role, "content": value}]
    return [dict(message) for message in value]


def _text(content: Any) -> str:
    """Message content as text: a string as is, a content-part list by its text parts, else its JSON."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [part.get("text", "") if isinstance(part, dict) else str(part) for part in content]
        return "\n".join(part for part in parts if part)
    return json.dumps(content, ensure_ascii=False)
