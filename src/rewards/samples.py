"""What a scorer reads of a finished episode, and how each view of it renders as text.

A :class:`ScoringSample` is one completion to score: the prompt the policy was given, every turn it
and its tools produced, what it finally answered, the row's reference and the tools it could call.
A term picks its VIEW of the sample — the final answer, the full transcript or a digest of it — and
the renderers here turn that view into the text a judge reads: numbered turns, the policy's
reasoning set apart from its visible text, every tool call with its arguments and every result
under the call it answers, and the turns the engine cut or the policy wasted marked as such, with the
calls a cut turn was writing marked as never run.
"""

import contextlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.inference.response import get_reasoning_text
from src.rewards.terms import View

Message = dict[str, Any]

# The turn flags an environment stamps on a sample message, each with the note a judge reads for it.
TURN_FLAG_NOTES = {
    "truncated": "cut by the engine at its length limit",
    "empty": "ended with neither visible text nor a tool call",
    "calls_rejected": "every tool call named a tool that does not exist, was refused unrun or ran and showed nothing",
}
# The sample-message key an environment puts the calls of a turn the engine cut while writing them under, and
# the note each renders with. They never ran, so they stay apart from ``tool_calls``, which the chat wire reads.
CUT_CALLS_KEY = "cut_tool_calls"
CUT_CALL_NOTE = "cut by the engine before the turn closed; never run"
# The keys of a message as the chat wire spells it; a sample message's other keys (the reasoning under
# either engine's spelling, the turn flags, a cut turn's calls) are the sample's own.
WIRE_KEYS = ("role", "content", "name", "tool_calls", "tool_call_id")

# How a cut text marks what it dropped: the head and the tail stay, so the answer or submission at
# the end of a long transcript is never what the cut removes.
CUT_MARKER = "\n…[{dropped} chars cut here]…\n"
# The share of a cut text kept before the marker; the rest of the budget keeps the end.
CUT_HEAD_SHARE = 2 / 3

# The digest's per-item budgets, enough to see what each turn did, not what it said at length: one for a
# turn's reasoning, text and tool result and the final answer, a shorter one for what is shown inline, a
# tool call's arguments and a tool's description.
DIGEST_CHARS = 400
DIGEST_INLINE_CHARS = 160

NO_FINAL_ANSWER = "(The episode ended without a final answer.)"


@dataclass(frozen=True)
class ScoringSample:
    """One completion to score. ``prompt`` is the conversation the policy was prompted with (system
    and task turns), ``completion`` its turns after that (assistant turns with their reasoning and
    tool calls, tool results), ``final_answer`` what the episode delivered — the final text answer,
    a submitted program — or ``None`` when it ended without one, ``reference`` the row's reference
    answer when it carries one, and ``tools`` the OpenAI tool schemas the policy could call."""

    prompt: list[Message]
    completion: list[Message]
    final_answer: str | None = None
    reference: Any = None
    tools: list[dict[str, Any]] | None = field(default=None, compare=False)


def final_assistant_text(messages: Sequence[Message]) -> str:
    """The visible text of the last assistant message, or ``""`` when there is none."""
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return _text(message.get("content"))
    return ""


def task_text(prompt: Sequence[Message]) -> str:
    """The user turns of the prompt, the task as the policy read it, without the system prompt."""
    return "\n\n".join(_text(message.get("content")) for message in prompt if message.get("role") == "user")


def scored_messages(sample: ScoringSample, view: View) -> list[Message]:
    """The conversation a scorer reads as messages: the prompt plus the whole completion (``full``) or
    plus one assistant message holding the final answer (``final``; empty when the episode delivered
    none, never a fragment), as the chat wire spells them — the sample-only keys (turn flags, a cut turn's
    calls, reasoning) dropped."""
    if view is View.FULL:
        messages = [*sample.prompt, *sample.completion]
    else:
        messages = [*sample.prompt, {"role": "assistant", "content": sample.final_answer or ""}]
    wire = []
    for message in messages:
        out = {"role": message.get("role", "?"), "content": message.get("content") or ""}
        out.update({key: message[key] for key in WIRE_KEYS if key not in out and message.get(key)})
        wire.append(out)
    return wire


def view_text(sample: ScoringSample, view: View, *, include_reasoning: bool, max_chars: int) -> str:
    """The text of one view of the sample, cut to ``max_chars`` with its end kept (:func:`cut_middle`)."""
    if view is View.FINAL:
        text = render_final_answer(sample)
    elif view is View.FULL:
        text = render_transcript(sample.completion, include_reasoning=include_reasoning)
    elif view is View.DIGEST:
        text = render_digest(sample, include_reasoning=include_reasoning)
    else:
        raise ValueError(f"unknown view {view!r}")
    return cut_middle(text, max_chars)


def render_final_answer(sample: ScoringSample) -> str:
    """The episode's final answer, or a note that there is none with the last assistant text after it,
    so a judge reads a fragment or a tool-call turn as what it is, never as the answer."""
    if sample.final_answer is not None:
        return sample.final_answer
    last = final_assistant_text(sample.completion)
    return f"{NO_FINAL_ANSWER}\nLast assistant turn:\n{last}" if last else NO_FINAL_ANSWER


def render_tools(tools: Sequence[dict[str, Any]]) -> str:
    """One line per tool: its name, its parameters and the head of its description."""
    lines = []
    for tool in tools:
        function = tool.get("function", tool) if isinstance(tool, Mapping) else {}
        parameters = function.get("parameters") or {}
        names = ", ".join((parameters.get("properties") or {}).keys())
        description = _head(_text(function.get("description")), DIGEST_INLINE_CHARS)
        lines.append(f"- {function.get('name', '?')}({names})" + (f": {description}" if description else ""))
    return "\n".join(lines)


def render_transcript(messages: Sequence[Message], *, include_reasoning: bool) -> str:
    """Every turn whole: numbered, the reasoning set apart, each tool call with its arguments (a cut
    turn's after them, marked :data:`CUT_CALL_NOTE`), each tool result under its name and call id."""
    return "\n\n".join(_render_turn(i, m, include_reasoning, digest=False) for i, m in enumerate(messages, 1))


def render_digest(sample: ScoringSample, *, include_reasoning: bool) -> str:
    """Every turn compactly — the head of each reasoning and text, each tool call with its arguments cut
    to their heads (verbatim, so a quote from a program matches), the head and tail of each result —
    then the final answer whole, the artifact an audit reads."""
    turns = [_render_turn(i, m, include_reasoning, digest=True) for i, m in enumerate(sample.completion, 1)]
    final = sample.final_answer if sample.final_answer is not None else NO_FINAL_ANSWER
    return "\n\n".join([*turns, f"Final answer:\n{final}"])


def cut_middle(text: str, max_chars: int) -> str:
    """``text`` cut to about ``max_chars`` with its head and its tail kept and the cut marked
    (:data:`CUT_MARKER`), so a reader sees how the text began and how it ended."""
    if len(text) <= max_chars:
        return text
    head = int(max_chars * CUT_HEAD_SHARE)
    tail = max_chars - head
    return text[:head] + CUT_MARKER.format(dropped=len(text) - head - tail) + (text[-tail:] if tail else "")


def samples_from_completions(
    prompts: Sequence[Any], completions: Sequence[Any], references: Sequence[Any] | None = None
) -> list[ScoringSample]:
    """Samples from TRL's shapes: each prompt or completion is a message list (conversational) or a
    plain string, which becomes one user or assistant message; the final answer is the last
    assistant text."""
    if references is None:
        references = [None] * len(prompts)
    samples = []
    for prompt, completion, reference in zip(prompts, completions, references, strict=True):
        messages = _as_messages(completion, "assistant")
        samples.append(
            ScoringSample(
                prompt=_as_messages(prompt, "user"),
                completion=messages,
                final_answer=final_assistant_text(messages),
                reference=reference,
            )
        )
    return samples


def _render_turn(index: int, message: Message, include_reasoning: bool, *, digest: bool) -> str:
    role = message.get("role", "?")
    name = message.get("name")
    call_id = message.get("tool_call_id")
    header = f"[{index}] {role}"
    if role == "tool":
        header += f" {name}" if name else ""
        header += f" (call {call_id})" if call_id else ""
    notes = [note for flag, note in TURN_FLAG_NOTES.items() if message.get(flag)]
    if notes:
        header += f"  — {'; '.join(notes)}"
    lines = [header]
    reasoning = get_reasoning_text(message) if include_reasoning else None
    if reasoning:
        body = _head(reasoning, DIGEST_CHARS) if digest else reasoning
        lines.append(f"<reasoning>\n{body}\n</reasoning>")
    content = _text(message.get("content"))
    if content:
        if digest:
            content = _head(content, DIGEST_CHARS) if role == "assistant" else cut_middle(content, DIGEST_CHARS)
        lines.append(content)
    argument_chars = DIGEST_INLINE_CHARS if digest else None
    for call in message.get("tool_calls") or []:
        lines.append(_render_call(call, argument_chars))
    for call in message.get(CUT_CALLS_KEY) or []:
        lines.append(_render_call(call, argument_chars, note=CUT_CALL_NOTE))
    return "\n".join(lines)


def _render_call(call: Any, argument_chars: int | None, *, note: str | None = None) -> str:
    """``→ name`` with the call id and ``note`` beside it, then every argument cut to its head past
    ``argument_chars`` (``None`` keeps each whole)."""
    function = call.get("function", call) if isinstance(call, Mapping) else {}
    call_id = call.get("id") if isinstance(call, Mapping) else None
    tags = "; ".join(tag for tag in (f"call {call_id}" if call_id else None, note) if tag)
    head = f"→ {function.get('name', '?')}" + (f" ({tags})" if tags else "")
    arguments = function.get("arguments", "")
    if isinstance(arguments, str):
        with contextlib.suppress(ValueError):
            arguments = json.loads(arguments) if arguments else {}
    if not isinstance(arguments, Mapping):
        return f"{head}: {_argument_text(arguments, argument_chars)}"
    if not arguments:
        return head
    # Every argument verbatim, a multi-line one on its own lines: a quote from a program the policy
    # submitted must match the text as the policy wrote it, not a JSON-escaped copy.
    lines = [head]
    for key, value in arguments.items():
        text = _argument_text(value, argument_chars)
        lines.append(f"  {key}:\n{text}" if "\n" in text else f"  {key}: {text}")
    return "\n".join(lines)


def _argument_text(value: Any, max_chars: int | None) -> str:
    """An argument as the policy wrote it (a non-string as JSON), cut to its head past ``max_chars``."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if max_chars is None else _head(text, max_chars)


def _head(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + f"…[{len(text) - max_chars} more chars]"


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
