"""What a scorer reads of a finished episode (:class:`ScoringSample`), and the text a judge reads of each view of it:
the final answer, the full transcript or a digest."""

import contextlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from halo.inference.response import get_reasoning_text
from halo.rewards.terms import View

# Each part of a judge's prompt goes between its own tags: a task's markdown headings and a program's comments would
# read as the prompt's own sections under markdown ones. Quoted text spells none of these tags (:func:`escape_tags`).
VIEW_TAGS = {View.FINAL: "final_answer", View.FULL: "transcript", View.DIGEST: "transcript_digest"}
_PART_TAGS = ("setting", "policy_instructions", "task", "policy_tools", "reference_answer", *VIEW_TAGS.values())
_REASONING_TAG = "reasoning"
# The ``<`` that opens or closes one of them, in any case and with any attributes.
_PROMPT_TAG = re.compile(rf"<(?=/?(?:{'|'.join((*_PART_TAGS, _REASONING_TAG))})\b)", re.IGNORECASE)

# Turn flags an environment stamps on a sample message, with the note a judge reads for each.
TURN_FLAG_NOTES = {
    "reasoning_capped": "its reasoning ran to the turn's cap and the engine closed it, so what follows was written past it",
    "truncated": "cut by the engine at its length limit",
    "empty": "ended with neither visible text nor a tool call",
    "calls_rejected": "every tool call named a tool that does not exist or was refused unrun",
}
# Calls of a turn the engine cut while writing them; they never ran, so they stay apart from ``tool_calls``.
CUT_TOOL_CALLS_KEY = "cut_tool_calls"
CUT_CALL_NOTE = "cut by the engine before the turn closed; never run"
# Ids of a turn's calls refused unrun, so a judge reads each as never run though the policy wrote it.
REFUSED_CALL_IDS_KEY = "refused_call_ids"
REFUSED_CALL_NOTE = "refused unrun; never run"
# A message's keys on the chat wire; any other sample-message key is the sample's own.
WIRE_KEYS = ("role", "content", "name", "tool_calls", "tool_call_id")

# Head and tail stay, so the answer or submission at the end of a long transcript is never what the cut removes.
CUT_MARKER = "\n…[{dropped} chars cut here]…\n"
_CUT_HEAD_SHARE = 2 / 3

# Digest per-item budgets: one for a turn's reasoning, text and tool result, a shorter one for a call's arguments.
DIGEST_CHARS = 400
DIGEST_INLINE_CHARS = 160

NO_FINAL_ANSWER = "(The episode ended without a final answer.)"

_SampleMessage = dict[str, Any]


@dataclass(frozen=True)
class ScoringSample:
    """One completion to score. ``final_answer`` is ``None`` when the episode delivered none; ``facts`` are what the
    environment establishes, which a judge check stands exempt under."""

    prompt: list[_SampleMessage]
    completion: list[_SampleMessage]
    final_answer: str | None = None
    reference: Any = None
    tools: list[dict[str, Any]] | None = field(default=None, compare=False)
    facts: frozenset[str] = frozenset()


def _text(content: Any) -> str:
    """A string as is, a content-part list by its text parts, else its JSON."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [part.get("text", "") if isinstance(part, dict) else str(part) for part in content]
        return "\n".join(part for part in parts if part)
    return json.dumps(content, ensure_ascii=False)


def final_assistant_text(messages: Sequence[_SampleMessage]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return _text(message.get("content"))
    return ""


def task_text(prompt: Sequence[_SampleMessage]) -> str:
    return "\n\n".join(_text(message.get("content")) for message in prompt if message.get("role") == "user")


def system_text(prompt: Sequence[_SampleMessage]) -> str:
    return "\n\n".join(_text(message.get("content")) for message in prompt if message.get("role") == "system")


def scored_messages(sample: ScoringSample, view: View) -> list[_SampleMessage]:
    """The prompt plus the whole completion (``full``) or one assistant message holding the final answer (``final``;
    empty when the episode delivered none, never a fragment), with only the chat wire's keys."""
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
    """What it quotes is escaped (:func:`escape_tags`); cut to ``max_chars`` with its end kept."""
    if view is View.FINAL:
        text = render_final_answer(sample)
    elif view is View.FULL:
        text = render_transcript(sample.completion, include_reasoning=include_reasoning, max_chars=max_chars)
    elif view is View.DIGEST:
        text = render_digest(sample, include_reasoning=include_reasoning)
    else:
        raise ValueError(f"unknown view {view!r}")
    return cut_middle(text, max_chars)


def shows_reasoning(sample: ScoringSample, view: View, *, include_reasoning: bool) -> bool:
    return include_reasoning and view is not View.FINAL and any(map(get_reasoning_text, sample.completion))


def _head(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + f"…[{len(text) - max_chars} more chars]"


def _argument_text(value: Any, max_chars: int | None) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if max_chars is None else _head(text, max_chars)


def _call_arguments(call: Any) -> Any:
    """A JSON string decoded (an empty one as ``{}``); text that does not parse stays text."""
    function = call.get("function", call) if isinstance(call, Mapping) else {}
    arguments = function.get("arguments", "")
    if isinstance(arguments, str):
        with contextlib.suppress(ValueError):
            arguments = json.loads(arguments) if arguments else {}
    return arguments


def _acted_text(message: _SampleMessage, *, digest: bool) -> str:
    """An assistant turn's visible text and its calls' arguments, or a tool result; a user turn (a nudge) has none."""
    role = message.get("role")
    content = _text(message.get("content"))
    if role == "tool":
        return cut_middle(content, DIGEST_CHARS) if digest else content
    if role != "assistant":
        return ""
    pieces = [_head(content, DIGEST_CHARS) if digest else content]
    for call in [*(message.get("tool_calls") or []), *(message.get(CUT_TOOL_CALLS_KEY) or [])]:
        arguments = _call_arguments(call)
        values = arguments.values() if isinstance(arguments, Mapping) else [arguments]
        pieces.extend(_argument_text(value, DIGEST_INLINE_CHARS if digest else None) for value in values)
    return "\n".join(piece for piece in pieces if piece)


def render_actions(sample: ScoringSample, view: View, *, max_chars: int) -> str:
    """The policy's actions and the tool results it received, without its reasoning or the environment's notes, so a
    quote of either is never taken for an action. Under ``final``, the answer or the last assistant text."""
    if view is View.FINAL:
        text = sample.final_answer if sample.final_answer is not None else final_assistant_text(sample.completion)
    elif view in (View.FULL, View.DIGEST):
        digest = view is View.DIGEST
        pieces = [_acted_text(message, digest=digest) for message in sample.completion]
        if digest and sample.final_answer is not None:
            pieces.append(sample.final_answer)
        text = "\n\n".join(piece for piece in pieces if piece)
    else:
        raise ValueError(f"unknown view {view!r}")
    return cut_middle(text, max_chars)


def render_final_answer(sample: ScoringSample) -> str:
    """The final answer, or a note that there is none with the last assistant text after it, so a judge reads a
    fragment or a tool-call turn as what it is."""
    if sample.final_answer is not None:
        return escape_tags(sample.final_answer)
    last = escape_tags(final_assistant_text(sample.completion))
    return f"{NO_FINAL_ANSWER}\nLast assistant turn:\n{last}" if last else NO_FINAL_ANSWER


def render_tools(tools: Sequence[dict[str, Any]]) -> str:
    blocks = []
    for tool in tools:
        function = tool.get("function", tool) if isinstance(tool, Mapping) else {}
        properties = (function.get("parameters") or {}).get("properties") or {}
        description = _text(function.get("description"))
        lines = [
            f"- {function.get('name', '?')}({', '.join(properties)})" + (f": {description}" if description else "")
        ]
        for name, spec in properties.items():
            detail = _text(spec.get("description")) if isinstance(spec, Mapping) else ""
            if detail:
                lines.append(f"    {name}: {detail}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _call_id(call: Any) -> str | None:
    return call.get("id") if isinstance(call, Mapping) else None


def _render_call(call: Any, argument_chars: int | None, *, note: str | None = None) -> str:
    """``→ name`` with the call id and ``note``, then every argument; ``argument_chars=None`` keeps each whole."""
    function = call.get("function", call) if isinstance(call, Mapping) else {}
    call_id = _call_id(call)
    tags = "; ".join(tag for tag in (f"call {call_id}" if call_id else None, note) if tag)
    head = f"→ {function.get('name', '?')}" + (f" ({tags})" if tags else "")
    arguments = _call_arguments(call)
    if not isinstance(arguments, Mapping):
        return f"{head}: {_argument_text(arguments, argument_chars)}"
    if not arguments:
        return head
    # Verbatim, a multi-line argument on its own lines: a quote from a program the policy submitted must match the text
    # as written, not a JSON-escaped copy.
    lines = [head]
    for key, value in arguments.items():
        text = _argument_text(value, argument_chars)
        lines.append(f"  {key}:\n{text}" if "\n" in text else f"  {key}: {text}")
    return "\n".join(lines)


def _reasoning(message: _SampleMessage) -> str:
    return escape_tags(get_reasoning_text(message) or "")


def _render_turn(
    index: int, message: _SampleMessage, include_reasoning: bool, *, digest: bool, reasoning_chars: int | None = None
) -> str:
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
    lines = [escape_tags(header)]
    reasoning = _reasoning(message) if include_reasoning else ""
    if reasoning:
        if digest:
            reasoning = _head(reasoning, DIGEST_CHARS)
        elif reasoning_chars is not None:
            reasoning = cut_middle(reasoning, reasoning_chars)
        lines.append(f"<{_REASONING_TAG}>\n{reasoning}\n</{_REASONING_TAG}>")
    content = escape_tags(_text(message.get("content")))
    if content:
        if digest:
            content = _head(content, DIGEST_CHARS) if role == "assistant" else cut_middle(content, DIGEST_CHARS)
        lines.append(content)
    argument_chars = DIGEST_INLINE_CHARS if digest else None
    refused = set(message.get(REFUSED_CALL_IDS_KEY) or ())
    for call in message.get("tool_calls") or []:
        lines.append(
            escape_tags(
                _render_call(call, argument_chars, note=REFUSED_CALL_NOTE if _call_id(call) in refused else None)
            )
        )
    for call in message.get(CUT_TOOL_CALLS_KEY) or []:
        lines.append(escape_tags(_render_call(call, argument_chars, note=CUT_CALL_NOTE)))
    return "\n".join(lines)


def _even_shares(lengths: Sequence[int], budget: int) -> list[int]:
    """Each item gets all of itself or an even share of what the shorter ones leave, whichever is less."""
    shares = [0] * len(lengths)
    left = budget
    order = sorted(range(len(lengths)), key=lengths.__getitem__)
    for rank, i in enumerate(order):
        shares[i] = min(lengths[i], left // (len(order) - rank))
        left -= shares[i]
    return shares


def render_transcript(
    messages: Sequence[_SampleMessage], *, include_reasoning: bool, max_chars: int | None = None
) -> str:
    """Every turn whole. Past ``max_chars`` the reasoning gives way first: every turn's is cut to its head and tail
    within an even share of what the rest leaves, so no action is cut to make room for a thought."""
    turns = [_render_turn(i, m, include_reasoning, digest=False) for i, m in enumerate(messages, 1)]
    text = "\n\n".join(turns)
    if not include_reasoning or max_chars is None or len(text) <= max_chars:
        return text
    reasoning = [len(_reasoning(message)) for message in messages]
    # What stays whole: the joins, every turn less its reasoning, and the marker each cut reasoning gains.
    fixed = len(text) - sum(reasoning) + len(CUT_MARKER.format(dropped=sum(reasoning))) * sum(map(bool, reasoning))
    shares = _even_shares(reasoning, max(0, max_chars - fixed))
    return "\n\n".join(
        _render_turn(i, m, include_reasoning, digest=False, reasoning_chars=share)
        for i, (m, share) in enumerate(zip(messages, shares, strict=True), 1)
    )


def render_digest(sample: ScoringSample, *, include_reasoning: bool) -> str:
    """Every turn compactly, call arguments cut verbatim so a quote from a program matches, then the final answer
    whole, the artifact an audit reads."""
    turns = [_render_turn(i, m, include_reasoning, digest=True) for i, m in enumerate(sample.completion, 1)]
    final = escape_tags(sample.final_answer) if sample.final_answer is not None else NO_FINAL_ANSWER
    return "\n\n".join([*turns, f"Final answer:\n{final}"])


def cut_middle(text: str, max_chars: int) -> str:
    """Head and tail kept and the cut marked, so a reader sees how the text began and how it ended."""
    if len(text) <= max_chars:
        return text
    head = int(max_chars * _CUT_HEAD_SHARE)
    tail = max_chars - head
    return text[:head] + CUT_MARKER.format(dropped=len(text) - head - tail) + (text[-tail:] if tail else "")


def escape_tags(text: str) -> str:
    """The ``<`` of every prompt tag it spells becomes ``&lt;``: a judge still reads it, but it neither opens nor
    closes a part or a reasoning block."""
    return _PROMPT_TAG.sub("&lt;", text)


def _as_messages(value: Any, role: str) -> list[_SampleMessage]:
    if isinstance(value, str):
        return [{"role": role, "content": value}]
    return [dict(message) for message in value]


def samples_from_completions(
    prompts: Sequence[Any], completions: Sequence[Any], references: Sequence[Any] | None = None
) -> list[ScoringSample]:
    """Each prompt or completion is a message list (conversational) or a plain string, which becomes one user or
    assistant message."""
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
