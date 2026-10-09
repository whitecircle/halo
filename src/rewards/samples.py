"""What a scorer reads of a finished episode, and how each view of it renders as text.

A :class:`ScoringSample` is one completion to score: the prompt the policy was given, every turn it
and its tools produced, what it finally answered, the row's reference and the tools it could call.
A term picks its VIEW of the sample — the final answer, the full transcript or a digest of it — and
the renderers here turn that view into the text a judge reads: numbered turns, the policy's
reasoning set apart from its visible text, every tool call with its arguments and every result
under the call it answers, and the turns the engine cut or the policy wasted marked as such, with the
calls a cut turn was writing marked as never run. Text a view quotes never spells one of the prompt's tags.
"""

import contextlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.inference.response import get_reasoning_text
from src.rewards.terms import View

Message = dict[str, Any]

# Each part of a judge's prompt goes between tags of its own, the view last, and a view sets a turn's reasoning apart
# in a block of its own: a task's markdown headings and a program's comments would read as the prompt's own sections
# under markdown ones. Text the prompt quotes spells none of these tags (:func:`escape_tags`).
VIEW_TAGS = {View.FINAL: "final_answer", View.FULL: "transcript", View.DIGEST: "transcript_digest"}
PART_TAGS = ("setting", "policy_instructions", "task", "policy_tools", "reference_answer", *VIEW_TAGS.values())
REASONING_TAG = "reasoning"
# The ``<`` that opens or closes one of them, in any case and with any attributes: a judge reads each as the tag.
_PROMPT_TAG = re.compile(rf"<(?=/?(?:{'|'.join((*PART_TAGS, REASONING_TAG))})\b)", re.IGNORECASE)

# The turn flags an environment stamps on a sample message, each with the note a judge reads for it.
TURN_FLAG_NOTES = {
    "reasoning_capped": "its reasoning ran to the turn's cap and the engine closed it, so what follows was written past it",
    "truncated": "cut by the engine at its length limit",
    "empty": "ended with neither visible text nor a tool call",
    "calls_rejected": "every tool call named a tool that does not exist or was refused unrun",
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
# turn's reasoning, text and tool result, a shorter one for a tool call's arguments.
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


def system_text(prompt: Sequence[Message]) -> str:
    """The system turns of the prompt, the instructions the policy was given."""
    return "\n\n".join(_text(message.get("content")) for message in prompt if message.get("role") == "system")


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
    """The text of one view of the sample, what it quotes escaped (:func:`escape_tags`), cut to ``max_chars`` with
    its end kept (:func:`cut_middle`)."""
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
    """Whether :func:`view_text` shows any of the sample's reasoning: a ``full`` or ``digest`` view shows each turn's
    with ``include_reasoning`` on, a ``final`` view never does."""
    return include_reasoning and view is not View.FINAL and any(map(get_reasoning_text, sample.completion))


def render_actions(sample: ScoringSample, view: View, *, max_chars: int) -> str:
    """What the policy wrote as its actions and the tool results it received, as ``view`` shows them: its visible
    text, every argument of every call (a call the engine cut while it was being written included) and each result,
    without the reasoning or the environment's own notes (turn headers and flags, call headers, nudges), so a quote
    of either is never taken for an action. The ``final`` view is the answer, or the last assistant text when there
    is none. Cut to ``max_chars`` like the view."""
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
    """The episode's final answer, or a note that there is none with the last assistant text after it,
    so a judge reads a fragment or a tool-call turn as what it is, never as the answer; either escaped
    (:func:`escape_tags`)."""
    if sample.final_answer is not None:
        return escape_tags(sample.final_answer)
    last = escape_tags(final_assistant_text(sample.completion))
    return f"{NO_FINAL_ANSWER}\nLast assistant turn:\n{last}" if last else NO_FINAL_ANSWER


def render_tools(tools: Sequence[dict[str, Any]]) -> str:
    """Each tool as the policy read it: its name and parameters, its description, then each described
    parameter's description on its own line."""
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


def render_transcript(messages: Sequence[Message], *, include_reasoning: bool, max_chars: int | None = None) -> str:
    """Every turn whole: numbered, the reasoning set apart, each tool call with its arguments (a cut
    turn's after them, marked :data:`CUT_CALL_NOTE`), each tool result under its name and call id.

    Past ``max_chars`` the reasoning gives way first: every turn's is cut to its head and tail (:func:`cut_middle`)
    within an even share of what the rest of the transcript leaves, so no action is cut to make room for a thought."""
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
    """Every turn compactly — the head of each reasoning and text, each tool call with its arguments cut
    to their heads (verbatim, so a quote from a program matches), the head and tail of each result —
    then the final answer whole, the artifact an audit reads."""
    turns = [_render_turn(i, m, include_reasoning, digest=True) for i, m in enumerate(sample.completion, 1)]
    final = escape_tags(sample.final_answer) if sample.final_answer is not None else NO_FINAL_ANSWER
    return "\n\n".join([*turns, f"Final answer:\n{final}"])


def cut_middle(text: str, max_chars: int) -> str:
    """``text`` cut to about ``max_chars`` with its head and its tail kept and the cut marked
    (:data:`CUT_MARKER`), so a reader sees how the text began and how it ended."""
    if len(text) <= max_chars:
        return text
    head = int(max_chars * CUT_HEAD_SHARE)
    tail = max_chars - head
    return text[:head] + CUT_MARKER.format(dropped=len(text) - head - tail) + (text[-tail:] if tail else "")


def escape_tags(text: str) -> str:
    """``text`` with the ``<`` of every prompt tag it spells (:data:`PART_TAGS`, :data:`REASONING_TAG`, in any case)
    written ``&lt;``: a judge still reads it, but it neither opens nor closes a part or a reasoning block."""
    return _PROMPT_TAG.sub("&lt;", text)


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


def _even_shares(lengths: Sequence[int], budget: int) -> list[int]:
    """``budget`` split over items of ``lengths``: each gets all of itself or an even share of what the shorter
    ones leave, whichever is less."""
    shares = [0] * len(lengths)
    left = budget
    order = sorted(range(len(lengths)), key=lengths.__getitem__)
    for rank, i in enumerate(order):
        shares[i] = min(lengths[i], left // (len(order) - rank))
        left -= shares[i]
    return shares


def _render_turn(
    index: int, message: Message, include_reasoning: bool, *, digest: bool, reasoning_chars: int | None = None
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
        lines.append(f"<{REASONING_TAG}>\n{reasoning}\n</{REASONING_TAG}>")
    content = escape_tags(_text(message.get("content")))
    if content:
        if digest:
            content = _head(content, DIGEST_CHARS) if role == "assistant" else cut_middle(content, DIGEST_CHARS)
        lines.append(content)
    argument_chars = DIGEST_INLINE_CHARS if digest else None
    for call in message.get("tool_calls") or []:
        lines.append(escape_tags(_render_call(call, argument_chars)))
    for call in message.get(CUT_CALLS_KEY) or []:
        lines.append(escape_tags(_render_call(call, argument_chars, note=CUT_CALL_NOTE)))
    return "\n".join(lines)


def _reasoning(message: Message) -> str:
    """A turn's reasoning as a view quotes it (:func:`escape_tags`), ``""`` when it has none."""
    return escape_tags(get_reasoning_text(message) or "")


def _acted_text(message: Message, *, digest: bool) -> str:
    """One message's share of :func:`render_actions`: an assistant turn's visible text and its calls' arguments, a
    tool result; a user turn (an environment's nudge) has none."""
    role = message.get("role")
    content = _text(message.get("content"))
    if role == "tool":
        return cut_middle(content, DIGEST_CHARS) if digest else content
    if role != "assistant":
        return ""
    pieces = [_head(content, DIGEST_CHARS) if digest else content]
    for call in [*(message.get("tool_calls") or []), *(message.get(CUT_CALLS_KEY) or [])]:
        function = call.get("function", call) if isinstance(call, Mapping) else {}
        arguments = function.get("arguments", "")
        if isinstance(arguments, str):
            with contextlib.suppress(ValueError):
                arguments = json.loads(arguments) if arguments else {}
        values = arguments.values() if isinstance(arguments, Mapping) else [arguments]
        pieces.extend(_argument_text(value, DIGEST_INLINE_CHARS if digest else None) for value in values)
    return "\n".join(piece for piece in pieces if piece)


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
