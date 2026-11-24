"""Notes a model keeps across an episode's turns, in named sections: the ``notes`` tool, the notes a message history
holds, and the view that shows them as the last turn of every prompt. The view is a function of the messages alone, so
the rollout's request and the trainer's per-turn render read one prompt."""

import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal, get_args

from halo.config_validation import require_int
from halo.environments.episodes.messages import Message
from halo.environments.tools.definitions import (
    NativeTool,
    NativeToolCall,
    ToolArgumentError,
    ToolCallRefused,
    ToolParameter,
)

NOTES_TOOL = "notes"
# Where a call's text before its first section header goes.
DEFAULT_SECTION = "main"
# A section header is a line ``## <name>``: the format the notes are shown in.
_SECTION_HEADER = re.compile(r"^## (\S.*)$", re.M)
# Leads every reply to a call that changed the notes; the replay reads it, so it is the one record of an applied call.
NOTES_SAVED_PREFIX = "Notes saved"

NotesMode = Literal["replace", "append"]
Notes = dict[str, str]


@dataclass(frozen=True)
class NotesConfig:
    """``max_chars``: the most the notes may hold, section names included."""

    max_chars: int = 6000

    def __post_init__(self) -> None:
        require_int("notes", ge=1, max_chars=self.max_chars)


def build_notes_tool(update: Callable[..., str]) -> NativeTool:
    return NativeTool(
        name=NOTES_TOOL,
        description=(
            "Your notes for this task, in named sections, shown in full at the end of every turn and never removed "
            "from view: keep in them what you will need later. One call writes any number of sections: start each "
            "with a line '## <name>' followed by its text. mode 'replace' (the default) sets each section you give "
            "and deletes one you give empty; 'append' adds each section's text to its end. Sections you leave out "
            "stay as they are. It does not run or submit code."
        ),
        parameters=[
            ToolParameter("text", "string", "One or more sections, each a line '## <name>' and its text"),
            ToolParameter(
                "mode", "string", "replace or append (default replace)", enum=list(get_args(NotesMode)), required=False
            ),
        ],
        handler=update,
    )


def notes_size(notes: Notes) -> int:
    return sum(len(name) + len(text) for name, text in notes.items())


def _sections(text: str) -> list[tuple[str, str]]:
    """``text`` split at its ``## <name>`` headers; what precedes the first one is :data:`DEFAULT_SECTION`'s."""
    headers = list(_SECTION_HEADER.finditer(text))
    bounds = [h.start() for h in headers] + [len(text)]
    sections = [(DEFAULT_SECTION, text[: bounds[0]])] if text[: bounds[0]].strip() else []
    sections += [(h.group(1).strip(), text[h.end() : bounds[i + 1]]) for i, h in enumerate(headers)]
    return [(name, body.strip("\n")) for name, body in sections]


def notes_after_call(notes: Notes, text: str, mode: str | None, max_chars: int) -> Notes:
    """The notes one call leaves, ``notes`` unchanged; ``mode`` is one the tool's binding admitted. A call with no
    section (empty text), or a result past ``max_chars``, is refused without stating the limit."""
    mode = mode or "replace"
    sections = _sections(text)
    if not sections:
        raise ToolCallRefused(
            "notes: the text is empty; write each section as a line '## <name>' and its text below it."
        )
    updated = dict(notes)
    for name, body in sections:
        if mode == "append":
            updated[name] = "\n".join(part for part in (updated.get(name, ""), body) if part)
        elif body.strip():
            updated[name] = body
        else:
            updated.pop(name, None)
    updated = {name: body for name, body in updated.items() if body.strip()}
    if notes_size(updated) > max_chars:
        raise ToolCallRefused(
            "Not saved: the notes would be over their size limit. Shorten a section, or delete one by giving it empty."
        )
    return updated


def notes_saved_reply(notes: Notes) -> str:
    return f"{NOTES_SAVED_PREFIX}; sections: {', '.join(notes) or 'none'}."


def _saved_notes_arguments(messages: Sequence[Message], tool: NativeTool) -> Iterator[dict[str, Any]]:
    """The arguments of each call to the notes ``tool`` whose reply says it saved, parsed and bound by that tool as its
    execution read them. A turn's replies follow it one per stored call, in call order, so a call pairs with its reply
    by position, never by an id. A saved call that no longer binds means the view and the execution read calls
    differently, and raises."""
    for i, message in enumerate(messages):
        if message.role != "assistant":
            continue
        for j, stored in enumerate(message.tool_calls or []):
            reply = messages[i + 1 + j] if i + 1 + j < len(messages) else None
            if reply is None or reply.role != "tool" or not reply.content.startswith(NOTES_SAVED_PREFIX):
                continue
            call = NativeToolCall.from_openai_format(stored)
            if call.name != tool.name:
                continue
            try:
                arguments = tool.bind(call.arguments)
            except ToolArgumentError as exc:
                raise RuntimeError(
                    f"the notes call at message {i} replied saved, but its arguments no longer bind"
                ) from exc
            yield arguments


def notes_from_messages(messages: Sequence[Message], tool: NativeTool, max_chars: int) -> Notes:
    """The notes a history holds: each saved call to the notes ``tool`` applied in call order."""
    notes: Notes = {}
    for arguments in _saved_notes_arguments(messages, tool):
        notes = notes_after_call(notes, arguments["text"], arguments.get("mode"), max_chars)
    return notes


def pinned_notes_block(notes: Notes) -> str:
    body = "\n\n".join(f"## {name}\n{text}" for name, text in notes.items()) or "(empty)"
    return f"<notes>\n{body}\n</notes>"


def notes_view(messages: Sequence[Message], config: NotesConfig, tool: NativeTool) -> list[Message]:
    """``messages`` as the model sees them for its next turn: the notes follow as a user turn of their own, never inside
    a tool's output. ``tool`` is the notes tool the environment registered, which binds the recorded calls."""
    view = list(messages)
    if view and view[-1].role in ("user", "tool"):
        notes = notes_from_messages(messages, tool, config.max_chars)
        view.append(Message.user(pinned_notes_block(notes)))
    return view
