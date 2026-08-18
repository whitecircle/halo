"""Locate the policy-generated token spans of a multi-turn trajectory inside ONE authoritative render.

Chat templates are not prefix-monotone: the render of the first N messages is generally NOT a prefix
of the render of the first N+1. The harmony template rewrites an assistant turn's terminator
(``<|return|>`` becomes ``<|end|>``) and drops its reasoning as soon as a final message exists;
Qwen3 keeps a turn's reasoning only until a later user message arrives, and injects an empty
``<think>`` block into a reasoning-free current turn. Concatenating independently rendered per-turn
deltas therefore trains token sequences that no serving-time render produces — including the
terminal stop token in the middle of an episode.

Every span here is an index range into a single full-conversation render, so what the policy is
trained on is exactly what the serving template emits for that trajectory. Block boundaries come
from two mechanisms, in order:

* an ANCHOR — a prefix render verified to be an exact prefix of the authoritative render, which
  pins the boundary by construction;
* the message's own block length, measured by re-rendering with that message DUPLICATED and diffing
  against the nearest anchor above it: every other block renders identically, so the length delta is
  that block. This covers the boundary right after a non-final assistant turn, which no prefix render
  can pin (a prefix render ending there necessarily makes that turn the last message, the case
  templates rewrite).

Within an assistant block, the trained span starts where the engine's generation prompt stops
agreeing with the authoritative render: whatever the prompt already contained was handed to the
policy, not emitted by it. Matching rather than requiring a prefix is what keeps templates that
rewrite their own thinking scaffolding once a turn is archived (GLM-4 ``<think>`` -> ``</think>``,
Qwen3.5's empty think block) decomposable.

Every boundary is therefore pinned by a render this module verified, never inferred from another
turn: a role header's token length is not turn-invariant (BPE merges its last token with the turn's
first content token differently per turn), so any carry-over would mislocate a span with nothing to
catch it. A turn whose own context does not anchor raises instead.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

# (messages, add_generation_prompt, include_thinking) -> token ids
RenderFn = Callable[[Sequence[Any], bool, bool], list[int]]

# Probe the reasoning-carrying render first; ``False`` is what a template hiding prior-turn CoT emits.
_ANCHOR_THINKING_ORDER = (True, False)


class TemplateSpanError(ValueError):
    """An assistant turn's span could not be located in the authoritative render."""


@dataclass(frozen=True)
class TrajectorySpans:
    """The authoritative render plus the half-open ranges of it that the policy generated.

    ``prompt_len`` is the length of the leading context (through the first assistant turn's
    template-injected header); ``turn_spans`` holds one ``[start, end)`` range per assistant turn,
    indexed into ``full_ids``.
    """

    full_ids: list[int]
    prompt_len: int
    turn_spans: list[tuple[int, int]]


def locate_assistant_spans(render: RenderFn, messages: Sequence[Any], first_assistant_idx: int) -> TrajectorySpans:
    """Render the whole trajectory once and locate every assistant turn's span inside it.

    Raises :class:`TemplateSpanError` when a boundary cannot be pinned — a template this toolkit
    cannot decompose. Callers must surface that instead of training a guessed span.
    """
    return _SpanLocator(render, messages, first_assistant_idx).resolve()


class _SpanLocator:
    """Resolves message-block boundaries of one trajectory in its authoritative render."""

    def __init__(self, render: RenderFn, messages: Sequence[Any], first_assistant_idx: int):
        self._render = render
        self._messages = list(messages)
        self._first = first_assistant_idx
        try:
            self._full = list(render(self._messages, False, True))
        except Exception as e:  # any template failure must become a TemplateSpanError
            # A raw template rejection escapes on one rank alone and hangs the collectives; converting
            # lets the caller raise TemplateSpanError on every rank.
            raise TemplateSpanError(f"the chat template cannot render this trajectory: {type(e).__name__}: {e}") from e
        # index -> (block start, the include_thinking flag whose render pinned it), or None.
        self._anchors: dict[int, tuple[int, bool] | None] = {}
        self._anchor_flag_order = _ANCHOR_THINKING_ORDER
        # The end of the last message's block is the end of the render — the one free boundary.
        self._block_starts: dict[int, int] = {len(self._messages): len(self._full)}

    def resolve(self) -> TrajectorySpans:
        prompt_len = self._anchor_prompt()
        spans: list[tuple[int, int]] = []
        prev_end = prompt_len
        for idx, message in enumerate(self._messages):
            if message.role != "assistant":
                continue
            start = self._generation_start(idx)
            end = self._block_start(idx + 1)
            if not (prev_end <= start < end <= len(self._full)):
                raise TemplateSpanError(
                    f"turn {idx}: located span [{start}, {end}) is not a forward, non-empty range inside "
                    f"the {len(self._full)}-token full render (previous turn ended at {prev_end})"
                )
            spans.append((start, end))
            prev_end = end
        return TrajectorySpans(self._full, prompt_len, spans)

    def _anchor_prompt(self) -> int:
        """Anchor the context that precedes the first assistant turn and return where it ends.

        This renders only the messages BEFORE that turn, so no template's last-assistant rewriting is
        in play; a failure here means the prompt itself does not open the authoritative render, and
        nothing downstream can be trusted. Seeding the boundary also terminates
        :meth:`_block_start`'s walk, which never has to descend past it.
        """
        anchored = self._anchor(self._first)
        if anchored is None:
            raise TemplateSpanError(
                f"turn {self._first}: the prompt render does not open the full-conversation render"
            )
        self._block_starts[self._first] = anchored[0]
        return self._generation_start(self._first)

    def _generation_start(self, index: int) -> int:
        """Where the policy's own tokens for the assistant turn at ``index`` begin: the point at which
        the engine's generation prompt stops agreeing with the authoritative render.

        Everything the prompt already contained was handed to the policy rather than emitted by it, so
        this needs the turn's own context anchored. Without an anchor the boundary is unknowable — a
        header's token length is not turn-invariant (BPE merges its last token with the turn's first
        content token differently per turn), so carrying one over would mislocate the span silently.
        """
        anchored = self._anchor(index)
        prompt = None if anchored is None else self._try_render(self._messages[:index], True, anchored[1])
        if anchored is None or prompt is None:
            raise TemplateSpanError(
                f"turn {index}: its context does not render as a prefix of the full render, so the "
                f"boundary between the template's generation prompt and the policy's own tokens is "
                f"unknowable (a trajectory with consecutive assistant messages reaches this)"
            )
        block_start = anchored[0]
        generated_start = self._common_prefix_len(prompt)
        if generated_start < block_start:
            # Anything before block_start belongs to another message: trusting it trains env-injected tokens.
            raise TemplateSpanError(
                f"turn {index}: the generation prompt diverges from the full render before this turn's "
                f"block ({generated_start} < {block_start}), so its own tokens cannot be delimited"
            )
        return generated_start

    def _block_start(self, index: int) -> int:
        """Where message ``index``'s rendered block begins (its role header included)."""
        if index not in self._block_starts:
            anchored = self._anchor(index)
            if anchored is not None:
                self._block_starts[index] = anchored[0]
            else:
                self._block_starts[index] = self._block_start(index - 1) + self._block_len(index - 1)
        return self._block_starts[index]

    def _block_len(self, index: int) -> int:
        """Token length of message ``index``'s block, measured by duplicating that message.

        Measured against the nearest anchor ABOVE ``index`` rather than the whole render: both sides
        then end on the same message, so the same last-message rewriting applies to both and the delta
        is the extra copy rendered non-finally — exactly how the block appears in the full render. The
        full render is the last resort (it anchors trivially), so a long trajectory pays a prefix
        render here instead of a whole-conversation one.
        """
        stop, base, include_thinking = self._next_anchor_above(index)
        duplicated = [*self._messages[: index + 1], *self._messages[index:stop]]
        rendered = self._try_render(duplicated, False, include_thinking)
        if rendered is None or len(rendered) <= base:
            raise TemplateSpanError(
                f"message {index}: the render does not grow when this message is duplicated, so its "
                f"block length cannot be measured"
            )
        return len(rendered) - base

    def _next_anchor_above(self, index: int) -> tuple[int, int, bool]:
        """``(message count, its anchored length, its include_thinking flag)`` for the shortest
        anchored prefix that still contains message ``index``. The whole message list always
        qualifies: the authoritative render is a prefix of itself."""
        for stop in range(index + 1, len(self._messages)):
            anchored = self._anchor(stop)
            if anchored is not None:
                return stop, anchored[0], anchored[1]
        return len(self._messages), len(self._full), True

    def _anchor(self, index: int) -> tuple[int, bool] | None:
        """Length of the render of ``messages[:index]`` when it is an exact prefix of the full render,
        with the ``include_thinking`` flag that produced it."""
        if index not in self._anchors:
            self._anchors[index] = self._resolve_anchor(index)
        return self._anchors[index]

    def _resolve_anchor(self, index: int) -> tuple[int, bool] | None:
        # Whichever flag last anchored is tried first (a template is consistent, so the winner repeats);
        # both orders agree on the answer, so this only skips the other flag's render.
        for include_thinking in self._anchor_flag_order:
            rendered = self._try_render(self._messages[:index], False, include_thinking)
            if rendered is not None and self._full[: len(rendered)] == list(rendered):
                self._anchor_flag_order = (include_thinking, not include_thinking)
                return len(rendered), include_thinking
        return None

    def _common_prefix_len(self, rendered: Sequence[int]) -> int:
        limit = min(len(rendered), len(self._full))
        matched = 0
        while matched < limit and rendered[matched] == self._full[matched]:
            matched += 1
        return matched

    def _try_render(
        self, messages: Sequence[Any], add_generation_prompt: bool, include_thinking: bool
    ) -> list[int] | None:
        """Render a PROBE conversation, or ``None`` when it does not render.

        Only probes go through here. A template legitimately rejects a truncated or duplicated message
        list (strict role alternation, a tool result with no preceding call), which makes that probe
        unavailable rather than the trajectory invalid; every caller either tries another probe or
        raises :class:`TemplateSpanError`, so a swallowed failure can never become a guessed span.
        """
        try:
            return self._render(messages, add_generation_prompt, include_thinking)
        except Exception:  # any template rejection means "this probe is unavailable"
            return None
