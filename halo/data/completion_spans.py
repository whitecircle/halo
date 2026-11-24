"""Completion spans: the marker-to-terminator search within one turn, its restart after a block the template writes
into the model's own turn, the named span policies, the labels built over them, and the refusal of a setup that would
train nothing."""

import warnings
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import torch
from transformers import PreTrainedTokenizerBase

from halo.data.chat_render import render_tool_round_probe
from halo.data.turn_markers import (
    reject_missing_response_marker,
    resolve_eos_token_ids,
    tokenize_response_template,
    verify_marker_renders,
)
from halo.kernels.logprobs import LABEL_IGNORE_INDEX

_WARN_DECODE_TOKENS = 128
_WARN_DECODE_CHARS = 400

PADDED_SPAN_POLICY: dict[str, bool] = {
    "include_marker": True,
    "bound_by_next_start": True,
    "eos_fallback_to_end": False,
}
# Packed rows carry no trailing pad, so only there may a terminator-less turn end at the last position.
PACKED_SPAN_POLICY: dict[str, bool] = {**PADDED_SPAN_POLICY, "eos_fallback_to_end": True}
SELF_DISTILL_SPAN_POLICY: dict[str, bool] = {
    "include_marker": False,
    "bound_by_next_start": False,
    "eos_fallback_to_end": True,
}
# Two answers to one probed tool round: their renders part where the answer starts.
_PROBE_ANSWERS = ("Sunny today.", "Rain later.")


class InjectedBlock(NamedTuple):
    """Blocks the chat template writes into the model's own turn after a stop the model's output ends at, the turn
    resuming after the last: they follow the stop ``opener``, each closing at the special token ``closer``, and another
    opens at the opener or at ``head``, the block's own first special token (Gemma 4's tool replies run
    ``<|tool_response>…<tool_response|><|tool_response>…``, Mistral's ``</s>[TOOL_RESULTS]…[/TOOL_RESULTS]
    [TOOL_RESULTS]…``). Scalars only, so a dataset map's cache key reads it by value."""

    opener: int
    head: int | None
    closer: int


class SpanBounds(NamedTuple):
    """What ends a completion span besides its terminator, read off the tokenizer and chat template once. A span ends
    before a turn of another role, which opens at ``turn_opener`` (the marker's first token where special) with no
    marker following, so a final answer gpt-oss closes with ``<|end|>`` never runs on through the next user turn to a
    call's ``<|call|>``; without an opener only the span policy's marker bound applies, as a plain first token
    (``###``) also opens markdown. A span ending at ``injected_block``'s opener restarts after it. Scalars and an
    :class:`InjectedBlock`, so a dataset map's cache key reads it by value."""

    turn_opener: int | None = None
    injected_block: InjectedBlock | None = None


def find_terminator_positions(
    token_ids: list[int] | torch.Tensor,
    eos_token_ids: frozenset[int],
) -> list[int]:
    seq = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids
    return [i for i, token in enumerate(seq) if token in eos_token_ids]


def filter_eos_after_responses(response_starts: list[int], eos_positions: list[int], uppers: list[int]) -> list[int]:
    """First EOS within each response's own turn, after its start and before its exclusive bound in ``uppers``; ``-1``
    (a no-op span) for a turn missing its own, which must not borrow a later turn's and unmask the turns between."""
    sorted_eos = sorted(eos_positions)
    filtered = []
    for start, upper in zip(response_starts, uppers, strict=True):
        following = next((eos for eos in sorted_eos if eos > start), upper)
        filtered.append(following if following < upper else -1)
    return filtered


def _find_response_starts(sequence: list[int], response_token_ids: list[int]) -> list[int]:
    """Where ``response_token_ids`` occurs contiguously, slicing only at first-token hits."""
    first = response_token_ids[0]
    template_len = len(response_token_ids)
    return [
        idx
        for idx, token in enumerate(sequence)
        if token == first and sequence[idx : idx + template_len] == response_token_ids
    ]


def _first_terminator(sequence: list[int], start: int, stop: int, eos_token_ids: frozenset[int], missing: int) -> int:
    """The first terminator position in ``[start, stop)``, else ``missing``."""
    return next((k for k in range(start, stop) if sequence[k] in eos_token_ids), missing)


def _next_turn(
    sequence: list[int], start: int, stop: int, turn_opener: int | None, response_token_ids: list[int]
) -> int:
    """The first position in ``[start, stop)`` where ``turn_opener`` opens a turn the marker does not, else ``stop``."""
    if turn_opener is None:
        return stop
    m = len(response_token_ids)
    return next(
        (k for k in range(start, stop) if sequence[k] == turn_opener and sequence[k : k + m] != response_token_ids),
        stop,
    )


def _resume_position(
    sequence: list[int], stop: int, block: InjectedBlock, upper: int, eos_token_ids: frozenset[int]
) -> int | None:
    """Past the closer of each block opened in a row from ``stop`` on, as parallel tool calls get one reply each;
    ``None`` where a block holds a terminator before its closer, which a reply never does, or nothing follows."""
    position = stop
    while position < upper and sequence[position] in (block.opener, block.head):
        limit = _first_terminator(sequence, position + 1, upper, eos_token_ids, upper)
        closer = next((k for k in range(position + 1, limit) if sequence[k] == block.closer), None)
        if closer is None:
            return None
        position = closer + 1
    return position if position < upper else None


def _resumed_spans(
    sequence: list[int],
    stop: int,
    upper: int,
    eos_token_ids: frozenset[int],
    injected_block: InjectedBlock | None,
    missing: int,
) -> list[tuple[int, int]]:
    """The spans a turn ending at ``stop`` resumes with after each block ``injected_block`` writes into it, each to its
    first terminator before ``upper`` (``missing`` where none), so the block itself never trains."""
    spans: list[tuple[int, int]] = []
    while injected_block is not None and stop != -1 and sequence[stop] == injected_block.opener:
        resume = _resume_position(sequence, stop, injected_block, upper, eos_token_ids)
        if resume is None:
            break
        stop = _first_terminator(sequence, resume, upper, eos_token_ids, missing)
        spans.append((resume, stop))
    return spans


def _scan_completion_spans_sequential(
    sequence: list[int],
    response_token_ids: list[int],
    eos_token_ids: frozenset[int],
    include_marker: bool,
    eos_fallback_to_end: bool,
    span_bounds: SpanBounds,
) -> tuple[list[int], list[int]]:
    """One scan (the self-distill policy): each span runs to the first terminator after its marker, consuming markers
    in between but ending before another role's turn; a terminator-less span no such turn cuts ends at the last
    position under ``eos_fallback_to_end``, else no-ops."""
    starts: list[int] = []
    ends: list[int] = []
    n, m = len(sequence), len(response_token_ids)
    i = 0
    while i <= n - m:
        if sequence[i : i + m] != response_token_ids:
            i += 1
            continue
        upper = _next_turn(sequence, i + m, n, span_bounds.turn_opener, response_token_ids)
        missing = n - 1 if eos_fallback_to_end and upper == n else -1
        end = _first_terminator(sequence, i + m, upper, eos_token_ids, missing)
        spans = [(i if include_marker else i + m, end)]
        spans += _resumed_spans(sequence, end, upper, eos_token_ids, span_bounds.injected_block, missing)
        for span_start, span_end in spans:
            starts.append(span_start)
            ends.append(span_end)
        last_start, last_end = spans[-1]
        i = last_end + 1 if last_end != -1 else max(last_start, i + m)
    return starts, ends


def resolve_completion_spans(
    sequence: list[int],
    response_token_ids: list[int],
    eos_token_ids: frozenset[int],
    train_on_last_assistant_only: bool = False,
    eos_fallback_to_end: bool = False,
    include_marker: bool = True,
    bound_by_next_start: bool = True,
    span_bounds: SpanBounds | None = None,
) -> tuple[list[int], list[int]]:
    """``(response_starts, eos_ends)``, each end the inclusive terminator or ``-1`` for a no-op span. Search
    ``input_ids``: labels lose the turn-ending EOS where pad == eos.

    ``include_marker`` starts a span at the template or after it. Every span ends before the next turn of another role
    ``span_bounds`` reads. ``bound_by_next_start`` requires the terminator before the next marker too, and its
    ``eos_fallback_to_end`` applies only to a sequence with no terminator at all; without it one sequential scan falls
    back per span. A span ending at the injected block's opener restarts after the block's closer, as a span of its
    own. ``train_on_last_assistant_only`` keeps only the final span."""
    if span_bounds is None:
        span_bounds = SpanBounds()
    if not bound_by_next_start:
        response_starts, eos_ends = _scan_completion_spans_sequential(
            sequence, response_token_ids, eos_token_ids, include_marker, eos_fallback_to_end, span_bounds
        )
    else:
        marker_starts = _find_response_starts(sequence, response_token_ids)
        next_starts = [*marker_starts[1:], len(sequence)] if marker_starts else []
        uppers = [
            _next_turn(sequence, marker + len(response_token_ids), upper, span_bounds.turn_opener, response_token_ids)
            for marker, upper in zip(marker_starts, next_starts, strict=True)
        ]
        eos_positions = find_terminator_positions(sequence, eos_token_ids)
        if eos_fallback_to_end and marker_starts and not eos_positions:
            eos_positions = [len(sequence) - 1]
        marker_ends = filter_eos_after_responses(marker_starts, eos_positions, uppers)
        # Shift after pairing so the bounds stay anchored on marker positions.
        offset = 0 if include_marker else len(response_token_ids)
        response_starts, eos_ends = [], []
        for marker, end, upper in zip(marker_starts, marker_ends, uppers, strict=True):
            for span_start, span_end in [
                (marker + offset, end),
                *_resumed_spans(sequence, end, upper, eos_token_ids, span_bounds.injected_block, -1),
            ]:
                response_starts.append(span_start)
                eos_ends.append(span_end)
    if train_on_last_assistant_only and response_starts:
        response_starts = [response_starts[-1]]
        eos_ends = [eos_ends[-1]]
    return response_starts, eos_ends


def warn_completion_span_miss(
    input_ids: torch.Tensor | list[int],
    response_prompt_template: str | list[int],
    tokenizer: PreTrainedTokenizerBase | None,
    *,
    matched_marker: bool,
    row_label: str,
) -> None:
    """Warn that a sequence trains no tokens: a wrong ``assistant_message_template`` misses on every row, zeroing the
    loss with no other signal. The decode is bounded on both axes, as 32k tokens per row would dominate the step."""
    if matched_marker:
        problem = f"Response key `{response_prompt_template}` matched but no completion span has a terminator in"
        hint = (
            "check for truncated final turns (e.g. increase `max_length`) "
            "or reconsider `train_on_last_assistant_only`."
        )
    else:
        problem = f"Could not find response key `{response_prompt_template}` in"
        hint = "consider increasing the `max_length`."

    if tokenizer is not None:
        elided = max(0, len(input_ids) - _WARN_DECODE_TOKENS)
        instance = f"{tokenizer.decode(input_ids[:_WARN_DECODE_TOKENS])}"[:_WARN_DECODE_CHARS]
        if elided:
            instance = f"{instance}… (+{elided} more tokens)"
    else:
        instance = row_label

    warnings.warn(
        f"{problem} the following instance: {instance} "
        f"This instance will be ignored in loss calculation. "
        f"Note, if this happens often, {hint}",
        stacklevel=3,
    )


def resolve_spans_or_warn(
    sequence: list[int],
    response_token_ids: list[int],
    eos_token_ids: frozenset[int],
    *,
    train_on_last_assistant_only: bool,
    span_policy: dict[str, bool],
    response_prompt_template: str | list[int],
    tokenizer: PreTrainedTokenizerBase | None,
    row_label: str,
    span_bounds: SpanBounds | None = None,
) -> tuple[list[int], list[int]] | None:
    """Spans, or ``None`` (fully masked) after warning, which tells an unmatched marker from an unclosed span."""
    response_starts, eos_ends = resolve_completion_spans(
        sequence,
        response_token_ids,
        eos_token_ids,
        train_on_last_assistant_only=train_on_last_assistant_only,
        span_bounds=span_bounds,
        **span_policy,
    )

    matched_marker = bool(response_starts)
    if not matched_marker or all(end == -1 for end in eos_ends):
        warn_completion_span_miss(
            sequence,
            response_prompt_template,
            tokenizer,
            matched_marker=matched_marker,
            row_label=row_label,
        )
        return None

    return response_starts, eos_ends


def build_span_labels(values, input_ids, spans: tuple[list[int], list[int]] | None, *, restore_end: bool = False):
    """``values`` inside each span, the ignore index elsewhere (everywhere when ``spans`` is ``None``). ``values`` is
    ``input_ids`` when every span token trains, or labels already masked when those must stay masked; ``restore_end``
    then trains each closing token from ``input_ids``, since pad == eos masks the stop signal too."""
    if isinstance(values, torch.Tensor):
        labels = torch.full_like(values, LABEL_IGNORE_INDEX)
    else:
        labels = [LABEL_IGNORE_INDEX] * len(values)
    for start, end in zip(*(spans or ((), ())), strict=True):
        labels[start : end + 1] = values[start : end + 1]
        if restore_end and start <= end < len(input_ids):
            labels[end] = input_ids[end]
    return labels


def _real_token_bounds(row: torch.Tensor, attention_mask_row: torch.Tensor | None) -> tuple[int, int]:
    """``[lo, hi)`` of real tokens by the mask (either padding side); the whole row without one, ``(0, 0)`` all pad."""
    if attention_mask_row is None:
        return 0, row.shape[0]
    real = attention_mask_row.nonzero().flatten()
    if real.numel() == 0:
        return 0, 0
    return int(real[0]), int(real[-1]) + 1


def mask_batch_to_completion_spans(
    batch: dict[str, Any],
    response_token_ids: list[int],
    eos_token_ids: frozenset[int],
    train_on_last_assistant_only: bool,
    response_prompt_template: str | list[int],
    tokenizer: PreTrainedTokenizerBase | None = None,
    span_policy: dict[str, bool] = PADDED_SPAN_POLICY,
    attention_mask: torch.Tensor | None = None,
    extra_ignore_token_ids: tuple[int, ...] = (),
    span_bounds: SpanBounds | None = None,
) -> dict[str, Any]:
    """Completion-only mask every row, training each span's tokens from ``input_ids`` so the turn-ending EOS trains
    even where pad == eos masked it. ``span_policy`` defaults to :data:`PADDED_SPAN_POLICY` (no end-of-sequence
    fallback, which would train pad on padded rows). ``attention_mask`` confines the search to real tokens, so the
    self-distill fallback cannot run into padding. ``extra_ignore_token_ids`` (image tokens) are masked after the
    refill, which would restore them."""
    for i in range(batch["labels"].shape[0]):
        input_ids = batch["input_ids"][i]
        lo, hi = _real_token_bounds(input_ids, None if attention_mask is None else attention_mask[i])
        new_labels = torch.full_like(batch["labels"][i], LABEL_IGNORE_INDEX)
        if lo < hi:
            real_ids = input_ids[lo:hi]
            spans = resolve_spans_or_warn(
                real_ids.tolist(),
                response_token_ids,
                eos_token_ids,
                train_on_last_assistant_only=train_on_last_assistant_only,
                span_policy=span_policy,
                response_prompt_template=response_prompt_template,
                tokenizer=tokenizer,
                row_label=f"example {i}",
                span_bounds=span_bounds,
            )
            new_labels[lo:hi] = build_span_labels(real_ids, real_ids, spans)
        batch["labels"][i] = new_labels

    for token_id in extra_ignore_token_ids:
        batch["labels"][batch["labels"] == token_id] = LABEL_IGNORE_INDEX

    return batch


def build_training_labels(
    input_ids: torch.Tensor,
    tokenizer,
    response_prompt_template: str | None,
    train_on_completions_only: bool,
    extra_ignore_token_ids: tuple[int, ...] = (),
    attention_mask: torch.Tensor | None = None,
    eos_token_ids: frozenset[int] | None = None,
    span_policy: dict[str, bool] = SELF_DISTILL_SPAN_POLICY,
    span_bounds: SpanBounds | None = None,
) -> torch.Tensor:
    """Labels from ``input_ids``: pad and ``extra_ignore_token_ids`` masked, and with ``train_on_completions_only``
    everything outside completions. Shared by text and VLM collators so both branches select the same tokens. Callers
    with an ``attention_mask`` must pass it: the fallback masks ``== pad_token_id``, erasing real EOS at pad == eos."""
    reject_missing_response_marker(response_prompt_template, train_on_completions_only, "completion-only labels")
    labels = input_ids.clone()
    if attention_mask is not None:
        labels[attention_mask == 0] = LABEL_IGNORE_INDEX
    elif tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = LABEL_IGNORE_INDEX
    for token_id in extra_ignore_token_ids:
        labels[labels == token_id] = LABEL_IGNORE_INDEX

    if not train_on_completions_only:
        return labels

    if eos_token_ids is None:
        eos_token_ids = resolve_eos_token_ids(tokenizer)
    batch = mask_batch_to_completion_spans(
        {"input_ids": input_ids, "labels": labels},
        tokenize_response_template(response_prompt_template, tokenizer),
        eos_token_ids,
        train_on_last_assistant_only=False,
        response_prompt_template=response_prompt_template,
        tokenizer=tokenizer,
        span_policy=span_policy,
        attention_mask=attention_mask,
        extra_ignore_token_ids=extra_ignore_token_ids,
        span_bounds=span_bounds,
    )
    return batch["labels"]


def _special_token_ids(tokenizer: PreTrainedTokenizerBase) -> set[int]:
    """The named special tokens and every added token flagged special, which ``all_special_ids`` leaves out (gpt-oss's
    ``<|start|>``)."""
    added = {token_id for token_id, token in tokenizer.added_tokens_decoder.items() if token.special}
    return set(tokenizer.all_special_ids) | added


def find_turn_opener(tokenizer: PreTrainedTokenizerBase, response_token_ids: list[int]) -> int | None:
    """The marker's first token where special: the token the template opens every turn with (``<|im_start|>``,
    gpt-oss's ``<|start|>``), or the marker itself (``[/INST]``); ``None`` for a plain one."""
    first = response_token_ids[0]
    return first if first in _special_token_ids(tokenizer) else None


def reject_untrainable_completions(
    render: Callable[[list[dict[str, Any]]], Any],
    probe_conversations: Sequence[list[dict[str, Any]]],
    tokenizer,
    response_prompt_template: str,
    eos_token_ids: frozenset[int],
    span_policy: dict[str, bool],
    subject: str,
) -> None:
    """Refuse completion-only labels that would train no token of any of ``probe_conversations``: a marker the
    template never emits, or no assistant turn ending in ``eos_token_ids``, as with a template borrowed from another
    family until ``eos_token`` names its closing token."""
    response_token_ids = tokenize_response_template(response_prompt_template, tokenizer)
    span_bounds = SpanBounds(turn_opener=find_turn_opener(tokenizer, response_token_ids))
    for text in verify_marker_renders(render, probe_conversations, response_prompt_template):
        input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        starts, ends = resolve_completion_spans(
            input_ids, response_token_ids, eos_token_ids, span_bounds=span_bounds, **span_policy
        )
        if starts and all(end == -1 for end in ends):
            terminators = sorted(tokenizer.convert_ids_to_tokens(sorted(eos_token_ids)))
            raise ValueError(
                f"{subject}: no assistant turn of the chat template ends in a turn terminator {terminators}, so "
                f"completion-only masking would train no token; the template renders {text!r}. Set eos_token to "
                f"the token the template closes an assistant turn with, or set train_on_completions_only: false."
            )


def find_injected_block(
    tokenizer: PreTrainedTokenizerBase,
    response_prompt_template: str | list[int],
    eos_token_ids: frozenset[int],
    *,
    template_owner: Any = None,
) -> InjectedBlock | None:
    """The block the chat template of ``template_owner`` (the tokenizer by default; a VLM run's processor) writes into
    the model's own turn after the stop a tool call ends at, the answer following with no marker: probed on one tool
    round rendered with two answers, the block running from the call span's stop to the answer, its closer its last
    special token, its head its first token where special. ``None`` where the template renders no tool round, or a
    terminator or the marker's first token opens another turn between."""
    owner = tokenizer if template_owner is None else template_owner
    renders = [
        render_tool_round_probe(owner, {"value": "1"}, with_tools=True, answer=answer) for answer in _PROBE_ANSWERS
    ]
    if not all(isinstance(text, str) for text in renders):
        return None
    response_token_ids = tokenize_response_template(response_prompt_template, tokenizer)
    first, second = (tokenizer(text, add_special_tokens=False)["input_ids"] for text in renders)
    answer_at = next((k for k, (a, b) in enumerate(zip(first, second, strict=False)) if a != b), None)
    if answer_at is None:
        return None
    starts, ends = resolve_completion_spans(first, response_token_ids, eos_token_ids)
    opened = [end for start, end in zip(starts, ends, strict=True) if start < answer_at]
    if not opened or not 0 <= opened[-1] < answer_at - 1:
        return None
    stop = opened[-1]
    block = first[stop + 1 : answer_at]
    if any(token in eos_token_ids or token == response_token_ids[0] for token in block):
        return None
    special = _special_token_ids(tokenizer)
    closer_at = max((k for k, token in enumerate(block) if token in special), default=None)
    if closer_at is None:
        return None
    head = block[0] if block[0] in special and closer_at > 0 else None
    return InjectedBlock(opener=first[stop], head=head, closer=block[closer_at])


def resolve_span_bounds(
    tokenizer: PreTrainedTokenizerBase,
    response_prompt_template: str | list[int],
    eos_token_ids: frozenset[int],
    *,
    template_owner: Any = None,
) -> SpanBounds:
    """The :class:`SpanBounds` of ``response_prompt_template`` under the chat template of ``template_owner`` (the
    tokenizer by default; a VLM run's processor)."""
    response_token_ids = tokenize_response_template(response_prompt_template, tokenizer)
    return SpanBounds(
        turn_opener=find_turn_opener(tokenizer, response_token_ids),
        injected_block=find_injected_block(
            tokenizer, response_token_ids, eos_token_ids, template_owner=template_owner
        ),
    )


def completion_labels(input_ids: torch.Tensor, completion_mask: torch.Tensor) -> torch.Tensor:
    return torch.where(completion_mask.bool(), input_ids, LABEL_IGNORE_INDEX)
