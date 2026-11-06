"""Completion spans: the marker-to-terminator search, the named span policies, the labels built over them, and the
refusal of a setup that would train nothing."""

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

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


def find_terminator_positions(
    token_ids: list[int] | torch.Tensor,
    eos_token_ids: frozenset[int],
) -> list[int]:
    seq = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids
    return [i for i, token in enumerate(seq) if token in eos_token_ids]


def filter_eos_after_responses(response_starts: list[int], eos_positions: list[int]) -> list[int]:
    """First EOS within each response's own turn, after its start and before the next one; ``-1`` (a no-op
    span) for a turn missing its own, which must not borrow a later turn's and unmask the turns between."""
    sorted_eos = sorted(eos_positions)
    filtered = []
    for i, start in enumerate(response_starts):
        upper = response_starts[i + 1] if i + 1 < len(response_starts) else None
        chosen = -1
        for eos in sorted_eos:
            if eos <= start:
                continue
            if upper is not None and eos >= upper:
                break
            chosen = eos
            break
        filtered.append(chosen)
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


def _scan_completion_spans_sequential(
    sequence: list[int],
    response_token_ids: list[int],
    eos_token_ids: frozenset[int],
    include_marker: bool,
    eos_fallback_to_end: bool,
) -> tuple[list[int], list[int]]:
    """One scan (the self-distill policy): each span runs to the first terminator after its marker, consuming markers
    in between; a terminator-less span ends at the last position under ``eos_fallback_to_end``, else no-ops."""
    starts: list[int] = []
    ends: list[int] = []
    n, m = len(sequence), len(response_token_ids)
    i = 0
    while i <= n - m:
        if sequence[i : i + m] != response_token_ids:
            i += 1
            continue
        span_start = i if include_marker else i + m
        end = next((k for k in range(i + m, n) if sequence[k] in eos_token_ids), -1)
        if end == -1:
            if not eos_fallback_to_end:
                starts.append(span_start)
                ends.append(-1)
                i += m
                continue
            end = n - 1
        starts.append(span_start)
        ends.append(end)
        i = end + 1
    return starts, ends


def resolve_completion_spans(
    sequence: list[int],
    response_token_ids: list[int],
    eos_token_ids: frozenset[int],
    train_on_last_assistant_only: bool = False,
    eos_fallback_to_end: bool = False,
    include_marker: bool = True,
    bound_by_next_start: bool = True,
) -> tuple[list[int], list[int]]:
    """``(response_starts, eos_ends)``, each end the inclusive terminator or ``-1`` for a no-op span. Search
    ``input_ids``: labels lose the turn-ending EOS where pad == eos.

    ``include_marker`` starts a span at the template or after it. ``bound_by_next_start`` requires the terminator
    before the next marker, and its ``eos_fallback_to_end`` applies only to a sequence with no terminator at all;
    without it one sequential scan falls back per span. ``train_on_last_assistant_only`` keeps only the final span."""
    if not bound_by_next_start:
        response_starts, eos_ends = _scan_completion_spans_sequential(
            sequence, response_token_ids, eos_token_ids, include_marker, eos_fallback_to_end
        )
    else:
        marker_starts = _find_response_starts(sequence, response_token_ids)
        eos_positions = find_terminator_positions(sequence, eos_token_ids)
        if eos_fallback_to_end and marker_starts and not eos_positions:
            eos_positions = [len(sequence) - 1]
        eos_ends = filter_eos_after_responses(marker_starts, eos_positions)
        # Shift after pairing so the next-start bound stays anchored on marker positions.
        offset = 0 if include_marker else len(response_token_ids)
        response_starts = [start + offset for start in marker_starts]
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
) -> tuple[list[int], list[int]] | None:
    """Spans, or ``None`` (fully masked) after warning, which tells an unmatched marker from an unclosed span."""
    response_starts, eos_ends = resolve_completion_spans(
        sequence,
        response_token_ids,
        eos_token_ids,
        train_on_last_assistant_only=train_on_last_assistant_only,
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
    )
    return batch["labels"]


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
    for text in verify_marker_renders(render, probe_conversations, response_prompt_template):
        input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        starts, ends = resolve_completion_spans(input_ids, response_token_ids, eos_token_ids, **span_policy)
        if starts and all(end == -1 for end in ends):
            terminators = sorted(tokenizer.convert_ids_to_tokens(sorted(eos_token_ids)))
            raise ValueError(
                f"{subject}: no assistant turn of the chat template ends in a turn terminator {terminators}, so "
                f"completion-only masking would train no token; the template renders {text!r}. Set eos_token to "
                f"the token the template closes an assistant turn with, or set train_on_completions_only: false."
            )


def completion_labels(input_ids: torch.Tensor, completion_mask: torch.Tensor) -> torch.Tensor:
    return torch.where(completion_mask.bool(), input_ids, LABEL_IGNORE_INDEX)
