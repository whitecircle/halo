"""Assistant-completion span resolution shared by every completion-only masking path.

Leaf module: the loss ignore index, the assistant-turn terminator set, the marker→terminator span
search, the span mask, the completion-only label builder and the construction-time chat-template
probe. It sits beside the collators rather than inside them because the offline label bake and the
PP losses read it too, and none of them should import a collator to reach a span helper.
"""

import warnings
from typing import Any

import torch
from accelerate import PartialState
from accelerate.logging import get_logger
from transformers import PreTrainedTokenizerBase

logger = get_logger(__name__)

# The HF cross_entropy ignore_index. Lives in this leaf so the collators, the preprocessing label
# bake and the PP losses read ONE definition.
LABEL_IGNORE_INDEX = -100

# A template-mismatch warning decodes the offending row; bounded on both axes so a wrong template
# (which misses on every row) cannot flood the log or pay a full 32k-token decode per row.
_WARN_DECODE_TOKENS = 128
_WARN_DECODE_CHARS = 400

# Named span policies for :func:`resolve_completion_spans`, so the runtime collators, the offline
# label bake and ``build_completion_only_labels`` below select one instead of re-typing the kwargs. PACKED differs
# only in ``eos_fallback_to_end``: a packed sequence carries no trailing padding, so a turn whose
# terminator the pack boundary cut away may end at the last position — under padding that position
# is pad. The offline bake picks by whether its artifact is packed, so a preprocessed run and the
# same YAML on a raw dataset train the same tokens for a terminator-less turn.
COLLATOR_SPAN_POLICY: dict[str, bool] = {
    "include_marker": True,
    "bound_by_next_start": True,
    "eos_fallback_to_end": False,
}
PACKED_SPAN_POLICY: dict[str, bool] = {**COLLATOR_SPAN_POLICY, "eos_fallback_to_end": True}
SELF_DISTILL_SPAN_POLICY: dict[str, bool] = {
    "include_marker": False,
    "bound_by_next_start": False,
    "eos_fallback_to_end": True,
}


def require_response_marker(
    response_prompt_template: str | list[int] | None,
    train_on_completions_only: bool,
    subject: str,
) -> None:
    """Refuse completion-only masking with no assistant marker to find the completions by.

    Called where the pair is first accepted — the collator constructors and the VLM dataset prep —
    rather than at the first batch: without the marker no path can build labels, and a refusal that
    waits for collation arrives after the model load and the whole dataset map. ``subject`` names
    the caller in the error.
    """
    if train_on_completions_only and response_prompt_template is None:
        raise ValueError(
            f"{subject}: train_on_completions_only=True requires assistant_message_template (the "
            f"response marker the model's chat template renders) so completion-only labels can be "
            f"built. Pass the marker, or set train_on_completions_only=False to train on the full "
            f"sequence."
        )


def tokenize_response_template(
    response_prompt_template: str | list[int],
    tokenizer: PreTrainedTokenizerBase,
) -> list[int]:
    """Tokenize a response template string, or validate pre-tokenized IDs.

    Raises ValueError if the result is empty (would match every position).
    """
    if isinstance(response_prompt_template, str):
        token_ids = tokenizer.encode(response_prompt_template, add_special_tokens=False)
    else:
        token_ids = list(response_prompt_template)

    if not token_ids:
        raise ValueError(
            f"response_prompt_template produced empty token IDs. "
            f"Template: {response_prompt_template!r}. "
            f"An empty template would match every position, training on all tokens."
        )
    return token_ids


def verify_marker_renders_in_chat_template(
    tokenizer: PreTrainedTokenizerBase, assistant_message_template: str
) -> None:
    """Raise when ``assistant_message_template`` never occurs in this tokenizer's rendered output.

    Probe the RENDERED text, not the template source — templates assemble the marker from pieces,
    and a marker the template never emits masks EVERY label, training zero tokens at loss ~0. Two
    variants because reasoning templates (gpt-oss harmony) emit the assistant marker only for a
    message carrying ``thinking``; the raise fires only when NO variant emits it, and a thinking-only
    marker is still caught by the per-batch all-masked warnings.
    """
    probe_conversations = (
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "thinking": "t"}],
    )
    rendered_texts = []
    probe_failures = []
    for conversation in probe_conversations:
        try:
            rendered = tokenizer.apply_chat_template(conversation, tokenize=False)
        except Exception as exc:  # template requires inputs this synthetic variant lacks — skip it
            probe_failures.append(f"{type(exc).__name__}: {exc}")
            continue
        # isinstance: only judge actual text — a tokenizer returning anything else cannot be probed.
        if isinstance(rendered, str):
            rendered_texts.append(rendered)
    if not rendered_texts and PartialState().is_main_process:
        # Nothing rendered, so the marker check below never ran. Silence here reads exactly like a
        # passing probe, and the defect it exists to catch — a marker the template never emits —
        # then shows up only as a run at loss ~0.
        logger.warning(
            "Could not verify assistant_message_template %r against this tokenizer's chat "
            "template: no probe conversation rendered (%s). Completion-only masking is UNCHECKED "
            "for this run — a marker the template never emits masks EVERY label and trains zero "
            "tokens; watch for the per-batch all-masked warnings.",
            assistant_message_template,
            "; ".join(probe_failures) or "apply_chat_template returned a non-text value",
        )
    if rendered_texts and all(assistant_message_template not in text for text in rendered_texts):
        raise ValueError(
            f"assistant_message_template {assistant_message_template!r} does not occur in this "
            f"tokenizer's rendered chat template, so completion-only masking would mask EVERY "
            f"label and the run would train zero tokens at loss ~0. Set the marker to the exact "
            f"substring the template emits before assistant content (inspect "
            f"tokenizer.apply_chat_template on one conversation), or set "
            f"train_on_completions_only: false."
        )


def ends_with_terminator(input_ids, tokenizer, eos_token_ids) -> bool:
    """Whether ``input_ids`` already ends a turn, ignoring the template's trailing whitespace.

    Chat templates close the final turn with ``<terminator>\\n``, so the LAST token is usually the
    newline rather than the terminator: a bare ``input_ids[-1] in eos_token_ids`` test reads that as
    unterminated and appends a second ender the model never emits (measured on Qwen3 —
    ``Four.<|im_end|>\\n<|im_end|>``). Same rule the SFT render path applies in
    ``_render_terminator_survived``; the walk stops at the first content token, so it costs one decode
    on an already-terminated row.
    """
    for token_id in reversed(input_ids):
        if token_id in eos_token_ids:
            return True
        if tokenizer.decode([token_id]).strip():
            return False
    return False


def resolve_eos_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    model_config: Any = None,
) -> frozenset[int]:
    """Every token id that ends an assistant turn.

    ``tokenizer.eos_token_id`` alone is unreliable: some chat templates (e.g. GLM-4) delimit turns with
    role markers listed only in ``config.eos_token_id``. Unions the model config's ``eos_token_id``
    (also nested under ``text_config`` for VLM/composite) with the tokenizer's eos;
    ``model_config=None`` degrades to tokenizer eos only. A distinct pad token is deliberately NOT a
    terminator: on a right-padded row whose turn lacks a real terminator, the first trailing pad would
    otherwise close the span — training the model to predict pad and bypassing the warn-and-mask path.
    (When pad == eos, the eos collect already covers it.)
    """
    ids: set[int] = set()

    def _collect(value: Any) -> None:
        if isinstance(value, int):
            ids.add(value)
        elif isinstance(value, (list, tuple)):
            ids.update(v for v in value if isinstance(v, int))

    _collect(tokenizer.eos_token_id)
    if model_config is not None:
        _collect(getattr(model_config, "eos_token_id", None))
        text_config = getattr(model_config, "text_config", None)
        if text_config is not None:
            _collect(getattr(text_config, "eos_token_id", None))

    return frozenset(ids)


def warn_if_pad_equals_eos(tokenizer: PreTrainedTokenizerBase) -> None:
    """Warn when pad == eos: completion masking must then rescue the turn-ending EOS from
    ``input_ids`` (the LM collator masks every pad position in labels)."""
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        warnings.warn(
            "The pad_token_id and eos_token_id values of this tokenizer are identical. "
            "The toolkit collators restore turn-ending EOS labels from input_ids at real-token "
            "positions (attention_mask == 1), so EOS is still trained and padding stays masked — "
            "no action needed with these collators; third-party LM collators may still mask "
            "every EOS-valued label.",
            stacklevel=3,
        )


def find_terminator_positions(
    token_ids: list[int] | torch.Tensor,
    eos_token_ids: frozenset[int],
) -> list[int]:
    """Positions of every assistant-turn terminator (any id in ``eos_token_ids``) in the sequence."""
    seq = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids
    return [i for i, token in enumerate(seq) if token in eos_token_ids]


def filter_eos_after_responses(response_starts: list[int], eos_positions: list[int]) -> list[int]:
    """First EOS within each response's OWN turn (after its start, before the next start).

    Bounding by the next start prevents a turn missing its own EOS from borrowing a later turn's,
    which would unmask intervening user/tool + a later assistant turn. Such a turn returns ``-1``
    (empty no-op span).
    """
    if not response_starts or not eos_positions:
        return []

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
    """Indices where ``response_token_ids`` occurs as a contiguous run in ``sequence``.

    Slices only at positions whose first token matches, so cost is proportional to the number
    of candidate hits, not sequence length × template length.
    """
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
    """Single left-to-right marker→terminator scan (the self-distill label policy).

    Each span ends at the FIRST terminator anywhere at/after the marker end — NOT bounded by the
    next marker start — and the marker search resumes past the span end, so a marker inside a
    terminator-less span is consumed by it. ``eos_fallback_to_end`` here is per-span: a
    terminator-less final span ends at the last position (else it is an empty ``-1`` no-op span).
    """
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
                i += m if m else 1
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
    """Locate assistant-completion spans for completion-only masking.

    Returns ``(response_starts, eos_ends)``: each span start paired with the inclusive terminator
    position ending its turn (``-1`` = empty no-op span). Search ``input_ids``, never ``labels`` — a
    labels search misses the turn-ending EOS masked to ignore_index when ``pad_token_id ==
    eos_token_id``. ``train_on_last_assistant_only`` keeps only the final span.

    Three named span policies, defaulting to the padded-collator one (:data:`COLLATOR_SPAN_POLICY`;
    the others are :data:`PACKED_SPAN_POLICY` and :data:`SELF_DISTILL_SPAN_POLICY`):

    - ``include_marker``: span starts AT the response template (collator) vs after it (self-distill).
    - ``bound_by_next_start=True`` (collator): a terminator must precede the next marker start, else
      the span is ``-1`` rather than borrowing a later turn's EOS; ``eos_fallback_to_end`` is then
      global — the last position substitutes only for a sequence with NO terminator at all (safe only
      without trailing padding, i.e. packed/flattened paths).
    - ``bound_by_next_start=False`` (self-distill): one sequential scan, each span running to the first
      terminator after its marker (consuming markers in between); ``eos_fallback_to_end`` is per span.
    """
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
        # Shift AFTER pairing so the next-start bound stays anchored on marker positions.
        offset = 0 if include_marker else len(response_token_ids)
        response_starts = [start + offset for start in marker_starts]
    if train_on_last_assistant_only and response_starts and eos_ends:
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
    """Warn that a sequence trains no tokens because completion-only masking found no usable span.

    A wrong ``assistant_message_template`` misses on every row, zeroing the loss with no other
    signal, so every completion collator (padded, packed, padding-free) reports it here. The decode
    is bounded on both axes — a full 32k-token decode per row would dominate the step.
    """
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
    """Spans of one sequence, or ``None`` after reporting that it has no usable one.

    ``None`` is the shared "trains zero tokens" verdict: every completion-only path (padded, packed,
    padding-free) then fully masks the sequence, differing only in how it refills the usable case.
    The report distinguishes whether the MARKER matched from whether a span closed — a matched turn
    with no terminator anywhere leaves ``eos_ends`` empty, and folding that in would blame the
    template instead.
    """
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


def _real_token_bounds(row: torch.Tensor, attention_mask_row: torch.Tensor | None) -> tuple[int, int]:
    """``[lo, hi)`` slice of ``row`` holding its real (non-padding) tokens; ``(0, 0)`` when all pad.

    Derived from the mask rather than assuming a side, so it holds for left- and right-padded
    batches alike. Without a mask the whole row is "real".
    """
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
    ignore_index: int,
    train_on_last_assistant_only: bool,
    response_prompt_template: str | list[int],
    tokenizer: PreTrainedTokenizerBase | None = None,
    span_policy: dict[str, bool] | None = None,
    attention_mask: torch.Tensor | None = None,
    extra_ignore_token_ids: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Completion-only mask every row of ``batch['labels']``.

    The batch masker behind "keep only assistant completions": the padded collator, the offline label
    bake and the self-distill / VLM label builder all land here, so a row masked offline and the same
    row masked at runtime carry the same targets. The packed and padding-free collators mask one
    sequence at a time — they copy their spans out of ``labels`` rather than ``input_ids``, to keep a
    document boundary's own masking — but share this function's :func:`resolve_spans_or_warn`
    preamble, under :data:`PACKED_SPAN_POLICY`.

    Spans are copied from ``input_ids`` so the turn-ending EOS is trained even when it was masked
    in labels (``pad_token_id == eos_token_id``). A row without a template match is fully masked
    with a warning.

    - ``span_policy`` selects the policy kwargs of :func:`resolve_completion_spans`, defaulting to
      :data:`COLLATOR_SPAN_POLICY` (no end-of-sequence fallback — padded rows would train pad).
    - ``attention_mask``, when given, confines the span search to each row's real tokens, so
      :data:`SELF_DISTILL_SPAN_POLICY`'s end-of-sequence fallback cannot run a terminator-less span
      into the padding. Callers that pass none search the whole row, padding included.
    - ``extra_ignore_token_ids`` (e.g. image tokens) are masked AFTER the span refill: the refill
      reads ``input_ids``, so masking them first would restore every one that sits inside a span.
    """
    policy = COLLATOR_SPAN_POLICY if span_policy is None else span_policy
    for i in range(batch["labels"].shape[0]):
        input_ids = batch["input_ids"][i]
        lo, hi = _real_token_bounds(input_ids, None if attention_mask is None else attention_mask[i])
        new_labels = torch.full_like(batch["labels"][i], ignore_index)
        if lo == hi:
            batch["labels"][i] = new_labels
            continue
        real_ids = input_ids[lo:hi]
        spans = resolve_spans_or_warn(
            real_ids.tolist(),
            response_token_ids,
            eos_token_ids,
            train_on_last_assistant_only=train_on_last_assistant_only,
            span_policy=policy,
            response_prompt_template=response_prompt_template,
            tokenizer=tokenizer,
            row_label=f"example {i}",
        )
        if spans is not None:
            response_starts, eos_ends = spans
            for start, end in zip(response_starts, eos_ends, strict=True):
                new_labels[lo + start : lo + end + 1] = real_ids[start : end + 1]
        batch["labels"][i] = new_labels

    for token_id in extra_ignore_token_ids:
        batch["labels"][batch["labels"] == token_id] = ignore_index

    return batch


def build_completion_only_labels(
    input_ids: torch.Tensor,
    tokenizer,
    response_prompt_template: str | None,
    train_on_completions_only: bool,
    extra_ignore_token_ids: tuple[int, ...] = (),
    attention_mask: torch.Tensor | None = None,
    eos_token_ids: frozenset[int] | None = None,
    span_policy: dict[str, bool] | None = None,
) -> torch.Tensor:
    """Loss labels from ``input_ids``: mask pad (+ any ``extra_ignore_token_ids``, e.g. image tokens),
    and optionally everything outside assistant completions.

    Shared by text and VLM collators so student and teacher branches select the SAME response tokens
    (OPD row alignment). Callers with an ``attention_mask`` MUST pass it: the value-based fallback
    masks by ``== pad_token_id``, which erases real eos tokens wherever ``pad_token_id ==
    eos_token_id`` (the Qwen default), and only fits an unpadded single example. ``span_policy``
    defaults to :data:`SELF_DISTILL_SPAN_POLICY` (start after the marker, end at the first terminator
    after it, unbounded by the next marker start).
    """
    require_response_marker(response_prompt_template, train_on_completions_only, "completion-only labels")
    labels = input_ids.clone()
    if attention_mask is not None:
        labels[attention_mask == 0] = LABEL_IGNORE_INDEX
    elif tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = LABEL_IGNORE_INDEX
    for token_id in extra_ignore_token_ids:
        labels[labels == token_id] = LABEL_IGNORE_INDEX

    if not train_on_completions_only:
        return labels

    # tokenizer.eos_token_id alone is unreliable: some templates (GLM-4) delimit turns with role
    # markers listed in config.eos_token_id instead of a per-turn <|endoftext|>.
    if eos_token_ids is None:
        eos_token_ids = resolve_eos_token_ids(tokenizer)
    batch = mask_batch_to_completion_spans(
        {"input_ids": input_ids, "labels": labels},
        tokenize_response_template(response_prompt_template, tokenizer),
        eos_token_ids,
        ignore_index=LABEL_IGNORE_INDEX,
        train_on_last_assistant_only=False,
        response_prompt_template=response_prompt_template,
        tokenizer=tokenizer,
        span_policy=SELF_DISTILL_SPAN_POLICY if span_policy is None else span_policy,
        attention_mask=attention_mask,
        extra_ignore_token_ids=extra_ignore_token_ids,
    )
    return batch["labels"]
