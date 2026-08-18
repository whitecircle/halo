"""SFT batch builders over precomputed rows: packing, padding-free Flash Attention, and the padded
(CP-route) collator.

position_ids reset at document boundaries from the `seq_lengths` field TRL's `pack_dataset()`
emits under the "bfd" and "bfd_split" strategies; "wrapped" omits it and is treated as one long sequence
(cross-document attention — avoid with FlashAttention).

TRL forces `padding_free=True` for `packing_strategy="bfd"` (rejects custom collators), so always
set `sft_config.padding_free=False` and `dataset_kwargs={"skip_prepare_dataset": True}` with these.
"""

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase
from transformers.data.data_collator import DefaultDataCollator

from src.data.spans import (
    LABEL_IGNORE_INDEX,
    PACKED_SPAN_POLICY,
    mask_batch_to_completion_spans,
    resolve_eos_token_ids,
    resolve_spans_or_warn,
    tokenize_response_template,
    warn_if_pad_equals_eos,
)

# transformers turns every position-0 reset into a varlen segment boundary, and the FA4 backward
# pays a fixed per-segment cost, so an unchunked pad tail costs one segment per pad token. Laying
# the tail out as a repeating ramp of this length bounds the segment count instead.
PAD_TAIL_SEGMENT_CHUNK = 256


def collate_preserving_precomputed_labels(
    super_call: Callable[[list[Any]], dict[str, Any]],
    examples: list[Any],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Any]:
    """Run the parent LM collator over ``examples`` while preserving precomputed ragged ``labels``.

    ``tokenizer.pad`` pads only model inputs, so ragged precomputed labels (offline-baked completion
    masks) fail the parent's tensor conversion: pop them, collate, then re-pad them to the batch
    shape with the ignore index on the tokenizer's padding side. Rows without precomputed labels go
    through the pad==eos restore instead; baked labels must not be touched by it. ``examples`` must
    already be a copy, since the pop mutates the rows.
    """
    has_precomputed = bool(examples) and all(isinstance(ex, Mapping) and "labels" in ex for ex in examples)
    label_rows = [ex.pop("labels") for ex in examples] if has_precomputed else None

    batch = super_call(examples)
    if label_rows is None:
        return restore_eos_labels_when_pad_equals_eos(batch, tokenizer)

    padded = torch.full_like(batch["labels"], LABEL_IGNORE_INDEX)
    for i, row in enumerate(label_rows):
        row_t = torch.as_tensor(row, dtype=padded.dtype)
        if tokenizer.padding_side == "left":
            padded[i, padded.shape[1] - len(row_t) :] = row_t
        else:
            padded[i, : len(row_t)] = row_t
    batch["labels"] = padded
    return batch


def restore_eos_labels_when_pad_equals_eos(
    batch: dict[str, Any], tokenizer: PreTrainedTokenizerBase
) -> dict[str, Any]:
    """Restore real turn-ending EOS labels masked by the parent LM collator when pad == eos.

    ``DataCollatorForLanguageModeling`` masks every ``pad_token_id`` position in the labels, which
    includes the turn-ending EOS when pad and EOS coincide (e.g. GLM-4's ``<|endoftext|>``, any
    family on the pad=eos fallback), leaving the model with no stop signal to learn. Restore labels
    at real-token positions (``attention_mask == 1``) from ``input_ids``; true padding
    (``attention_mask == 0``) stays masked. No-op when pad != eos. Must not be applied over
    precomputed labels, whose baked ``-100`` spans are authoritative.
    """
    if (
        tokenizer.pad_token_id is not None
        and tokenizer.pad_token_id == tokenizer.eos_token_id
        and "labels" in batch
        and "attention_mask" in batch
    ):
        real = batch["attention_mask"].bool()
        batch["labels"] = torch.where(real, batch["input_ids"], batch["labels"])
    return batch


def pad_tail_positions(n: int, dtype: torch.dtype = torch.long) -> torch.Tensor:
    """Position ids for an ``n``-token pad tail: a ramp restarting every ``PAD_TAIL_SEGMENT_CHUNK``."""
    return torch.arange(n, dtype=dtype) % PAD_TAIL_SEGMENT_CHUNK


def flatten_packed_batch(batch: dict[str, Any], real_lengths: list[int]) -> dict[str, Any]:
    """Concatenate the rows of a packed batch into one ``[1, total]`` row, dropping inter-row padding.

    transformers' FA packed-sequence detection (``_is_packed_sequence``) only engages at batch
    size 1: a ``[B>1, L]`` packed batch with no attention_mask falls through to plain dense-causal
    attention and attends across document boundaries, so the rows must merge.

    ``real_lengths`` (per-row real token counts) keeps the merge cheap. A plain reshape keeps each
    row's pad tail, and every pad carries position_id 0, giving one length-1 attention segment per
    pad token once transformers derives cu_seqlens; a mostly-empty row (the dataset's final partial
    pack) co-batched with a full one contributes tens of thousands of them, and lockstep FSDP
    collectives impose that cost on every rank. Real tokens never attend to pads and pad labels are
    already ``-100``, so dropping them changes no trained value.
    """
    rows, cols = batch["input_ids"].shape
    keep = torch.cat([torch.arange(n, dtype=torch.long) + i * cols for i, n in enumerate(real_lengths)])
    # ndim == 2 exactly: the gather indexes with token strides, so [B, L, X] would be misindexed.
    return {
        k: v.reshape(1, -1)[:, keep] if isinstance(v, torch.Tensor) and v.ndim == 2 and v.shape == (rows, cols) else v
        for k, v in batch.items()
    }


class DataCollatorForCausalLMWithPadding(DataCollatorForLanguageModeling):
    """Padded causal-LM batches (CP route) that preserve precomputed ``labels``.

    The parent builds labels from ``input_ids`` and cannot convert ragged precomputed ``labels``
    (``tokenizer.pad`` pads only model inputs); this pads them with the ignore index instead,
    keeping offline prompt-masking intact. On the raw-labels path the turn-ending EOS is restored
    when pad == eos, since the parent masks every pad-valued label.
    """

    def torch_call(self, examples: list) -> dict:
        examples = [dict(ex) for ex in examples]  # pop must not mutate the caller's rows
        return collate_preserving_precomputed_labels(super().torch_call, examples, self.tokenizer)


class DataCollatorWithPacking(DataCollatorForLanguageModeling):
    """Collator for packed sequences: per-document position_ids that reset at each
    boundary, drops the dense attention_mask so Flash Attention derives a
    block-diagonal cu_seqlens from those position_ids (a dense all-ones mask would
    leak attention across documents), and flattens the packed mini-batch to a single
    ``[1, total_tokens]`` row — transformers' packed-sequence detection only engages
    at batch size 1 (see :func:`flatten_packed_batch`).

    Cross-document isolation comes from the position_ids: FA derives varlen cu_seqlens from them,
    and transformers materializes a ``packed_sequence_mask`` from them for eager/SDPA whenever
    ``attention_mask`` is None (``masking_utils``). Both routes require the model to plumb
    position_ids to its attention interface (flash) or its mask construction (dense), which is
    family-dependent; ``agent-docs/data/collators.md`` has the per-family isolation matrix. The
    non-flash path pays a dense ``[L, L]`` mask over the flattened row, where ``L`` is the whole
    batch's token count, so the cost grows with the batch squared;
    :func:`src.data.collators.factory.select_data_collator` warns there and rejects ``padding_free``.

    Requires right padding: position_ids are laid out from index 0 forward, so leading pads would
    become an attended document and displace a real one out of the loss.
    """

    # Columns that must survive HF's signature-based column pruning; a caller pinning columns itself
    # unions this in, since without ``seq_lengths`` every row collates as a single document.
    required_dataset_columns: tuple[str, ...] = ("seq_lengths",)

    # Flatten the packed ``[B, L]`` batch to ``[1, B*L]`` (:func:`flatten_packed_batch`). PP turns
    # this off, since its dim-0 microbatch split already performs the flatten.
    flatten_to_single_row: bool = True

    _warned_no_seq_lengths: bool = False

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pad_to_multiple_of: int | None = None,
        return_seq_idx: bool = False,
        return_flash_attn_kwargs: bool = False,
    ):
        if getattr(tokenizer, "padding_side", "right") != "right":
            raise ValueError(
                f"DataCollatorWithPacking requires a right-padding tokenizer, got "
                f"padding_side={tokenizer.padding_side!r}. Packed position_ids are laid out from "
                f"index 0 forward, so leading pads become an attended document and push a real "
                f"document out of the loss entirely. The training loaders force right padding; set "
                f"tokenizer.padding_side='right' if you are constructing this collator directly."
            )
        super().__init__(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=pad_to_multiple_of, return_tensors="pt")
        self.return_seq_idx = return_seq_idx
        self.return_flash_attn_kwargs = return_flash_attn_kwargs

    def torch_call(self, examples: list[list[int] | Any | dict[str, Any]]) -> dict[str, Any]:
        """Collate examples, adding resetting position_ids for packed data."""
        # Copy mappings before popping seq_lengths/labels — the pops must not mutate the caller's rows.
        examples = [dict(ex) if isinstance(ex, Mapping) else ex for ex in examples]
        has_packing = any("seq_lengths" in ex for ex in examples if isinstance(ex, dict))

        if not has_packing and not self._warned_no_seq_lengths:
            self._warned_no_seq_lengths = True
            warnings.warn(
                "DataCollatorWithPacking received rows without `seq_lengths`: each row collates as "
                "ONE document, so any concatenated documents inside it attend across each other. "
                "This is what `packing_strategy: wrapped` produces (pack_dataset emits no "
                "boundaries there); use `bfd`/`bfd_split` for document isolation, or ignore this "
                "if cross-document attention is intended.",
                stacklevel=2,
            )

        # seq_lengths can't be tensorized (variable lengths); pop before the parent collates.
        # One entry per example, non-dict rows included: a shorter list would misalign the zip
        # below and re-attach a row's lengths to a different row.
        seq_lengths_list = []
        if has_packing:
            for ex in examples:
                seq_lengths_list.append(ex.pop("seq_lengths", None) if isinstance(ex, dict) else None)

        batch = collate_preserving_precomputed_labels(super().torch_call, examples, self.tokenizer)

        if has_packing:
            for ex, seq_lens in zip(examples, seq_lengths_list, strict=True):
                if isinstance(ex, dict) and seq_lens is not None:
                    ex["seq_lengths"] = seq_lens
            batch = self._handle_packing(batch, examples)
            batch = self._mask_packed_labels(batch, examples)
            if self.flatten_to_single_row:
                batch = flatten_packed_batch(batch, self._real_row_lengths(batch, examples))
                if self.pad_to_multiple_of:
                    batch = self._pad_flattened_tail(batch)
            if self.return_seq_idx:
                # Segment ids for the conv/linear-attention mixers (LFM2 ShortConv, GatedDeltaNet
                # conv); attention isolation alone leaves those mixers crossing document boundaries.
                batch["seq_idx"] = ((batch["position_ids"] == 0).cumsum(dim=1) - 1).to(torch.int32)
            if self.return_flash_attn_kwargs and self.flatten_to_single_row:
                # GatedDeltaNet's chunked delta rule reads ``cu_seq_lens_q`` and nothing model-side
                # derives it. Only defined on the flattened [1, total] row — PP has no convention.
                positions = batch["position_ids"][0]
                starts = (positions == 0).nonzero(as_tuple=True)[0]
                cu_seq_lens = torch.cat([starts, torch.tensor([positions.numel()])]).to(torch.int32)
                batch["cu_seq_lens_q"] = batch["cu_seq_lens_k"] = cu_seq_lens
                batch["max_length_q"] = batch["max_length_k"] = int(cu_seq_lens.diff().max())

        return batch

    @staticmethod
    def _real_row_lengths(batch: dict[str, Any], examples: list[Any]) -> list[int]:
        """Per-row real token counts: sum of the row's ``seq_lengths``, else the full padded width.

        A row without ``seq_lengths`` was collated as one document over its whole width (see
        ``_handle_packing``); its pads are indistinguishable without the lengths, so they are kept.
        Rows are index-aligned with ``examples`` by the parent collation.
        """
        width = batch["input_ids"].shape[1]
        lengths = []
        for i in range(batch["input_ids"].shape[0]):
            example = examples[i] if i < len(examples) else None
            seq_lengths = example.get("seq_lengths") if isinstance(example, dict) else None
            lengths.append(min(sum(seq_lengths), width) if isinstance(seq_lengths, list) else width)
        return lengths

    def _pad_flattened_tail(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Re-pad the flattened row to ``pad_to_multiple_of`` as a chunked-ramp tail.

        ``pad_tail_positions`` keeps the tail a handful of no-op documents — re-padding with
        position-0 tokens would rebuild the per-pad length-1 segments the flatten just removed.
        """
        total = batch["input_ids"].shape[1]
        missing = -total % self.pad_to_multiple_of
        if not missing:
            return batch
        batch["input_ids"] = torch.cat(
            [
                batch["input_ids"],
                torch.full((1, missing), self.tokenizer.pad_token_id, dtype=batch["input_ids"].dtype),
            ],
            dim=1,
        )
        batch["labels"] = torch.cat(
            [batch["labels"], torch.full((1, missing), LABEL_IGNORE_INDEX, dtype=batch["labels"].dtype)], dim=1
        )
        batch["position_ids"] = torch.cat(
            [batch["position_ids"], pad_tail_positions(missing, batch["position_ids"].dtype).unsqueeze(0)], dim=1
        )
        return batch

    def _mask_packed_labels(self, batch: dict[str, Any], examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Per-row label hook, run on the ``[B, L]`` packed batch after ``_handle_packing`` and
        before flattening (per-example row indexing is meaningless on the ``[1, N]`` shape).
        Subclasses apply completion masking here; the base collator masks nothing extra."""
        return batch

    def _handle_packing(self, batch: dict[str, Any], examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Build position_ids that reset to 0 at each document start within the pack."""
        seq_len = batch["input_ids"].shape[1]

        position_ids = torch.zeros_like(batch["input_ids"])

        for i, example in enumerate(examples):
            if isinstance(example, dict) and "seq_lengths" in example:
                seq_lengths = example["seq_lengths"]
                if isinstance(seq_lengths, list):
                    offset = 0
                    for length in seq_lengths:
                        position_ids[i, offset : offset + length] = torch.arange(length)
                        offset += length
                    # The kept-rows (PP) path ships this tail to the kernels, so it must be chunked.
                    position_ids[i, offset:] = pad_tail_positions(seq_len - offset)
                else:
                    position_ids[i] = torch.arange(seq_len)
            else:
                position_ids[i] = torch.arange(seq_len)

        batch["position_ids"] = position_ids

        # Dropping the dense all-ones mask lets FA derive cu_seqlens from position_ids; keeping it leaks.
        batch.pop("attention_mask", None)

        # At a boundary (position_ids==0) the causal shift predicts across blocked attention — noise,
        # not loss. Pad tails are not this line's job; every other route already ignores them.
        batch["labels"][batch["position_ids"] == 0] = LABEL_IGNORE_INDEX

        return batch


class DataCollatorForCompletionOnlyLMWithPacking(DataCollatorWithPacking):
    """Packing collator (from DataCollatorWithPacking) plus completion-only masking:
    loss is computed only on assistant-response tokens. Works with packed and
    non-packed sequences.

    response_prompt_template (str or token IDs) marks assistant response start.
    train_on_last_assistant_only masks all but the last assistant message.
    """

    def __init__(
        self,
        response_prompt_template: str | list[int],
        tokenizer: PreTrainedTokenizerBase,
        ignore_index: int = LABEL_IGNORE_INDEX,
        pad_to_multiple_of: int | None = None,
        train_on_last_assistant_only: bool = False,
        eos_token_ids: frozenset[int] | None = None,
        return_seq_idx: bool = False,
        return_flash_attn_kwargs: bool = False,
    ):
        super().__init__(
            tokenizer=tokenizer,
            pad_to_multiple_of=pad_to_multiple_of,
            return_seq_idx=return_seq_idx,
            return_flash_attn_kwargs=return_flash_attn_kwargs,
        )

        self.response_prompt_template = response_prompt_template
        self.ignore_index = ignore_index
        self.train_on_last_assistant_only = train_on_last_assistant_only
        self.response_token_ids = tokenize_response_template(response_prompt_template, self.tokenizer)
        self.eos_token_ids = eos_token_ids if eos_token_ids is not None else resolve_eos_token_ids(self.tokenizer)

        warn_if_pad_equals_eos(self.tokenizer)

    def torch_call(self, examples: list[list[int] | Any | dict[str, Any]]) -> dict[str, Any]:
        # Packed rows are masked in the parent's flow via _mask_packed_labels (pre-flattening).
        batch = super().torch_call(examples)

        has_packing = any("seq_lengths" in ex for ex in examples if isinstance(ex, dict))
        if not has_packing:
            # Non-packed: unmask only template→EOS spans, exactly as DataCollatorForCompletionOnlyLM does.
            batch = mask_batch_to_completion_spans(
                batch,
                self.response_token_ids,
                self.eos_token_ids,
                self.ignore_index,
                self.train_on_last_assistant_only,
                self.response_prompt_template,
                tokenizer=self.tokenizer,
            )

        return batch

    def _mask_packed_labels(self, batch: dict[str, Any], examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Completion-only masking for packed sequences: mask each original document separately."""
        for i, example in enumerate(examples):
            labels = batch["labels"][i]
            input_ids = batch["input_ids"][i]
            if not isinstance(example, dict) or "seq_lengths" not in example:
                batch["labels"][i] = self._mask_sequence(labels, input_ids=input_ids)
                continue

            new_labels = torch.full_like(labels, self.ignore_index)
            offset = 0
            for seq_len in example["seq_lengths"]:
                seq_labels = labels[offset : offset + seq_len]
                seq_input_ids = input_ids[offset : offset + seq_len]
                new_labels[offset : offset + seq_len] = self._mask_sequence(seq_labels, input_ids=seq_input_ids)
                offset += seq_len

            batch["labels"][i] = new_labels

        return batch

    def _mask_sequence(self, labels: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Completion-only mask a single sequence. When input_ids is given, detect template/EOS in
        it (unaffected by the position_ids==0 boundary masking on labels); the mask applies to labels.

        Uses :data:`~src.data.spans.PACKED_SPAN_POLICY`, the same policy the offline bake takes for
        a packed artifact, so a terminator-less final turn trains the same tokens either way.
        """
        detect = input_ids if input_ids is not None else labels
        spans = resolve_spans_or_warn(
            detect.tolist(),
            self.response_token_ids,
            self.eos_token_ids,
            train_on_last_assistant_only=self.train_on_last_assistant_only,
            span_policy=PACKED_SPAN_POLICY,
            response_prompt_template=self.response_prompt_template,
            tokenizer=self.tokenizer,
            row_label="packed document",
        )
        if spans is None:
            return torch.full_like(labels, self.ignore_index)
        response_starts, eos_ends = spans

        new_labels = torch.full_like(labels, self.ignore_index)
        for start, end in zip(response_starts, eos_ends, strict=False):
            # Copy from `labels`, not input_ids: a template starting at a doc boundary must keep its mask.
            new_labels[start : end + 1] = labels[start : end + 1]
            # Rescue the turn-ending EOS dropped by the copy when pad_token_id == eos_token_id.
            if input_ids is not None and start <= end < len(input_ids):
                new_labels[end] = input_ids[end]

        return new_labels


def _convert_flattened_batch_to_tensors(batch: dict[str, Any]) -> dict[str, Any]:
    """Convert a flattened/packed batch to tensors with the proper per-key dtypes.

    Shared by the flattening collators: labels/position_ids/input_ids/seq_idx gain
    a batch dimension (int64 for the first three, int32 for seq_idx);
    max_length_q/max_length_k stay Python ints.
    """
    int_64_keys = {"labels", "position_ids", "input_ids"}
    batch_dim_keys = {"labels", "position_ids", "input_ids", "seq_idx"}
    py_int_keys = {"max_length_q", "max_length_k"}

    for k, v in batch.items():
        if k in py_int_keys:
            continue
        if k in batch_dim_keys:
            v = [v]
        dtype = torch.int64 if k in int_64_keys else torch.int32
        batch[k] = torch.tensor(v, dtype=dtype)

    return batch


@dataclass
class DataCollatorWithFlattening(DefaultDataCollator):
    """Padding-free Flash Attention collator (no completion masking): flattens the
    mini-batch into a single [1, total_tokens] sequence. Requires the model to use
    `attn_implementation="flash_attention_2"`.
    """

    tokenizer: PreTrainedTokenizerBase = None
    return_flash_attn_kwargs: bool = True
    return_seq_idx: bool = False

    def _sample_labels(self, labels: list[int], input_ids: list[int]) -> list[int]:
        """Per-sample label hook (provided labels, or an input_ids copy); subclasses mask here.

        ``input_ids`` is the detection source: provided labels can carry baked ``-100`` spans that
        hide the response marker / EOS, so span search must never run on labels.
        """
        return labels

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Flatten the batch, routing each sample's labels through ``_sample_labels``."""
        is_labels_provided = "labels" in features[0]

        batch = {"input_ids": [], "labels": [], "position_ids": []}
        if self.return_seq_idx:
            batch["seq_idx"] = []
        if self.return_flash_attn_kwargs:
            cu_seq_lens = [0]
            max_length = 0

        for seq_idx, sample in enumerate(features):
            input_ids = sample["input_ids"]
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()

            batch["input_ids"] += input_ids

            if is_labels_provided:
                labels = sample["labels"]
                if isinstance(labels, torch.Tensor):
                    labels = labels.tolist()
            else:
                labels = input_ids.copy()

            labels = self._sample_labels(labels, input_ids)

            # The first label of every sample is dropped: the flattened row has no preceding token
            # to predict it from, and the loss shift would otherwise read across the document boundary.
            batch["labels"] += [LABEL_IGNORE_INDEX] + labels[1:]

            batch["position_ids"] += list(range(len(input_ids)))

            if self.return_seq_idx:
                batch["seq_idx"] += [seq_idx] * len(input_ids)

            if self.return_flash_attn_kwargs:
                cu_seq_lens.append(cu_seq_lens[-1] + len(input_ids))
                max_length = max(max_length, len(input_ids))

        if self.return_flash_attn_kwargs:
            batch["cu_seq_lens_q"] = cu_seq_lens
            batch["cu_seq_lens_k"] = cu_seq_lens
            batch["max_length_q"] = max_length
            batch["max_length_k"] = max_length

        return _convert_flattened_batch_to_tensors(batch)


@dataclass
class DataCollatorWithFlatteningAndCompletionMask(DataCollatorWithFlattening):
    """Padding-free flattening collator (see :class:`DataCollatorWithFlattening`) plus
    completion-only masking via the per-sample label hook.

    response_prompt_template (str or token IDs) marks assistant response start.
    train_on_last_assistant_only masks all but the last assistant message.
    """

    response_prompt_template: str | list[int] = None
    ignore_index: int = LABEL_IGNORE_INDEX
    train_on_last_assistant_only: bool = False
    eos_token_ids: frozenset[int] | None = None

    def __post_init__(self):
        if self.tokenizer is None:
            raise ValueError("tokenizer must be provided")

        if self.response_prompt_template is not None:
            self.response_token_ids = tokenize_response_template(self.response_prompt_template, self.tokenizer)
        else:
            self.response_token_ids = None

        if self.eos_token_ids is None:
            self.eos_token_ids = resolve_eos_token_ids(self.tokenizer)

        warn_if_pad_equals_eos(self.tokenizer)

    def _sample_labels(self, labels: list[int], input_ids: list[int]) -> list[int]:
        if self.response_token_ids is None:
            return labels
        return self._apply_completion_mask(labels, input_ids)

    def _apply_completion_mask(self, labels: list[int], input_ids: list[int]) -> list[int]:
        """Completion-only mask labels (unmask only template→EOS spans). Operates on the flattened,
        padding-free sequence, so a terminator-less turn falls back to end-of-sequence
        (:data:`~src.data.spans.PACKED_SPAN_POLICY`, the same named policy the packing collator
        and the offline bake for a packed artifact take).

        Spans are detected in ``input_ids``, not in labels: precomputed labels with the marker or
        EOS baked to ``-100`` would match nothing and zero the loss.
        """
        spans = resolve_spans_or_warn(
            input_ids,
            self.response_token_ids,
            self.eos_token_ids or frozenset(),
            train_on_last_assistant_only=self.train_on_last_assistant_only,
            span_policy=PACKED_SPAN_POLICY,
            response_prompt_template=self.response_prompt_template,
            tokenizer=self.tokenizer,
            row_label="flattened sample",
        )
        if spans is None:
            return [self.ignore_index] * len(labels)
        response_starts, eos_ends = spans

        new_labels = [self.ignore_index] * len(labels)
        for start, end in zip(response_starts, eos_ends, strict=False):
            for i in range(start, min(end + 1, len(labels))):
                new_labels[i] = labels[i]

        return new_labels
