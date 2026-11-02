"""Flattened-row SFT collators: packed rows whose position_ids reset at each document of the ``seq_lengths``
TRL's ``pack_dataset()`` emits ("bfd"/"bfd_split"; "wrapped" rows collate as one document), and padding-free
batches for a varlen flash kernel. TRL forces ``padding_free`` for "bfd" and rejects custom collators, so
these run with ``padding_free=False`` and ``skip_prepare_dataset``."""

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase
from transformers.data.data_collator import DefaultDataCollator

from halo.data.collators.padded import collate_preserving_precomputed_labels
from halo.data.completion_spans import (
    PACKED_SPAN_POLICY,
    build_span_labels,
    mask_batch_to_completion_spans,
    resolve_spans_or_warn,
)
from halo.data.turn_markers import (
    reject_missing_response_marker,
    resolve_eos_token_ids,
    tokenize_response_template,
    warn_if_pad_equals_eos,
)
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.models.segment_markers import (
    SegmentMarkers,
    positions_restart,
    segment_marker_kwargs,
    single_row_varlen_kwargs,
)

# Every position-0 reset is a varlen segment the FA4 backward pays for; a ramped pad tail bounds their count.
PAD_TAIL_SEGMENT_CHUNK = 256


def pad_tail_positions(n: int, dtype: torch.dtype = torch.long) -> torch.Tensor:
    """Position ids for an ``n``-token pad tail: a ramp restarting every ``PAD_TAIL_SEGMENT_CHUNK``."""
    return torch.arange(n, dtype=dtype) % PAD_TAIL_SEGMENT_CHUNK


def flatten_packed_batch(batch: dict[str, Any], real_lengths: list[int]) -> dict[str, Any]:
    """Concatenate the rows of a packed batch into one ``[1, total]`` row, dropping inter-row padding.

    transformers detects a packed sequence only at batch size 1; a ``[B>1, L]`` packed batch without an
    attention_mask attends across documents. The pads go because each would be its own length-1 segment
    (a final partial pack contributes tens of thousands), and dropping them changes no trained value.
    """
    rows, cols = batch["input_ids"].shape
    keep = torch.cat([torch.arange(n, dtype=torch.long) + i * cols for i, n in enumerate(real_lengths)])
    # ndim == 2 exactly: the gather indexes with token strides, so [B, L, X] would be misindexed.
    return {
        k: v.reshape(1, -1)[:, keep] if isinstance(v, torch.Tensor) and v.ndim == 2 and v.shape == (rows, cols) else v
        for k, v in batch.items()
    }


def _packed_position_ids(input_ids: torch.Tensor, examples: list[Any]) -> torch.Tensor:
    """Position ids restarting at 0 at each document of a row's ``seq_lengths``; a row without them counts
    as one document over its whole width."""
    seq_len = input_ids.shape[1]
    position_ids = torch.zeros_like(input_ids)
    for i, example in enumerate(examples):
        seq_lengths = example.get("seq_lengths") if isinstance(example, dict) else None
        if not isinstance(seq_lengths, list):
            position_ids[i] = torch.arange(seq_len)
            continue
        offset = 0
        for length in seq_lengths:
            position_ids[i, offset : offset + length] = torch.arange(length)
            offset += length
        # The kept-rows (PP) path ships this tail to the kernels, so it must be chunked.
        position_ids[i, offset:] = pad_tail_positions(seq_len - offset)
    return position_ids


class DataCollatorWithPacking(DataCollatorForLanguageModeling):
    """Collator for packed rows: position_ids reset at each document, no attention_mask, and the batch
    flattened to one ``[1, total_tokens]`` row (:func:`flatten_packed_batch`).

    Isolation comes from the document boundaries the position_ids reset at: flash runs its varlen kernel over
    the ``cu_seq_lens`` handed over below, or derives them from position_ids where none are, and eager/SDPA
    build a packed-sequence mask from them, a dense ``[L, L]`` over the whole flattened batch. Both need the
    family to plumb position_ids or the varlen set through (the matrix is in ``agent-docs/data/collators.md``). Requires right
    padding, since leading pads would become an attended document.

    ``emit_varlen_always`` emits the varlen set (``cu_seq_lens_q/k``, ``max_length_q/k``) on every
    flattened row, for a mixer that reads it; ``emit_varlen_on_flash_restart`` emits it where transformers would derive
    the same set from position_ids itself, on a flash run's multi-document row; ``emit_varlen_one_document`` on a batch
    that is one document (:func:`~halo.models.segment_markers.single_row_varlen_kwargs`), which transformers
    would test for packing before its dense kernel. Handed over, the set spares every attention layer that
    derivation or test and its device syncs.
    """

    # Kept through HF's signature-column pruning: without ``seq_lengths`` every row is one document.
    required_dataset_columns: tuple[str, ...] = ("seq_lengths",)

    # PP turns this off: its dim-0 microbatch split keeps the rows apart.
    flatten_to_single_row: bool = True

    # The run's opt-in, recorded by the collator factory so a trainer's own check admits what the factory did.
    allow_cross_document_linear_attention: bool = False

    _warned_no_seq_lengths: bool = False

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        return_seq_idx: bool = False,
        emit_varlen_always: bool = False,
        emit_varlen_on_flash_restart: bool = False,
        emit_varlen_one_document: bool = False,
    ):
        if getattr(tokenizer, "padding_side", "right") != "right":
            raise ValueError(
                f"DataCollatorWithPacking requires a right-padding tokenizer, got "
                f"padding_side={tokenizer.padding_side!r}. Packed position_ids are laid out from "
                f"index 0 forward, so leading pads become an attended document and push a real "
                f"document out of the loss entirely. The training loaders force right padding; set "
                f"tokenizer.padding_side='right' if you are constructing this collator directly."
            )
        super().__init__(tokenizer=tokenizer, mlm=False, return_tensors="pt")
        self.return_seq_idx = return_seq_idx
        self.emit_varlen_always = emit_varlen_always
        self.emit_varlen_on_flash_restart = emit_varlen_on_flash_restart
        self.emit_varlen_one_document = emit_varlen_one_document

    def torch_call(self, examples: list[list[int] | Any | dict[str, Any]]) -> dict[str, Any]:
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

        # Popped before the parent tensorizes, one entry per example so the zip below stays row-aligned.
        seq_lengths_list = []
        if has_packing:
            for ex in examples:
                seq_lengths_list.append(ex.pop("seq_lengths", None) if isinstance(ex, dict) else None)

        batch = collate_preserving_precomputed_labels(super().torch_call, examples, self.tokenizer)

        if has_packing:
            for ex, seq_lens in zip(examples, seq_lengths_list, strict=True):
                if isinstance(ex, dict) and seq_lens is not None:
                    ex["seq_lengths"] = seq_lens
            batch["position_ids"] = _packed_position_ids(batch["input_ids"], examples)
            # Dropping the dense all-ones mask lets FA take its varlen path; keeping it leaks.
            batch.pop("attention_mask", None)
            # At a boundary (position_ids==0) the causal shift predicts across blocked attention — noise,
            # not loss. Pad tails are not this line's job; every other route already ignores them.
            batch["labels"][batch["position_ids"] == 0] = LABEL_IGNORE_INDEX
            batch = self._mask_packed_labels(batch, examples)
            if self.flatten_to_single_row:
                batch = flatten_packed_batch(batch, self._real_row_lengths(batch, examples))
            # The varlen set is only defined on the flattened [1, total] row — PP keeps its rows.
            varlen = self.emit_varlen_always or (
                self.emit_varlen_on_flash_restart and positions_restart(batch["position_ids"])
            )
            markers = SegmentMarkers(seq_idx=self.return_seq_idx, cu_seq_lens=varlen and self.flatten_to_single_row)
            batch.update(segment_marker_kwargs(batch["position_ids"], markers))
        if self.emit_varlen_one_document:
            batch.update(single_row_varlen_kwargs(batch))

        return batch

    @staticmethod
    def _real_row_lengths(batch: dict[str, Any], examples: list[Any]) -> list[int]:
        """Per-row real token counts: the sum of the row's ``seq_lengths``, else the full padded width (a row
        without them is one document whose pads cannot be told apart)."""
        width = batch["input_ids"].shape[1]
        lengths = []
        for i in range(batch["input_ids"].shape[0]):
            example = examples[i] if i < len(examples) else None
            seq_lengths = example.get("seq_lengths") if isinstance(example, dict) else None
            lengths.append(min(sum(seq_lengths), width) if isinstance(seq_lengths, list) else width)
        return lengths

    def _mask_packed_labels(self, batch: dict[str, Any], examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Per-row label hook on the ``[B, L]`` batch before flattening; subclasses mask completions here."""
        return batch


class DataCollatorForCompletionOnlyLMWithPacking(DataCollatorWithPacking):
    """:class:`DataCollatorWithPacking` plus completion-only masking, packed or not; ``response_prompt_template``
    (text or token ids) marks the response start."""

    def __init__(
        self,
        response_prompt_template: str | list[int],
        tokenizer: PreTrainedTokenizerBase,
        train_on_last_assistant_only: bool = False,
        eos_token_ids: frozenset[int] | None = None,
        return_seq_idx: bool = False,
        emit_varlen_always: bool = False,
        emit_varlen_on_flash_restart: bool = False,
        emit_varlen_one_document: bool = False,
    ):
        super().__init__(
            tokenizer=tokenizer,
            return_seq_idx=return_seq_idx,
            emit_varlen_always=emit_varlen_always,
            emit_varlen_on_flash_restart=emit_varlen_on_flash_restart,
            emit_varlen_one_document=emit_varlen_one_document,
        )

        self.response_prompt_template = response_prompt_template
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
                self.train_on_last_assistant_only,
                self.response_prompt_template,
                tokenizer=self.tokenizer,
            )

        return batch

    def _mask_packed_labels(self, batch: dict[str, Any], examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Completion-only mask each packed document separately."""
        for i, example in enumerate(examples):
            labels = batch["labels"][i]
            input_ids = batch["input_ids"][i]
            if not isinstance(example, dict) or "seq_lengths" not in example:
                batch["labels"][i] = self._mask_sequence(labels, input_ids=input_ids)
                continue

            new_labels = torch.full_like(labels, LABEL_IGNORE_INDEX)
            offset = 0
            for seq_len in example["seq_lengths"]:
                seq_labels = labels[offset : offset + seq_len]
                seq_input_ids = input_ids[offset : offset + seq_len]
                new_labels[offset : offset + seq_len] = self._mask_sequence(seq_labels, input_ids=seq_input_ids)
                offset += seq_len

            batch["labels"][i] = new_labels

        return batch

    def _mask_sequence(self, labels: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Completion-only mask one document under :data:`~halo.data.completion_spans.PACKED_SPAN_POLICY`, the
        offline bake's policy for a packed artifact. Spans are searched in ``input_ids`` but train ``labels``,
        so a template starting at a document boundary keeps that position's mask."""
        spans = resolve_spans_or_warn(
            input_ids.tolist(),
            self.response_token_ids,
            self.eos_token_ids,
            train_on_last_assistant_only=self.train_on_last_assistant_only,
            span_policy=PACKED_SPAN_POLICY,
            response_prompt_template=self.response_prompt_template,
            tokenizer=self.tokenizer,
            row_label="packed document",
        )
        return build_span_labels(labels, input_ids, spans, restore_end=True)


@dataclass
class DataCollatorWithFlattening(DefaultDataCollator):
    """Padding-free collator: the mini-batch as one ``[1, total_tokens]`` sequence for a varlen flash kernel
    (:data:`~halo.models.attention.selection.VARLEN_ATTN_IMPLEMENTATIONS`)."""

    tokenizer: PreTrainedTokenizerBase = None
    emit_varlen_always: bool = True
    return_seq_idx: bool = False
    # The run's opt-in, recorded by the collator factory so a trainer's own check admits what the factory did.
    allow_cross_document_linear_attention: bool = False

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        is_labels_provided = "labels" in features[0]

        batch = {"input_ids": [], "labels": [], "position_ids": []}

        for sample in features:
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

        batch = {key: torch.tensor([value], dtype=torch.int64) for key, value in batch.items()}
        markers = SegmentMarkers(seq_idx=self.return_seq_idx, cu_seq_lens=self.emit_varlen_always)
        batch.update(segment_marker_kwargs(batch["position_ids"], markers))
        return batch

    def _sample_labels(self, labels: list[int], input_ids: list[int]) -> list[int]:
        """Per-sample label hook over the provided labels or an ``input_ids`` copy; subclasses mask here."""
        return labels


@dataclass
class DataCollatorWithFlatteningAndCompletionMask(DataCollatorWithFlattening):
    """:class:`DataCollatorWithFlattening` plus completion-only masking through the per-sample label hook."""

    response_prompt_template: str | list[int] = None
    train_on_last_assistant_only: bool = False
    eos_token_ids: frozenset[int] | None = None

    def __post_init__(self):
        if self.tokenizer is None:
            raise ValueError("tokenizer must be provided")
        reject_missing_response_marker(self.response_prompt_template, True, type(self).__name__)
        self.response_token_ids = tokenize_response_template(self.response_prompt_template, self.tokenizer)

        if self.eos_token_ids is None:
            self.eos_token_ids = resolve_eos_token_ids(self.tokenizer)

        warn_if_pad_equals_eos(self.tokenizer)

    def _sample_labels(self, labels: list[int], input_ids: list[int]) -> list[int]:
        """Completion-only mask one flattened sample under :data:`~halo.data.completion_spans.PACKED_SPAN_POLICY`
        (a padding-free sequence has no pad for a terminator-less turn to run into). Spans are searched in
        ``input_ids``: precomputed labels with the marker or EOS baked to ``-100`` would match nothing."""
        spans = resolve_spans_or_warn(
            input_ids,
            self.response_token_ids,
            self.eos_token_ids,
            train_on_last_assistant_only=self.train_on_last_assistant_only,
            span_policy=PACKED_SPAN_POLICY,
            response_prompt_template=self.response_prompt_template,
            tokenizer=self.tokenizer,
            row_label="flattened sample",
        )
        return build_span_labels(labels, input_ids, spans)
