"""Padded LM collators: completion-only masking, TRL's SFT collator with the one-document varlen set, and the CP
route's label-preserving padder."""

from collections.abc import Callable, Mapping
from typing import Any

import torch
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling as TRLDataCollatorForLanguageModeling

from halo.data.completion_spans import mask_batch_to_completion_spans
from halo.data.turn_markers import resolve_eos_token_ids, tokenize_response_template, warn_if_pad_equals_eos
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.models.segment_markers import single_row_varlen_kwargs


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """Every label outside the assistant completions is ignored. Pass ``response_prompt_template`` as token ids where
    the tokenizer encodes it context-dependently."""

    def __init__(
        self,
        response_prompt_template: str | list[int],
        *args,
        train_on_last_assistant_only: bool = False,
        eos_token_ids: frozenset[int] | None = None,
        emit_varlen_one_document: bool = False,
        **kwargs,
    ):
        super().__init__(*args, mlm=False, **kwargs)

        self.response_prompt_template = response_prompt_template
        self.train_on_last_assistant_only = train_on_last_assistant_only
        self.response_token_ids = tokenize_response_template(response_prompt_template, self.tokenizer)
        self.eos_token_ids = eos_token_ids if eos_token_ids is not None else resolve_eos_token_ids(self.tokenizer)
        self.emit_varlen_one_document = emit_varlen_one_document

        warn_if_pad_equals_eos(self.tokenizer)

    def torch_call(self, examples: list[list[int] | Any | dict[str, Any]]) -> dict[str, Any]:
        batch = mask_batch_to_completion_spans(
            super().torch_call(examples),
            self.response_token_ids,
            self.eos_token_ids,
            self.train_on_last_assistant_only,
            self.response_prompt_template,
            tokenizer=self.tokenizer,
        )
        if self.emit_varlen_one_document:
            batch.update(single_row_varlen_kwargs(batch))
        return batch


class DataCollatorForLanguageModelingWithVarlen(TRLDataCollatorForLanguageModeling):
    """TRL's SFT collator, handing a one-document batch its varlen set."""

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = super().torch_call(examples)
        batch.update(single_row_varlen_kwargs(batch))
        return batch


def collate_preserving_precomputed_labels(
    super_call: Callable[[list[Any]], dict[str, Any]],
    examples: list[Any],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Any]:
    """``tokenizer.pad`` pads only model inputs, so ragged precomputed ``labels`` would fail the parent's tensor
    conversion: pop them, collate, then re-pad with the ignore index. Only rows without precomputed labels get the pad
    == eos restore. ``examples`` must already be a copy, since the pop mutates rows."""
    has_precomputed = bool(examples) and all(isinstance(ex, Mapping) and "labels" in ex for ex in examples)
    label_rows = [ex.pop("labels") for ex in examples] if has_precomputed else None

    batch = super_call(examples)
    if label_rows is None:
        # The parent masks every pad-valued label, so pad == eos (GLM-4) loses the stop signal: restore real positions.
        pad_is_eos = tokenizer.pad_token_id is not None and tokenizer.pad_token_id == tokenizer.eos_token_id
        if pad_is_eos and "labels" in batch and "attention_mask" in batch:
            batch["labels"] = torch.where(batch["attention_mask"].bool(), batch["input_ids"], batch["labels"])
        return batch

    padded = torch.full_like(batch["labels"], LABEL_IGNORE_INDEX)
    for i, row in enumerate(label_rows):
        row_t = torch.as_tensor(row, dtype=padded.dtype)
        if tokenizer.padding_side == "left":
            padded[i, padded.shape[1] - len(row_t) :] = row_t
        else:
            padded[i, : len(row_t)] = row_t
    batch["labels"] = padded
    return batch


class DataCollatorForCausalLMWithPadding(DataCollatorForLanguageModeling):
    """CP route: the parent cannot convert ragged precomputed ``labels``, so they are padded with the ignore index,
    keeping offline prompt-masking intact."""

    def torch_call(self, examples: list) -> dict:
        examples = [dict(ex) for ex in examples]  # pop must not mutate the caller's rows
        return collate_preserving_precomputed_labels(super().torch_call, examples, self.tokenizer)
