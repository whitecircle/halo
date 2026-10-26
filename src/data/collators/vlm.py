"""VLM data collators: the runtime render-and-collate path, the preprocessed-artifact padder and the SDPG
self-distillation branch, all composing :mod:`src.data.vlm_render` so a runtime batch and an offline bake
of the same row tokenize identically."""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from src.data.collators.self_distill import SelfDistillBranchMixin, require_aligned_responses
from src.data.completion_spans import COLLATOR_SPAN_POLICY, build_completion_only_labels
from src.data.turn_markers import reject_missing_response_marker, resolve_eos_token_ids
from src.data.vlm_render import (
    SEQUENCE_ALIGNED_VISION_KEYS,
    VLM_OUTPUT_FEATURES,
    reject_over_length_vlm_batch,
    render_vlm_text,
    resolve_image_token_ids,
    run_vlm_processor,
    vlm_row_tools,
)
from src.kernels.logprobs import LABEL_IGNORE_INDEX
from src.log import warn_once

logger = logging.getLogger(__name__)


class VLMDataCollator:
    """Data collator for Vision Language Models."""

    # Same span policy as the text runtime collators; the SDPG subclass overrides it.
    _COMPLETION_SPAN_POLICY: dict[str, bool] | None = COLLATOR_SPAN_POLICY

    def __init__(
        self,
        processor,
        tokenizer,
        max_length=2048,
        response_prompt_template=None,
        train_on_completions_only=False,
        model_config=None,
    ):
        reject_missing_response_marker(response_prompt_template, train_on_completions_only, type(self).__name__)
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.response_prompt_template = response_prompt_template
        self.train_on_completions_only = train_on_completions_only
        self.eos_token_ids = resolve_eos_token_ids(tokenizer, model_config)
        # Construction-time, like the eos set: the union reads the tokenizer's full vocab dict, which
        # is a per-batch cost of the vocabulary's size for a verdict that cannot change between batches.
        self.image_token_ids = tuple(resolve_image_token_ids(tokenizer, processor))

    def __call__(self, examples):
        batch = self._process_batch(examples, lambda example: example.get("history", []), "batch")
        batch["labels"] = self._build_labels(batch["input_ids"], batch.get("attention_mask"))
        return batch

    def _process_batch(self, examples, history_of: Callable[[dict[str, Any]], list], subject: str):
        """Render each row's ``history_of`` conversation and run the processor over the batch, refusing an
        over-length sequence: truncating would desync expanded image placeholders from the vision tensors,
        and a runtime batch cannot drop rows without desyncing DP ranks."""
        texts = [self._render(history_of(example), example) for example in examples]
        images = [image for example in examples for image in example.get("images") or ()]
        batch = run_vlm_processor(self.processor, texts, images, padding=True)
        reject_over_length_vlm_batch(batch, self.max_length, subject)
        return batch

    def _render(self, history: list[dict[str, Any]], example: dict[str, Any]) -> str:
        """Chat-template one conversation with the row's tools."""
        return render_vlm_text(self.processor, history, tools=vlm_row_tools(example))

    def _build_labels(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        """Pad and image tokens masked, and with ``train_on_completions_only`` everything outside the completions."""
        return build_completion_only_labels(
            input_ids,
            self.tokenizer,
            self.response_prompt_template,
            self.train_on_completions_only,
            extra_ignore_token_ids=self.image_token_ids,
            attention_mask=attention_mask,
            eos_token_ids=self.eos_token_ids,
            span_policy=self._COMPLETION_SPAN_POLICY,
        )


def _stacked_vision_tensors(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """``pixel_values`` and ``image_grid_thw`` concatenated over every image-carrying row of a preprocessed
    batch, in order (a mixed batch has text-only rows)."""
    tensors = {}
    pixel_values = [
        torch.from_numpy(
            np.frombuffer(ex["pixel_values"], dtype=np.float16).reshape(ex["pixel_values_shape"]).astype(np.float32)
        )
        for ex in examples
        if ex.get("pixel_values") is not None
    ]
    if pixel_values:
        # For Qwen-VL pixel_values are already flattened per image.
        tensors["pixel_values"] = torch.cat(pixel_values, dim=0)
    grids = []
    for ex in examples:
        grid = ex.get("image_grid_thw")
        if grid is None:
            continue
        if isinstance(grid, list):
            grids.extend(grid if isinstance(grid[0], list) else [grid])
        else:
            grids.append(grid.tolist() if hasattr(grid, "tolist") else grid)
    if grids:
        tensors["image_grid_thw"] = torch.tensor(grids, dtype=torch.long)
    return tensors


class PreprocessedVLMDataCollator:
    """Pad and stack pre-tokenized VLM rows (:data:`~src.data.vlm_render.VLM_OUTPUT_FEATURES`)."""

    # Unioned into HF's signature columns, which would otherwise prune pixel_values_shape and the rest.
    required_dataset_columns: tuple[str, ...] = tuple(VLM_OUTPUT_FEATURES)

    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id or 0
        # Per collator instance: one line about this run's preprocessed lengths, not one per batch.
        self._warned: set[str] = set()

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Pad only: re-truncating without trimming pixel_values/image_grid_thw desyncs placeholders.
        max_len = max(len(ex["input_ids"]) for ex in examples)
        if max_len > self.max_length:
            warn_once(
                logger,
                self._warned,
                "truncation",
                f"PreprocessedVLMDataCollator: a preprocessed sequence ({max_len} tokens) exceeds "
                f"max_length ({self.max_length}). VLM data is NOT truncated here (it would desync image "
                f"tokens from pixel_values); it is used at its preprocessed length. Re-run preparation "
                f"with a smaller max_length to reduce it.",
            )
        pads = [max_len - len(ex["input_ids"]) for ex in examples]
        batch = {
            key: torch.tensor(
                [ex[key] + [fill] * pad for ex, pad in zip(examples, pads, strict=True)], dtype=torch.long
            )
            for key, fill in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", LABEL_IGNORE_INDEX))
        }
        batch.update(_stacked_vision_tensors(examples))
        return batch


class SelfDistillVLMDataCollator(SelfDistillBranchMixin, VLMDataCollator):
    """VLM collator for SDPG-style self-distillation (arXiv:2606.04036): the student batch plus a hinted
    ``teacher_*`` branch whose labels select the same response tokens (:func:`require_aligned_responses`).

    The text-only hint leaves the image grid unchanged, so the trainer reuses the student's image tensors.
    Takes the self-distill span policy, like :class:`SelfDistillTextCollator`; ``hint_template=None``
    builds no teacher branch. Optionally emits ``confidence_weights``.
    """

    _COMPLETION_SPAN_POLICY = None

    def __init__(
        self,
        *args,
        hint_template: str | None,
        answer_field: str | None = "answer",
        solution_field: str | None = "solution",
        confidence_field: str | None = None,
        confidence_power: float = 4.0,
        confidence_normalizer: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._init_self_distill_fields(
            hint_template, answer_field, solution_field, confidence_field, confidence_power, confidence_normalizer
        )

    def audit_row(self, example: dict[str, Any]) -> dict[str, Any]:
        """Refuse a row whose two branches would supervise different tokens; returns no columns. Reads the
        rendered text as the processor tokenizes it, before the image placeholders (all in the user turn)
        expand; mapped at prep by ``audit_self_distill_row``, where the raise is world-uniform."""
        if not self.builds_teacher_branch:
            return {}
        history = example.get("history", [])
        labels = []
        for branch in (history, self._teacher_history(history, example)):
            text = self._render(branch, example)
            input_ids = torch.tensor([self.tokenizer(text, add_special_tokens=False)["input_ids"]])
            labels.append(self._build_labels(input_ids, torch.ones_like(input_ids)))
        require_aligned_responses({"labels": labels[0], "teacher_labels": labels[1]})
        return {}

    def __call__(self, examples):
        batch = super().__call__(examples)

        if self.builds_teacher_branch:
            # Never truncated either: the hinted teacher is the longer branch, so truncation would cut response
            # tokens the student keeps and break OPD row alignment.
            teacher_batch = self._process_batch(
                examples,
                lambda example: self._teacher_history(example.get("history", []), example),
                "teacher branch (student history + privileged hint)",
            )
            batch["teacher_input_ids"] = teacher_batch["input_ids"]
            batch["teacher_attention_mask"] = teacher_batch["attention_mask"]
            batch["teacher_labels"] = self._build_labels(
                teacher_batch["input_ids"], teacher_batch.get("attention_mask")
            )
            # Per-token tensors (GLM-4.6V mm_token_type_ids) must be the teacher's own — its sequence is longer.
            for key in SEQUENCE_ALIGNED_VISION_KEYS:
                if key in teacher_batch:
                    batch[f"teacher_{key}"] = teacher_batch[key]
            require_aligned_responses(batch)

        self._add_confidence_weights(batch, examples)
        return batch
