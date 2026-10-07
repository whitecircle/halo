"""VLM data collators: the runtime render-and-collate path, the preprocessed-artifact padder and the
SDPG self-distillation branch.

Every VLM primitive they compose — the chat render, the processor call, the over-length refusal —
lives in :mod:`src.data.vlm`, so a runtime batch and an offline bake of the same row
tokenize identically.
"""

import logging
from typing import Any

import numpy as np
import torch

from src.data.collators.self_distill import SelfDistillBranchMixin, require_aligned_responses
from src.data.spans import (
    COLLATOR_SPAN_POLICY,
    LABEL_IGNORE_INDEX,
    build_completion_only_labels,
    require_response_marker,
    resolve_eos_token_ids,
)
from src.data.vlm import (
    SEQUENCE_ALIGNED_VISION_KEYS,
    VLM_OUTPUT_FEATURES,
    get_image_token_ids,
    raise_if_over_length,
    render_vlm_text,
    run_vlm_processor,
    vlm_row_tools,
)
from src.log import warn_once

logger = logging.getLogger(__name__)

__all__ = [
    "VLMDataCollator",
    "PreprocessedVLMDataCollator",
    "SelfDistillVLMDataCollator",
]


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
        require_response_marker(response_prompt_template, train_on_completions_only, type(self).__name__)
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.response_prompt_template = response_prompt_template
        self.train_on_completions_only = train_on_completions_only
        self.eos_token_ids = resolve_eos_token_ids(tokenizer, model_config)
        # Construction-time, like the eos set: the union reads the tokenizer's full vocab dict, which
        # is a per-batch cost of the vocabulary's size for a verdict that cannot change between batches.
        self.image_token_ids = tuple(get_image_token_ids(tokenizer, processor))

    def _render(self, history: list[dict[str, Any]], example: dict[str, Any]) -> str:
        """Chat-template one conversation with the row's tools. Shared by the student and
        self-distill teacher branches so both render through the exact same call."""
        return render_vlm_text(self.processor, history, tools=vlm_row_tools(example))

    def __call__(self, examples):
        texts = []
        all_images = []

        for example in examples:
            history = example.get("history", [])
            images = example.get("images", [])

            texts.append(self._render(history, example))

            if images:
                all_images.extend(images)

        # NEVER truncate: cutting expanded image placeholders desyncs text from pixel_values/
        # image_grid_thw, and a runtime batch cannot drop rows without desyncing DP ranks.
        batch = run_vlm_processor(self.processor, texts, all_images, padding=True)
        raise_if_over_length(batch, self.max_length, "batch")
        batch["labels"] = self._build_labels(batch["input_ids"], batch.get("attention_mask"))
        return batch

    def _build_labels(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        """Build loss labels: mask pad + image tokens, optionally keep only assistant completions.
        Shared by the student batch and the self-distill teacher branch so both select the SAME
        response tokens (required for OPD row alignment).
        """
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


class PreprocessedVLMDataCollator:
    """Pad/stack pre-tokenized VLM data (no tokenization at collation time).

    Expects per-example: input_ids, attention_mask, labels, pixel_values
    (float16 bytes), pixel_values_shape, and optionally image_grid_thw (Qwen-VL).
    """

    # The trainer's dataloader mixin unions these into HF's signature-column set. Without the
    # declaration, TRL's SFT signature list drops every stored column it does not name
    # (pixel_values_shape, attention_mask, ...) before collation, and the first batch crashes.
    required_dataset_columns: tuple[str, ...] = tuple(VLM_OUTPUT_FEATURES)

    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id or 0
        # Per collator instance: one line about this run's preprocessed lengths, not one per batch.
        self._warned: set[str] = set()

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate preprocessed VLM examples into a batch."""
        input_ids_list = [ex["input_ids"] for ex in examples]
        attention_mask_list = [ex["attention_mask"] for ex in examples]
        labels_list = [ex["labels"] for ex in examples]

        # Pad only: re-truncating without trimming pixel_values/image_grid_thw desyncs placeholders.
        max_len = max(len(ids) for ids in input_ids_list)
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

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for input_ids, attention_mask, labels in zip(input_ids_list, attention_mask_list, labels_list, strict=False):
            padding_len = max_len - len(input_ids)

            batch_input_ids.append(input_ids + [self.pad_token_id] * padding_len)
            batch_attention_mask.append(attention_mask + [0] * padding_len)
            batch_labels.append(labels + [LABEL_IGNORE_INDEX] * padding_len)

        batch = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

        # Collect vision tensors from EVERY example in order: a mixed batch has text-only rows, and
        # keying off examples[0] would drop or misalign the batch's vision tensors.
        pixel_values_list = []
        for ex in examples:
            pv_bytes = ex.get("pixel_values")
            if pv_bytes is None:
                continue
            pv_array = np.frombuffer(pv_bytes, dtype=np.float16).reshape(ex["pixel_values_shape"])
            pixel_values_list.append(torch.from_numpy(pv_array.astype(np.float32)))
        if pixel_values_list:
            # For Qwen-VL pixel_values are already flattened per image.
            batch["pixel_values"] = torch.cat(pixel_values_list, dim=0)

        grid_list = []
        for ex in examples:
            grid = ex.get("image_grid_thw")
            if grid is None:
                continue
            if isinstance(grid, list):
                grid_list.extend(grid if isinstance(grid[0], list) else [grid])
            else:
                grid_list.append(grid.tolist() if hasattr(grid, "tolist") else grid)
        if grid_list:
            batch["image_grid_thw"] = torch.tensor(grid_list, dtype=torch.long)

        return batch


class SelfDistillVLMDataCollator(SelfDistillBranchMixin, VLMDataCollator):
    """VLM collator for SDPG-style self-distillation (arXiv:2606.04036).

    Emits the student batch plus a teacher branch (``teacher_*``) whose last user turn carries a
    privileged gold-answer hint. The assistant response stays byte-identical across both sequences and
    teacher labels reuse the parent ``_build_labels``, so both branches select the SAME tokens (OPD row
    alignment, :func:`require_aligned_responses`). The text-only hint leaves the image grid unchanged,
    so the trainer reuses the student's image tensors. Uses the self-distill span policy
    (``_COMPLETION_SPAN_POLICY = None``), matching :class:`SelfDistillTextCollator`.
    ``hint_template=None`` builds no teacher branch (the OPD term is off). Optionally emits
    ``confidence_weights``.
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
        """Refuse a row whose two branches would supervise different tokens; returns no columns.

        Read off the rendered text before the processor expands the image placeholders: those sit in
        the user turn, outside every label span. Tokenized as ``run_vlm_processor`` tokenizes the
        collated text (no added special tokens), so the audited ids are the collated ones. Mapped at
        prep by ``audit_self_distill_row``, where the raise is world-uniform; at collate time it would
        be rank-local.
        """
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
            # Same template path as the student, so the assistant response tokens stay byte-identical.
            teacher_texts = []
            all_images = []
            for example in examples:
                images = example.get("images", [])
                teacher_texts.append(self._render(self._teacher_history(example.get("history", []), example), example))
                if images:
                    all_images.extend(images)

            # NEVER truncate the teacher: it is systematically longer than the student (the hint), so
            # right-truncation cuts response tokens the student keeps, breaking OPD row alignment.
            teacher_batch = run_vlm_processor(self.processor, teacher_texts, all_images, padding=True)
            raise_if_over_length(teacher_batch, self.max_length, "teacher branch (student history + privileged hint)")

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
