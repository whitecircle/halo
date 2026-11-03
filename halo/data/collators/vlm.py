"""VLM collators: runtime render-and-collate and the preprocessed-artifact padder, both composing
:mod:`halo.data.vlm_render` so a runtime batch and an offline bake tokenize identically."""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from halo.data.completion_spans import (
    PADDED_SPAN_POLICY,
    build_training_labels,
    reject_untrainable_completions,
)
from halo.data.turn_markers import reject_missing_response_marker, resolve_eos_token_ids
from halo.data.vlm_render import (
    VLM_OUTPUT_FEATURES,
    reject_over_length_vlm_batch,
    render_vlm_text,
    resolve_image_token_ids,
    run_vlm_processor,
    vlm_row_tools,
)
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.log import warn_once
from halo.models.attention.prefix_lm import has_prefix_lm_attention, prefix_lm_token_type_ids

logger = logging.getLogger(__name__)

# Two exchanges in the VLM render's parts form, so a template ending a turn at the next role marker (GLM) closes one.
_PROBE_CONVERSATION = [
    {"role": role, "content": [{"type": "text", "text": text}]}
    for role, text in (("user", "q"), ("assistant", "a"), ("user", "q"), ("assistant", "a"))
]


def _reject_rendered_bos(processor, tokenizer) -> None:
    """Refuse a chat template that renders ``<bos>`` for a prefix-LM processor, which puts one after each row's
    last image itself: a row would hold two, as would every prompt the same template renders for serving."""
    bos = tokenizer.bos_token
    if bos and bos in render_vlm_text(processor, _PROBE_CONVERSATION):
        raise ValueError(
            f"The chat template renders {bos}, which this prefix-LM (PaliGemma) processor already puts after each "
            f"row's last image, so every row would hold two. Remove the BOS from the chat template (a Gemma "
            f"template opens with {{{{ bos_token }}}})."
        )


class VLMDataCollator:
    # Same span policy as the text runtime collators; the SDPG subclass overrides it.
    _COMPLETION_SPAN_POLICY: dict[str, bool] | None = PADDED_SPAN_POLICY

    def __init__(
        self,
        processor,
        tokenizer,
        max_length: int,
        *,
        train_on_completions_only: bool,
        response_prompt_template=None,
        model_config=None,
    ):
        reject_missing_response_marker(response_prompt_template, train_on_completions_only, type(self).__name__)
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.response_prompt_template = response_prompt_template
        self.train_on_completions_only = train_on_completions_only
        self.eos_token_ids = resolve_eos_token_ids(tokenizer, model_config)
        # At construction: the union reads the full vocab dict, too costly per batch for a verdict that cannot change.
        self.image_token_ids = tuple(resolve_image_token_ids(tokenizer, processor))
        # The processor marks no suffix in a rendered conversation (all 0: one bidirectional prefix).
        self.prefix_lm = has_prefix_lm_attention(model_config)
        if self.prefix_lm:
            _reject_rendered_bos(processor, tokenizer)
        if train_on_completions_only:
            reject_untrainable_completions(
                lambda conversation: render_vlm_text(processor, conversation),
                _PROBE_CONVERSATION,
                tokenizer,
                response_prompt_template,
                self.eos_token_ids,
                self._COMPLETION_SPAN_POLICY,
                type(self).__name__,
            )

    def __call__(self, examples):
        return self._collate(examples, lambda example: example.get("history", []), "batch")

    def _collate(self, examples, history_of: Callable[[dict[str, Any]], list], subject: str):
        """The processed batch with its labels, and a prefix-LM model's ``token_type_ids``."""
        batch = self._process_batch(examples, history_of, subject)
        batch["labels"] = self._build_labels(batch["input_ids"], batch.get("attention_mask"))
        if self.prefix_lm:
            self._mask_appended_newline(batch)
            batch["token_type_ids"] = prefix_lm_token_type_ids(batch["labels"])
        return batch

    def _process_batch(self, examples, history_of: Callable[[dict[str, Any]], list], subject: str):
        """Refuses an over-length sequence: truncating would desync expanded image placeholders from the vision
        tensors, and a runtime batch cannot drop rows without desyncing DP ranks."""
        if self.prefix_lm and not all(example.get("images") for example in examples):
            # PaliGemma's processor puts <bos> after each row's last <image>: in a row without one it lands
            # six characters into the text, and a batch without any image raises inside the processor.
            raise ValueError(
                f"VLM {subject} holds a row without an image on a prefix-LM (PaliGemma) model, whose processor "
                f"splices <bos> into such a row's text. Drop the text-only rows from the image dataset, or train "
                f"them in a separate text-only run."
            )
        texts = [self._render(history_of(example), example) for example in examples]
        row_images = [example.get("images") or () for example in examples]
        batch = run_vlm_processor(self.processor, texts, row_images, padding=True)
        reject_over_length_vlm_batch(batch, self.max_length, subject)
        return batch

    def _render(self, history: list[dict[str, Any]], example: dict[str, Any]) -> str:
        return render_vlm_text(self.processor, history, tools=vlm_row_tools(example))

    def _mask_appended_newline(self, batch) -> None:
        """Mask the newline a prefix-LM processor appends to the text (PaliGemma's prefix/suffix separator), which
        lands after the conversation ends."""
        attention_mask = batch["attention_mask"]
        last_positions = attention_mask.shape[1] - 1 - attention_mask.flip(-1).argmax(dim=-1)
        for row, position in enumerate(last_positions.tolist()):
            if self.tokenizer.decode([int(batch["input_ids"][row, position])]).endswith("\n"):
                batch["labels"][row, position] = LABEL_IGNORE_INDEX

    def _build_labels(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        """Pad and image tokens masked, and with ``train_on_completions_only`` everything outside the completions."""
        return build_training_labels(
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
    """Concatenated over every image-carrying row, in order (a mixed batch has text-only rows)."""
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
    """Pad and stack pre-tokenized VLM rows (:data:`~halo.data.vlm_render.VLM_OUTPUT_FEATURES`)."""

    # Unioned into HF's signature columns, which would otherwise prune pixel_values_shape and the rest.
    required_dataset_columns: tuple[str, ...] = tuple(VLM_OUTPUT_FEATURES)

    def __init__(self, tokenizer, max_length: int):
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
