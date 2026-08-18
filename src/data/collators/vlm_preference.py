"""Batch-time collation of vision-language preference rows for Bradley-Terry reward modeling.

Takes the rows :func:`~src.data.pipeline.preferences.render_vlm_preference_row` produces, expands
their image placeholders through the processor, and emits TRL's ``[chosen ⧺ rejected]`` layout plus
the vision tensors concatenated row-major over that layout. The reward head scores the whole
sequence, so unlike the SMPO collator there is no prompt/completion split: each side is one rendered
conversation, and the image placeholders sit in the shared prompt prefix of both.
"""

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from transformers import ProcessorMixin
from transformers.data.data_collator import DataCollatorMixin

from src.data.pipeline.preferences import MARGIN_COLUMN, VLM_PREFERENCE_COLUMNS
from src.data.vlm import raise_if_over_length, run_vlm_processor


@dataclass
class DataCollatorForVLMPreference(DataCollatorMixin):
    """Collate rendered VLM preference rows into TRL's ``[chosen ⧺ rejected]`` reward batch.

    One processor call covers the whole ``2 * batch_size`` row block, so ``input_ids`` /
    ``attention_mask`` and every vision tensor come out aligned by construction. The row order is
    load-bearing: ``RewardTrainer.compute_loss`` chunks the pooled scores in two and reads the first
    half as chosen, so a swapped half inverts every preference without changing a shape.

    Both sides of a pair render the same prompt and therefore the same images, so the image list is
    the batch's row-major images repeated once per half, matching the text block it feeds.
    """

    processor: ProcessorMixin | None = None
    max_length: int | None = None
    return_tensors: str = "pt"

    # The trainer's dataloader mixin unions these into HF's signature-column set. Without the
    # declaration TRL's reward signature list (chosen_ids / rejected_ids / margin) drops every
    # column this collator reads, and the first batch crashes on a missing key.
    required_dataset_columns: ClassVar[tuple[str, ...]] = VLM_PREFERENCE_COLUMNS

    def __post_init__(self):
        if self.processor is None:
            raise ValueError("DataCollatorForVLMPreference requires a processor.")

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [example["chosen_text"] for example in examples] + [example["rejected_text"] for example in examples]
        per_row_images = [list(example.get("images") or []) for example in examples]
        images = [image for row_images in per_row_images + per_row_images for image in row_images]

        batch = dict(run_vlm_processor(self.processor, texts, images, padding=True))
        # The dataset map already drops rows whose text alone is over budget; this is the backstop
        # for rows that image expansion pushes over it.
        raise_if_over_length(batch, self.max_length, "preference batch")
        if MARGIN_COLUMN in examples[0]:
            batch[MARGIN_COLUMN] = torch.tensor([example[MARGIN_COLUMN] for example in examples], dtype=torch.float)
        return batch
