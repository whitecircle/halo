"""Batch-time collation of vision-language preference rows for Bradley-Terry reward modeling: each side is
one rendered conversation (:func:`~src.data.pipeline.preferences.render_vlm_preference_row`), expanded
through the processor into TRL's ``[chosen ⧺ rejected]`` layout."""

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from transformers import ProcessorMixin
from transformers.data.data_collator import DataCollatorMixin

from src.data.pipeline.preferences import MARGIN_COLUMN, VLM_PREFERENCE_COLUMNS
from src.data.vlm_render import reject_over_length_vlm_batch, run_vlm_processor


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

    # Unioned into HF's signature columns, which name none of the columns this collator reads.
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
        reject_over_length_vlm_batch(batch, self.max_length, "preference batch")
        if MARGIN_COLUMN in examples[0]:
            batch[MARGIN_COLUMN] = torch.tensor([example[MARGIN_COLUMN] for example in examples], dtype=torch.float)
        return batch
