"""Collation of vision-language preference rows for Bradley-Terry reward modeling."""

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from transformers import ProcessorMixin
from transformers.data.data_collator import DataCollatorMixin

from halo.data.pipeline.preferences import MARGIN_COLUMN, VLM_PREFERENCE_COLUMNS
from halo.data.vlm_render import reject_over_length_vlm_batch, run_vlm_processor


@dataclass
class DataCollatorForVLMPreference(DataCollatorMixin):
    """Collate into TRL's ``[chosen ⧺ rejected]`` batch. One processor call covers the whole ``2 * batch_size`` block,
    so text and vision tensors align by construction. The order is load-bearing: ``RewardTrainer.compute_loss`` reads
    the first half as chosen, so a swap inverts every preference without changing a shape."""

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

        batch = dict(run_vlm_processor(self.processor, texts, per_row_images + per_row_images, padding=True))
        # The dataset map drops rows whose text alone is over budget; this catches rows image expansion pushes over it.
        reject_over_length_vlm_batch(batch, self.max_length, "preference batch")
        if MARGIN_COLUMN in examples[0]:
            batch[MARGIN_COLUMN] = torch.tensor([example[MARGIN_COLUMN] for example in examples], dtype=torch.float)
        return batch
