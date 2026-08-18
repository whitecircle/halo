"""SMPO preference collators: the text pair-padder and its VLM subclass.

Both emit the same batch keys, so the trainer's chosen|rejected concat is written once: prompts are
LEFT-padded (the completion boundary must align across a batch), completions right-padded. The VLM
subclass adds image processing at collation — per-row pixel tensors are orders of magnitude larger
than the encoded image, so they never enter the Arrow cache — and concatenates the resulting vision
tensors row-major over that same layout.

A leaf, like :mod:`src.data.collators.vlm_preference`: its processor/vision imports stay out of every
collator import, and the SMPO trainer reaches it by name.
"""

from dataclasses import dataclass
from typing import Any

import torch
from transformers import ProcessorMixin
from transformers.data.data_collator import DataCollatorMixin
from trl.trainer.utils import pad

from src.data.pipeline.rendered import probe_tokenizer_specials
from src.data.vlm import run_vlm_processor

_TEXT_SIDES = ("prompt", "chosen", "rejected")

# Text keys the SMPO collators emit; anything else is a VLM vision tensor. Derived from the sides the collator pads.
PREFERENCE_BATCH_KEYS = frozenset(
    {f"{side}_{suffix}" for side in _TEXT_SIDES for suffix in ("input_ids", "attention_mask")}
)

# Per-token type tensors (M-RoPE): 1:1 with input_ids, so they ride the pad/concat transforms, never a row-major cat.
SEQUENCE_ALIGNED_VISION_KEYS = frozenset({"mm_token_type_ids", "token_type_ids"})


@dataclass
class DataCollatorForSMPO(DataCollatorMixin):
    """Data collator for SMPO preference data; pads prompt/chosen/rejected to per-batch max lengths."""

    pad_token_id: int
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_input_ids = [torch.tensor(ex["prompt_input_ids"]) for ex in examples]
        prompt_attention_mask = [torch.ones_like(ids) for ids in prompt_input_ids]
        chosen_input_ids = [torch.tensor(ex["chosen_input_ids"]) for ex in examples]
        chosen_attention_mask = [torch.ones_like(ids) for ids in chosen_input_ids]
        rejected_input_ids = [torch.tensor(ex["rejected_input_ids"]) for ex in examples]
        rejected_attention_mask = [torch.ones_like(ids) for ids in rejected_input_ids]

        output = {
            "prompt_input_ids": pad(prompt_input_ids, padding_value=self.pad_token_id, padding_side="left"),
            "prompt_attention_mask": pad(prompt_attention_mask, padding_value=0, padding_side="left"),
            "chosen_input_ids": pad(chosen_input_ids, padding_value=self.pad_token_id),
            "chosen_attention_mask": pad(chosen_attention_mask, padding_value=0),
            "rejected_input_ids": pad(rejected_input_ids, padding_value=self.pad_token_id),
            "rejected_attention_mask": pad(rejected_attention_mask, padding_value=0),
        }

        return output


@dataclass
class DataCollatorForVLMSMPO(DataCollatorForSMPO):
    """SMPO collator for VLM rows: templated ``prompt_text`` + ``images``, pre-tokenized completions.

    Image placeholder expansion and pixel-tensor extraction happen here, not in the dataset map:
    per-row pixel tensors would bloat the Arrow cache by orders of magnitude. Emits the standard
    SMPO keys plus every extra processor tensor concatenated row-major over the batch.

    Prompts are never truncated — cutting expanded image placeholder tokens while the vision tensors
    keep every patch desyncs text from vision, so an over-``max_prompt_length`` prompt is a hard
    error. Completions are text-only and already length-budgeted by ``tokenize_vlm_preference_row``.
    """

    processor: ProcessorMixin | None = None
    max_prompt_length: int | None = None

    def __post_init__(self):
        if self.processor is None:
            raise ValueError("DataCollatorForVLMSMPO requires a processor.")

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        tokenizer = self.processor.tokenizer
        # Same probe gate as the text path (``tokenize_preference_row``): run_vlm_processor
        # tokenizes with add_special_tokens=False, so BOS is only owed when the tokenizer's own
        # post-processor emits one. A nominal bos_token it never emits (gpt-oss/Bailing shape) must
        # NOT be injected — that trains on a token the served policy never sees.
        bos_token_id = tokenizer.bos_token_id if probe_tokenizer_specials(tokenizer).adds_leading_bos else None

        enriched_examples = []
        vision_features: dict[str, list[torch.Tensor]] = {}
        sequence_features: dict[str, list[torch.Tensor]] = {}
        for example in examples:
            images = example.get("images") or []
            encoded = run_vlm_processor(self.processor, example["prompt_text"], images)

            prompt_ids = encoded["input_ids"][0]
            bos_prepended = bos_token_id is not None and (
                prompt_ids.numel() == 0 or int(prompt_ids[0]) != bos_token_id
            )
            if bos_prepended:
                prompt_ids = torch.cat([prompt_ids.new_tensor([bos_token_id]), prompt_ids])
            if self.max_prompt_length is not None and prompt_ids.numel() > self.max_prompt_length:
                raise ValueError(
                    f"VLM prompt expands to {prompt_ids.numel()} tokens, over "
                    f"max_prompt_length={self.max_prompt_length}. VLM prompts cannot be truncated "
                    f"(cutting expanded image placeholder tokens desyncs them from pixel_values); "
                    f"raise max_prompt_length or pre-filter over-length rows."
                )
            enriched_examples.append({**example, "prompt_input_ids": prompt_ids.tolist()})

            for key, value in encoded.items():
                if key in ("input_ids", "attention_mask"):
                    continue
                if key in SEQUENCE_ALIGNED_VISION_KEYS:
                    aligned = value[0]
                    if bos_prepended:  # keep 1:1 with prompt_input_ids (BOS is a text position → 0)
                        aligned = torch.cat([aligned.new_zeros(1), aligned])
                    sequence_features.setdefault(key, []).append(aligned)
                else:
                    vision_features.setdefault(key, []).append(value)

        output = super().torch_call(enriched_examples)
        for key, values in vision_features.items():
            output[key] = torch.cat(values, dim=0)
        for key, values in sequence_features.items():
            output[key] = pad(values, padding_value=0, padding_side="left")
        return output
