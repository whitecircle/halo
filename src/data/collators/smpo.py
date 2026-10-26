"""SMPO preference collators: the text pair-padder and its VLM subclass, emitting the same batch keys.
Prompts are left-padded so the completion boundary aligns across a batch; completions are right-padded."""

from dataclasses import dataclass
from typing import Any

import torch
from transformers import ProcessorMixin
from transformers.data.data_collator import DataCollatorMixin
from trl.trainer.utils import pad

from src.data.pipeline.chat_render import lacks_emitted_bos
from src.data.vlm_render import SEQUENCE_ALIGNED_VISION_KEYS, run_vlm_processor

# Text keys the SMPO collators emit; anything else is a VLM vision tensor.
PREFERENCE_BATCH_KEYS = frozenset(
    {f"{side}_{suffix}" for side in ("prompt", "chosen", "rejected") for suffix in ("input_ids", "attention_mask")}
)


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
    """SMPO collator for VLM rows (templated ``prompt_text`` + ``images``, pre-tokenized completions): the
    images expand here rather than in the Arrow cache, and every extra processor tensor is concatenated
    row-major. Prompts are never truncated (that desyncs placeholders from pixels): an over-length one raises.
    """

    processor: ProcessorMixin | None = None
    max_prompt_length: int | None = None

    def __post_init__(self):
        if self.processor is None:
            raise ValueError("DataCollatorForVLMSMPO requires a processor.")

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        tokenizer = self.processor.tokenizer
        enriched_examples = []
        vision_features: dict[str, list[torch.Tensor]] = {}
        sequence_features: dict[str, list[torch.Tensor]] = {}
        for example in examples:
            images = example.get("images") or []
            encoded = run_vlm_processor(self.processor, example["prompt_text"], images)

            # run_vlm_processor tokenizes with add_special_tokens=False, so the BOS the tokenizer's
            # own post-processor emits is restored here, as on the text path.
            prompt_ids = encoded["input_ids"][0].tolist()
            bos_prepended = lacks_emitted_bos(prompt_ids, tokenizer)
            if bos_prepended:
                prompt_ids = [tokenizer.bos_token_id, *prompt_ids]
            if self.max_prompt_length is not None and len(prompt_ids) > self.max_prompt_length:
                raise ValueError(
                    f"VLM prompt expands to {len(prompt_ids)} tokens, over "
                    f"max_prompt_length={self.max_prompt_length}. VLM prompts cannot be truncated "
                    f"(cutting expanded image placeholder tokens desyncs them from pixel_values); "
                    f"raise max_prompt_length or pre-filter over-length rows."
                )
            enriched_examples.append({**example, "prompt_input_ids": prompt_ids})

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


def flush_rows_left(attention_mask: torch.Tensor, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Roll each row left by its leading-pad count, so every pad trails its real tokens; ``tensors``
    share the mask's shape and move with it, and the mask returns first. Unlike TRL's ``flush_left``
    no column is truncated, so nothing is read back from the device."""
    width = attention_mask.size(1)
    leading_pads = (attention_mask.cumsum(dim=1) == 0).sum(dim=1, keepdim=True)
    index = (torch.arange(width, device=attention_mask.device) + leading_pads) % width
    return tuple(tensor.gather(1, index) for tensor in (attention_mask, *tensors))
