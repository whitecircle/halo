"""SMPO preference collators: the text pair-padder and its VLM subclass, emitting the same batch keys, and
``concatenate_pairs``, which joins a collated batch into the trainer's one ``[chosen ⧺ rejected]`` forward.
Prompts are left-padded so the completion boundary aligns across a batch; completions are right-padded."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import ProcessorMixin
from transformers.data.data_collator import DataCollatorMixin
from trl.trainer.utils import pad

from src.data.completion_spans import completion_labels
from src.data.pipeline.chat_render import lacks_emitted_bos
from src.data.vlm_render import SEQUENCE_ALIGNED_VISION_KEYS, run_vlm_processor
from src.kernels.logprobs import LABEL_IGNORE_INDEX

# Text keys the SMPO collators emit; anything else is a VLM vision tensor.
_PREFERENCE_BATCH_KEYS = frozenset(
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


def _flush_rows_left(attention_mask: torch.Tensor, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Roll each row left by its leading-pad count, so every pad trails its real tokens; ``tensors``
    share the mask's shape and move with it, and the mask returns first. Unlike TRL's ``flush_left``
    no column is truncated, so nothing is read back from the device."""
    width = attention_mask.size(1)
    leading_pads = (attention_mask.cumsum(dim=1) == 0).sum(dim=1, keepdim=True)
    index = (torch.arange(width, device=attention_mask.device) + leading_pads) % width
    return tuple(tensor.gather(1, index) for tensor in (attention_mask, *tensors))


def concatenate_pairs(batch: dict[str, torch.Tensor], pad_token_id: int, cp_size: int = 1) -> dict[str, torch.Tensor]:
    """One ``[chosen ⧺ rejected]`` batch (``input_ids``/``attention_mask``/``labels``) for a single forward.

    Every row is flushed left (:func:`_flush_rows_left`), so a model's default positions are its real tokens'
    positions: attention that reads absolute positions (Mistral4's llama-4 query scale) or cuts KV windows at fixed
    indices (DeepSeek-V4's compressors) scores a padded row as unpadded, and the CP attention, which ignores the
    mask, never reaches a pad from a real token. Under CP the rows are then right-padded to a multiple of ``cp_size``.
    """
    max_completion_len = max(
        batch["chosen_input_ids"].shape[1],
        batch["rejected_input_ids"].shape[1],
    )

    def _pad(t: torch.Tensor, fill: int) -> torch.Tensor:
        return F.pad(t, (0, max_completion_len - t.shape[1]), value=fill)

    chosen_padded = _pad(batch["chosen_input_ids"], pad_token_id)
    rejected_padded = _pad(batch["rejected_input_ids"], pad_token_id)
    chosen_mask = _pad(batch["chosen_attention_mask"], 0)
    rejected_mask = _pad(batch["rejected_attention_mask"], 0)

    batch_size = batch["prompt_input_ids"].shape[0]

    chosen_full_ids = torch.cat([batch["prompt_input_ids"], chosen_padded], dim=1)
    rejected_full_ids = torch.cat([batch["prompt_input_ids"], rejected_padded], dim=1)
    chosen_full_mask = torch.cat([batch["prompt_attention_mask"], chosen_mask], dim=1)
    rejected_full_mask = torch.cat([batch["prompt_attention_mask"], rejected_mask], dim=1)

    chosen_labels = torch.cat(
        [
            torch.full_like(batch["prompt_input_ids"], LABEL_IGNORE_INDEX),
            completion_labels(chosen_padded, chosen_mask),
        ],
        dim=1,
    )
    rejected_labels = torch.cat(
        [
            torch.full_like(batch["prompt_input_ids"], LABEL_IGNORE_INDEX),
            completion_labels(rejected_padded, rejected_mask),
        ],
        dim=1,
    )

    output = {
        "input_ids": torch.cat([chosen_full_ids, rejected_full_ids], dim=0),
        "attention_mask": torch.cat([chosen_full_mask, rejected_full_mask], dim=0),
        "labels": torch.cat([chosen_labels, rejected_labels], dim=0),
        "num_chosen": batch_size,
    }

    # The rejected rows repeat the same prompts, so vision tensors need an identical second copy;
    # sequence-aligned token types must first extend over the completion region to stay 1:1.
    for key, value in batch.items():
        if key in _PREFERENCE_BATCH_KEYS:
            continue
        if key in SEQUENCE_ALIGNED_VISION_KEYS:
            value = F.pad(value, (0, max_completion_len), value=0)
        output[key] = torch.cat([value, value], dim=0)

    flushed = [
        "attention_mask",
        "input_ids",
        "labels",
        *(key for key in SEQUENCE_ALIGNED_VISION_KEYS if key in output),
    ]
    output.update(zip(flushed, _flush_rows_left(*(output[key] for key in flushed)), strict=True))
    if cp_size > 1:
        fills = {"input_ids": pad_token_id, "labels": LABEL_IGNORE_INDEX}
        tail = -output["attention_mask"].size(1) % cp_size
        for key in flushed:
            output[key] = F.pad(output[key], (0, tail), value=fills.get(key, 0))

    return output
