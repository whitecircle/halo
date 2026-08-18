"""Offline GRPO batch collation: pad a tokenized prompt/completion group to the batch's shape.

The padding sides and the substituted-completion mask below are the batch layout
``OfflineGRPOTrainer.compute_loss`` expects.
"""

from dataclasses import dataclass
from typing import Any

import torch
from accelerate.logging import get_logger
from trl.trainer.utils import pad

logger = get_logger(__name__, log_level="INFO")

# Reference per-token log-probs, one per completion token, written into the tokenized dataset by the
# trainer's pipeline-parallel reference sweep (``OfflineGRPOTrainer._pp_precompute_reference_logps``)
# and collated below, aligned position-for-position with ``completion_input_ids``.
REF_PER_TOKEN_LOGPS_COLUMN = "ref_per_token_logps"


@dataclass
class OfflineGRPODataCollatorWithPadding:
    """Pads tokenized prompt/completion inputs to the batch's max length (prompts left-padded)."""

    pad_token_id: int = 0

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_input_ids = []
        completion_input_ids = []
        substituted_completion: list[bool] = []
        ref_logps: list[torch.Tensor] = []
        has_ref_logps = REF_PER_TOKEN_LOGPS_COLUMN in features[0]

        for example in features:
            prompt_ids = example["prompt_input_ids"]
            if not prompt_ids:
                logger.warning("Empty prompt_input_ids found, using pad token")
                prompt_ids = [self.pad_token_id]
            prompt_input_ids.append(torch.tensor(prompt_ids, dtype=torch.long))

            comp_ids = example["completion_input_ids"]
            substituted_completion.append(not comp_ids)
            if has_ref_logps:
                # Row-aligned with the tokenized completion: a length mismatch means the column was
                # produced under other length caps or another tokenizer, which would anchor the KL
                # to the wrong tokens. A substituted row carries none.
                values = example[REF_PER_TOKEN_LOGPS_COLUMN]
                if len(values) != len(comp_ids):
                    raise ValueError(
                        f"{REF_PER_TOKEN_LOGPS_COLUMN} holds {len(values)} values for a completion of "
                        f"{len(comp_ids)} tokens; the reference log-probs must be one per completion "
                        f"token under the run's own tokenization and length caps."
                    )
                ref_logps.append(torch.tensor(values, dtype=torch.float32))
            if not comp_ids:
                logger.warning("Empty completion_input_ids found, using pad token")
                comp_ids = [self.pad_token_id]
            completion_input_ids.append(torch.tensor(comp_ids, dtype=torch.long))

        prompt_attention_mask = [torch.ones_like(input_ids) for input_ids in prompt_input_ids]
        # A substituted row is masked out: the pad token stands in for a completion the policy never
        # produced, so training it would reinforce that token at this row's advantage.
        completion_attention_mask = [
            torch.zeros_like(input_ids) if substituted else torch.ones_like(input_ids)
            for input_ids, substituted in zip(completion_input_ids, substituted_completion, strict=True)
        ]

        pad_value = self.pad_token_id

        output = {
            "prompt_input_ids": pad(prompt_input_ids, padding_value=pad_value, padding_side="left"),
            "prompt_attention_mask": pad(prompt_attention_mask, padding_value=0, padding_side="left"),
            "completion_input_ids": pad(completion_input_ids, padding_value=pad_value),
            "completion_attention_mask": pad(completion_attention_mask, padding_value=0),
            "group_id": torch.tensor([ex["group_id"] for ex in features]),
            "group_size": torch.tensor([ex["group_size"] for ex in features]),
            "advantage": torch.tensor([ex["advantage"] for ex in features]),
        }
        if has_ref_logps:
            output[REF_PER_TOKEN_LOGPS_COLUMN] = pad(ref_logps, padding_value=0)

        return output
