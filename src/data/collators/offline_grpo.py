"""Offline GRPO batch collation: pad a tokenized prompt/completion group to the batch's shape.

The padding sides and the substituted-completion mask below are the batch layout
``OfflineGRPOTrainer.compute_loss`` expects.
"""

from dataclasses import dataclass
from typing import Any

import torch
from accelerate.logging import get_logger
from trl.trainer.utils import pad

from src.data.spans import LABEL_IGNORE_INDEX

logger = get_logger(__name__, log_level="INFO")

# Raw reference scores align with completion tokens, independent of the scoring parallelism.
REF_PER_TOKEN_LOGPS_COLUMN = "ref_per_token_logps"


def _validate_reference_rows(features: list[dict[str, Any]]) -> bool:
    """Both layouts require references to match the same raw completion tokens."""
    if not features:
        raise ValueError("Offline GRPO needs at least one row per batch")
    has_reference = REF_PER_TOKEN_LOGPS_COLUMN in features[0]
    for row in features:
        if (REF_PER_TOKEN_LOGPS_COLUMN in row) != has_reference:
            raise ValueError("Offline GRPO cannot mix rows with and without reference log-probs")
        if has_reference:
            values, tokens = row[REF_PER_TOKEN_LOGPS_COLUMN], row["completion_input_ids"]
            if len(values) != len(tokens):
                raise ValueError(
                    f"{REF_PER_TOKEN_LOGPS_COLUMN} holds {len(values)} values for a completion of "
                    f"{len(tokens)} tokens; the raw reference must match tokenization: scores must be "
                    "one per completion token under the run's own length caps."
                )
    return has_reference


@dataclass
class OfflineGRPODataCollatorWithPadding:
    """Pads tokenized prompt/completion inputs to the batch's max length (prompts left-padded)."""

    pad_token_id: int = 0

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_input_ids = []
        completion_input_ids = []
        substituted_completion: list[bool] = []
        ref_logps: list[torch.Tensor] = []
        has_ref_logps = _validate_reference_rows(features)

        for example in features:
            # Never empty: the trainer's tokenize map refuses a prompt with no tokens.
            prompt_input_ids.append(torch.tensor(example["prompt_input_ids"], dtype=torch.long))

            comp_ids = example["completion_input_ids"]
            substituted_completion.append(not comp_ids)
            if has_ref_logps:
                values = example[REF_PER_TOKEN_LOGPS_COLUMN]
                ref_logps.append(torch.tensor(values or [0.0], dtype=torch.float32))
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


@dataclass
class OfflineGRPOCPDataCollatorWithPadding:
    """Right-pad whole rows so CP attention never sees a pad before a real token.

    The supervision and optional raw reference grid are indexed by the target token's position
    in ``input_ids``. The CP scorer shifts both grids at shard boundaries after its hidden forward.
    """

    pad_token_id: int = 0
    cp_size: int = 1

    def __post_init__(self):
        if self.cp_size < 1:
            raise ValueError(f"cp_size must be positive, got {self.cp_size}")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        has_reference = _validate_reference_rows(features)

        rows: list[list[int]] = []
        labels: list[list[int]] = []
        references: list[list[float]] = []
        for row in features:
            prompt = row["prompt_input_ids"]
            completion = row["completion_input_ids"]
            rows.append([*prompt, *(completion or [self.pad_token_id])])
            labels.append(
                [LABEL_IGNORE_INDEX] * len(prompt) + (list(completion) if completion else [LABEL_IGNORE_INDEX])
            )
            if has_reference:
                values = row[REF_PER_TOKEN_LOGPS_COLUMN]
                references.append([0.0] * len(prompt) + list(values or [0.0]))

        width = max(len(row) for row in rows)
        width = max(self.cp_size, ((width + self.cp_size - 1) // self.cp_size) * self.cp_size)
        token_rows = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        attention = torch.zeros_like(token_rows)
        target_rows = torch.full_like(token_rows, LABEL_IGNORE_INDEX)
        reference_rows = torch.zeros((len(rows), width), dtype=torch.float32) if has_reference else None
        for index, (tokens, targets) in enumerate(zip(rows, labels, strict=True)):
            token_rows[index, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            target_rows[index, : len(targets)] = torch.tensor(targets, dtype=torch.long)
            attention[index, : len(tokens)] = 1
            if not features[index]["completion_input_ids"]:
                attention[index, len(features[index]["prompt_input_ids"])] = 0
            if reference_rows is not None:
                reference_rows[index, : len(references[index])] = torch.tensor(references[index], dtype=torch.float32)

        output = {
            "input_ids": token_rows,
            "attention_mask": attention,
            "labels": target_rows,
            "group_id": torch.tensor([row["group_id"] for row in features]),
            "group_size": torch.tensor([row["group_size"] for row in features]),
            "advantage": torch.tensor([row["advantage"] for row in features]),
        }
        if reference_rows is not None:
            output[REF_PER_TOKEN_LOGPS_COLUMN] = reference_rows
        return output
