"""CP offline-GRPO rows stay right-padded and target/reference aligned."""

import pytest
import torch
from accelerate import PartialState

from src.data.collators.offline_grpo import (
    REF_PER_TOKEN_LOGPS_COLUMN,
    OfflineGRPOCPDataCollatorWithPadding,
    OfflineGRPODataCollatorWithPadding,
)
from src.trainers.grpo.offline import tokenize_offline_grpo_rows


class _Tokenizer:
    bos_token_id = None
    eos_token_id = 99

    def __call__(self, text, *, truncation=False, max_length=None, **kwargs):
        tokens = [{"p": 1, "a": 2, "b": 3, "c": 4, "d": 5, "e": 6, "end": 99}[word] for word in text.split()]
        return {"input_ids": tokens[:max_length] if truncation else tokens}

    def decode(self, token_ids):
        return "token"


_TOKENIZE_KWARGS = {
    "processing_class": _Tokenizer(),
    "max_prompt_length": 4,
    "max_completion_length": 3,
    "advantage_method": "z_norm",
    "best_completion_emphasis": 0.0,
}


def _row(prompt, completion, *, reference=None):
    row = {
        "prompt_input_ids": prompt,
        "completion_input_ids": completion,
        "group_id": 1,
        "group_size": 2,
        "advantage": 0.5,
    }
    if reference is not None:
        row[REF_PER_TOKEN_LOGPS_COLUMN] = reference
    return row


def test_cp_rows_keep_completion_targets_and_reference_in_full_row_positions():
    batch = OfflineGRPOCPDataCollatorWithPadding(pad_token_id=0, cp_size=4)(
        [
            _row([11, 12], [21, 22, 23], reference=[-1.0, -2.0, -3.0]),
            _row([13], [24], reference=[-4.0]),
        ]
    )
    assert batch["input_ids"].shape == (2, 8)
    assert batch["input_ids"][0].tolist() == [11, 12, 21, 22, 23, 0, 0, 0]
    assert batch["attention_mask"][1].tolist() == [1, 1, 0, 0, 0, 0, 0, 0]
    assert batch["labels"][0].tolist() == [-100, -100, 21, 22, 23, -100, -100, -100]
    assert batch["labels"][1].tolist() == [-100, 24, -100, -100, -100, -100, -100, -100]
    torch.testing.assert_close(
        batch[REF_PER_TOKEN_LOGPS_COLUMN][0], torch.tensor([0.0, 0.0, -1.0, -2.0, -3.0, 0, 0, 0])
    )
    # The causal shift predicts each completion token from the position immediately before it.
    shifted_targets = batch["labels"][:, 1:]
    shifted_refs = batch[REF_PER_TOKEN_LOGPS_COLUMN][:, 1:]
    torch.testing.assert_close(shifted_refs[shifted_targets != -100], torch.tensor([-1.0, -2.0, -3.0, -4.0]))


@pytest.mark.parametrize("cp", [False, True], ids=["local", "cp"])
def test_empty_completion_is_inert_and_mixed_reference_rows_fail(cp):
    PartialState()  # the substituted-completion warning logs through accelerate's logger
    collator = (
        OfflineGRPOCPDataCollatorWithPadding(pad_token_id=7, cp_size=2)
        if cp
        else OfflineGRPODataCollatorWithPadding(pad_token_id=7)
    )
    batch = collator([_row([7], [])])
    if cp:
        assert batch["input_ids"].tolist() == [[7, 7]]
        assert batch["attention_mask"].tolist() == [[1, 0]]
        assert batch["labels"].tolist() == [[-100, -100]]
    else:
        assert batch["completion_input_ids"].tolist() == [[7]]
        assert batch["completion_attention_mask"].tolist() == [[0]]
    with pytest.raises(ValueError, match="at least one row"):
        collator([])
    with pytest.raises(ValueError, match="mix rows"):
        collator([_row([1], [2]), _row([1], [2], reference=[-1.0])])
    with pytest.raises(ValueError, match="must match tokenization"):
        collator([_row([1], [2, 3], reference=[-1.0])])


@pytest.mark.parametrize("cp_size", [0, -2])
def test_cp_collator_refuses_a_nonpositive_cp_size_at_construction(cp_size):
    with pytest.raises(ValueError, match="cp_size must be positive"):
        OfflineGRPOCPDataCollatorWithPadding(pad_token_id=0, cp_size=cp_size)


def test_grouped_tokenization_drops_unused_supplied_scores():
    batch = {
        "prompt": ["p"],
        "completions": [["a b", "c d e"]],
        "rewards": [[1.0, 0.0]],
        REF_PER_TOKEN_LOGPS_COLUMN: [[[-0.1, -0.2], [-0.4, -0.5, -0.6]]],
    }
    expanded = tokenize_offline_grpo_rows(batch, [7], **_TOKENIZE_KWARGS)
    assert len(expanded["completion_input_ids"]) == 2
    assert REF_PER_TOKEN_LOGPS_COLUMN not in expanded


def test_tokenization_does_not_invent_reference_scores():
    batch = {"prompt": ["p"], "completions": [["a"]], "rewards": [[1.0]]}
    expanded = tokenize_offline_grpo_rows(batch, [0], **_TOKENIZE_KWARGS)
    assert REF_PER_TOKEN_LOGPS_COLUMN not in expanded


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
