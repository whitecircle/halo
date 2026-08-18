#!/usr/bin/env python3
"""Preprocessed PACKED datasets must reach the packing collator: metadata.packed forcing
sft_config.packing=False BEFORE collator selection lands the TRL default collator and dense
cross-document attention. The packing collator must also preserve ragged precomputed labels —
the parent LM collator rebuilds labels from input_ids and silently discards offline completion
masks.

Usage:
    python tests/cpu/data/test_preprocessed_packed_collator.py
"""

import json
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from accelerate import PartialState
from datasets import Dataset, DatasetDict

PartialState()

from src.data.collators.packing import DataCollatorForCompletionOnlyLMWithPacking, DataCollatorWithPacking

PAD = 0
EOS = 1
IGNORE = -100
RESP_1, RESP_2 = 10, 11


def make_tokenizer(pad_token_id: int = PAD, eos_token_id: int = EOS) -> MagicMock:
    """Mock tokenizer with the minimal interface the LM collators need."""
    tok = MagicMock()
    tok.pad_token_id = pad_token_id
    tok.eos_token_id = eos_token_id
    tok.bos_token_id = None
    tok.padding_side = "right"
    tok.model_max_length = 4096
    tok.encode.return_value = [RESP_1, RESP_2]

    def _pad(features, **kwargs):
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": []}
        for f in features:
            ids = list(f["input_ids"])
            mask = list(f.get("attention_mask", [1] * len(ids)))
            pad_len = max_len - len(ids)
            batch["input_ids"].append(ids + [pad_token_id] * pad_len)
            batch["attention_mask"].append(mask + [0] * pad_len)
        return {k: torch.tensor(v) for k, v in batch.items()}

    tok.pad.side_effect = _pad
    return tok


def test_completion_packing_collator_accepts_ragged_precomputed_labels():
    """The completion-masking packed collator re-masks from input_ids and must not crash on
    ragged precomputed labels."""
    collator = DataCollatorForCompletionOnlyLMWithPacking(
        response_prompt_template=[RESP_1, RESP_2],
        tokenizer=make_tokenizer(),
    )
    examples = [
        {
            "input_ids": [20, RESP_1, RESP_2, 30, EOS, 21, RESP_1, RESP_2, 31, EOS],
            "labels": [IGNORE, RESP_1, RESP_2, 30, EOS, IGNORE, RESP_1, RESP_2, 31, EOS],
            "seq_lengths": [5, 5],
        },
    ]
    batch = collator.torch_call(examples)
    labels = batch["labels"][0].tolist()
    assert labels[3] == 30 and labels[8] == 31, f"completion tokens must stay trained, got {labels}"
    assert labels[0] == IGNORE and labels[5] == IGNORE, "prompt tokens must stay masked"


def _write_metadata(dataset_dir: str, packed: bool, max_length: int = 64) -> None:
    metadata = {
        "version": "1.0",
        "preprocessed": True,
        "model_name": "stub-model",
        "tokenizer_vocab_size": 100,
        "max_length": max_length,
        "packed": packed,
        "packing_strategy": "bfd" if packed else None,
        "train_on_completions_only": False,
        "is_vlm": False,
        "has_pixel_values": False,
        "num_shards": 1,
        "total_train_examples": 2,
        "total_test_examples": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
        "config": {},
    }
    with open(os.path.join(dataset_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)


def _prepare_preprocessed(packed: bool, train_on_completions_only: bool = False):
    """Drive sft.py's _prepare_text_data over a fake preprocessed dataset; returns the collator."""
    from scripts.training.sft import _prepare_text_data

    tmp = tempfile.mkdtemp()
    try:
        _write_metadata(tmp, packed=packed)
        rows = {"input_ids": [[5, 6, 7], [8, 9]], "labels": [[5, 6, 7], [8, 9]]}
        if packed:
            rows["seq_lengths"] = [[3], [2]]
        ds = DatasetDict({"train": Dataset.from_dict(rows), "test": Dataset.from_dict(rows)})

        args = SimpleNamespace(
            dataset=tmp,
            train_on_completions_only=train_on_completions_only,
            assistant_message_template="<|assistant|>" if train_on_completions_only else None,
            train_on_last_assistant_only=False,
        )
        sft_config = SimpleNamespace(max_length=64, packing=False, padding_free=False, per_device_train_batch_size=1)
        model_config = SimpleNamespace(model_name_or_path="stub-model")
        parallelism_config = SimpleNamespace(is_cp_mode=False, cp_size=1, pp_size=1)

        _train, _eval, _gen, collator = _prepare_text_data(
            ds, True, args, sft_config, model_config, make_tokenizer(), parallelism_config, hf_config=None
        )
        return collator
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_preprocessed_packed_selects_packing_collator():
    """metadata.packed must select the packing collator even though sft_config.packing is False
    for TRL: under the TRL default collator documents attend across pack boundaries."""
    collator = _prepare_preprocessed(packed=True)
    assert isinstance(collator, DataCollatorWithPacking), (
        f"packed preprocessed data must get the packing collator, got {type(collator).__name__}"
    )

    batch = collator.torch_call([{"input_ids": [5, 6, 7, 8, 9, 12], "seq_lengths": [3, 3]}])
    assert batch["position_ids"][0].tolist() == [0, 1, 2, 0, 1, 2]


def test_preprocessed_unpacked_keeps_default_collator():
    """Un-packed preprocessed data keeps the default (None → TRL) collator path."""
    assert _prepare_preprocessed(packed=False) is None


def test_preprocessed_baked_labels_are_authoritative():
    """train_on_completions_only must NOT select a re-masking collator for preprocessed data —
    the labels are baked offline, and runtime re-masking would overwrite them (e.g. a packed
    chunk whose response marker landed in the previous chunk loses its trained tokens)."""
    collator = _prepare_preprocessed(packed=True, train_on_completions_only=True)
    assert isinstance(collator, DataCollatorWithPacking)
    assert not isinstance(collator, DataCollatorForCompletionOnlyLMWithPacking), (
        "preprocessed data selected the completion-masking collator — baked labels would be overwritten"
    )
    assert _prepare_preprocessed(packed=False, train_on_completions_only=True) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
