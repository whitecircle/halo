#!/usr/bin/env python
"""A split the length filter empties must fail loud in sft.py, before packing hides the cause.

The chat processor drops (never truncates) every conversation longer than ``max_length``. Packing an
emptied split raises ``Columns ['seq_lengths'] not in the dataset`` from deep inside the packer,
which names neither the split nor ``max_length``. Both splits are checked: a guard covering ``train``
alone lets an eval split of long conversations crash that way.

Usage:
    python tests/cpu/data/test_sft_empty_split_guard.py
"""

from types import SimpleNamespace

import datasets
import pytest
from accelerate import PartialState
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

PartialState()

from scripts.training.sft import _prepare_text_data

_MAX_LENGTH = 64
_SHORT = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
_LONG = [{"role": "user", "content": "word " * 400}, {"role": "assistant", "content": "word " * 400}]


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        dataset=None,
        conversation_field="messages",
        system_prompt=None,
        model_supports_system_role=True,
        interleaved_thinking=False,
        tools_field=None,
        train_on_completions_only=False,
        assistant_message_template=None,
        train_on_last_assistant_only=False,
    )


def _sft_config() -> SimpleNamespace:
    return SimpleNamespace(
        max_length=_MAX_LENGTH,
        packing=True,
        eval_packing=True,
        packing_strategy="bfd",
        padding_free=False,
        dataset_num_proc=None,
        per_device_train_batch_size=1,
    )


@pytest.fixture(autouse=True)
def _writable_dataset_cache(tmp_path, monkeypatch):
    """The map passes spill through the datasets cache, which the CPU tier mounts read-only."""
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path)


def _prepare(train_messages, test_messages):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    ds = DatasetDict(
        {
            "train": Dataset.from_dict({"messages": train_messages}),
            "test": Dataset.from_dict({"messages": test_messages}),
        }
    )
    return _prepare_text_data(
        ds,
        False,
        _args(),
        _sft_config(),
        SimpleNamespace(model_name_or_path="Qwen/Qwen3-0.6B"),
        tokenizer,
        SimpleNamespace(is_cp_mode=False, cp_size=1, pp_size=1),
        hf_config=None,
    )


def test_emptied_eval_split_names_the_split_and_the_cap():
    """Every eval conversation over max_length: raise here, not inside the packer."""
    with pytest.raises(ValueError, match=f"All evaluation rows were dropped.*max_length={_MAX_LENGTH}"):
        _prepare([_SHORT] * 4, [_LONG] * 2)


def test_emptied_train_split_names_the_split_and_the_cap():
    with pytest.raises(ValueError, match=f"All training rows were dropped.*max_length={_MAX_LENGTH}"):
        _prepare([_LONG] * 2, [_SHORT] * 2)


def test_both_splits_within_the_cap_still_pack():
    """Anti-over-rejection: the guard must not fire on a dataset the filter kept."""
    train_dataset, eval_dataset, _generate, _collator = _prepare([_SHORT] * 4, [_SHORT] * 2)

    assert len(train_dataset) > 0 and len(eval_dataset) > 0
    assert "seq_lengths" in train_dataset.column_names
    assert "seq_lengths" in eval_dataset.column_names


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
