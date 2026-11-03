#!/usr/bin/env python
"""sft.py's raw-text split preparation: a split the length filter empties fails loud before the packer hides the
cause behind ``Columns ['seq_lengths'] not in the dataset``, and the test split is mapped for generation only
when the run generates examples."""

from types import SimpleNamespace

import datasets
import pytest
from accelerate import PartialState
from datasets import Dataset, DatasetDict
from transformers import Qwen3Config

PartialState()

from halo.data.collators.padded import DataCollatorForLanguageModelingWithVarlen
from scripts.training.sft import _prepare_text_data
from tests.common.models import QWEN3_0_6B, TINY_QWEN3_CONFIG
from tests.common.packed_flash_varlen import VARLEN_KEYS
from tests.common.tokenizers import load_cached_tokenizer

_MAX_LENGTH = 64
_SHORT = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
_LONG = [{"role": "user", "content": "word " * 400}, {"role": "assistant", "content": "word " * 400}]


def _args(generate_eval_examples: bool) -> SimpleNamespace:
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
        tree_packing=False,
        generate_eval_examples=generate_eval_examples,
        allow_cross_document_linear_attention=False,
    )


def _sft_config(**overrides) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            "max_length": _MAX_LENGTH,
            "packing": True,
            "eval_packing": True,
            "packing_strategy": "bfd",
            "padding_free": False,
            "dataset_num_proc": None,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "eval_strategy": "no",
            "pad_to_multiple_of": None,
            **overrides,
        }
    )


@pytest.fixture(autouse=True)
def _writable_dataset_cache(tmp_path, monkeypatch):
    """The map passes spill through the datasets cache; a per-test dir keeps them off the shared cache."""
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path)


def _prepare(train_messages, test_messages, generate_eval_examples: bool = False, hf_config=None, **sft_overrides):
    tokenizer = load_cached_tokenizer(QWEN3_0_6B)
    ds = DatasetDict(
        {
            "train": Dataset.from_dict({"messages": train_messages}),
            "test": Dataset.from_dict({"messages": test_messages}),
        }
    )
    return _prepare_text_data(
        ds,
        None,
        _args(generate_eval_examples),
        _sft_config(**sft_overrides),
        SimpleNamespace(model_name_or_path=QWEN3_0_6B),
        tokenizer,
        tokenizer,
        SimpleNamespace(is_cp_mode=False, cp_size=1, pp_size=1),
        hf_config=hf_config,
    )


def test_emptied_eval_split_names_the_split_and_the_cap():
    with pytest.raises(ValueError, match=f"All evaluation rows were dropped.*max_length={_MAX_LENGTH}"):
        _prepare([_SHORT] * 4, [_LONG] * 2)


def test_emptied_train_split_names_the_split_and_the_cap():
    with pytest.raises(ValueError, match=f"All training rows were dropped.*max_length={_MAX_LENGTH}"):
        _prepare([_LONG] * 2, [_SHORT] * 2)


def test_both_splits_within_the_cap_still_pack():
    train_dataset, eval_dataset, _generate, _collator = _prepare([_SHORT] * 4, [_SHORT] * 2)

    assert len(train_dataset) > 0 and len(eval_dataset) > 0
    assert "seq_lengths" in train_dataset.column_names
    assert "seq_lengths" in eval_dataset.column_names


def test_the_test_split_is_mapped_for_generation_only_when_examples_are_generated():
    _train, _eval, skipped, _collator = _prepare([_SHORT] * 4, [_SHORT] * 2)
    assert skipped is None
    _train, _eval, generate, _collator = _prepare([_SHORT] * 4, [_SHORT] * 2, generate_eval_examples=True)
    assert len(generate) == 2
    assert "input_ids" in generate.column_names


@pytest.mark.parametrize("pad_to_multiple_of", [None, _MAX_LENGTH])
def test_unpacked_rows_on_a_flash_run_take_trls_collator_with_the_one_document_set(pad_to_multiple_of):
    flash = Qwen3Config(**TINY_QWEN3_CONFIG)
    flash._attn_implementation = "flash_attention_2"
    train_dataset, _eval, _generate, collator = _prepare(
        [_SHORT] * 4, [_SHORT] * 2, hf_config=flash, packing=False, pad_to_multiple_of=pad_to_multiple_of
    )

    assert type(collator) is DataCollatorForLanguageModelingWithVarlen
    assert collator.pad_to_multiple_of == pad_to_multiple_of
    width = len(train_dataset[0]["input_ids"])
    batch = collator([train_dataset[0]])
    if pad_to_multiple_of is None:
        assert batch["cu_seq_lens_q"].tolist() == [0, width] and batch["max_length_q"] == width
    else:
        # The row is shorter than the multiple, so the padded batch keeps transformers' mask path.
        assert width % pad_to_multiple_of and batch["input_ids"].shape[1] == pad_to_multiple_of
        assert not any(key in batch for key in VARLEN_KEYS)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
