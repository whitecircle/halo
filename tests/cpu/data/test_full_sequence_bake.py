"""A full-sequence bake (``train_on_completions_only`` off) stores the rows the runtime path collates, with no
``labels`` column, and packed rows with no ``attention_mask``: every collator such an artifact reaches derives
both from ``input_ids``, so the copies the bake used to store only doubled the artifact. Each collator must batch
the lean rows exactly as it batched rows carrying them, with ``pad != eos`` and with ``pad == eos`` (the
turn-ending EOS a derived label must keep).

Usage:
    python tests/cpu/data/test_full_sequence_bake.py
"""

import pytest
import torch
from datasets import Dataset
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling as TRLDataCollatorForLanguageModeling

from halo.data.collators.packing import DataCollatorWithFlattening, DataCollatorWithPacking
from halo.data.collators.padded import DataCollatorForCausalLMWithPadding
from halo.data.preprocessed.metadata import PreprocessingConfig
from halo.data.preprocessed.preparation import preprocess_dataset
from tests.common.tokenizers import load_cached_tokenizer

MODEL = "Qwen/Qwen3-0.6B"
MAX_LENGTH = 48
DOCUMENTS = [
    "Two plus two is four.",
    "The quick brown fox jumps over the lazy dog, then naps in the warm afternoon sun.",
    "Short.",
    "A longer document that keeps going so that a packed row has to hold it beside the others, "
    "and the batch has rows of different lengths to pad.",
]
CONVERSATIONS = [
    [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello there."}],
    [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "Four."}],
]


@pytest.fixture(params=[False, True], ids=["pad!=eos", "pad==eos"])
def tokenizer(request):
    tokenizer = load_cached_tokenizer(MODEL)
    tokenizer.padding_side = "right"
    if request.param:
        tokenizer.pad_token = tokenizer.eos_token
    assert (tokenizer.pad_token_id == tokenizer.eos_token_id) == request.param
    return tokenizer


def _bake(tmp_path, monkeypatch, tokenizer, *, packed: bool, mode: str = "text") -> list[dict]:
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "cache"))
    if mode == "text":
        corpus = Dataset.from_dict({"text": DOCUMENTS})
        knobs = {"mode": "text"}
    else:
        corpus = Dataset.from_dict({"prompt": CONVERSATIONS})
        knobs = {"conversation_field": "prompt"}
    config = PreprocessingConfig(
        model_name_or_path=MODEL,
        max_length=MAX_LENGTH,
        pack_sequences=packed,
        train_on_completions_only=False,
        num_proc=1,
        **knobs,
    )
    return [dict(row) for row in preprocess_dataset(corpus, tokenizer, config)["train"]]


def _collate(collator, rows: list[dict]) -> dict:
    return collator([dict(row) for row in rows])


def _assert_same_batches(collator, rows: list[dict]) -> None:
    """The lean rows against the same rows carrying the labels copy and all-ones mask the bake used to store."""
    with_copy = [
        {**row, "labels": list(row["input_ids"]), "attention_mask": [1] * len(row["input_ids"])} for row in rows
    ]
    lean, baked = _collate(collator, rows), _collate(collator, with_copy)
    assert set(lean) == set(baked), f"{type(collator).__name__}: {sorted(lean)} vs {sorted(baked)}"
    for key, value in baked.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(lean[key], value), f"{type(collator).__name__} batches {key} differently"
        else:
            assert lean[key] == value, f"{type(collator).__name__} batches {key} differently"


@pytest.mark.parametrize("mode", ["text", "chat"])
def test_a_full_sequence_bake_stores_no_labels(tmp_path, monkeypatch, tokenizer, mode):
    for packed in (False, True):
        rows = _bake(tmp_path / f"{mode}-{packed}", monkeypatch, tokenizer, packed=packed, mode=mode)
        assert rows and all("labels" not in row for row in rows), f"{mode} packed={packed}: {sorted(rows[0])}"
        assert all(("attention_mask" in row) != packed for row in rows), f"{mode} packed={packed}: {sorted(rows[0])}"


def test_packed_rows_batch_alike_with_and_without_the_copy(tmp_path, monkeypatch, tokenizer):
    """The packing collator, flattened as a dense run takes it and kept per row as PP takes it."""
    rows = _bake(tmp_path, monkeypatch, tokenizer, packed=True)
    assert any(len(row["seq_lengths"]) > 1 for row in rows), "a packed row must hold several documents"
    for flatten in (True, False):
        collator = DataCollatorWithPacking(tokenizer, return_flash_attn_kwargs=flatten)
        collator.flatten_to_single_row = flatten
        _assert_same_batches(collator, rows)


def test_unpacked_rows_batch_alike_with_and_without_the_copy(tmp_path, monkeypatch, tokenizer):
    """The CP padded collator, the padding-free one and TRL's own, which an unpacked full-sequence run takes."""
    rows = _bake(tmp_path, monkeypatch, tokenizer, packed=False)
    assert len({len(row["input_ids"]) for row in rows}) > 1, "rows of one length would pad nothing"
    for collator in (
        DataCollatorForCausalLMWithPadding(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=2),
        DataCollatorWithFlattening(tokenizer=tokenizer, return_flash_attn_kwargs=True),
        TRLDataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id),
    ):
        _assert_same_batches(collator, rows)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
