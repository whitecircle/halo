#!/usr/bin/env python
"""``configure_run_tokenizer`` marks a model whose embeddings it grew, and the mark reads through every wrapper.

The rows a grow adds exist in no base checkpoint, so the adapter savers ask :func:`vocabulary_grown` whether the run
grew the vocabulary rather than re-reading the base config, which PEFT skips offline. Pinned on a tiny Qwen3: tokens
past the padded rows grow the embedding to the tokenizer's length and mark it, read through a toolkit wrapper (the CP
model's protocol) and through PEFT around one; tokens that fit the padding, or none, leave it unmarked.

    python tests/cpu/model_loading/test_grow_vocabulary.py
"""

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from halo.model_loading.tokenizer_setup import vocabulary_grown
from tests.common.grown_vocab import FITTING_TOKENS, GROWING_TOKENS, WORDS, grow_vocabulary
from tests.common.tiny_models import TINY_DENSE_FAMILY, tiny_family_model, word_level_tokenizer


class _ToolkitWrapper(nn.Module):
    """A wrapper under the toolkit unwrap protocol, as the CP model declares it."""

    _toolkit_inner_model_attr = "model"

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model


def _configured(added: list[str] | None):
    tokenizer = word_level_tokenizer(WORDS)
    torch.manual_seed(0)
    model = tiny_family_model(TINY_DENSE_FAMILY, tokenizer)
    rows = model.get_input_embeddings().num_embeddings
    return model, grow_vocabulary(model, tokenizer, added), rows


def test_tokens_past_the_padding_grow_the_embedding_and_mark_the_model():
    model, tokenizer, rows = _configured(GROWING_TOKENS)
    assert model.get_input_embeddings().num_embeddings == len(tokenizer) > rows
    assert model.get_output_embeddings().weight.shape[0] == len(tokenizer)
    assert vocabulary_grown(model)
    assert vocabulary_grown(_ToolkitWrapper(model))
    assert vocabulary_grown(get_peft_model(_ToolkitWrapper(model), LoraConfig(r=2, target_modules=["q_proj"])))


@pytest.mark.parametrize("added", [None, FITTING_TOKENS], ids=["none", "within-padding"])
def test_no_grow_leaves_the_model_unmarked(added):
    model, _, rows = _configured(added)
    assert model.get_input_embeddings().num_embeddings == rows
    assert not vocabulary_grown(model)
    assert not vocabulary_grown(get_peft_model(model, LoraConfig(r=2, target_modules=["q_proj"])))


@pytest.mark.parametrize(
    ("added", "grows"),
    [(GROWING_TOKENS, True), (FITTING_TOKENS, False), (None, False)],
    ids=["past-padding", "within-padding", "none"],
)
def test_a_grow_says_so_once_with_its_adapter_save_consequence(caplog, added, grows):
    """The grow makes every adapter save carry the embedding and head whole, so it is logged once, naming both
    lengths; a tokenizer that fits the rows logs nothing."""
    with caplog.at_level("INFO", logger="halo.model_loading.tokenizer_setup"):
        model, tokenizer, rows = _configured(added)
    lines = [record.getMessage() for record in caplog.records if "grew" in record.getMessage()]
    expected = f"vocabulary from {rows} to {len(tokenizer)} rows; a PEFT adapter save of this run carries"
    assert len(lines) == int(grows) and all(expected in line for line in lines), lines


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
