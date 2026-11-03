#!/usr/bin/env python
"""``add_tokens_to_model`` never shrinks the embedding. Resizing down to ``len(tokenizer)`` drops the padding rows that
hold high special tokens (harmony EOS, <|return|>, <|call|>), and the served model then cannot emit its stop tokens."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from halo.checkpoint.tools import vocab_patch as pv

_BASE_VOCAB = 100  # len(tokenizer): tokens occupy rows [0, 100)
_PADDED_ROWS = 128  # the embedding is padded larger than the vocab
_HIDDEN = 8


class _FakeTokenizer:
    """Encodes an unknown multi-char pattern to several ids, so it passes the patch filter; ``len()`` counts added
    tokens, as HF's does."""

    def __init__(self, base_vocab: int):
        self._size = base_vocab
        self._added: dict[str, int] = {}

    def __len__(self) -> int:
        return self._size

    def encode(self, text: str, add_special_tokens: bool = False):
        if text in self._added:
            return [self._added[text]]
        return [0, 1] if len(text) > 1 else [0]

    def decode(self, ids):
        # Only used by the post-patch verification print.
        rev = {v: k for k, v in self._added.items()}
        return "".join(rev.get(i, "?") for i in ids)

    def add_tokens(self, new_tokens):
        added = 0
        for tok in new_tokens:
            if tok not in self._added:
                self._added[tok] = self._size
                self._size += 1
                added += 1
        return added


class _StubModelWithPaddedEmbedding(nn.Module):
    """The gpt-oss embedding shape: ``padded_rows`` > ``len(tokenizer)``. Its resize appends rows and keeps existing
    ones, as HF's does when growing."""

    def __init__(self, padded_rows: int, hidden: int):
        super().__init__()
        self.embed = nn.Embedding(padded_rows, hidden)
        self.lm_head = nn.Linear(hidden, padded_rows, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def resize_token_embeddings(self, new_num_tokens: int):
        old = self.embed.weight.shape[0]
        if new_num_tokens == old:
            return
        new_embed = nn.Embedding(new_num_tokens, self.embed.weight.shape[1])
        new_head = nn.Linear(self.lm_head.weight.shape[1], new_num_tokens, bias=False)
        with torch.no_grad():
            keep = min(old, new_num_tokens)
            new_embed.weight[:keep] = self.embed.weight[:keep]
            new_head.weight[:keep] = self.lm_head.weight[:keep]
        self.embed = new_embed
        self.lm_head = new_head


def _build_model_and_tokenizer():
    torch.manual_seed(0)
    model = _StubModelWithPaddedEmbedding(_PADDED_ROWS, _HIDDEN)
    # Distinctive padding rows, so a byte-identity check is meaningful.
    with torch.no_grad():
        model.embed.weight[_BASE_VOCAB:] = torch.randn(_PADDED_ROWS - _BASE_VOCAB, _HIDDEN) + 50.0
        model.lm_head.weight[_BASE_VOCAB:] = torch.randn(_PADDED_ROWS - _BASE_VOCAB, _HIDDEN) - 50.0
    tok = _FakeTokenizer(_BASE_VOCAB)
    return model, tok


def test_embedding_never_shrinks_below_padded_size():
    """Resizing down to len(tokenizer) would drop special-token rows."""
    model, tok = _build_model_and_tokenizer()
    patterns = ["alpha", "beta", "gamma"]

    num_added = pv.add_tokens_to_model(model, tok, patterns)

    assert num_added == 3
    assert len(tok) == _BASE_VOCAB + 3 < _PADDED_ROWS
    assert model.get_input_embeddings().weight.shape[0] == _PADDED_ROWS, "embedding shrank below padded size"
    assert model.get_output_embeddings().weight.shape[0] == _PADDED_ROWS, "lm_head shrank below padded size"


def test_high_special_token_rows_preserved_byte_identical():
    """Rows above the new vocab boundary hold the harmony stop tokens; dropping them leaves the served model unable to
    emit them."""
    model, tok = _build_model_and_tokenizer()
    before_in = model.get_input_embeddings().weight.detach().clone()
    before_out = model.get_output_embeddings().weight.detach().clone()

    num_added = pv.add_tokens_to_model(model, tok, ["alpha", "beta", "gamma"])
    new_vocab = _BASE_VOCAB + num_added  # 103; rows [103:128] are the high special tokens

    after_in = model.get_input_embeddings().weight.detach()
    after_out = model.get_output_embeddings().weight.detach()
    assert torch.equal(after_in[new_vocab:], before_in[new_vocab:]), "high input special-token rows changed"
    assert torch.equal(after_out[new_vocab:], before_out[new_vocab:]), "high output special-token rows changed"


def test_new_token_rows_init_to_pre_patch_mean():
    """A wrong slice (e.g. mean over padded rows) would differ."""
    model, tok = _build_model_and_tokenizer()
    expected_in_mean = model.get_input_embeddings().weight[:_BASE_VOCAB].mean(dim=0).detach().clone()
    expected_out_mean = model.get_output_embeddings().weight[:_BASE_VOCAB].mean(dim=0).detach().clone()

    num_added = pv.add_tokens_to_model(model, tok, ["alpha", "beta", "gamma"])

    after_in = model.get_input_embeddings().weight.detach()
    after_out = model.get_output_embeddings().weight.detach()
    for row in range(_BASE_VOCAB, _BASE_VOCAB + num_added):
        assert torch.allclose(after_in[row], expected_in_mean, atol=1e-6), f"input row {row} != original-vocab mean"
        assert torch.allclose(after_out[row], expected_out_mean, atol=1e-6), f"output row {row} != original-vocab mean"
    # The mean differs from the padding rows, so the assertion can fail.
    assert not torch.allclose(expected_in_mean, after_in[_BASE_VOCAB + num_added], atol=1e-6)


def test_grow_when_vocab_exceeds_padding():
    """The no-shrink guard must not also block legitimate growth."""
    model, tok = _build_model_and_tokenizer()
    # More patterns than the 28 free padding rows.
    patterns = [f"tok_{i}" for i in range(40)]
    num_added = pv.add_tokens_to_model(model, tok, patterns)

    assert num_added == 40
    new_vocab = _BASE_VOCAB + 40
    assert new_vocab > _PADDED_ROWS
    assert model.get_input_embeddings().weight.shape[0] == new_vocab, "embedding failed to grow past padding"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
