#!/usr/bin/env python
"""CPU test for ``scripts/before_training/patch_vocab.py`` :func:`add_tokens_to_model`.

The invariant (patch_vocab.py ~211-224): never shrink — only grow when ``new_vocab_size`` exceeds the
model's existing (padded) embedding size; added tokens reuse the existing padding rows. Resizing the
embedding DOWN to ``len(tokenizer)`` drops the padding rows that hold high special tokens (harmony
EOS / <|return|> / <|call|> …), and the served model then cannot emit its stop tokens and generates
degenerate garbage.

These tests reconstruct the gpt-oss shape with a tiny stub model whose input/output embeddings are
PADDED LARGER than ``len(tokenizer)``, plus a fake tokenizer, and assert the load-bearing invariants:
  (a) the embedding row count never shrinks below the original padded size;
  (b) the freshly-initialized new-token rows equal the pre-patch mean over the ORIGINAL vocab rows;
  (c) the high special-token rows in the padding region are byte-identical before/after.
``assert-finite`` is explicitly insufficient here — dropped/overwritten rows are still finite.

Run: ``python tests/cpu/models/test_vocab_patch.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tests.common.utils import load_script_module

pv = load_script_module("scripts/before_training/patch_vocab.py")


# --------------------------------------------------------------------------- #
# Tiny stubs reproducing the gpt-oss shape: tokenizer vocab < padded embedding.
# --------------------------------------------------------------------------- #

_BASE_VOCAB = 100  # len(tokenizer): "normal" tokens occupy rows [0, 100)
_PADDED_ROWS = 128  # the embedding is padded LARGER than the vocab (efficiency padding)
_HIDDEN = 8


class _FakeTokenizer:
    """Minimal tokenizer: a fixed base vocab plus ``add_tokens`` that appends new ids.

    Patterns to add are made multi-token by ``encode`` (so they pass the patch filter); already-known
    single tokens encode to one id. ``len()`` reflects base vocab + added tokens, mirroring HF.
    """

    def __init__(self, base_vocab: int):
        self._size = base_vocab
        self._added: dict[str, int] = {}

    def __len__(self) -> int:
        return self._size

    def encode(self, text: str, add_special_tokens: bool = False):
        if text in self._added:
            return [self._added[text]]  # already a single token
        # Unknown multi-char text → pretend it is >1 token so the patch picks it up.
        return [0, 1] if len(text) > 1 else [0]

    def decode(self, ids):
        # Only used by the post-patch verification print; round-trips an added single token.
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
    """Stub exposing the gpt-oss embedding shape: ``padded_rows`` > ``len(tokenizer)``.

    Implements the three accessors ``add_tokens_to_model`` calls: ``get_input_embeddings``,
    ``get_output_embeddings`` and ``resize_token_embeddings`` (grow-only row append, preserving
    existing rows — exactly what HF's resize does when growing)."""

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
    # Make the padding rows distinctive so a byte-identity check is meaningful (not all-zeros / equal).
    with torch.no_grad():
        model.embed.weight[_BASE_VOCAB:] = torch.randn(_PADDED_ROWS - _BASE_VOCAB, _HIDDEN) + 50.0
        model.lm_head.weight[_BASE_VOCAB:] = torch.randn(_PADDED_ROWS - _BASE_VOCAB, _HIDDEN) - 50.0
    tok = _FakeTokenizer(_BASE_VOCAB)
    return model, tok


def test_embedding_never_shrinks_below_padded_size():
    """The new vocab (base + a few added) fits inside the padding, so the row count must stay
    at the padded size — resizing DOWN to len(tokenizer) drops special-token rows."""
    model, tok = _build_model_and_tokenizer()
    patterns = ["alpha", "beta", "gamma"]  # multi-char → multi-token → added (3 tokens)

    num_added = pv.add_tokens_to_model(model, tok, patterns)

    assert num_added == 3
    # len(tokenizer) after adding = 103, still < 128 → must NOT shrink.
    assert len(tok) == _BASE_VOCAB + 3 < _PADDED_ROWS
    assert model.get_input_embeddings().weight.shape[0] == _PADDED_ROWS, "embedding shrank below padded size"
    assert model.get_output_embeddings().weight.shape[0] == _PADDED_ROWS, "lm_head shrank below padded size"


def test_high_special_token_rows_preserved_byte_identical():
    """Rows in the padding region ABOVE the new vocab boundary (the harmony stop tokens) must be
    untouched — byte-identical before and after. Dropping them was the garbage-generation cause."""
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
    """New-token rows must be the mean over the ORIGINAL vocab rows [0, base) computed BEFORE the
    resize — that is the documented init. A wrong slice (e.g. mean over padded rows) would differ."""
    model, tok = _build_model_and_tokenizer()
    expected_in_mean = model.get_input_embeddings().weight[:_BASE_VOCAB].mean(dim=0).detach().clone()
    expected_out_mean = model.get_output_embeddings().weight[:_BASE_VOCAB].mean(dim=0).detach().clone()

    num_added = pv.add_tokens_to_model(model, tok, ["alpha", "beta", "gamma"])

    after_in = model.get_input_embeddings().weight.detach()
    after_out = model.get_output_embeddings().weight.detach()
    for row in range(_BASE_VOCAB, _BASE_VOCAB + num_added):
        assert torch.allclose(after_in[row], expected_in_mean, atol=1e-6), f"input row {row} != original-vocab mean"
        assert torch.allclose(after_out[row], expected_out_mean, atol=1e-6), f"output row {row} != original-vocab mean"
    # And the mean is genuinely distinct from the padding rows (so the assertion can actually fail).
    assert not torch.allclose(expected_in_mean, after_in[_BASE_VOCAB + num_added], atol=1e-6)


def test_grow_when_vocab_exceeds_padding():
    """When the added tokens overflow the padding, the embedding DOES grow (resize up) — the
    no-shrink guard must not also block legitimate growth."""
    model, tok = _build_model_and_tokenizer()
    # Add more distinct multi-token patterns than the available padding (28 free rows).
    patterns = [f"tok_{i}" for i in range(40)]
    num_added = pv.add_tokens_to_model(model, tok, patterns)

    assert num_added == 40
    new_vocab = _BASE_VOCAB + 40  # 140 > 128 padded
    assert new_vocab > _PADDED_ROWS
    assert model.get_input_embeddings().weight.shape[0] == new_vocab, "embedding failed to grow past padding"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
