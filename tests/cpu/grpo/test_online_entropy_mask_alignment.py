#!/usr/bin/env python
"""``top_entropy_quantile`` protection must fail loud when its completion ids do not match the mask.

The mixin stashes the completion ids in ``_get_per_token_logps_and_entropies`` and unions their
special-token positions into ``get_high_entropy_mask`` on the SAME micro-batch. A stash from a
different micro-batch (or none at all) must not fall through to TRL's plain mask, which silently
drops the structural tokens from the trained set — the template rot the mixin exists to prevent.

    python tests/cpu/grpo/test_online_entropy_mask_alignment.py
"""

import pytest
import torch
from accelerate import PartialState

from src.trainers.grpo.mixins.entropy_mask import ProtectedTokenEntropyMixin

# The mixin logs through accelerate's adapter, which needs the process state to exist.
PartialState()

SPECIAL_ID = 100


class _Tokenizer:
    all_special_ids = [SPECIAL_ID]

    def get_added_vocab(self):
        return {}


class _NoSpecialsTokenizer:
    all_special_ids: list[int] = []

    def get_added_vocab(self):
        return {}


class _PlainQuantile:
    def get_high_entropy_mask(self, entropies, mask, threshold):
        cutoff = torch.quantile(entropies[mask.bool()].float(), threshold)
        return (entropies >= cutoff) & mask.bool()

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, *a, **kw):
        return None, None


class _Harness(ProtectedTokenEntropyMixin, _PlainQuantile):
    def __init__(self, tokenizer=None):
        self.processing_class = tokenizer if tokenizer is not None else _Tokenizer()


def _batch(rows: int, length: int):
    completion_ids = torch.full((rows, length), 7)
    completion_ids[:, 0] = SPECIAL_ID  # the lowest-entropy position, outside the plain quantile
    entropies = torch.linspace(0.01, 5.0, rows * length).view(rows, length)
    return completion_ids, entropies, torch.ones(rows, length, dtype=torch.long)


def test_a_stash_from_another_micro_batch_raises_instead_of_dropping_protection():
    harness = _Harness()
    other_ids, _, other_mask = _batch(rows=2, length=6)
    harness._get_per_token_logps_and_entropies(None, other_ids, other_mask, other_ids.size(1))
    _, entropies, mask = _batch(rows=1, length=5)
    with pytest.raises(RuntimeError, match="different micro-batches"):
        harness.get_high_entropy_mask(entropies, mask, 0.8)


def test_a_mask_requested_before_any_stash_raises():
    _, entropies, mask = _batch(rows=1, length=5)
    with pytest.raises(RuntimeError, match="lost its completion ids"):
        _Harness().get_high_entropy_mask(entropies, mask, 0.8)


def test_an_aligned_stash_protects_the_special_token():
    harness = _Harness()
    completion_ids, entropies, mask = _batch(rows=1, length=5)
    harness._get_per_token_logps_and_entropies(None, completion_ids, mask, completion_ids.size(1))
    got = harness.get_high_entropy_mask(entropies, mask, 0.8)
    assert bool(got[0, 0]), "the special token must be unioned back into the trained set"


def test_nothing_to_protect_is_not_a_mismatch():
    """A tokenizer exposing no specials has nothing to align; the plain mask stands, warned once."""
    harness = _Harness(_NoSpecialsTokenizer())
    _, entropies, mask = _batch(rows=1, length=5)
    plain = _PlainQuantile().get_high_entropy_mask(entropies, mask, 0.8)
    assert torch.equal(harness.get_high_entropy_mask(entropies, mask, 0.8), plain)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
