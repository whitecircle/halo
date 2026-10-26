#!/usr/bin/env python
"""The profiling benchmarks' corpus gives a model nothing to fit and is the same on every rank and run.

``make bench`` holds each row's throughput and peak memory to a band drawn from five runs, so the corpus must
not let a row's work drift within a run or between runs. An MoE model routes each token by its content, and on
a corpus it can fit its routing (each EP rank's load, step time and peak) moves as it fits, differently on
every run under non-deterministic dispatch. Uniform token ids over the regular vocabulary cannot be fitted,
every token of every row is real and trained on, and a fixed seed makes every rank build the same rows.

Run: python tests/cpu/conventions/test_benchmark_corpus.py
"""

from collections import Counter
from types import SimpleNamespace

import pytest

from tests.common.datasets import create_benchmark_dataset

# Only ``vocab_size`` is read: the corpus is token ids, never text.
TOKENIZER = SimpleNamespace(vocab_size=1000)
SEQ_LEN, NUM_SAMPLES = 256, 64
# Uniform ids put 16.4 of the 16384 tokens on each id; a filler that repeats a word puts thousands on one.
MAX_TOKEN_SHARE = 0.005


@pytest.fixture(scope="module")
def corpus():
    return create_benchmark_dataset(TOKENIZER, SEQ_LEN, NUM_SAMPLES)


def test_every_rank_builds_the_same_rows(corpus):
    again = create_benchmark_dataset(TOKENIZER, SEQ_LEN, NUM_SAMPLES)

    assert again["input_ids"] == corpus["input_ids"]
    assert create_benchmark_dataset(TOKENIZER, SEQ_LEN, NUM_SAMPLES, seed=1)["input_ids"] != corpus["input_ids"]


def test_every_token_is_a_regular_vocabulary_id_attended_and_trained_on(corpus):
    assert len(corpus) == NUM_SAMPLES
    for ids, mask, labels in zip(corpus["input_ids"], corpus["attention_mask"], corpus["labels"], strict=True):
        assert len(ids) == SEQ_LEN
        assert all(0 <= token < TOKENIZER.vocab_size for token in ids)
        assert mask == [1] * SEQ_LEN, "a padded position carries no work"
        assert labels == ids


def test_no_token_dominates_the_corpus(corpus):
    counts = Counter(token for ids in corpus["input_ids"] for token in ids)
    total = SEQ_LEN * NUM_SAMPLES

    assert max(counts.values()) / total < MAX_TOKEN_SHARE
    assert len(counts) == TOKENIZER.vocab_size
    assert len({tuple(ids) for ids in corpus["input_ids"]}) == NUM_SAMPLES


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
