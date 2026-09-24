#!/usr/bin/env python
"""A tokenizer whose content cannot be hashed keys the dataset-map cache on ``name_or_path`` — visibly.

``_tokenizer_content_sig`` hashes a tokenizer's content so a resume from a checkpoint path reuses the
cache. When reading that content raises (a fast backend with a custom Python component, a slow
tokenizer without ``get_vocab``), the key degrades to the path: every resume leg re-runs the map and a
changed tokenizer at the same path reuses stale token ids. The degradation must warn once per type.

Run: pytest tests/cpu/data/test_tokenizer_content_sig_fallback.py
"""

import logging

import pytest
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import WordLevel

from src.data.pipeline import processing
from src.data.pipeline.processing import _get_kwargs_fingerprint, _tokenizer_content_sig

_LOGGER = "src.data.pipeline.processing"
_WARNING_TEXT = "cannot hash the content of tokenizer"


class _NoOpPreTokenizer:
    def pre_tokenize(self, pretok):
        pass


class _CustomComponentTokenizer:
    """A fast tokenizer whose backend carries a custom Python pre-tokenizer, which ``to_str`` refuses."""

    vocab_size = 2

    def __init__(self, name_or_path):
        self.name_or_path = name_or_path
        self.backend_tokenizer = Tokenizer(WordLevel(vocab={"[UNK]": 0, "a": 1}, unk_token="[UNK]"))
        self.backend_tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(_NoOpPreTokenizer())


class _NoVocabTokenizer:
    """A slow tokenizer on the ``PreTrainedTokenizerBase.get_vocab`` default."""

    name_or_path = "org/slow"
    vocab_size = 2

    def get_vocab(self):
        raise NotImplementedError()


class _ContentlessTokenizer:
    """Exposes neither a backend nor a vocab: the path key is the only term, and nothing failed."""

    name_or_path = "org/contentless"
    vocab_size = 2


@pytest.fixture(autouse=True)
def _rearm_warning(monkeypatch):
    monkeypatch.setattr(processing, "_CONTENT_SIG_FAILED_WARNED", set())


def _content_warnings(caplog):
    return [r for r in caplog.records if _WARNING_TEXT in r.getMessage()]


def test_unserializable_backend_warns_once_and_keys_on_path(caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert _tokenizer_content_sig(_CustomComponentTokenizer("org/custom")) is None
        _tokenizer_content_sig(_CustomComponentTokenizer("org/custom"))

    warnings = _content_warnings(caplog)
    assert len(warnings) == 1, f"expected one warning per tokenizer type, got {len(warnings)}"
    message = warnings[0].getMessage()
    assert "_CustomComponentTokenizer" in message
    assert "cannot be serialized" in message, "the warning must carry the cause"
    assert "org/custom" in message, "the warning must name the path the key fell back to"

    same = _get_kwargs_fingerprint({"tokenizer": _CustomComponentTokenizer("org/custom")})
    moved = _get_kwargs_fingerprint({"tokenizer": _CustomComponentTokenizer("checkpoints/run/checkpoint-10")})
    assert same == _get_kwargs_fingerprint({"tokenizer": _CustomComponentTokenizer("org/custom")})
    assert same != moved, "without a content hash the path is the only term that separates tokenizers"


def test_missing_get_vocab_warns(caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert _tokenizer_content_sig(_NoVocabTokenizer()) is None

    warnings = _content_warnings(caplog)
    assert len(warnings) == 1
    assert "NotImplementedError" in warnings[0].getMessage()


def test_tokenizer_without_content_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert _tokenizer_content_sig(_ContentlessTokenizer()) is None

    assert not _content_warnings(caplog), "nothing raised, so there is nothing to report"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
