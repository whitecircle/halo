#!/usr/bin/env python
"""``PreloadedTransformer`` must sit on sentence-transformers' CURRENT module surface.

The wrapper is the only module in the toolkit that subclasses a sentence-transformers base class, so
it is the one that goes stale when ST moves a path or renames a method. Both failure modes are
silent today and hard removals later: a deprecated import path still resolves (with a
``DeprecationWarning`` nobody reads) until the release that deletes it, and ST's
``Module.__getattr__`` still routes a retired method name to its replacement until it does not — at
which point ``SentenceTransformer.get_embedding_dimension`` returns ``None`` and every loss sized off
the embedding dimension is built wrong rather than failing.

Run: python tests/cpu/trainers/test_preloaded_transformer_st_api.py
     (or: pytest -m cpu tests/cpu/trainers/test_preloaded_transformer_st_api.py)
"""

import importlib
import sys
import warnings

import pytest
from sentence_transformers import SentenceTransformer
from sentence_transformers.base.modules.input_module import InputModule
from sentence_transformers.base.modules.module import Module
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model

from tests.common.models import TINY_QWEN3_CONFIG

_MODULE = "halo.trainers.embedding.preloaded_transformer"


def _reimport_under_deprecation_errors():
    """Import the wrapper afresh with ``DeprecationWarning`` promoted to an error.

    ST raises its path warnings from the package's import hook, which fires only on a real import —
    a cached ``sentence_transformers.models`` swallows them and would make this assertion vacuous.
    So the retired package tree is evicted for the duration and every evicted module restored after,
    leaving the classes other tests already hold untouched.
    """
    evicted = {
        name: module
        for name, module in list(sys.modules.items())
        if name == _MODULE or name.startswith("sentence_transformers.models")
    }
    for name in evicted:
        del sys.modules[name]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            return importlib.import_module(_MODULE)
    finally:
        for name in list(sys.modules):
            if name == _MODULE or name.startswith("sentence_transformers.models"):
                del sys.modules[name]
        sys.modules.update(evicted)


def test_the_wrapper_imports_off_no_deprecated_sentence_transformers_path():
    """``sentence_transformers.models.InputModule`` is doubly deprecated (package AND module); the
    live path is ``sentence_transformers.base.modules.input_module``."""
    module = _reimport_under_deprecation_errors()
    assert issubclass(module.PreloadedTransformer, InputModule)


def test_the_wrapper_declares_no_method_name_st_has_retired():
    """Read off ST's OWN rename table rather than a hand-kept list, so the next rename is caught by
    the same assertion. Declaring a retired name is not a warning here — ST resolves it through
    ``__getattr__`` only while the alias survives."""
    module = _reimport_under_deprecation_errors()
    declared = set(vars(module.PreloadedTransformer))
    retired = declared & set(Module._DEPRECATED_METHOD_RENAMES)
    assert not retired, f"PreloadedTransformer declares retired ST method name(s): {sorted(retired)}"


def test_the_wrapper_preprocesses_rather_than_tokenizes():
    """``InputModule.tokenize`` is the retired entry point: a module that overrides it instead of
    ``preprocess`` is reached through a deprecation fallback, which drops the per-batch ``max_length``
    sentence-transformers' collator passes and mangles a prompted text pair."""
    module = _reimport_under_deprecation_errors()
    assert "preprocess" in vars(module.PreloadedTransformer)
    assert module.PreloadedTransformer.tokenize is InputModule.tokenize


def test_the_wrapper_answers_the_dimension_query_sentence_transformer_makes():
    """``SentenceTransformer.get_embedding_dimension`` walks its modules asking for this name first;
    a wrapper that answers only under a retired alias rides the deprecation shim to get found."""
    module = _reimport_under_deprecation_errors()
    assert "get_embedding_dimension" in vars(module.PreloadedTransformer)


def test_the_collator_length_and_the_prompt_reach_the_tokenizer():
    """sentence-transformers' training collator passes ``max_length`` per batch and the trainer's
    ``prompts`` as ``prompt``; both must shape the tokens as they do on ST's own ``Transformer``."""
    module = _reimport_under_deprecation_errors()
    backend = Tokenizer(models.WordLevel({"[PAD]": 0, "[UNK]": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]")
    wrapper = module.PreloadedTransformer(Qwen3Model(Qwen3Config(**TINY_QWEN3_CONFIG)), tokenizer, max_seq_length=8)
    model = SentenceTransformer(modules=[wrapper], device="cpu")
    text = " ".join(["word"] * 20)

    assert model.preprocess([text])["input_ids"].shape[1] == 8
    assert model.preprocess([text], max_length=4)["input_ids"].shape[1] == 4
    assert model.preprocess([("a b", "c")], prompt="p q ")["input_ids"].shape[1] == 5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
