#!/usr/bin/env python
"""``PreloadedTransformer`` must sit on sentence-transformers' current module surface.
A retired name works silently until a release drops it, and then every loss sized off the dimension is built wrong."""

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
    """Import the wrapper afresh with ``DeprecationWarning`` as an error.
    ST warns from its import hook, so a cached ``sentence_transformers.models`` would make the assert vacuous."""
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
    """``sentence_transformers.models.InputModule`` is doubly deprecated (package and module); the live path is
    ``sentence_transformers.base.modules.input_module``."""
    module = _reimport_under_deprecation_errors()
    assert issubclass(module.PreloadedTransformer, InputModule)


def test_the_wrapper_declares_no_method_name_st_has_retired():
    """Read off ST's own rename table, so the next rename is caught.
    ST resolves a retired name through ``__getattr__`` only while the alias survives."""
    module = _reimport_under_deprecation_errors()
    declared = set(vars(module.PreloadedTransformer))
    retired = declared & set(Module._DEPRECATED_METHOD_RENAMES)
    assert not retired, f"PreloadedTransformer declares retired ST method name(s): {sorted(retired)}"


def test_the_wrapper_preprocesses_rather_than_tokenizes():
    """Overriding the retired ``InputModule.tokenize`` goes through a deprecation fallback, which drops the collator's
    per-batch ``max_length`` and mangles a prompted text pair."""
    module = _reimport_under_deprecation_errors()
    assert "preprocess" in vars(module.PreloadedTransformer)
    assert module.PreloadedTransformer.tokenize is InputModule.tokenize


def test_the_wrapper_answers_the_dimension_query_sentence_transformer_makes():
    """``SentenceTransformer.get_embedding_dimension`` asks its modules for this name first; an alias rides a shim."""
    module = _reimport_under_deprecation_errors()
    assert "get_embedding_dimension" in vars(module.PreloadedTransformer)


def test_the_collator_length_and_the_prompt_reach_the_tokenizer():
    """ST's training collator passes ``max_length`` per batch and the trainer's ``prompts`` as ``prompt``."""
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
