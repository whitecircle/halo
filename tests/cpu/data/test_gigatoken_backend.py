#!/usr/bin/env python3
"""Tests for the gigatoken tokenizer backend.

Tests needing the real Rust encoder skip when the optional extra is absent; the equivalence-probe
tests at the end drive a fake backend instead, so the guard that rejects a divergent tokenizer is
pinned in every environment (including images built before the extra landed)."""

import pickle
import sys

import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from src.data.pipeline import tokenizer_backend as tb
from src.data.pipeline.preprocessed_metadata import PreprocessingConfig
from src.data.pipeline.preprocessing import tokenize_dataset
from src.data.pipeline.tokenizer_backend import (
    TOKENIZER_BACKENDS,
    resolve_processor_backend,
    resolve_tokenizer_backend,
)

MODEL_NAME = "Qwen/Qwen3-0.6B"
VLM_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
VLM_MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"  # pin: hub main can drift


class PreprocessingArgsStub:
    """Minimal CommonScriptArguments stand-in for setup_model_and_tokenizer."""

    pad_token = bos_token = eos_token = chat_template = added_special_tokens = None
    force_chat_template = False

    def __init__(self, tokenizer_backend: str) -> None:
        self.tokenizer_backend = tokenizer_backend


CONVERSATIONS = [
    [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you! Ünïcodé 🚀"},
    ],
    [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
    ],
]


def _load_tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:  # offline / no cached snapshot
        pytest.skip(f"tokenizer unavailable offline: {e}")


def test_unknown_backend_rejected():
    """An unknown backend name must fail loud, listing the valid choices."""
    with pytest.raises(ValueError, match="tokenizer_backend"):
        resolve_tokenizer_backend(object(), "typo")


def test_hf_backend_is_passthrough():
    """backend='hf' must return the tokenizer itself (no wrapping, no probes)."""
    sentinel = object()
    assert resolve_tokenizer_backend(sentinel, "hf") is sentinel


def test_vlm_processor_backend_parity():
    """A processor with the gigatoken proxy installed must produce identical input_ids
    (image pads included), pixel_values, and grid to the stock processor."""
    pytest.importorskip("gigatoken")
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(VLM_MODEL_NAME, revision=VLM_MODEL_REVISION)
    except Exception as e:  # offline / no cached snapshot
        pytest.skip(f"VLM processor unavailable offline: {e}")

    img = Image.fromarray(np.full((64, 64, 3), [100, 150, 200], dtype=np.uint8))
    conv = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color? Ünïcodé 🚀"}]},
        {"role": "assistant", "content": "Blue-ish."},
    ]
    text = processor.apply_chat_template(conv, tokenize=False)
    base = processor(text=[text], images=[img], return_tensors="pt")

    resolve_processor_backend(processor, "gigatoken")
    swapped = processor(text=[text], images=[img], return_tensors="pt")

    assert swapped["input_ids"].tolist() == base["input_ids"].tolist()
    assert (swapped["pixel_values"] == base["pixel_values"]).all()
    assert swapped["image_grid_thw"].tolist() == base["image_grid_thw"].tolist()


def test_proxy_matches_hf_tokenizer():
    """Proxy token IDs must equal the HF tokenizer's on chat-rendered text (special tokens
    included), plain text, and under truncation; fingerprinted metadata must delegate."""
    pytest.importorskip("gigatoken")
    tokenizer = _load_tokenizer()
    proxy = resolve_tokenizer_backend(tokenizer, "gigatoken")

    rendered = tokenizer.apply_chat_template(CONVERSATIONS[0], tokenize=False)
    for text in (rendered, "plain text with numbers 12345", "Ünïcodé — em—dash… 🚀🔥"):
        for add_special_tokens in (False, True):
            expected = tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]
            actual = list(proxy(text, add_special_tokens=add_special_tokens)["input_ids"])
            assert actual == expected, f"proxy diverges on {text[:40]!r}"

    expected = tokenizer(rendered, truncation=True, max_length=16)["input_ids"]
    actual = list(proxy(rendered, truncation=True, max_length=16)["input_ids"])
    assert actual == expected, "proxy diverges under truncation"

    # The dataset cache fingerprint reads these off the tokenizer — they must delegate.
    assert proxy.name_or_path == tokenizer.name_or_path
    assert proxy.vocab_size == tokenizer.vocab_size
    assert len(proxy) == len(tokenizer)
    assert proxy.chat_template == tokenizer.chat_template
    assert proxy.eos_token_id == tokenizer.eos_token_id


def test_verifier_catches_truncation_only_divergence(monkeypatch):
    """The startup verifier must probe truncated encodes — production paths tokenize with
    truncation=True, so a backend diverging only there must be rejected at resolve time."""
    pytest.importorskip("gigatoken")
    from src.data.pipeline import tokenizer_backend as tb

    tokenizer = _load_tokenizer()
    real_call = tb.GigatokenTokenizerProxy.__call__

    def diverge_when_truncating(self, text=None, **kwargs):
        out = real_call(self, text, **kwargs)
        if kwargs.get("truncation"):
            out["input_ids"] = list(out["input_ids"])[:-1]
        return out

    monkeypatch.setattr(tb.GigatokenTokenizerProxy, "__call__", diverge_when_truncating)
    with pytest.raises(ValueError, match="diverges"):
        tb.resolve_tokenizer_backend(tokenizer, "gigatoken")


def test_proxy_pickle_roundtrip():
    """datasets.map workers pickle the processor closure; the proxy must survive and re-encode
    identically (the Rust backend is dropped and rebuilt lazily)."""
    pytest.importorskip("gigatoken")
    tokenizer = _load_tokenizer()
    proxy = resolve_tokenizer_backend(tokenizer, "gigatoken")

    restored = pickle.loads(pickle.dumps(proxy))
    text = tokenizer.apply_chat_template(CONVERSATIONS[1], tokenize=False)
    assert list(restored(text)["input_ids"]) == tokenizer(text)["input_ids"]


def test_proxy_transparency():
    """Trainers receive the proxy as processing_class: isinstance must report the wrapped type,
    attribute writes must reach the wrapped tokenizer (so its bound methods see them), and
    re-resolving must be a no-op."""
    pytest.importorskip("gigatoken")
    tokenizer = _load_tokenizer()
    proxy = resolve_tokenizer_backend(tokenizer, "gigatoken")

    assert isinstance(proxy, type(tokenizer))

    original_side = tokenizer.padding_side
    proxy.padding_side = "left"
    assert tokenizer.padding_side == "left"
    tokenizer.padding_side = original_side

    assert resolve_tokenizer_backend(proxy, "gigatoken") is proxy


def test_setup_model_and_tokenizer_resolves_backend():
    """setup_model_and_tokenizer must return the backend-resolved tokenizer — the single seam
    through which every training script gets the gigatoken proxy."""
    pytest.importorskip("gigatoken")
    from src.data.pipeline.tokenizer_backend import GigatokenTokenizerProxy
    from src.models.loading.tokenizer_setup import setup_model_and_tokenizer

    tokenizer = _load_tokenizer()
    hf_args = PreprocessingArgsStub("hf")
    assert setup_model_and_tokenizer(hf_args, None, tokenizer, 512) is tokenizer

    proxy = setup_model_and_tokenizer(PreprocessingArgsStub("gigatoken"), None, tokenizer, 512)
    assert type(proxy) is GigatokenTokenizerProxy
    text = tokenizer.apply_chat_template(CONVERSATIONS[0], tokenize=False)
    assert list(proxy(text)["input_ids"]) == tokenizer(text)["input_ids"]


def test_reward_preprocess_backend_parity():
    """The Bradley-Terry reward map (shared non-SFT tokenization path) must produce identical
    rows with the proxy."""
    pytest.importorskip("gigatoken")
    from src.data.pipeline.preferences import build_reward_preprocess_fn

    tokenizer = _load_tokenizer()
    proxy = resolve_tokenizer_backend(tokenizer, "gigatoken")
    examples = {
        "prompt": [[{"role": "user", "content": "Best emoji? 🚀"}]],
        "chosen": [[{"role": "assistant", "content": "Rocket."}]],
        "rejected": [[{"role": "assistant", "content": "None."}]],
    }
    base = build_reward_preprocess_fn(tokenizer, max_length=512)(examples)
    swapped = build_reward_preprocess_fn(proxy, max_length=512)(examples)
    assert {k: [list(v) for v in vs] for k, vs in swapped.items()} == base


@pytest.mark.parametrize("mode", ["chat", "text"])
def test_tokenize_dataset_backend_parity(mode):
    """tokenize_dataset must produce identical rows under both backends, chat and text mode."""
    pytest.importorskip("gigatoken")
    tokenizer = _load_tokenizer()

    if mode == "chat":
        dataset = Dataset.from_dict({"conversation": CONVERSATIONS})
        # The config default is the training-side "prompt"; this fixture names its column otherwise.
        extra = {"conversation_field": "conversation"}
    else:
        dataset = Dataset.from_dict({"text": ["A raw pretraining document.\n\nWith newlines.", "Another one. 🚀"]})
        extra = {"mode": "text"}

    results = {}
    for backend in TOKENIZER_BACKENDS:
        config = PreprocessingConfig(
            model_name_or_path=MODEL_NAME,
            max_length=512,
            tokenizer_backend=backend,
            **extra,
        )
        tokenized = tokenize_dataset(dataset, tokenizer, config)
        results[backend] = tokenized[:]

    assert results["gigatoken"]["input_ids"] == results["hf"]["input_ids"]
    assert results["gigatoken"]["attention_mask"] == results["hf"]["attention_mask"]
    assert results["gigatoken"]["labels"] == results["hf"]["labels"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


class _FakeGigatoken:
    """Stand-in for the optional ``gigatoken`` module.

    ``Tokenizer(hf).as_hf()`` returns a callable that re-encodes with the wrapped HF tokenizer, then
    applies ``corrupt`` to the ids. ``corrupt=None`` is a faithful backend.
    """

    def __init__(self, corrupt=None):
        self._corrupt = corrupt
        outer = self

        class Tokenizer:
            def __init__(self, hf):
                self._hf = hf

            def as_hf(self):
                def _encode(text, **kwargs):
                    out = dict(self._hf(text, **kwargs))
                    if outer._corrupt is not None:
                        out["input_ids"] = outer._corrupt(list(out["input_ids"]), kwargs)
                    return out

                return _encode

        self.Tokenizer = Tokenizer


def test_a_faithful_backend_is_accepted_without_the_package(monkeypatch):
    """Anti-vacuity for the divergence test below: with an id-identical fake backend the verifier
    must ACCEPT, so a verifier that rejected everything could not pass both tests."""
    monkeypatch.setattr(tb, "gigatoken", _FakeGigatoken())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    resolved = tb.resolve_tokenizer_backend(tokenizer, "gigatoken")
    assert type(resolved) is tb.GigatokenTokenizerProxy


def test_a_backend_that_diverges_only_under_truncation_is_rejected(monkeypatch):
    """The equivalence probe must sweep truncation, not just the plain encode.

    A Rust backend that agrees untruncated but drops or keeps one token at the ``max_length``
    boundary silently shifts every label in a packed dataset. Dropping the truncation leg of
    ``_verify_backend_equivalence`` makes this pass — and needs no gigatoken install to catch.
    """
    monkeypatch.setattr(
        tb, "gigatoken", _FakeGigatoken(corrupt=lambda ids, kw: ids[:-1] if kw.get("truncation") and ids else ids)
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    with pytest.raises(ValueError, match="diverges"):
        tb.resolve_tokenizer_backend(tokenizer, "gigatoken")


def test_a_backend_that_diverges_on_special_tokens_is_rejected(monkeypatch):
    """The probe must sweep ``add_special_tokens`` in both states: a backend that mishandles BOS/EOS
    shifts every sequence by one token."""
    monkeypatch.setattr(
        tb,
        "gigatoken",
        _FakeGigatoken(corrupt=lambda ids, kw: ([0] + ids) if kw.get("add_special_tokens") else ids),
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    with pytest.raises(ValueError, match="diverges"):
        tb.resolve_tokenizer_backend(tokenizer, "gigatoken")
