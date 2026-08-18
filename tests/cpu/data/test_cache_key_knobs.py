#!/usr/bin/env python
"""Cache-key sensitivity tests for the REAL create_llm_processor closures.

The text/LLM SFT path builds its map function with create_llm_processor, which captures
output-affecting knobs (conversation_field, system_prompt, add_generation_prompt,
max_length, ...) as closure FREE VARIABLES — not fn_kwargs. The deterministic cache file
name therefore depends entirely on _get_closure_fingerprint capturing those scalars:
a knob that changes tokenized output but is invisible to the cache key makes two
distinct runs collide on one cache file and silently reuse stale tokens.

These tests drive the actual factory (not a hand-rolled stand-in) through
_build_cache_file_name and assert:
  - changing each output-affecting knob CHANGES the cache name, and
  - a captured-but-output-irrelevant dict does NOT (the negative control below).

Run: pytest tests/cpu/data/test_cache_key_knobs.py
"""

from datasets import Dataset

from src.data.pipeline.processing import _build_cache_file_name, _get_closure_fingerprint
from src.data.pipeline.row_processors import create_llm_processor


class _MockTokenizer:
    """Minimal tokenizer with the surface create_llm_processor + the fingerprinter need."""

    name_or_path = "org/mock-model"
    vocab_size = 32000
    bos_token = None
    chat_template = "{{ messages }}"

    def __len__(self):
        return self.vocab_size

    def __call__(self, text, **kwargs):
        # Deterministic, length-stable; create_llm_processor only inspects len(input_ids).
        return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    def apply_chat_template(self, conversation, **kwargs):
        return "formatted"


def _ds() -> Dataset:
    return Dataset.from_dict({"value": list(range(8))})


def _name_for(**proc_kwargs) -> str:
    """Build a processor with the given knobs and return its cache file name."""
    tok = _MockTokenizer()
    proc = create_llm_processor(tok, max_length=512, **proc_kwargs)
    return _build_cache_file_name("map", proc, _ds(), "tokenize", {})


def test_conversation_field_changes_cache_name():
    """conversation_field 'messages' vs 'conversation' selects a different source column —
    different tokens — so the cache name MUST differ."""
    name_messages = _name_for(conversation_field="messages")
    name_conversation = _name_for(conversation_field="conversation")
    assert name_messages != name_conversation, (
        "conversation_field change did not change the cache name — runs over different "
        "source columns would collide on one stale cache file"
    )


def test_system_prompt_changes_cache_name():
    """A None vs a concrete system_prompt prepends different tokens; cache name MUST differ."""
    name_none = _name_for(system_prompt=None)
    name_set = _name_for(system_prompt="You are a helpful assistant.")
    assert name_none != name_set, "system_prompt change did not change the cache name"


def test_add_generation_prompt_changes_cache_name():
    """add_generation_prompt True vs False changes the templated string (and which turns are
    kept); cache name MUST differ."""
    name_true = _name_for(add_generation_prompt=True)
    name_false = _name_for(add_generation_prompt=False)
    assert name_true != name_false, "add_generation_prompt change did not change the cache name"


def test_max_length_changes_cache_name():
    """max_length is the truncation bound — different output for long rows; cache name MUST differ."""
    tok = _MockTokenizer()
    proc_512 = create_llm_processor(tok, max_length=512, conversation_field="messages")
    proc_4096 = create_llm_processor(tok, max_length=4096, conversation_field="messages")
    name_512 = _build_cache_file_name("map", proc_512, _ds(), "tokenize", {})
    name_4096 = _build_cache_file_name("map", proc_4096, _ds(), "tokenize", {})
    assert name_512 != name_4096, "max_length change did not change the cache name"


def test_identical_knobs_same_cache_name():
    """Two processors with identical knobs must reuse the same cache (determinism)."""
    name_a = _name_for(conversation_field="messages", system_prompt="sys")
    name_b = _name_for(conversation_field="messages", system_prompt="sys")
    assert name_a == name_b, "identical processor knobs produced different cache names (no reuse)"


def test_irrelevant_captured_object_does_not_change_cache_name():
    """A captured dict that is neither a scalar nor a tokenizer (the rejection sentinel
    create_llm_processor builds) must NOT reach the closure fingerprint: it does not affect the
    tokenized output of valid rows, and keying on it would needlessly re-tokenize the corpus.

    The negative control for the tests above — pinned on _get_closure_fingerprint directly, since
    the sentinel is derived from the tokenizer and no longer settable from outside."""

    def make(max_length, sentinel):
        def process_row(row):
            return sentinel if len(row) > max_length else row

        return process_row

    fp_a = _get_closure_fingerprint(make(512, {"input_ids": None}))
    fp_b = _get_closure_fingerprint(make(512, {"input_ids": None, "extra": None}))
    assert fp_a, "the captured scalar must reach the fingerprint at all"
    assert fp_a == fp_b, (
        "an output-irrelevant captured dict changed the closure fingerprint — the cache key is "
        "keying on non-output-affecting captures and would needlessly re-tokenize"
    )
    assert fp_a != _get_closure_fingerprint(make(4096, {"input_ids": None})), (
        "the control is inert: the fingerprint ignored the output-affecting scalar too"
    )


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
