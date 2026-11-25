#!/usr/bin/env python
"""Cache-key sensitivity of the real ``build_chat_row_processor`` closures, whose output-affecting knobs are free
variables and reach the file name only through ``_closure_fingerprint``. A knob invisible to the key makes two
distinct runs silently share stale tokens."""

import pytest
from datasets import Dataset

from halo.data.pipeline.map_cache_key import _closure_fingerprint, map_cache_file_name
from halo.data.pipeline.text_rows import build_chat_row_processor


class _MockTokenizer:
    """Minimal tokenizer with the surface build_chat_row_processor + the fingerprinter need."""

    name_or_path = "org/mock-model"
    vocab_size = 32000
    bos_token = None
    chat_template = "{{ messages }}"

    def __len__(self):
        return self.vocab_size

    def __call__(self, text, **kwargs):
        # Deterministic, length-stable; build_chat_row_processor only inspects len(input_ids).
        return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    def apply_chat_template(self, conversation, **kwargs):
        return "formatted"


def _ds() -> Dataset:
    return Dataset.from_dict({"value": list(range(8))})


def _name_for(**proc_kwargs) -> str:
    tok = _MockTokenizer()
    knobs = {"conversation_field": "messages", "eos_token_ids": frozenset(), **proc_kwargs}
    proc = build_chat_row_processor(tok, max_length=512, **knobs)
    return map_cache_file_name("map", proc, _ds(), "tokenize", {})


def test_conversation_field_changes_cache_name():
    name_messages = _name_for(conversation_field="messages")
    name_conversation = _name_for(conversation_field="conversation")
    assert name_messages != name_conversation, (
        "conversation_field change did not change the cache name — runs over different "
        "source columns would collide on one stale cache file"
    )


def test_system_prompt_changes_cache_name():
    name_none = _name_for(system_prompt=None)
    name_set = _name_for(system_prompt="You are a helpful assistant.")
    assert name_none != name_set, "system_prompt change did not change the cache name"


def test_add_generation_prompt_changes_cache_name():
    name_true = _name_for(add_generation_prompt=True)
    name_false = _name_for(add_generation_prompt=False)
    assert name_true != name_false, "add_generation_prompt change did not change the cache name"


def test_max_length_changes_cache_name():
    tok = _MockTokenizer()
    knobs = {"conversation_field": "messages", "eos_token_ids": frozenset()}
    proc_512 = build_chat_row_processor(tok, max_length=512, **knobs)
    proc_4096 = build_chat_row_processor(tok, max_length=4096, **knobs)
    name_512 = map_cache_file_name("map", proc_512, _ds(), "tokenize", {})
    name_4096 = map_cache_file_name("map", proc_4096, _ds(), "tokenize", {})
    assert name_512 != name_4096, "max_length change did not change the cache name"


def test_eos_token_ids_change_cache_name():
    """The terminators decide whether a row's last turn is closed, so two models on one tokenizer that list
    different ones tokenize a corpus differently."""
    name_eos_only = _name_for(eos_token_ids=frozenset({2}))
    name_with_role_marker = _name_for(eos_token_ids=frozenset({2, 5}))
    assert name_eos_only != name_with_role_marker, "eos_token_ids change did not change the cache name"
    assert name_eos_only == _name_for(eos_token_ids=frozenset({2})), "one terminator set produced two cache names"


def test_irrelevant_captured_object_does_not_change_cache_name():
    """A captured dict that is neither scalar nor tokenizer (the rejection sentinel) must not reach the fingerprint,
    or the corpus re-tokenizes needlessly. Pinned on ``_closure_fingerprint``, since the sentinel is not settable
    from outside."""

    def make(max_length, sentinel):
        def process_row(row):
            return sentinel if len(row) > max_length else row

        return process_row

    fp_a = _closure_fingerprint(make(512, {"input_ids": None}))
    fp_b = _closure_fingerprint(make(512, {"input_ids": None, "extra": None}))
    assert fp_a, "the captured scalar must reach the fingerprint at all"
    assert fp_a == fp_b, (
        "an output-irrelevant captured dict changed the closure fingerprint — the cache key is "
        "keying on non-output-affecting captures and would needlessly re-tokenize"
    )
    assert fp_a != _closure_fingerprint(make(4096, {"input_ids": None})), (
        "the control is inert: the fingerprint ignored the output-affecting scalar too"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
