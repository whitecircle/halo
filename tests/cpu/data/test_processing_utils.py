#!/usr/bin/env python
"""
Tests for dataset processing fingerprinting utilities:
get_function_identifier(), _get_kwargs_fingerprint(), maybe_parse_json(),
tools_field threading, and cache-key invalidation.

Run: python tests/cpu/data/test_processing_utils.py
"""

import functools
import json

import pytest

from src.data.pipeline import row_processors
from src.data.pipeline.conversation import maybe_parse_json
from src.data.pipeline.preferences import apply_chat_template_to_preference_data
from src.data.pipeline.processing import _build_cache_file_name, _get_kwargs_fingerprint, get_function_identifier
from src.data.pipeline.row_processors import (
    apply_chat_template_to_conversations,
    create_llm_processor,
    create_vlm_processor,
    prepare_generative_row,
)

# get_function_identifier tests


def _helper_func_a(x):
    return x + 1


def _helper_func_b(x):
    return x * 2


def test_function_id_deterministic():
    """Same function should always produce the same identifier."""
    id1 = get_function_identifier(_helper_func_a)
    id2 = get_function_identifier(_helper_func_a)
    assert id1 == id2, f"Expected same ID, got '{id1}' and '{id2}'"


def test_function_id_different_functions():
    """Different functions should produce different identifiers."""
    id_a = get_function_identifier(_helper_func_a)
    id_b = get_function_identifier(_helper_func_b)
    assert id_a != id_b, f"Expected different IDs, both got '{id_a}'"


def test_function_id_contains_name():
    """Identifier should contain the function name."""
    fid = get_function_identifier(_helper_func_a)
    assert "_helper_func_a" in fid, f"Expected '_helper_func_a' in '{fid}'"


def test_function_id_with_lambda():
    """Lambda identifiers carry the '<lambda>' name and a code hash."""
    # A raw lambda's __name__ is "<lambda>" (a `def`, by contrast, is named after its identifier).
    fid = get_function_identifier(lambda x: x + 10)
    assert isinstance(fid, str)
    # Source-hash path yields "<name>_<hash>"; the lambda's __name__ is "<lambda>".
    assert fid.startswith("<lambda>_"), f"Expected '<lambda>_...' prefix, got '{fid}'"
    assert len(fid) > len("<lambda>_")  # a non-empty hash is appended


def test_function_id_nested_function():
    """Nested (local) function should produce a valid identifier."""

    def inner_func(y):
        return y**2

    fid = get_function_identifier(inner_func)
    assert isinstance(fid, str)
    assert "inner_func" in fid


def test_function_id_partial_distinguishes_bound_tokenizer():
    """functools.partial binding different tokenizers must produce different identifiers.

    A partial exposes no source/__code__ and no __closure__, so an identifier that reads only those
    collapses to the constant "functools.unknown": two partials over the same dataset binding
    different models' tokenizers then share one cache file and reuse the wrong token IDs.
    """
    p_a = functools.partial(prepare_generative_row, tokenizer=_StubTokenizer("modelA", 201088), max_length=4096)
    p_b = functools.partial(prepare_generative_row, tokenizer=_StubTokenizer("modelB", 262155), max_length=4096)
    id_a = get_function_identifier(p_a)
    id_b = get_function_identifier(p_b)
    assert id_a != id_b, f"partials binding different tokenizers collided on '{id_a}'"
    assert "prepare_generative_row" in id_a, f"expected wrapped func name in '{id_a}'"


def test_function_id_partial_distinguishes_bound_scalar():
    """functools.partial binding a different max_length must change the identifier."""
    tok = _StubTokenizer("modelA", 201088)
    p_short = functools.partial(prepare_generative_row, tokenizer=tok, max_length=4096)
    p_long = functools.partial(prepare_generative_row, tokenizer=tok, max_length=36000)
    assert get_function_identifier(p_short) != get_function_identifier(p_long)


def test_function_id_partial_deterministic():
    """Same wrapped func + same bound values => same identifier (cross-rank cache stability)."""

    def mk():
        return functools.partial(prepare_generative_row, tokenizer=_StubTokenizer("modelA", 201088), max_length=4096)

    assert get_function_identifier(mk()) == get_function_identifier(mk())


# _get_kwargs_fingerprint tests


def test_kwargs_fingerprint_empty():
    """Empty kwargs should return empty string."""
    result = _get_kwargs_fingerprint({})
    assert result == "", f"Expected '', got '{result}'"


def test_kwargs_fingerprint_distinguishes_name_or_path():
    """Two tokenizers with different name_or_path must NOT collide on the same fingerprint.

    This is the load-bearing behavior: the cache key must change when the tokenizer changes,
    otherwise one model's cached token IDs are reused for another (OOB embedding gather).
    """

    class _Tok:
        def __init__(self, nop):
            self.name_or_path = nop

    fp_a = _get_kwargs_fingerprint({"tokenizer": _Tok("Qwen/Qwen3-4B")})
    fp_b = _get_kwargs_fingerprint({"tokenizer": _Tok("meta-llama/Llama-3-8B")})
    assert fp_a != fp_b, "Different tokenizers must produce different fingerprints"


def test_kwargs_fingerprint_distinguishes_vocab_size():
    """Configs differing only in vocab_size must produce different fingerprints."""

    class _Cfg:
        def __init__(self, vs):
            self.vocab_size = vs

    fp_a = _get_kwargs_fingerprint({"config": _Cfg(32000)})
    fp_b = _get_kwargs_fingerprint({"config": _Cfg(151936)})
    assert fp_a != fp_b


def test_kwargs_fingerprint_distinguishes_chat_template():
    """Tokenizers identical except for chat_template must NOT collide on the fn_kwargs/partial path.

    A --chat_template/--force_chat_template override mutates tokenizer.chat_template in place
    without changing name_or_path/vocab_size, so the tokenizer-identity key must fold the template
    in — otherwise a partial-bound tokenizer reuses another template's cached tokens.
    """

    class _Tok:
        def __init__(self, template):
            self.name_or_path = "same/model"
            self.vocab_size = 1000
            self.chat_template = template

    fp_a = _get_kwargs_fingerprint({"tokenizer": _Tok("{{ messages }}")})
    fp_b = _get_kwargs_fingerprint({"tokenizer": _Tok("{{ messages }}<eot>")})
    fp_a2 = _get_kwargs_fingerprint({"tokenizer": _Tok("{{ messages }}")})
    assert fp_a != fp_b, "Different chat_template must produce different fingerprints"
    assert fp_a == fp_a2, "Same chat_template must be stable"


def test_kwargs_fingerprint_distinguishes_primitive_values():
    """Different primitive values for the same key change the fingerprint."""
    assert _get_kwargs_fingerprint({"max_length": 1024}) != _get_kwargs_fingerprint({"max_length": 4096})


def test_kwargs_fingerprint_opaque_object_uses_type_name():
    """An object with neither name_or_path nor vocab_size nor primitive falls back to its type name.

    Two distinct instances of the same opaque type therefore share a fingerprint (only the type
    name is captured), while a different type changes it.
    """

    class _OpaqueA:
        pass

    class _OpaqueB:
        pass

    fp_a1 = _get_kwargs_fingerprint({"obj": _OpaqueA()})
    fp_a2 = _get_kwargs_fingerprint({"obj": _OpaqueA()})
    fp_b = _get_kwargs_fingerprint({"obj": _OpaqueB()})
    assert fp_a1 == fp_a2, "Same opaque type → same fingerprint (only type name captured)"
    assert fp_a1 != fp_b, "Different opaque type → different fingerprint"


def test_kwargs_fingerprint_sorted_keys():
    """Key order should not affect fingerprint (keys are sorted internally)."""
    fp1 = _get_kwargs_fingerprint({"a": 1, "b": 2, "c": 3})
    fp2 = _get_kwargs_fingerprint({"c": 3, "a": 1, "b": 2})
    assert fp1 == fp2, f"Different key orders should produce same fingerprint: '{fp1}' vs '{fp2}'"


# maybe_parse_json tests


def test_maybe_parse_json_passthrough_list():
    """Already-parsed list values should be returned as-is."""
    value = [{"role": "user", "content": "hi"}]
    assert maybe_parse_json(value) is value


def test_maybe_parse_json_passes_through_invalid_string():
    """A non-JSON string is returned unchanged (the documented fallback): a mis-specified
    conversation_field then surfaces as a clear chat-template error downstream instead of an opaque
    JSONDecodeError deep inside a multiprocess map worker. Valid JSON still parses."""
    assert maybe_parse_json("{not valid json") == "{not valid json"
    assert maybe_parse_json("plain text") == "plain text"
    assert maybe_parse_json("") == ""
    # Valid JSON strings still decode to Python objects...
    assert maybe_parse_json('{"role": "user"}') == {"role": "user"}
    assert maybe_parse_json("[1, 2]") == [1, 2]
    # ...and non-string values pass through unchanged.
    assert maybe_parse_json([{"role": "user"}]) == [{"role": "user"}]


# tools_field threading tests (apply_chat_template_to_conversations + processor)


class _RecordingTokenizer:
    """Minimal tokenizer stub that records apply_chat_template kwargs.

    The recorded calls let us assert that ``tools`` is forwarded only when the
    dataset row carries a non-empty value at ``tools_field``.
    """

    bos_token = None
    pad_token = "<pad>"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        self.calls.append({"messages": messages, "tokenize": tokenize, **kwargs})
        # Return a deterministic string that depends on tools so callers can
        # observe propagation if they care.
        tools = kwargs.get("tools")
        suffix = f"|tools={len(tools)}" if tools else "|no-tools"
        # Per-message suffix keeps the render prefix-consistent (template(prompt) is a prefix of
        # template(prompt + completion)) — the invariant the preference helper enforces.
        return "".join(m["content"] + suffix for m in messages)

    def __call__(self, text, **kwargs):
        # Trivial whitespace tokenizer for length checks.
        ids = list(range(len(text.split())))
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_apply_chat_template_forwards_tools_when_present():
    tokenizer = _RecordingTokenizer()
    tools = [{"type": "function", "function": {"name": "calc"}}]
    row = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": tools,
    }
    apply_chat_template_to_conversations(row, tokenizer, tools_field="tools")
    assert tokenizer.calls and tokenizer.calls[-1].get("tools") == tools


def test_apply_chat_template_omits_tools_when_field_missing():
    tokenizer = _RecordingTokenizer()
    row = {"messages": [{"role": "user", "content": "hi"}]}
    apply_chat_template_to_conversations(row, tokenizer, tools_field="tools")
    assert tokenizer.calls and "tools" not in tokenizer.calls[-1]


def test_apply_chat_template_parses_json_tools():
    tokenizer = _RecordingTokenizer()
    tools = [{"type": "function", "function": {"name": "calc"}}]
    row = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": json.dumps(tools),
    }
    apply_chat_template_to_conversations(row, tokenizer, tools_field="tools")
    assert tokenizer.calls[-1].get("tools") == tools


def test_apply_chat_template_parses_json_messages():
    """Conversation column stored as JSON string should be decoded transparently."""
    tokenizer = _RecordingTokenizer()
    row = {
        "messages": json.dumps([{"role": "user", "content": "hi"}]),
    }
    apply_chat_template_to_conversations(row, tokenizer)
    assert tokenizer.calls[-1]["messages"] == [{"role": "user", "content": "hi"}]


def test_create_llm_processor_threads_tools():
    """create_llm_processor must propagate tools_field into the closure."""
    tokenizer = _RecordingTokenizer()
    tools = [{"type": "function", "function": {"name": "calc"}}]
    processor = create_llm_processor(
        tokenizer=tokenizer,
        max_length=128,
        tools_field="tools",
    )
    row = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": tools,
    }
    processor(row)
    # First call is the length-check apply_chat_template.
    assert any(call.get("tools") == tools for call in tokenizer.calls)


def test_preference_helper_threads_tools():
    """apply_chat_template_to_preference_data forwards tools to all 3 fields."""
    tokenizer = _RecordingTokenizer()
    tools = [{"type": "function", "function": {"name": "calc"}}]
    row = {
        "prompt": [{"role": "user", "content": "q"}],
        "chosen": [{"role": "assistant", "content": "a1"}],
        "rejected": [{"role": "assistant", "content": "a2"}],
        "tools": tools,
    }
    apply_chat_template_to_preference_data(row, tokenizer, tools_field="tools")
    # Three apply_chat_template calls (prompt/chosen/rejected) — all must carry tools.
    tools_calls = [c for c in tokenizer.calls if c.get("tools") == tools]
    assert len(tools_calls) == 3, f"Expected 3 calls with tools, got {len(tools_calls)} of {len(tokenizer.calls)}"


def test_preference_helper_omits_tools_when_field_unset():
    """No tools_field arg => no tools kwarg sent."""
    tokenizer = _RecordingTokenizer()
    row = {
        "prompt": [{"role": "user", "content": "q"}],
        "chosen": [{"role": "assistant", "content": "a1"}],
        "rejected": [{"role": "assistant", "content": "a2"}],
    }
    apply_chat_template_to_preference_data(row, tokenizer)
    assert all("tools" not in c for c in tokenizer.calls)


def test_preference_helper_parses_json_tools():
    """JSON-string tools column must be decoded before forwarding."""
    tokenizer = _RecordingTokenizer()
    tools = [{"type": "function", "function": {"name": "calc"}}]
    row = {
        "prompt": [{"role": "user", "content": "q"}],
        "chosen": [{"role": "assistant", "content": "a1"}],
        "rejected": [{"role": "assistant", "content": "a2"}],
        "tool_specs": json.dumps(tools),
    }
    apply_chat_template_to_preference_data(
        row,
        tokenizer,
        tools_field="tool_specs",
    )
    assert all(c.get("tools") == tools for c in tokenizer.calls)


def test_prepare_generative_row_threads_tools():
    """prepare_generative_row must forward tools_field through to the tokenizer."""
    tokenizer = _RecordingTokenizer()
    tools = [{"type": "function", "function": {"name": "calc"}}]
    row = {
        "prompt": [{"role": "user", "content": "hi"}],
        "tools": tools,
    }
    prepare_generative_row(row, tokenizer, max_length=128, tools_field="tools")
    assert tokenizer.calls and tokenizer.calls[-1].get("tools") == tools


# VLM processor tests (no real tokenizer/collator — exercise the row shape)


def _stub_process_vlm(conversation, system_prompt=None, model_supports_system_role=True):
    """Stub VLM conversation processor: passthrough conversation, no images."""
    return list(conversation), []


@pytest.fixture
def _passthrough_vlm_conversation(monkeypatch):
    """Swap the module-global image extractor for a passthrough, so these row-shape tests do not
    need real PIL payloads."""
    monkeypatch.setattr(row_processors, "process_vlm_conversation", _stub_process_vlm)


def test_vlm_processor_emits_tools_json_when_field_set(_passthrough_vlm_conversation):
    """tools list at row[tools_field] should land as a JSON string on the example."""
    tools = [{"type": "function", "function": {"name": "calc"}}]
    processor = create_vlm_processor(tools_field="tools")
    row = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": tools,
    }
    example = processor(row)
    assert example["tools_json"] is not None
    assert json.loads(example["tools_json"]) == tools


def test_vlm_processor_passes_through_json_string_tools(_passthrough_vlm_conversation):
    """Pre-serialized JSON tool strings should be stored as-is, not double-encoded."""
    tools_str = json.dumps([{"type": "function", "function": {"name": "calc"}}])
    processor = create_vlm_processor(tools_field="tools")
    row = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": tools_str,
    }
    example = processor(row)
    # Stored verbatim — round-trip yields the original list, not a string of a string.
    assert example["tools_json"] == tools_str
    assert isinstance(json.loads(example["tools_json"]), list)


def test_vlm_processor_omits_tools_json_when_field_unset(_passthrough_vlm_conversation):
    """tools_json must always be present (schema stable) and None when no tools."""
    processor = create_vlm_processor()
    row = {"messages": [{"role": "user", "content": "hi"}]}
    example = processor(row)
    assert "tools_json" in example
    assert example["tools_json"] is None


def test_vlm_processor_handles_missing_tools_column(_passthrough_vlm_conversation):
    """Setting tools_field on rows that lack the column should not raise."""
    processor = create_vlm_processor(tools_field="tools")
    row = {"messages": [{"role": "user", "content": "hi"}]}
    example = processor(row)
    assert example["tools_json"] is None


# Cache key invalidation tests


class _StubDataset:
    """Stand-in that satisfies _build_cache_file_name's fingerprinting needs."""

    _fingerprint = "abc123"

    def __len__(self):
        return 10


def test_cache_key_extras_change_filename():
    """Toggling cache_key_extras must change the cache file name."""

    def fn(row):
        return row

    ds = _StubDataset()
    name_a = _build_cache_file_name(
        "map",
        fn,
        ds,
        "desc",
        {},
        cache_key_extras={"tools_field": None},
    )
    name_b = _build_cache_file_name(
        "map",
        fn,
        ds,
        "desc",
        {},
        cache_key_extras={"tools_field": "tools"},
    )
    assert name_a != name_b, f"Expected different cache names when tools_field flips, both got '{name_a}'"


def test_cache_key_extras_stable_for_same_inputs():
    """Same extras must produce the same cache file name (deterministic)."""

    def fn(row):
        return row

    ds = _StubDataset()
    extras = {"tools_field": "tools", "system_prompt": "hello"}
    name_1 = _build_cache_file_name("map", fn, ds, "desc", {}, cache_key_extras=extras)
    name_2 = _build_cache_file_name("map", fn, ds, "desc", {}, cache_key_extras=extras)
    assert name_1 == name_2


def test_cache_key_extras_none_matches_empty_dict():
    """Passing None or {} for cache_key_extras must collapse to the same key."""

    def fn(row):
        return row

    ds = _StubDataset()
    name_none = _build_cache_file_name("map", fn, ds, "desc", {}, cache_key_extras=None)
    name_empty = _build_cache_file_name("map", fn, ds, "desc", {}, cache_key_extras={})
    assert name_none == name_empty


class _StubTokenizer:
    """Minimal tokenizer stand-in for closure-fingerprint tests."""

    def __init__(self, name_or_path, vocab_size):
        self.name_or_path = name_or_path
        self.vocab_size = vocab_size
        self.bos_token = None

    def __len__(self):
        return self.vocab_size

    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    def apply_chat_template(self, *args, **kwargs):
        return "x"


def test_closure_tokenizer_changes_cache_filename():
    """Processors closing over different tokenizers must NOT share a cache file.

    The processor captures the tokenizer as a closure free variable (invisible
    to the source hash and fn_kwargs), so without the closure fingerprint a
    262k-vocab tokenizer's cached token IDs are silently reused for a
    201k-vocab model, gathering out of bounds at the first step.
    """
    ds = _StubDataset()
    proc_a = create_llm_processor(
        tokenizer=_StubTokenizer("modelA", 201088),
        max_length=4096,
        conversation_field="conversation",
        use_padding=False,
    )
    proc_b = create_llm_processor(
        tokenizer=_StubTokenizer("modelB", 262155),
        max_length=4096,
        conversation_field="conversation",
        use_padding=False,
    )
    name_a = _build_cache_file_name("map", proc_a, ds, "Processing dataset", {})
    name_b = _build_cache_file_name("map", proc_b, ds, "Processing dataset", {})
    assert name_a != name_b, f"Different tokenizers collided on '{name_a}'"


def test_closure_max_length_changes_cache_filename():
    """Same tokenizer but a different max_length must change the cache file."""
    ds = _StubDataset()
    tok = _StubTokenizer("modelA", 201088)
    proc_short = create_llm_processor(
        tokenizer=tok,
        max_length=4096,
        conversation_field="conversation",
        use_padding=False,
    )
    proc_long = create_llm_processor(
        tokenizer=tok,
        max_length=36000,
        conversation_field="conversation",
        use_padding=False,
    )
    name_short = _build_cache_file_name("map", proc_short, ds, "Processing dataset", {})
    name_long = _build_cache_file_name("map", proc_long, ds, "Processing dataset", {})
    assert name_short != name_long, f"max_length change collided on '{name_short}'"


def test_closure_chat_template_changes_cache_filename():
    """Same tokenizer identity but a different chat_template must change the cache file.

    Regression: --chat_template/--force_chat_template overrides tokenizer.chat_template in place;
    the closure fingerprint keys the tokenizer by identity, so without folding in the template two
    runs over the same dataset+model differing only in the template silently shared a cache file.
    """
    ds = _StubDataset()
    tok_a = _StubTokenizer("modelA", 201088)
    tok_a.chat_template = "{{ messages }}"
    tok_b = _StubTokenizer("modelA", 201088)
    tok_b.chat_template = "{{ messages }}<|eot|>"
    proc_a = create_llm_processor(
        tokenizer=tok_a, max_length=4096, conversation_field="conversation", use_padding=False
    )
    proc_b = create_llm_processor(
        tokenizer=tok_b, max_length=4096, conversation_field="conversation", use_padding=False
    )
    name_a = _build_cache_file_name("map", proc_a, ds, "Processing dataset", {})
    name_b = _build_cache_file_name("map", proc_b, ds, "Processing dataset", {})
    assert name_a != name_b, f"chat_template change collided on '{name_a}'"


def test_closure_fingerprint_deterministic():
    """Identical tokenizer + config must yield the same cache file (cross-rank)."""
    ds = _StubDataset()

    def mk():
        return create_llm_processor(
            tokenizer=_StubTokenizer("modelA", 201088),
            max_length=4096,
            conversation_field="conversation",
            use_padding=False,
        )

    name_1 = _build_cache_file_name("map", mk(), ds, "Processing dataset", {})
    name_2 = _build_cache_file_name("map", mk(), ds, "Processing dataset", {})
    assert name_1 == name_2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
