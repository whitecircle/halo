#!/usr/bin/env python
import functools
import json

import pytest

from halo.data.conversation import resolve_json_value
from halo.data.pipeline import vlm_rows
from halo.data.pipeline.map_cache_key import function_cache_id, kwargs_fingerprint, map_cache_file_name
from halo.data.pipeline.preferences import render_preference_row
from halo.data.pipeline.text_rows import build_chat_row_processor, prepare_generative_row, render_row_conversation
from halo.data.pipeline.vlm_rows import build_vlm_row_processor


def _helper_func_a(x):
    return x + 1


def _helper_func_b(x):
    return x * 2


def test_function_id_deterministic():
    id1 = function_cache_id(_helper_func_a)
    id2 = function_cache_id(_helper_func_a)
    assert id1 == id2, f"Expected same ID, got '{id1}' and '{id2}'"


def test_function_id_different_functions():
    id_a = function_cache_id(_helper_func_a)
    id_b = function_cache_id(_helper_func_b)
    assert id_a != id_b, f"Expected different IDs, both got '{id_a}'"


def test_function_id_contains_name():
    fid = function_cache_id(_helper_func_a)
    assert "_helper_func_a" in fid, f"Expected '_helper_func_a' in '{fid}'"


def test_function_id_with_lambda():
    fid = function_cache_id(lambda x: x + 10)
    assert isinstance(fid, str)
    assert fid.startswith("<lambda>_"), f"Expected '<lambda>_...' prefix, got '{fid}'"
    assert len(fid) > len("<lambda>_")


def test_function_id_nested_function():
    def inner_func(y):
        return y**2

    fid = function_cache_id(inner_func)
    assert isinstance(fid, str)
    assert "inner_func" in fid


def test_function_id_partial_distinguishes_bound_tokenizer():
    """A partial exposes no source, __code__ or __closure__, so an identifier reading only those collapses to
    ``functools.unknown`` and partials over different tokenizers share one cache file."""
    p_a = functools.partial(prepare_generative_row, tokenizer=_StubTokenizer("modelA", 201088), max_length=4096)
    p_b = functools.partial(prepare_generative_row, tokenizer=_StubTokenizer("modelB", 262155), max_length=4096)
    id_a = function_cache_id(p_a)
    id_b = function_cache_id(p_b)
    assert id_a != id_b, f"partials binding different tokenizers collided on '{id_a}'"
    assert "prepare_generative_row" in id_a, f"expected wrapped func name in '{id_a}'"


def test_function_id_partial_distinguishes_bound_scalar():
    tok = _StubTokenizer("modelA", 201088)
    p_short = functools.partial(prepare_generative_row, tokenizer=tok, max_length=4096)
    p_long = functools.partial(prepare_generative_row, tokenizer=tok, max_length=36000)
    assert function_cache_id(p_short) != function_cache_id(p_long)


def test_function_id_partial_deterministic():
    def mk():
        return functools.partial(prepare_generative_row, tokenizer=_StubTokenizer("modelA", 201088), max_length=4096)

    assert function_cache_id(mk()) == function_cache_id(mk())


def test_kwargs_fingerprint_empty():
    result = kwargs_fingerprint({})
    assert result == "", f"Expected '', got '{result}'"


def test_kwargs_fingerprint_distinguishes_name_or_path():
    """Otherwise one model's cached token IDs are reused for another (OOB embedding gather)."""

    class _Tok:
        def __init__(self, nop):
            self.name_or_path = nop

    fp_a = kwargs_fingerprint({"tokenizer": _Tok("Qwen/Qwen3-4B")})
    fp_b = kwargs_fingerprint({"tokenizer": _Tok("meta-llama/Llama-3-8B")})
    assert fp_a != fp_b, "Different tokenizers must produce different fingerprints"


def test_kwargs_fingerprint_distinguishes_vocab_size():
    class _Cfg:
        def __init__(self, vs):
            self.vocab_size = vs

    fp_a = kwargs_fingerprint({"config": _Cfg(32000)})
    fp_b = kwargs_fingerprint({"config": _Cfg(151936)})
    assert fp_a != fp_b


def test_kwargs_fingerprint_distinguishes_chat_template():
    """A --chat_template override mutates tokenizer.chat_template in place without changing name_or_path or
    vocab_size, so the identity key must fold the template in."""

    class _Tok:
        def __init__(self, template):
            self.name_or_path = "same/model"
            self.vocab_size = 1000
            self.chat_template = template

    fp_a = kwargs_fingerprint({"tokenizer": _Tok("{{ messages }}")})
    fp_b = kwargs_fingerprint({"tokenizer": _Tok("{{ messages }}<eot>")})
    fp_a2 = kwargs_fingerprint({"tokenizer": _Tok("{{ messages }}")})
    assert fp_a != fp_b, "Different chat_template must produce different fingerprints"
    assert fp_a == fp_a2, "Same chat_template must be stable"


def test_kwargs_fingerprint_distinguishes_primitive_values():
    assert kwargs_fingerprint({"max_length": 1024}) != kwargs_fingerprint({"max_length": 4096})


def test_kwargs_fingerprint_opaque_object_uses_type_name():
    """Two instances of one opaque type share a fingerprint, since only the type name is captured."""

    class _OpaqueA:
        pass

    class _OpaqueB:
        pass

    fp_a1 = kwargs_fingerprint({"obj": _OpaqueA()})
    fp_a2 = kwargs_fingerprint({"obj": _OpaqueA()})
    fp_b = kwargs_fingerprint({"obj": _OpaqueB()})
    assert fp_a1 == fp_a2, "Same opaque type → same fingerprint (only type name captured)"
    assert fp_a1 != fp_b, "Different opaque type → different fingerprint"


def test_kwargs_fingerprint_sorted_keys():
    fp1 = kwargs_fingerprint({"a": 1, "b": 2, "c": 3})
    fp2 = kwargs_fingerprint({"c": 3, "a": 1, "b": 2})
    assert fp1 == fp2, f"Different key orders should produce same fingerprint: '{fp1}' vs '{fp2}'"


def test_resolve_json_value_passthrough_list():
    value = [{"role": "user", "content": "hi"}]
    assert resolve_json_value(value) is value


def test_resolve_json_value_passes_through_invalid_string():
    """A mis-specified conversation_field then surfaces as a clear chat-template error, not a JSONDecodeError inside
    a map worker."""
    assert resolve_json_value("{not valid json") == "{not valid json"
    assert resolve_json_value("plain text") == "plain text"
    assert resolve_json_value("") == ""
    assert resolve_json_value('{"role": "user"}') == {"role": "user"}
    assert resolve_json_value("[1, 2]") == [1, 2]
    assert resolve_json_value([{"role": "user"}]) == [{"role": "user"}]


class _RecordingTokenizer:
    """Records apply_chat_template kwargs, so a test can assert ``tools`` is forwarded only when the row carries a
    value."""

    bos_token = None
    pad_token = "<pad>"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        self.calls.append({"messages": messages, "tokenize": tokenize, **kwargs})
        tools = kwargs.get("tools")
        suffix = f"|tools={len(tools)}" if tools else "|no-tools"
        # A per-message suffix keeps the render prefix-consistent, the invariant the preference helper enforces.
        return "".join(m["content"] + suffix for m in messages)

    def __call__(self, text, **kwargs):
        ids = list(range(len(text.split())))
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_apply_chat_template_forwards_tools_when_present():
    tokenizer = _RecordingTokenizer()
    tools = [{"type": "function", "function": {"name": "calc"}}]
    row = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": tools,
    }
    render_row_conversation(row, tokenizer, conversation_field="messages", tools_field="tools")
    assert tokenizer.calls and tokenizer.calls[-1].get("tools") == tools


def test_apply_chat_template_omits_tools_when_field_missing():
    tokenizer = _RecordingTokenizer()
    row = {"messages": [{"role": "user", "content": "hi"}]}
    render_row_conversation(row, tokenizer, conversation_field="messages", tools_field="tools")
    assert tokenizer.calls and "tools" not in tokenizer.calls[-1]


def test_apply_chat_template_parses_json_tools():
    tokenizer = _RecordingTokenizer()
    tools = [{"type": "function", "function": {"name": "calc"}}]
    row = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": json.dumps(tools),
    }
    render_row_conversation(row, tokenizer, conversation_field="messages", tools_field="tools")
    assert tokenizer.calls[-1].get("tools") == tools


def test_apply_chat_template_parses_json_messages():
    tokenizer = _RecordingTokenizer()
    row = {
        "messages": json.dumps([{"role": "user", "content": "hi"}]),
    }
    render_row_conversation(row, tokenizer, conversation_field="messages")
    assert tokenizer.calls[-1]["messages"] == [{"role": "user", "content": "hi"}]


def test_build_chat_row_processor_threads_tools():
    tokenizer = _RecordingTokenizer()
    tools = [{"type": "function", "function": {"name": "calc"}}]
    processor = build_chat_row_processor(
        tokenizer=tokenizer,
        max_length=128,
        conversation_field="messages",
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
    tokenizer = _RecordingTokenizer()
    tools = [{"type": "function", "function": {"name": "calc"}}]
    row = {
        "prompt": [{"role": "user", "content": "q"}],
        "chosen": [{"role": "assistant", "content": "a1"}],
        "rejected": [{"role": "assistant", "content": "a2"}],
        "tools": tools,
    }
    render_preference_row(row, tokenizer, tools_field="tools")
    tools_calls = [c for c in tokenizer.calls if c.get("tools") == tools]
    assert len(tools_calls) == 3, f"Expected 3 calls with tools, got {len(tools_calls)} of {len(tokenizer.calls)}"


def test_preference_helper_omits_tools_when_field_unset():
    tokenizer = _RecordingTokenizer()
    row = {
        "prompt": [{"role": "user", "content": "q"}],
        "chosen": [{"role": "assistant", "content": "a1"}],
        "rejected": [{"role": "assistant", "content": "a2"}],
    }
    render_preference_row(row, tokenizer)
    assert all("tools" not in c for c in tokenizer.calls)


def test_preference_helper_parses_json_tools():
    tokenizer = _RecordingTokenizer()
    tools = [{"type": "function", "function": {"name": "calc"}}]
    row = {
        "prompt": [{"role": "user", "content": "q"}],
        "chosen": [{"role": "assistant", "content": "a1"}],
        "rejected": [{"role": "assistant", "content": "a2"}],
        "tool_specs": json.dumps(tools),
    }
    render_preference_row(
        row,
        tokenizer,
        tools_field="tool_specs",
    )
    assert all(c.get("tools") == tools for c in tokenizer.calls)


def test_prepare_generative_row_threads_tools():
    tokenizer = _RecordingTokenizer()
    tools = [{"type": "function", "function": {"name": "calc"}}]
    row = {
        "prompt": [{"role": "user", "content": "hi"}],
        "tools": tools,
    }
    prepare_generative_row(row, tokenizer, max_length=128, tools_field="tools")
    assert tokenizer.calls and tokenizer.calls[-1].get("tools") == tools


def _stub_process_vlm(conversation, system_prompt=None, model_supports_system_role=True):
    return list(conversation), []


@pytest.fixture
def _passthrough_vlm_conversation(monkeypatch):
    """Swap the image extractor for a passthrough, so these row-shape tests need no real PIL payloads."""
    monkeypatch.setattr(vlm_rows, "extract_conversation_images", _stub_process_vlm)


def test_vlm_processor_emits_tools_json_when_field_set(_passthrough_vlm_conversation):
    tools = [{"type": "function", "function": {"name": "calc"}}]
    processor = build_vlm_row_processor(conversation_field="messages", tools_field="tools")
    row = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": tools,
    }
    example = processor(row)
    assert example["tools_json"] is not None
    assert json.loads(example["tools_json"]) == tools


def test_vlm_processor_passes_through_json_string_tools(_passthrough_vlm_conversation):
    tools_str = json.dumps([{"type": "function", "function": {"name": "calc"}}])
    processor = build_vlm_row_processor(conversation_field="messages", tools_field="tools")
    row = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": tools_str,
    }
    example = processor(row)
    # Stored verbatim, so a round trip yields the original list and not a string of a string.
    assert example["tools_json"] == tools_str
    assert isinstance(json.loads(example["tools_json"]), list)


def test_vlm_processor_omits_tools_json_when_field_unset(_passthrough_vlm_conversation):
    """tools_json is always present (schema stable) and None when no tools."""
    processor = build_vlm_row_processor(conversation_field="messages")
    row = {"messages": [{"role": "user", "content": "hi"}]}
    example = processor(row)
    assert "tools_json" in example
    assert example["tools_json"] is None


def test_vlm_processor_handles_missing_tools_column(_passthrough_vlm_conversation):
    processor = build_vlm_row_processor(conversation_field="messages", tools_field="tools")
    row = {"messages": [{"role": "user", "content": "hi"}]}
    example = processor(row)
    assert example["tools_json"] is None


class _StubDataset:
    """Stand-in that satisfies map_cache_file_name's fingerprinting needs."""

    _fingerprint = "abc123"

    def __len__(self):
        return 10


def test_cache_key_extras_change_filename():
    def fn(row):
        return row

    ds = _StubDataset()
    name_a = map_cache_file_name(
        "map",
        fn,
        ds,
        "desc",
        {},
        cache_key_extras={"tools_field": None},
    )
    name_b = map_cache_file_name(
        "map",
        fn,
        ds,
        "desc",
        {},
        cache_key_extras={"tools_field": "tools"},
    )
    assert name_a != name_b, f"Expected different cache names when tools_field flips, both got '{name_a}'"


def test_cache_key_extras_stable_for_same_inputs():
    def fn(row):
        return row

    ds = _StubDataset()
    extras = {"tools_field": "tools", "system_prompt": "hello"}
    name_1 = map_cache_file_name("map", fn, ds, "desc", {}, cache_key_extras=extras)
    name_2 = map_cache_file_name("map", fn, ds, "desc", {}, cache_key_extras=extras)
    assert name_1 == name_2


def test_cache_key_extras_none_matches_empty_dict():
    def fn(row):
        return row

    ds = _StubDataset()
    name_none = map_cache_file_name("map", fn, ds, "desc", {}, cache_key_extras=None)
    name_empty = map_cache_file_name("map", fn, ds, "desc", {}, cache_key_extras={})
    assert name_none == name_empty


class _StubTokenizer:
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
    """The processor captures the tokenizer as a free variable, invisible to the source hash and fn_kwargs, so a
    262k-vocab tokenizer's IDs would be reused for a 201k-vocab model and gather out of bounds at the first step."""
    ds = _StubDataset()
    proc_a = build_chat_row_processor(
        tokenizer=_StubTokenizer("modelA", 201088),
        max_length=4096,
        conversation_field="conversation",
        use_padding=False,
    )
    proc_b = build_chat_row_processor(
        tokenizer=_StubTokenizer("modelB", 262155),
        max_length=4096,
        conversation_field="conversation",
        use_padding=False,
    )
    name_a = map_cache_file_name("map", proc_a, ds, "Processing dataset", {})
    name_b = map_cache_file_name("map", proc_b, ds, "Processing dataset", {})
    assert name_a != name_b, f"Different tokenizers collided on '{name_a}'"


def test_closure_max_length_changes_cache_filename():
    ds = _StubDataset()
    tok = _StubTokenizer("modelA", 201088)
    proc_short = build_chat_row_processor(
        tokenizer=tok,
        max_length=4096,
        conversation_field="conversation",
        use_padding=False,
    )
    proc_long = build_chat_row_processor(
        tokenizer=tok,
        max_length=36000,
        conversation_field="conversation",
        use_padding=False,
    )
    name_short = map_cache_file_name("map", proc_short, ds, "Processing dataset", {})
    name_long = map_cache_file_name("map", proc_long, ds, "Processing dataset", {})
    assert name_short != name_long, f"max_length change collided on '{name_short}'"


def test_closure_chat_template_changes_cache_filename():
    """An override replaces tokenizer.chat_template in place and the closure fingerprint keys the tokenizer by
    identity, so the template must be folded in."""
    ds = _StubDataset()
    tok_a = _StubTokenizer("modelA", 201088)
    tok_a.chat_template = "{{ messages }}"
    tok_b = _StubTokenizer("modelA", 201088)
    tok_b.chat_template = "{{ messages }}<|eot|>"
    proc_a = build_chat_row_processor(
        tokenizer=tok_a, max_length=4096, conversation_field="conversation", use_padding=False
    )
    proc_b = build_chat_row_processor(
        tokenizer=tok_b, max_length=4096, conversation_field="conversation", use_padding=False
    )
    name_a = map_cache_file_name("map", proc_a, ds, "Processing dataset", {})
    name_b = map_cache_file_name("map", proc_b, ds, "Processing dataset", {})
    assert name_a != name_b, f"chat_template change collided on '{name_a}'"


def test_closure_fingerprint_deterministic():
    ds = _StubDataset()

    def mk():
        return build_chat_row_processor(
            tokenizer=_StubTokenizer("modelA", 201088),
            max_length=4096,
            conversation_field="conversation",
            use_padding=False,
        )

    name_1 = map_cache_file_name("map", mk(), ds, "Processing dataset", {})
    name_2 = map_cache_file_name("map", mk(), ds, "Processing dataset", {})
    assert name_1 == name_2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
