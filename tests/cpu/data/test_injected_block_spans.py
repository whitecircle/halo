"""A span restarts after a block the chat template writes into the model's own turn: Gemma 4 renders a tool's reply
inside the turn that called it, so a row trains the call and its ``<|tool_response>`` stop, then the answer and
``<turn|>``, never the reply, on every completion route (padded, packed, padding-free, self-distillation, VLM, tree rows
and both bakes); Mistral 4 resumes after ``[/TOOL_RESULTS]`` alike, and a template opening a new turn for the answer
yields no block."""

import json
from pathlib import Path

import pytest
from datasets import Dataset

from halo.data.collators.packing import (
    DataCollatorForCompletionOnlyLMWithPacking,
    DataCollatorWithFlatteningAndCompletionMask,
)
from halo.data.collators.padded import DataCollatorForCompletionOnlyLM
from halo.data.collators.self_distill import SelfDistillTextCollator
from halo.data.collators.vlm import VLMDataCollator
from halo.data.completion_spans import (
    PADDED_SPAN_POLICY,
    SELF_DISTILL_SPAN_POLICY,
    InjectedBlock,
    SpanBounds,
    find_injected_block,
    resolve_completion_spans,
)
from halo.data.pipeline.text_rows import tokenize_training_row
from halo.data.preprocessed.metadata import PreprocessingConfig
from halo.data.preprocessed.tokenization import tokenize_dataset, tokenize_vlm_dataset
from halo.data.turn_markers import resolve_eos_token_ids
from halo.data.vlm_render import extract_conversation_images
from tests.common.models import GEMMA4_26B_A4B, GLM4_FLASH, GPT_OSS_20B_OPENAI, MISTRAL3_119B_MOE, QWEN3_0_6B
from tests.common.repo import REPO_ROOT
from tests.common.tokenizers import (
    load_cached_config,
    load_cached_generation_config,
    load_cached_processor,
    load_cached_tokenizer,
    resolve_cached_eos_token_ids,
)
from tests.common.tool_rounds import CALL, TOOL_ROUND, TOOLS
from tests.common.trained_spans import trained_pieces, tree_row_labels

MARKER, OPEN, CLOSE, EOS, HEAD = 10, 50, 51, 106, 60
BLOCK = InjectedBlock(opener=OPEN, head=None, closer=CLOSE)
GEMMA4_MARKER = "<|turn>model\n"
GEMMA4_BUNDLED_TEMPLATE = REPO_ROOT / "jinja-templates" / "gemma4" / "gemma4-reasoning-effort.jinja"
CALL_SPAN = '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|><|tool_response>'
ANSWER_SPAN = "It is sunny.<turn|>"


def _spans(sequence, injected_block=BLOCK, **policy):
    starts, ends = resolve_completion_spans(
        sequence, [MARKER], frozenset({OPEN, EOS}), span_bounds=SpanBounds(injected_block=injected_block), **policy
    )
    return list(zip(starts, ends, strict=True))


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        ([5, MARKER, 20, OPEN, 30, 31, CLOSE, 40, 41, EOS], [(1, 3), (7, 9)]),
        # One reply per parallel call: the turn resumes after the last of them.
        ([MARKER, 20, OPEN, 30, CLOSE, OPEN, 31, CLOSE, 40, EOS], [(0, 2), (8, 9)]),
        # A block cut before its closer leaves nothing to resume.
        ([MARKER, 20, OPEN, 30, 31], [(0, 2)]),
        # Only the opener's block restarts: a closer after a turn's own end opens nothing.
        ([MARKER, 20, EOS, CLOSE, 40, EOS], [(0, 2)]),
        # The next turn's marker bounds the search for the closer.
        ([MARKER, 20, OPEN, 30, MARKER, 40, CLOSE, 41, EOS], [(0, 2), (4, 8)]),
        # The resumed answer may call again, its reply restarting once more.
        ([MARKER, 20, OPEN, 30, CLOSE, 40, OPEN, 31, CLOSE, 41, EOS], [(0, 2), (5, 6), (9, 10)]),
        # A terminator before the closer is no reply's: the closer after it opens nothing.
        ([MARKER, 20, OPEN, 30, EOS, 31, CLOSE, 40, EOS], [(0, 2)]),
    ],
)
def test_a_span_restarts_after_each_block_its_turn_holds(sequence, expected):
    assert _spans(sequence, **PADDED_SPAN_POLICY) == expected


def test_blocks_following_the_stop_restart_after_the_last_of_them():
    """Mistral's shape: the stop is no part of the block, whose own head opens each further one."""
    block = InjectedBlock(opener=EOS, head=HEAD, closer=CLOSE)
    sequence = [MARKER, 20, EOS, HEAD, 30, CLOSE, HEAD, 31, CLOSE, 40, EOS, MARKER, 41, EOS]
    assert _spans(sequence, injected_block=block, **PADDED_SPAN_POLICY) == [(0, 2), (9, 10), (11, 13)]


def test_the_self_distillation_scan_restarts_alike():
    sequence = [MARKER, 20, OPEN, 30, CLOSE, 40, EOS]
    assert _spans(sequence, **SELF_DISTILL_SPAN_POLICY) == [(1, 2), (5, 6)]


def test_the_last_assistant_span_is_the_answer_after_the_block():
    starts, ends = resolve_completion_spans(
        [MARKER, 20, OPEN, 30, CLOSE, 40, EOS],
        [MARKER],
        frozenset({OPEN, EOS}),
        train_on_last_assistant_only=True,
        span_bounds=SpanBounds(injected_block=BLOCK),
    )
    assert (starts, ends) == ([5], [6])


def test_without_a_block_a_span_ends_at_the_stop_alone():
    assert _spans([MARKER, 20, OPEN, 30, CLOSE, 40, EOS], injected_block=None, **PADDED_SPAN_POLICY) == [(0, 2)]


def _gemma4(template: Path | None = None):
    tokenizer = load_cached_tokenizer(GEMMA4_26B_A4B)
    if template is not None:
        tokenizer.chat_template = template.read_text(encoding="utf-8")
    config = load_cached_config(GEMMA4_26B_A4B)
    generation_config = load_cached_generation_config(GEMMA4_26B_A4B)
    return tokenizer, config, generation_config, resolve_eos_token_ids(tokenizer, config, generation_config)


@pytest.mark.parametrize("template", [None, GEMMA4_BUNDLED_TEMPLATE], ids=["hub", "bundled"])
def test_gemma4s_tool_reply_is_the_block_its_template_injects(template):
    tokenizer, _, _, eos_token_ids = _gemma4(template)
    expected = InjectedBlock(
        opener=tokenizer.convert_tokens_to_ids("<|tool_response>"),
        head=None,
        closer=tokenizer.convert_tokens_to_ids("<tool_response|>"),
    )
    assert find_injected_block(tokenizer, GEMMA4_MARKER, eos_token_ids) == expected
    processor = load_cached_processor(GEMMA4_26B_A4B)
    if template is not None:
        processor.chat_template = tokenizer.chat_template
    assert find_injected_block(tokenizer, GEMMA4_MARKER, eos_token_ids, template_owner=processor) == expected


@pytest.mark.parametrize(
    ("model", "marker"),
    [
        (QWEN3_0_6B, "<|im_start|>assistant\n"),
        (GPT_OSS_20B_OPENAI, "<|start|>assistant"),
        (GLM4_FLASH, "<|assistant|>"),
    ],
)
def test_a_template_opening_a_turn_for_the_answer_injects_no_block(model, marker):
    tokenizer = load_cached_tokenizer(model)
    assert find_injected_block(tokenizer, marker, resolve_cached_eos_token_ids(model, tokenizer)) is None


def test_mistral4_resumes_after_its_tool_results():
    """Its hub template writes the results after the call's ``</s>`` and the answer after them with no ``[/INST]``."""
    tokenizer = load_cached_tokenizer(MISTRAL3_119B_MOE)
    eos_token_ids = resolve_cached_eos_token_ids(MISTRAL3_119B_MOE, tokenizer)
    block = find_injected_block(tokenizer, "[/INST]", eos_token_ids)
    assert block == InjectedBlock(
        opener=tokenizer.eos_token_id,
        head=tokenizer.convert_tokens_to_ids("[TOOL_RESULTS]"),
        closer=tokenizer.convert_tokens_to_ids("[/TOOL_RESULTS]"),
    )
    second = {**CALL, "id": "call_1", "function": {"name": "get_weather", "arguments": {"city": "Rome"}}}
    conversation = [
        TOOL_ROUND[0],
        {"role": "assistant", "content": "", "tool_calls": [CALL, second]},
        TOOL_ROUND[2],
        {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "RAINY"},
        TOOL_ROUND[3],
    ]
    row = {"messages": conversation, "tools": TOOLS}
    ids = tokenize_training_row(row, tokenizer, eos_token_ids, conversation_field="messages", tools_field="tools")
    ids = ids["input_ids"]
    collator = DataCollatorForCompletionOnlyLM(
        response_prompt_template="[/INST]", tokenizer=tokenizer, eos_token_ids=eos_token_ids
    )
    pieces = trained_pieces(tokenizer, ids, collator([{"input_ids": ids}])["labels"][0].tolist())
    assert pieces[-1] == "It is sunny.</s>"
    assert not any("SUNNY" in piece or "RAINY" in piece for piece in pieces)


def _gemma4_row(template: Path | None):
    tokenizer, config, generation_config, eos_token_ids = _gemma4(template)
    row = {"messages": TOOL_ROUND, "tools": TOOLS}
    ids = tokenize_training_row(row, tokenizer, eos_token_ids, conversation_field="messages", tools_field="tools")
    return tokenizer, config, generation_config, eos_token_ids, ids["input_ids"]


@pytest.mark.parametrize("template", [None, GEMMA4_BUNDLED_TEMPLATE], ids=["hub", "bundled"])
@pytest.mark.parametrize("route", ["padded", "packed", "padding_free"])
def test_gemma4_text_routes_train_the_call_and_the_answer_never_the_reply(template, route):
    tokenizer, _, _, eos_token_ids, ids = _gemma4_row(template)
    tokenizer.padding_side = "right"
    kwargs = {"response_prompt_template": GEMMA4_MARKER, "tokenizer": tokenizer, "eos_token_ids": eos_token_ids}
    if route == "padded":
        labels = DataCollatorForCompletionOnlyLM(**kwargs)([{"input_ids": ids}])["labels"][0].tolist()
    elif route == "packed":
        # Two documents in one packed row, each masked on its own.
        packed = DataCollatorForCompletionOnlyLMWithPacking(**kwargs)
        batch = packed([{"input_ids": ids + ids, "seq_lengths": [len(ids), len(ids)]}])
        labels = batch["labels"][0, : len(ids)].tolist()
        assert batch["labels"][0, len(ids) + 1 :].tolist() == labels[1:], "the second document masks alike"
    else:
        batch = DataCollatorWithFlatteningAndCompletionMask(**kwargs)([{"input_ids": ids}])
        labels = batch["labels"][0].tolist()
    assert trained_pieces(tokenizer, ids, labels) == [f"{GEMMA4_MARKER}{CALL_SPAN}", ANSWER_SPAN]


def test_gemma4_self_distillation_rows_train_the_call_and_the_answer_never_the_reply():
    tokenizer, config, generation_config, _, ids = _gemma4_row(None)
    collator = SelfDistillTextCollator(
        tokenizer,
        max_length=4096,
        conversation_field="messages",
        hint_template=None,
        train_on_completions_only=True,
        answer_field=None,
        solution_field=None,
        confidence_power=1.0,
        response_prompt_template=GEMMA4_MARKER,
        tools_field="tools",
        model_config=config,
        generation_config=generation_config,
    )
    row = collator.encode_row({"messages": TOOL_ROUND, "tools": TOOLS})
    assert row["input_ids"] == ids
    assert trained_pieces(tokenizer, row["input_ids"], row["labels"]) == [CALL_SPAN, ANSWER_SPAN]


def test_gemma4_vlm_collator_trains_the_call_and_the_answer_never_the_reply():
    tokenizer, config, generation_config, _ = _gemma4(None)
    processor = load_cached_processor(GEMMA4_26B_A4B)
    collator = VLMDataCollator(
        processor,
        tokenizer,
        4096,
        train_on_completions_only=True,
        response_prompt_template=GEMMA4_MARKER,
        model_config=config,
        generation_config=generation_config,
    )
    history, images = extract_conversation_images(TOOL_ROUND)
    batch = collator([{"history": history, "images": images, "tools_json": json.dumps(TOOLS)}])
    ids, labels = batch["input_ids"][0].tolist(), batch["labels"][0].tolist()
    assert trained_pieces(tokenizer, ids, labels) == [f"{GEMMA4_MARKER}{CALL_SPAN}", ANSWER_SPAN]


@pytest.mark.parametrize("pack_sequences", [False, True], ids=["unpacked", "packed"])
def test_the_gemma4_bake_labels_its_rows_as_the_runtime_collator_does(pack_sequences, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf_datasets"))
    tokenizer, _, _, eos_token_ids, ids = _gemma4_row(None)
    config = PreprocessingConfig(
        model_name_or_path=GEMMA4_26B_A4B,
        max_length=4096,
        conversation_field="messages",
        tools_field="tools",
        train_on_completions_only=True,
        assistant_message_template=GEMMA4_MARKER,
        pack_sequences=pack_sequences,
        num_proc=1,
    )
    dataset = Dataset.from_dict({"messages": [json.dumps(TOOL_ROUND)], "tools": [json.dumps(TOOLS)]})
    baked = tokenize_dataset(dataset, tokenizer, config)
    assert baked["input_ids"][0] == ids
    assert trained_pieces(tokenizer, ids, baked["labels"][0]) == [f"{GEMMA4_MARKER}{CALL_SPAN}", ANSWER_SPAN]
    runtime = DataCollatorForCompletionOnlyLM(
        response_prompt_template=GEMMA4_MARKER, tokenizer=tokenizer, eos_token_ids=eos_token_ids
    )([{"input_ids": ids}])["labels"][0]
    assert baked["labels"][0] == runtime.tolist()


def test_gemma4_tree_rows_train_the_call_and_the_answer_never_the_reply():
    tokenizer, _, _, eos_token_ids, ids = _gemma4_row(None)
    tree_ids, labels = tree_row_labels(tokenizer, TOOL_ROUND, TOOLS, eos_token_ids=eos_token_ids, marker=GEMMA4_MARKER)
    assert tree_ids == ids
    assert trained_pieces(tokenizer, ids, labels) == [f"{GEMMA4_MARKER}{CALL_SPAN}", ANSWER_SPAN]


def test_the_gemma4_vlm_bake_labels_its_rows_as_the_runtime_vlm_collator_does(tmp_path, monkeypatch):
    """The VLM bake renders no tools column, so the row declares none on both sides."""
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf_datasets"))
    tokenizer, config, generation_config, _ = _gemma4(None)
    processor = load_cached_processor(GEMMA4_26B_A4B)
    bake_config = PreprocessingConfig(
        model_name_or_path=GEMMA4_26B_A4B,
        max_length=4096,
        conversation_field="messages",
        train_on_completions_only=True,
        assistant_message_template=GEMMA4_MARKER,
        is_vlm=True,
        num_proc=1,
    )
    baked = tokenize_vlm_dataset(Dataset.from_dict({"messages": [json.dumps(TOOL_ROUND)]}), processor, bake_config)
    ids, labels = baked["input_ids"][0], baked["labels"][0]
    assert trained_pieces(tokenizer, ids, labels) == [f"{GEMMA4_MARKER}{CALL_SPAN}", ANSWER_SPAN]
    collator = VLMDataCollator(
        processor,
        tokenizer,
        4096,
        train_on_completions_only=True,
        response_prompt_template=GEMMA4_MARKER,
        model_config=config,
        generation_config=generation_config,
    )
    history, images = extract_conversation_images(TOOL_ROUND)
    batch = collator([{"history": history, "images": images}])
    assert batch["input_ids"][0].tolist() == ids
    assert batch["labels"][0].tolist() == labels


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
