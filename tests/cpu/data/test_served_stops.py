"""A turn ends at the ids a served model stops at: the generation config's list joins the tokenizer's and the model
config's, so a tool call ends at its stop (gpt-oss's ``<|call|>``, Gemma 4's ``<|tool_response>``) with no eos appended
after it, while a pad only the generation config lists stays no terminator. A span ends before another role's turn, so
an answer gpt-oss closes with ``<|end|>`` never trains on through the next user turn to a later call's stop."""

import json
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset

from halo.data.collators.packing import (
    DataCollatorForCompletionOnlyLMWithPacking,
    DataCollatorWithFlatteningAndCompletionMask,
)
from halo.data.collators.padded import DataCollatorForCompletionOnlyLM
from halo.data.collators.self_distill import SelfDistillTextCollator
from halo.data.completion_spans import find_turn_opener, resolve_span_bounds
from halo.data.pipeline.text_rows import tokenize_training_row
from halo.data.preprocessed.metadata import PreprocessingConfig
from halo.data.preprocessed.tokenization import tokenize_dataset
from halo.data.turn_markers import resolve_eos_token_ids
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from tests.common.models import (
    GEMMA3_4B_IT,
    GEMMA4_26B_A4B,
    GPT_OSS_20B_OPENAI,
    NEMOTRON_3_5_SUPER_VL,
    QWEN3_0_6B,
    QWEN3_5_MOE_35B,
)
from tests.common.repo import REPO_ROOT
from tests.common.tokenizers import (
    load_cached_config,
    load_cached_generation_config,
    load_cached_tokenizer,
    resolve_cached_eos_token_ids,
)
from tests.common.tool_rounds import TOOL_ROUND, TOOLS
from tests.common.trained_spans import trained_pieces, tree_row_labels

GPT_OSS_FINAL = "<|start|>assistant<|channel|>final<|message|>"
GPT_OSS_HARMONY_TEMPLATE = REPO_ROOT / "jinja-templates" / "gpt-oss" / "gpt-oss-harmony.jinja"


def _served_terminators(model):
    tokenizer = load_cached_tokenizer(model)
    return tokenizer, resolve_cached_eos_token_ids(model, tokenizer)


def _row(tokenizer, eos_token_ids, conversation) -> list[int]:
    row = {"messages": conversation, "tools": TOOLS}
    return tokenize_training_row(row, tokenizer, eos_token_ids, conversation_field="messages", tools_field="tools")[
        "input_ids"
    ]


def test_the_generation_config_adds_its_stops_and_drops_a_pad_only_it_lists():
    tokenizer = SimpleNamespace(eos_token_id=1, pad_token_id=0)
    model_config = SimpleNamespace(eos_token_id=[1, 7])
    assert resolve_eos_token_ids(tokenizer, model_config, SimpleNamespace(eos_token_id=[1, 0, 50])) == {1, 7, 50}
    # A pad the model config lists beside the role markers stays, as GLM-4 lists its <|endoftext|>.
    listed_pad = SimpleNamespace(eos_token_id=[0, 1])
    assert resolve_eos_token_ids(tokenizer, listed_pad, SimpleNamespace(eos_token_id=[0, 50])) == {0, 1, 50}
    assert resolve_eos_token_ids(tokenizer, model_config, None) == {1, 7}


def test_qwen3_lists_its_pad_among_the_served_stops_and_it_ends_no_turn():
    tokenizer, eos_token_ids = _served_terminators(QWEN3_0_6B)
    assert tokenizer.pad_token_id in load_cached_generation_config(QWEN3_0_6B).eos_token_id
    assert eos_token_ids == {tokenizer.eos_token_id}


@pytest.mark.parametrize(
    ("model", "stop"),
    [(GPT_OSS_20B_OPENAI, "<|call|>"), (GEMMA4_26B_A4B, "<|tool_response>")],
    ids=["gpt-oss", "gemma4"],
)
def test_a_row_ending_on_a_tool_call_ends_at_the_served_stop(model, stop):
    """No eos follows the stop: gpt-oss would train ``<|call|><|return|>`` and Gemma 4 ``<|tool_response><eos>``."""
    tokenizer, eos_token_ids = _served_terminators(model)
    assert tokenizer.convert_tokens_to_ids(stop) in eos_token_ids
    ids = _row(tokenizer, eos_token_ids, TOOL_ROUND[:2])
    assert tokenizer.convert_ids_to_tokens(ids[-1]) == stop


def test_a_gpt_oss_tool_call_turn_inside_a_row_trains_through_its_call():
    """Unless ``<|call|>`` ends it, the call turn meets the next marker first and trains nothing."""
    tokenizer, eos_token_ids = _served_terminators(GPT_OSS_20B_OPENAI)
    ids = _row(tokenizer, eos_token_ids, TOOL_ROUND)
    collator = DataCollatorForCompletionOnlyLM(
        response_prompt_template="<|start|>assistant", tokenizer=tokenizer, eos_token_ids=eos_token_ids
    )
    labels = collator([{"input_ids": ids}])["labels"][0]
    trained = tokenizer.decode(torch.tensor(ids)[labels != LABEL_IGNORE_INDEX])
    assert '{"city": "Paris"}<|call|>' in trained
    assert "It is sunny.<|return|>" in trained
    assert "SUNNY" not in trained


def test_gpt_oss_opens_every_turn_at_a_special_token_all_special_ids_leaves_out():
    tokenizer, eos_token_ids = _served_terminators(GPT_OSS_20B_OPENAI)
    start = tokenizer.convert_tokens_to_ids("<|start|>")
    assert start not in tokenizer.all_special_ids, "the added-token read is moot: all_special_ids lists <|start|>"
    assert resolve_span_bounds(tokenizer, GPT_OSS_FINAL, eos_token_ids).turn_opener == start
    # A plain first token also opens markdown inside an answer, so it bounds no span.
    assert find_turn_opener(tokenizer, tokenizer.encode("### Answer", add_special_tokens=False)) is None


def _final_then_call(thinking: bool) -> list[dict]:
    """A final answer, then a user turn and a tool round in one row; the stock template renders the final channel for
    every answer, harmony only for one that carries ``thinking``."""

    def answer(content: str, reasoning: str) -> dict:
        return {"role": "assistant", "content": content, **({"thinking": reasoning} if thinking else {})}

    return [
        {"role": "user", "content": "USER-ONE"},
        answer("FINAL-ONE", "T1"),
        {"role": "user", "content": "USER-TWO"},
        *TOOL_ROUND[1:3],
        answer("FINAL-TWO", "T2"),
    ]


def _gpt_oss_route_labels(route, tokenizer, eos_token_ids, conversation, ids) -> list[int]:
    kwargs = {"response_prompt_template": GPT_OSS_FINAL, "tokenizer": tokenizer, "eos_token_ids": eos_token_ids}
    if route == "padded":
        return DataCollatorForCompletionOnlyLM(**kwargs)([{"input_ids": ids}])["labels"][0].tolist()
    if route == "packed":
        collator = DataCollatorForCompletionOnlyLMWithPacking(**kwargs)
        return collator([{"input_ids": ids, "seq_lengths": [len(ids)]}])["labels"][0].tolist()
    if route == "padding_free":
        return DataCollatorWithFlatteningAndCompletionMask(**kwargs)([{"input_ids": ids}])["labels"][0].tolist()
    if route == "self_distill":
        collator = SelfDistillTextCollator(
            tokenizer,
            max_length=4096,
            conversation_field="messages",
            hint_template=None,
            train_on_completions_only=True,
            answer_field=None,
            solution_field=None,
            confidence_power=1.0,
            response_prompt_template=GPT_OSS_FINAL,
            tools_field="tools",
            model_config=load_cached_config(GPT_OSS_20B_OPENAI),
            generation_config=load_cached_generation_config(GPT_OSS_20B_OPENAI),
        )
        row = collator.encode_row({"messages": conversation, "tools": TOOLS})
        assert row["input_ids"] == ids
        return row["labels"]
    if route == "tree":
        tree_ids, labels = tree_row_labels(
            tokenizer, conversation, TOOLS, eos_token_ids=eos_token_ids, marker=GPT_OSS_FINAL
        )
        assert tree_ids == ids
        return labels
    config = PreprocessingConfig(
        model_name_or_path=GPT_OSS_20B_OPENAI,
        max_length=4096,
        conversation_field="messages",
        tools_field="tools",
        train_on_completions_only=True,
        assistant_message_template=GPT_OSS_FINAL,
        num_proc=1,
    )
    dataset = Dataset.from_dict({"messages": [json.dumps(conversation)], "tools": [json.dumps(TOOLS)]})
    baked = tokenize_dataset(dataset, tokenizer, config)
    assert baked["input_ids"][0] == ids
    return baked["labels"][0]


@pytest.mark.parametrize(
    ("template", "thinking"), [(None, False), (GPT_OSS_HARMONY_TEMPLATE, True)], ids=["hub", "harmony"]
)
@pytest.mark.parametrize("route", ["padded", "packed", "padding_free", "self_distill", "tree", "bake"])
def test_a_gpt_oss_answer_never_trains_through_the_next_user_turn_to_a_calls_stop(
    route, template, thinking, tmp_path, monkeypatch
):
    """The earlier answer closes at ``<|end|>``, no terminator, and ``<|call|>`` is one: bounded only by the next final
    marker, its span would run through ``USER-TWO`` and the call to that stop."""
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf_datasets"))
    tokenizer, eos_token_ids = _served_terminators(GPT_OSS_20B_OPENAI)
    if template is not None:
        tokenizer.chat_template = template.read_text(encoding="utf-8")
    tokenizer.padding_side = "right"
    conversation = _final_then_call(thinking)
    ids = _row(tokenizer, eos_token_ids, conversation)
    labels = _gpt_oss_route_labels(route, tokenizer, eos_token_ids, conversation, ids)
    answer = "FINAL-TWO<|return|>" if route == "self_distill" else f"{GPT_OSS_FINAL}FINAL-TWO<|return|>"
    assert trained_pieces(tokenizer, ids, labels) == [answer]


@pytest.mark.parametrize("model", [GEMMA4_26B_A4B, GEMMA3_4B_IT, QWEN3_5_MOE_35B, NEMOTRON_3_5_SUPER_VL])
def test_a_text_only_load_resolves_the_composite_loads_set(model):
    """``text_only_model`` swaps the model config for its text sub-config, which Gemma 3/4 leave at ``<eos>`` alone while
    the top level lists ``<end_of_turn>``/``<turn|>`` too; the generation config, read whatever class loads the
    checkpoint, keeps the turn end."""
    tokenizer = load_cached_tokenizer(model)
    config = load_cached_config(model)
    generation_config = load_cached_generation_config(model)
    composite = resolve_eos_token_ids(tokenizer, config, generation_config)
    assert resolve_eos_token_ids(tokenizer, config.get_text_config(), generation_config) == composite


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
