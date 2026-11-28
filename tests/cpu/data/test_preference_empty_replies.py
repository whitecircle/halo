#!/usr/bin/env python3
"""A preference pair whose chosen or rejected reply is empty or blank is dropped before the chat-template maps, under
every template: Mistral Small 4's refuses to render an empty assistant turn and would stop the run, and an accepting
template would train on a pair that only teaches when to end the turn."""

import logging
from types import SimpleNamespace

import pytest
from datasets import Dataset, DatasetDict
from jinja2 import TemplateError

from halo.data.pipeline.preferences import drop_empty_reply_pairs, has_a_reply_on_both_sides, is_empty_reply
from halo.entrypoints import data_setup
from tests.common.models import MISTRAL3_119B_MOE
from tests.common.tiny_models import CONTENT_ONLY_CHAT_TEMPLATE, word_level_tokenizer
from tests.common.tokenizers import load_cached_tokenizer

PROMPT = [{"role": "user", "content": "Capital of France?"}]
REAL = "Paris."
CALL = {"type": "function", "function": {"name": "lookup", "arguments": {"city": "Paris"}}}


def _pair(chosen, rejected="France capital is Paris I think."):
    return {
        "prompt": PROMPT,
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }


@pytest.mark.parametrize(
    ("message", "empty"),
    [
        ({"role": "assistant", "content": None}, True),
        ({"role": "assistant", "content": ""}, True),
        ({"role": "assistant", "content": " \n\t"}, True),
        ({"role": "assistant", "content": []}, True),
        ({"role": "assistant", "content": [{"type": "text", "text": "  "}]}, True),
        ({"role": "assistant", "content": REAL}, False),
        ({"role": "assistant", "content": [{"type": "text", "text": REAL}]}, False),
        ({"role": "assistant", "content": None, "tool_calls": [CALL]}, False),
        ({"role": "assistant", "content": "", "tool_calls": [CALL]}, False),
    ],
    ids=[
        "none",
        "empty",
        "blank",
        "no_chunks",
        "blank_chunk",
        "text",
        "text_chunk",
        "tool_call_no_content",
        "tool_call_empty_content",
    ],
)
def test_an_empty_reply_has_no_content_and_no_tool_call(message, empty):
    assert is_empty_reply(message) is empty


@pytest.mark.parametrize("side", ["chosen", "rejected"])
def test_a_full_conversation_side_is_judged_by_its_last_assistant_turn(side):
    """ultrafeedback_binarized and the Tulu-3 mixtures repeat the prompt inside each side."""
    row = {
        "chosen": [*PROMPT, {"role": "assistant", "content": REAL}],
        "rejected": [*PROMPT, {"role": "assistant", "content": "Lyon."}],
    }
    assert has_a_reply_on_both_sides(row)
    row[side][-1]["content"] = " "
    assert not has_a_reply_on_both_sides(row)


@pytest.mark.parametrize("shape", ["user_last", "missing"])
def test_a_side_that_does_not_end_on_an_assistant_turn_is_left_to_normalization(shape):
    """The filter keeps it, so normalize_preference_row still refuses the mis-mapped dataset by name."""
    row = {"prompt": PROMPT, "rejected": [{"role": "assistant", "content": REAL}]}
    if shape == "user_last":
        row["chosen"] = [{"role": "user", "content": ""}]
    assert has_a_reply_on_both_sides(row)


def _prepare(tokenizer, rows, tmp_path, monkeypatch):
    prepared = {}
    monkeypatch.setattr(data_setup, "log_script_dataset_examples", lambda datasets, *_: prepared.update(datasets))
    args = SimpleNamespace(
        tools_field=None, reasoning_effort_field=None, generate_eval_examples=True, num_eval_examples=1
    )
    training_config = SimpleNamespace(dataset_num_proc=None, output_dir=str(tmp_path), report_to=["none"])
    splits = DatasetDict(train=Dataset.from_list(rows), test=Dataset.from_list(rows))
    data_setup.prepare_script_preference_data(
        args, training_config, splits, tokenizer, is_vlm_data=False, generation_max_prompt_length=None
    )
    return prepared


def test_a_template_that_refuses_an_empty_reply_trains_on_the_other_pairs(tmp_path, monkeypatch):
    tokenizer = load_cached_tokenizer(MISTRAL3_119B_MOE)
    with pytest.raises(TemplateError, match="Assistant message must have"):
        tokenizer.apply_chat_template([*PROMPT, {"role": "assistant", "content": ""}], tokenize=False)

    prepared = _prepare(tokenizer, [_pair(REAL), _pair(REAL, rejected="")], tmp_path, monkeypatch)

    for split in ("train", "test"):
        assert len(prepared[split]) == 1, f"the {split} split kept the empty-reply pair"
    assert len(prepared["generate"]) == 1, "the generation prompts were taken from the unfiltered test split"


@pytest.mark.parametrize("side", ["chosen", "rejected"])
@pytest.mark.parametrize("empty", ["", None, "   "], ids=["empty", "none", "blank"])
def test_an_accepting_template_loses_the_empty_reply_pair_too(tmp_path, monkeypatch, caplog, side, empty):
    """The content-only template renders an empty turn, as Qwen3's and Gemma's do, so only the filter removes it."""
    tokenizer = word_level_tokenizer()
    tokenizer.chat_template = CONTENT_ONLY_CHAT_TEMPLATE
    degenerate = _pair(empty) if side == "chosen" else _pair(REAL, rejected=empty)

    with caplog.at_level(logging.INFO, logger="halo.data.pipeline.rejected_rows"):
        prepared = _prepare(tokenizer, [_pair(REAL), degenerate], tmp_path, monkeypatch)

    for split in ("train", "test"):
        assert len(prepared[split]) == 1, f"the {split} split kept the empty-reply pair"
        assert REAL in prepared[split][0]["chosen"], f"the {split} split dropped the real pair"
    assert "Filtered out 1/2 examples (50.0%)" in caplog.text


def test_a_reply_holding_tool_calls_and_no_content_is_kept():
    call = {"role": "assistant", "content": None, "tool_calls": [CALL]}
    rows = Dataset.from_list([{**_pair(REAL), "rejected": [call]}])
    assert len(drop_empty_reply_pairs(rows, "train", num_proc=None)) == 1


def test_a_split_the_filter_empties_raises():
    rows = Dataset.from_list([_pair(""), _pair(REAL, rejected=" ")])
    with pytest.raises(ValueError, match="All 2 pairs of the train split have an empty or blank"):
        drop_empty_reply_pairs(rows, "train", num_proc=None)


def test_a_split_with_no_rows_to_begin_with_is_not_refused():
    rows = Dataset.from_list([_pair(REAL)]).select([])
    assert len(drop_empty_reply_pairs(rows, "test", num_proc=None)) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
