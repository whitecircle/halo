#!/usr/bin/env python3
"""A preference run's template columns reach every render its script prep runs: each row's tools and reasoning-effort
level in the train and eval chat templates and in the eval-generation prompts, on gpt-oss's harmony template."""

import json
from types import SimpleNamespace

import pytest
from datasets import Dataset, DatasetDict

from halo.entrypoints import data_setup
from tests.common.models import GPT_OSS_20B
from tests.common.tokenizers import load_cached_tokenizer

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Look up the weather.",
            "parameters": {"type": "object"},
        },
    }
]
ROW = {
    "prompt": [{"role": "user", "content": "Will it rain?"}],
    "chosen": [{"role": "assistant", "content": "Probably."}],
    "rejected": [{"role": "assistant", "content": "Never."}],
    "effort": "high",
    "tools": json.dumps(TOOLS),
}


def _states_the_rows_columns(text: str) -> bool:
    return "Reasoning: high" in text and "lookup_weather" in text


def test_the_preference_prep_renders_each_rows_level_and_tools(monkeypatch, tmp_path):
    tokenizer = load_cached_tokenizer(GPT_OSS_20B)
    default = tokenizer.apply_chat_template(ROW["prompt"], tokenize=False, add_generation_prompt=True)
    assert not _states_the_rows_columns(default), "the template states both by default: this case tests nothing"
    prepared = {}
    monkeypatch.setattr(data_setup, "log_script_dataset_examples", lambda datasets, *_: prepared.update(datasets))
    args = SimpleNamespace(
        tools_field="tools", reasoning_effort_field="effort", generate_eval_examples=True, num_eval_examples=1
    )
    training_config = SimpleNamespace(dataset_num_proc=None, output_dir=str(tmp_path), report_to=["none"])
    splits = DatasetDict(train=Dataset.from_list([ROW]), test=Dataset.from_list([ROW]))

    data_setup.prepare_script_preference_data(
        args, training_config, splits, tokenizer, is_vlm_data=False, generation_max_prompt_length=None
    )

    for split in ("train", "test"):
        assert _states_the_rows_columns(prepared[split][0]["prompt"]), f"the {split} render dropped a column"
    assert _states_the_rows_columns(tokenizer.decode(prepared["generate"][0]["input_ids"])), (
        "the generation prompt dropped a column"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
