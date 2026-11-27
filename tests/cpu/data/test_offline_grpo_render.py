#!/usr/bin/env python3
"""Offline-GRPO text prep: completions render as ``template(prompt + completion)`` minus the rendered-prompt prefix
(standalone raises on strict templates and injects a mid-sequence BOS) at the row's level, and the eval-generation
prompt's null cap means no cap."""

import json
from functools import partial
from types import SimpleNamespace

import pytest
from accelerate import PartialState
from datasets import Dataset, DatasetDict

PartialState()

import scripts.training.grpo.offline as offline_script
from halo.configs.grpo_offline import OfflineGRPOConfig, OfflineGRPOScriptArguments
from halo.configs.script_arguments import ModelArguments
from halo.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN
from halo.data.pipeline.text_rows import prepare_generative_row
from halo.distributed.parallelism_config import ParallelismConfig
from halo.trainers.grpo.offline.offline_rules import reject_unsupported_reference_input
from tests.common.models import GPT_OSS_20B
from tests.common.tokenizers import load_cached_tokenizer


class StrictTemplateTokenizer:
    """Qwen3.5-style STRICT template: raises on a conversation not opening with system/user."""

    bos_token = None

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        if not messages or messages[0]["role"] not in ("system", "user"):
            raise ValueError("Conversation must start with a system or user turn (strict template)")
        return "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)


class BosTemplateTokenizer:
    """Lenient BOS-emitting template: standalone completion renders gain a leading BOS."""

    bos_token = "<bos>"

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        return self.bos_token + "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)


ROW = {
    "prompt": [{"role": "user", "content": "What is 2+2?"}],
    "completions": [
        [{"role": "assistant", "content": "4"}],
        [{"role": "assistant", "content": "five"}],
    ],
    "rewards": [1.0, 0.0],
}


def test_offline_grpo_render_survives_strict_template():
    out = offline_script.render_offline_grpo_row(
        {**ROW, "completions": [list(c) for c in ROW["completions"]]}, StrictTemplateTokenizer(), None, None
    )
    assert out["prompt"] == "<|user|>What is 2+2?<|end|>"
    assert out["completions"] == ["<|assistant|>4<|end|>", "<|assistant|>five<|end|>"]


class EffortTemplateTokenizer(StrictTemplateTokenizer):
    """Writes the row's level ahead of the conversation, as an effort template's system block does."""

    def apply_chat_template(self, messages, tokenize=False, reasoning_effort=None, **kwargs):
        return f"<|effort|>{reasoning_effort}<|end|>" + super().apply_chat_template(messages, tokenize, **kwargs)


def test_offline_grpo_render_carries_the_row_level():
    """The prompt states the row's level, and each completion still splits off the prompt rendered at that level."""
    row = {**ROW, "completions": [list(c) for c in ROW["completions"]], "effort": "high"}
    out = offline_script.render_offline_grpo_row(row, EffortTemplateTokenizer(), None, "effort")
    assert out["prompt"] == "<|effort|>high<|end|><|user|>What is 2+2?<|end|>"
    assert out["completions"] == ["<|assistant|>4<|end|>", "<|assistant|>five<|end|>"]


class _RenderMapReached(Exception):
    """Raised by the stand-in chat-template map, carrying the row function ``main()`` handed it."""


def test_main_renders_each_rows_level(monkeypatch, tmp_path):
    """``main()``'s map is the one path from the configured level column to the template."""
    args = OfflineGRPOScriptArguments(dataset="org/offline", reasoning_effort_field="effort")
    config = OfflineGRPOConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, report_to="none")
    splits = DatasetDict(train=Dataset.from_list([ROW]), test=Dataset.from_list([ROW]))
    tokenizer = EffortTemplateTokenizer()

    def reach(dataset, function, **_kwargs):
        raise _RenderMapReached(function)

    stubs = {
        "parse_script_args": lambda *_: (args, config, ModelArguments(model_name_or_path="local"), SimpleNamespace()),
        "init_training_script": lambda *a, **k: SimpleNamespace(parallelism_config=ParallelismConfig()),
        "load_script_datasets": lambda *a, **k: (splits, False),
        "default_padded_attn_implementation": lambda *a, **k: "sdpa",
        "load_model_and_tokenizer": lambda *a, **k: (None, tokenizer),
        "apply_prompt_completion_window": lambda *a, **k: (tokenizer, None),
        "configure_adapter_training": lambda *a, **k: None,
        "_load_kl_reference": lambda *a, **k: None,
        "log_model_info": lambda *a, **k: None,
        "coordinated_map": reach,
    }
    for name, stub in stubs.items():
        monkeypatch.setattr(offline_script, name, stub)
    with pytest.raises(_RenderMapReached) as reached:
        offline_script.main()

    render = reached.value.args[0]
    row = {**ROW, "completions": [list(c) for c in ROW["completions"]], "effort": "high"}
    assert render(row)["prompt"] == "<|effort|>high<|end|><|user|>What is 2+2?<|end|>"


def test_main_generation_prompts_render_each_rows_level_and_tools(monkeypatch, tmp_path):
    """``main()``'s eval-generation map is the one path from the template columns to the generation prompts."""
    tokenizer = load_cached_tokenizer(GPT_OSS_20B)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "description": "Look up the weather.",
                "parameters": {"type": "object"},
            },
        }
    ]
    row = {**ROW, "effort": "high", "tools": json.dumps(tools)}
    args = OfflineGRPOScriptArguments(
        dataset="org/offline", reasoning_effort_field="effort", tools_field="tools", generate_eval_examples=True
    )
    config = OfflineGRPOConfig(output_dir=str(tmp_path), use_cpu=True, bf16=False, report_to="none")
    splits = DatasetDict(train=Dataset.from_list([row]), test=Dataset.from_list([row]))
    maps = []

    def reach_the_generation_map(dataset, function, **_kwargs):
        maps.append(function)
        if len(maps) == 2:
            raise _RenderMapReached(function)
        return dataset

    stubs = {
        "parse_script_args": lambda *_: (args, config, ModelArguments(model_name_or_path="local"), SimpleNamespace()),
        "init_training_script": lambda *a, **k: SimpleNamespace(parallelism_config=ParallelismConfig()),
        "load_script_datasets": lambda *a, **k: (splits, False),
        "default_padded_attn_implementation": lambda *a, **k: "sdpa",
        "load_model_and_tokenizer": lambda *a, **k: (None, tokenizer),
        "apply_prompt_completion_window": lambda *a, **k: (tokenizer, None),
        "configure_adapter_training": lambda *a, **k: None,
        "_load_kl_reference": lambda *a, **k: None,
        "log_model_info": lambda *a, **k: None,
        "coordinated_map": reach_the_generation_map,
    }
    for name, stub in stubs.items():
        monkeypatch.setattr(offline_script, name, stub)
    with pytest.raises(_RenderMapReached) as reached:
        offline_script.main()

    prompt = tokenizer.decode(reached.value.args[0](row)["input_ids"])
    assert "Reasoning: high" in prompt and "lookup_weather" in prompt
    default = tokenizer.decode(reached.value.args[0](ROW)["input_ids"])
    assert "Reasoning: high" not in default and "lookup_weather" not in default


def test_offline_grpo_rendered_supplied_references_are_refused_by_the_trainer():
    scores = [[-0.5, -0.75], [-1.25]]
    raw = Dataset.from_list([{**ROW, REF_PER_TOKEN_LOGPS_COLUMN: scores}])
    rendered = raw.map(
        partial(
            offline_script.render_offline_grpo_row,
            tokenizer=StrictTemplateTokenizer(),
            tools_field=None,
            reasoning_effort_field=None,
        )
    )
    assert rendered["completions"][0] == ["<|assistant|>4<|end|>", "<|assistant|>five<|end|>"]
    with pytest.raises(ValueError, match="Supplied ref_per_token_logps"):
        reject_unsupported_reference_input(rendered, None)


def test_offline_grpo_render_no_mid_sequence_bos():
    tok = BosTemplateTokenizer()
    out = offline_script.render_offline_grpo_row(
        {**ROW, "completions": [list(c) for c in ROW["completions"]]}, tok, None, None
    )
    for completion in out["completions"]:
        assert tok.bos_token not in completion, f"mid-sequence BOS in completion: {completion!r}"
        # The prefix-strip invariant: prompt + completion reconstructs the full rendered conversation.
    full = tok.apply_chat_template(ROW["prompt"] + ROW["completions"][0])
    assert out["prompt"] + out["completions"][0] == full


class HFTruncationTokenizer:
    """HF's truncation contract: ``truncation=True`` with ``max_length=None`` resolves the bound against
    ``model_max_length``."""

    model_max_length = 4
    pad_token_id = 0
    padding_side = "right"

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None, padding=False):
        ids = [100 + i for i, _ in enumerate(text.split())]
        if truncation:
            ids = ids[: self.model_max_length if max_length is None else max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        return " ".join(m["content"] for m in messages)

    def pad(self, encoded, **kwargs):
        return encoded


def test_generative_row_null_prompt_cap_is_no_cap():
    """``truncation=True`` with a nullable cap would cut an uncapped prompt at ``model_max_length``, silently
    shortening the prompts eval generations are judged on."""
    row = {"prompt": [{"role": "user", "content": "one two three four five six"}]}
    out = prepare_generative_row(row, HFTruncationTokenizer(), max_length=None)
    assert len(out["input_ids"]) == 6, f"uncapped generation prompt truncated: {out['input_ids']}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
