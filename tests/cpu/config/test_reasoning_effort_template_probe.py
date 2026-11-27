#!/usr/bin/env python
"""``reasoning_effort_field`` is refused at setup under a chat template that renders no level, in a closed conversation
or on its generation prompt, where every row's level would vanish from its render, and accepted under one that states
the level in either or refuses one it does not take. A VLM run is probed on its processor's own template, which renders
its rows, and not on the template of the tokenizer the processor holds."""

import types

import pytest
from datasets import Dataset, DatasetDict

from halo.configs.script_arguments import CommonScriptArguments
from halo.data.pipeline.vlm_rows import prepare_vlm_dataset
from halo.distributed.tensor_parallel.hf_plan import is_input_embedding_tp_sharded
from halo.model_loading.tokenizer_setup import configure_run_tokenizer, mark_processor_rendered_rows
from tests.common.models import (
    COMMAND_A_PLUS,
    GLM5_3_FLASH,
    GPT_OSS_20B,
    MISTRAL3_119B_MOE,
    PINNED_REVISIONS,
    QWEN3_0_6B,
    QWEN3_VL_2B,
)
from tests.common.repo import REPO_ROOT
from tests.common.tokenizers import load_cached_processor, load_cached_tokenizer

TEMPLATES = REPO_ROOT / "jinja-templates"
EFFORT_TEMPLATE = "qwen3/qwen3.6-reasoning-effort.jinja"
STATING = (EFFORT_TEMPLATE, "gemma4/gemma4-reasoning-effort.jinja", "gpt-oss/gpt-oss-harmony.jinja")
# The bundled Mistral 4 template pins its level at none.
LEVEL_BLIND = ("qwen3/qwen3-native.jinja", "glm/glm-native.jinja", "mistral4/mistral4-instruct.jinja")
_TURNS = "{%- for message in messages %}<|{{ message.role }}|>{{ message.content }}<|end|>{%- endfor %}"
# Command A's shape: the level is read only as 'none', and only in the generation prompt.
GENERATION_PROMPT_READER = (
    _TURNS + "{%- if add_generation_prompt %}<|assistant|>"
    "{%- if reasoning_effort is defined and reasoning_effort | lower == 'none' %}<|no_think|>{%- endif %}"
    "{%- endif %}"
)
GENERATION_PROMPT_BLIND = _TURNS + "{%- if add_generation_prompt %}<|assistant|><|no_think|>{%- endif %}"
MEDIUM_READER = _TURNS + "{%- if reasoning_effort is defined and reasoning_effort == 'medium' %}<|medium|>{%- endif %}"


def _configure(tokenizer, template: str | None = None):
    chat_template = (
        {} if template is None else {"chat_template": str(TEMPLATES / template), "force_chat_template": True}
    )
    args = CommonScriptArguments(reasoning_effort_field="effort", **chat_template)
    return configure_run_tokenizer(args, None, tokenizer, None, embeddings_sharded=is_input_embedding_tp_sharded)


def _shipping(template: str):
    tokenizer = load_cached_tokenizer(QWEN3_0_6B)
    tokenizer.chat_template = template
    return tokenizer


def test_a_stock_template_rendering_no_level_is_refused():
    with pytest.raises(
        ValueError,
        match=rf"{QWEN3_0_6B} ships renders .* no level, 'none', 'low', 'medium' and 'high', whether .*qwen3\.6-",
    ):
        _configure(load_cached_tokenizer(QWEN3_0_6B))


def test_a_template_reading_the_level_only_on_its_generation_prompt_is_accepted():
    """The level shows on no closed conversation, only on the generation prompt an online run's prompts end on."""
    _configure(_shipping(GENERATION_PROMPT_READER))


def test_a_template_reading_only_the_medium_level_is_accepted():
    """``medium`` is a row's commonest level, so a template stating that one alone still reads rows' levels."""
    _configure(_shipping(MEDIUM_READER))


def test_a_template_blind_to_the_level_on_its_generation_prompt_is_refused():
    """Its open render differs from its closed one at every level, which says nothing of the level."""
    with pytest.raises(
        ValueError, match="'none', 'low', 'medium' and 'high', whether closed by an assistant turn or open on its"
    ):
        _configure(_shipping(GENERATION_PROMPT_BLIND))


@pytest.mark.parametrize("template", LEVEL_BLIND)
def test_a_run_template_rendering_no_level_is_refused(template):
    with pytest.raises(ValueError, match="the run's chat_template renders a conversation alike"):
        _configure(load_cached_tokenizer(QWEN3_0_6B), template)


@pytest.mark.parametrize("template", STATING)
def test_a_template_stating_the_level_is_accepted(template):
    _configure(load_cached_tokenizer(QWEN3_0_6B), template)


@pytest.mark.parametrize(
    "model_name",
    [GPT_OSS_20B, MISTRAL3_119B_MOE, COMMAND_A_PLUS],
    ids=["gpt-oss", "mistral4-refuses-low", "command-a-reads-none-on-its-generation-prompt"],
)
def test_a_shipped_template_reading_the_level_is_accepted(model_name):
    _configure(load_cached_tokenizer(model_name))


def _vlm_args() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        dataset="the loaded dataset",
        conversation_field="conversation",
        images_field=None,
        system_prompt=None,
        model_supports_system_role=True,
        tools_field=None,
        reasoning_effort_field="effort",
        interleaved_thinking=False,
        train_on_completions_only=False,
        assistant_message_template=None,
    )


def _prepare(processor):
    dataset = Dataset.from_list([{"conversation": [{"role": "user", "content": "hi"}], "effort": "low"}])
    return prepare_vlm_dataset(
        DatasetDict({"train": dataset}), _vlm_args(), processor, processor.tokenizer, 128, 1, eos_token_ids=frozenset()
    )


def test_a_vlm_run_is_probed_on_its_processors_own_template(tmp_path, monkeypatch):
    """Qwen3-VL's tokenizer stating the level proves nothing for rows its processor renders."""
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf_datasets"))
    processor = load_cached_processor(QWEN3_VL_2B, revision=PINNED_REVISIONS[QWEN3_VL_2B])
    processor.tokenizer.chat_template = (TEMPLATES / EFFORT_TEMPLATE).read_text()

    with pytest.raises(ValueError, match="renders a conversation alike at no level"):
        _prepare(processor)
    _prepare(load_cached_processor(GLM5_3_FLASH, revision=PINNED_REVISIONS[GLM5_3_FLASH]))


def test_a_vlm_runs_tokenizer_template_leaves_the_verdict_to_its_processors(tmp_path, monkeypatch):
    """Rows its processor renders never meet a level-blind template on the tokenizer it holds, so the run's setup
    refuses nothing there once marked, and the processor's own template, which states the level, takes the rows."""
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf_datasets"))
    processor = load_cached_processor(GLM5_3_FLASH, revision=PINNED_REVISIONS[GLM5_3_FLASH])
    processor.tokenizer.chat_template = (TEMPLATES / "glm/glm-native.jinja").read_text()
    with pytest.raises(ValueError, match="renders a conversation alike at no level"):
        _configure(processor.tokenizer)

    mark_processor_rendered_rows(processor.tokenizer)
    _configure(processor.tokenizer)
    assert _prepare(processor)["train"].num_rows == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
