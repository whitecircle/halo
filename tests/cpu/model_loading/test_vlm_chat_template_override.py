#!/usr/bin/env python
"""A VLM run's chat-template override reaches the processor it renders and exports through. A processor carries its
own template, written over its tokenizer's at save."""

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen3_5Config, Qwen3_5ForConditionalGeneration

from halo.checkpoint.parallel.save import save_checkpoint
from halo.configs.script_arguments import CommonScriptArguments
from halo.data.preprocessed import prepare_dataset_tool
from halo.distributed.tensor_parallel.hf_plan import is_input_embedding_tp_sharded
from halo.entrypoints.model_setup import install_resolved_tokenizer
from halo.model_loading.tokenizer_setup import configure_run_tokenizer
from tests.common.checkpoint_io import save_context
from tests.common.models import QWEN3_5_9B, TINY_QWEN35_CONFIG
from tests.common.repo import load_script_module
from tests.common.tokenizers import load_cached_processor

OVERRIDE = "{% for message in messages %}<run>{% for part in message['content'] %}{{ part['text'] }}{% endfor %}</run>{% endfor %}"
MESSAGES = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
OVERRIDE_RENDER = "<run>hello</run>"
PROCESSOR_ONLY = OVERRIDE.replace("run>", "processor>")
PROCESSOR_ONLY_RENDER = "<processor>hello</processor>"


def _render(processor) -> str:
    return processor.apply_chat_template(MESSAGES, tokenize=False)


def _setup_run(args: CommonScriptArguments, processor=None):
    """After the tokenizer seam."""
    processor = processor or load_cached_processor(QWEN3_5_9B)
    tokenizer = configure_run_tokenizer(
        args, None, processor.tokenizer, None, embeddings_sharded=is_input_embedding_tp_sharded
    )
    return install_resolved_tokenizer(processor, tokenizer)


def _export(processing_class, output_dir: str) -> None:
    save_checkpoint(save_context(nn.Linear(4, 4), toolkit_fsdp2=True, tokenizer=processing_class), output_dir)


def test_a_training_run_renders_and_exports_its_override(tmp_path):
    processing_class = _setup_run(CommonScriptArguments(chat_template=OVERRIDE, force_chat_template=True))
    assert _render(processing_class) == OVERRIDE_RENDER, "the VLM rows render on the checkpoint's template"

    _export(processing_class, str(tmp_path))

    exported = AutoProcessor.from_pretrained(str(tmp_path))
    assert _render(exported) == OVERRIDE_RENDER, "the export ships the checkpoint's template, not the run's"
    assert exported.tokenizer.chat_template == OVERRIDE


def test_a_run_without_an_override_keeps_the_processors_own_template(tmp_path):
    """One the checkpoint shipped for the processor alone stays the processor's."""
    processor = load_cached_processor(QWEN3_5_9B)
    processor.chat_template = PROCESSOR_ONLY
    assert processor.tokenizer.chat_template != PROCESSOR_ONLY, "premise: the two templates differ"
    processing_class = _setup_run(CommonScriptArguments(), processor)

    assert _render(processing_class) == PROCESSOR_ONLY_RENDER
    _export(processing_class, str(tmp_path))
    assert _render(AutoProcessor.from_pretrained(str(tmp_path))) == PROCESSOR_ONLY_RENDER


def test_prepare_dataset_renders_vlm_rows_with_the_override():
    load_cached_processor(QWEN3_5_9B)  # Skips where the processor is not cached.
    args = SimpleNamespace(
        model_name_or_path=QWEN3_5_9B,
        trust_remote_code=False,
        min_pixels=None,
        max_pixels=None,
        pad_token=None,
        eos_token=None,
        bos_token=None,
        chat_template=OVERRIDE,
    )

    assert _render(prepare_dataset_tool.load_vlm_processor(args)) == OVERRIDE_RENDER


def test_patch_vocab_exports_the_override(tmp_path, monkeypatch):
    source, out = tmp_path / "source", tmp_path / "patched"
    torch.manual_seed(0)
    vision = {"depth": 1, "hidden_size": 16, "intermediate_size": 16, "num_heads": 2}
    config = Qwen3_5Config(
        text_config=dict(TINY_QWEN35_CONFIG),
        vision_config={**vision, "out_hidden_size": TINY_QWEN35_CONFIG["hidden_size"]},
    )
    Qwen3_5ForConditionalGeneration(config).save_pretrained(source)
    load_cached_processor(QWEN3_5_9B).save_pretrained(source)
    patch_vocab = load_script_module("scripts/before_training/patch_vocab.py")
    monkeypatch.setattr(
        sys,
        "argv",
        ["patch_vocab.py", "--model_id", str(source), "--output_dir", str(out), "--chat_template", OVERRIDE],
    )

    patch_vocab.main()

    assert _render(AutoProcessor.from_pretrained(str(out))) == OVERRIDE_RENDER


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
