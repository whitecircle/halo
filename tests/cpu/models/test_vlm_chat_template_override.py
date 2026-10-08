#!/usr/bin/env python
"""A VLM run's chat-template override reaches the processor it renders and exports through.

A processor carries its own ``chat_template``, apart from its tokenizer's: ``apply_chat_template``
renders with it, and ``save_pretrained`` writes it over the ``chat_template.jinja`` its tokenizer
wrote. The toolkit applies ``chat_template`` / ``force_chat_template`` to the tokenizer, so unless the
processor adopts it a VLM run renders its rows on the checkpoint's original template and ships that
template too. Each entry point that sets the override is driven here on a cached Qwen3.5 processor:
the training seam (``setup_model_and_tokenizer`` + ``install_resolved_tokenizer``) through the FSDP2
saver, ``prepare_dataset --vlm``'s processor setup, and ``patch_vocab.py``'s export.

    python tests/cpu/models/test_vlm_chat_template_override.py
"""

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from accelerate import PartialState
from transformers import AutoProcessor, Qwen3_5Config, Qwen3_5ForConditionalGeneration

from src.args.common_script_args import CommonScriptArguments
from src.distributed.checkpoint.save import save_fsdp2_checkpoint
from src.distributed.tensor_parallel.state_dict import input_embeddings_tp_sharded
from src.models.loading.tokenizer_setup import setup_model_and_tokenizer
from src.training.script_runner import install_resolved_tokenizer
from tests.common.models import QWEN3_5_9B, TINY_QWEN35_CONFIG
from tests.common.tokenizers import load_cached_processor
from tests.common.utils import load_script_module

PartialState()  # the setup seam and the savers log through accelerate's logger

OVERRIDE = "{% for message in messages %}<run>{% for part in message['content'] %}{{ part['text'] }}{% endfor %}</run>{% endfor %}"
MESSAGES = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
OVERRIDE_RENDER = "<run>hello</run>"
PROCESSOR_ONLY = OVERRIDE.replace("run>", "processor>")
PROCESSOR_ONLY_RENDER = "<processor>hello</processor>"


def _render(processor) -> str:
    return processor.apply_chat_template(MESSAGES, tokenize=False)


def _setup_run(args: CommonScriptArguments, processor=None):
    """The processing class a VLM training script hands its trainer, after the tokenizer seam."""
    processor = processor or load_cached_processor(QWEN3_5_9B)
    tokenizer = setup_model_and_tokenizer(
        args, None, processor.tokenizer, None, embeddings_sharded=input_embeddings_tp_sharded
    )
    return install_resolved_tokenizer(processor, tokenizer)


def _export(processing_class, output_dir: str) -> None:
    ctx = SimpleNamespace(
        model=nn.Linear(4, 4),
        is_save_rank=True,
        max_shard_size="5GB",
        training_checkpoint=False,
        tokenizer=processing_class,
    )
    save_fsdp2_checkpoint(ctx, output_dir)


def test_a_training_run_renders_and_exports_its_override(tmp_path):
    processing_class = _setup_run(CommonScriptArguments(chat_template=OVERRIDE, force_chat_template=True))
    assert _render(processing_class) == OVERRIDE_RENDER, "the VLM rows render on the checkpoint's template"

    _export(processing_class, str(tmp_path))

    exported = AutoProcessor.from_pretrained(str(tmp_path))
    assert _render(exported) == OVERRIDE_RENDER, "the export ships the checkpoint's template, not the run's"
    assert exported.tokenizer.chat_template == OVERRIDE


def test_a_run_without_an_override_keeps_the_processors_own_template(tmp_path):
    """Only a template the run chose moves to the processor: one the checkpoint shipped for the processor
    alone (a legacy ``chat_template.json`` beside a tokenizer template) stays the processor's."""
    processor = load_cached_processor(QWEN3_5_9B)
    processor.chat_template = PROCESSOR_ONLY
    assert processor.tokenizer.chat_template != PROCESSOR_ONLY, "premise: the two templates differ"
    processing_class = _setup_run(CommonScriptArguments(), processor)

    assert _render(processing_class) == PROCESSOR_ONLY_RENDER
    _export(processing_class, str(tmp_path))
    assert _render(AutoProcessor.from_pretrained(str(tmp_path))) == PROCESSOR_ONLY_RENDER


def test_prepare_dataset_renders_vlm_rows_with_the_override():
    prepare_dataset = load_script_module("scripts/before_training/prepare_dataset.py")
    load_cached_processor(QWEN3_5_9B)  # skips where the processor is not cached
    args = SimpleNamespace(
        model_name=QWEN3_5_9B,
        trust_remote_code=False,
        min_pixels=None,
        max_pixels=None,
        pad_token=None,
        eos_token=None,
        bos_token=None,
        chat_template=OVERRIDE,
    )

    assert _render(prepare_dataset.setup_vlm_processor(args)) == OVERRIDE_RENDER


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
