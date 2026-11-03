#!/usr/bin/env python
"""A preprocessed SFT run must end with the chat template its rows were rendered with, the one it exports. A run
applies its own ``chat_template`` only where the checkpoint ships none or ``force_chat_template`` is set, so the
startup check holds its final tokenizer or processor to the template the bake records."""

import json
import logging
import os

import pytest
from datasets import Dataset, DatasetDict

from halo.configs.sft import SFTScriptArguments
from halo.data.preprocessed.metadata import (
    PreprocessingConfig,
    load_preprocessed_metadata,
    verify_preprocessed_dataset,
)
from halo.data.preprocessed.preparation import preprocess_dataset
from halo.data.preprocessed.publishing import apply_tokenizer_overrides, load_vlm_processor
from halo.data.sources.paths import METADATA_FILE
from halo.distributed.tensor_parallel.hf_plan import is_input_embedding_tp_sharded
from halo.entrypoints.model_setup import install_resolved_tokenizer
from halo.model_loading.tokenizer_setup import configure_run_tokenizer
from tests.common.models import QWEN3_0_6B, QWEN3_5_9B
from tests.common.tokenizers import load_cached_processor, load_cached_tokenizer

MAX_LENGTH = 128
TURNS = [{"role": "user", "content": "Two plus two?"}, {"role": "assistant", "content": "Four."}]
OVERRIDE = "{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}</{{ m['role'] }}>{% endfor %}"
OVERRIDE_RENDER = "<user>Two plus two?</user><assistant>Four.</assistant>"
OTHER_CHECKPOINT_TEMPLATE = "{% for m in messages %}[{{ m['role'] }}] {{ m['content'] }}\n{% endfor %}"
# The VLM path hands the template content parts.
VLM_OVERRIDE = "{% for m in messages %}<run>{% for p in m['content'] %}{{ p['text'] }}{% endfor %}</run>{% endfor %}"


def _bake(output_dir, rows: dict, tokenizer_or_processor, **overrides):
    """The ``metadata.json`` a bake of ``rows`` writes, read back the way training reads it, and the baked rows."""
    config = PreprocessingConfig(max_length=MAX_LENGTH, num_proc=1, **overrides)
    baked = preprocess_dataset(
        DatasetDict({"train": Dataset.from_dict(rows)}), tokenizer_or_processor, config, output_dir=str(output_dir)
    )["train"]
    return load_preprocessed_metadata(str(output_dir)), baked


def _bake_chat(output_dir, **overrides):
    """A two-turn Qwen3 chat bake as ``prepare_dataset.py`` runs it: overrides applied to the tokenizer first."""
    config_overrides = {"model_name_or_path": QWEN3_0_6B, **overrides}
    tokenizer = load_cached_tokenizer(QWEN3_0_6B)
    apply_tokenizer_overrides(tokenizer, PreprocessingConfig(**config_overrides))
    metadata, baked = _bake(output_dir, {"prompt": [TURNS]}, tokenizer, **config_overrides)
    return metadata, tokenizer.decode(baked[0]["input_ids"])


def _run_tokenizer(args: SFTScriptArguments, shipped_template: str | None = None):
    """The tokenizer a training run ends with after its tokenizer setup; ``shipped_template`` stands in for a
    checkpoint shipping another template than the bake's."""
    tokenizer = load_cached_tokenizer(QWEN3_0_6B)
    if shipped_template is not None:
        tokenizer.chat_template = shipped_template
    return configure_run_tokenizer(args, None, tokenizer, MAX_LENGTH, embeddings_sharded=is_input_embedding_tp_sharded)


def _verify(metadata, args: SFTScriptArguments, processing_class, model: str = QWEN3_0_6B) -> None:
    verify_preprocessed_dataset(
        metadata,
        required_max_length=MAX_LENGTH,
        required_model=model,
        render_args=args,
        run_processing_class=processing_class,
    )


def _as_predating_the_recorded_template(output_dir):
    """Re-read the artifact with ``rendered_chat_template`` dropped from its stamp, as an earlier build wrote it."""
    path = os.path.join(output_dir, METADATA_FILE)
    with open(path) as f:
        payload = json.load(f)
    del payload["rendered_chat_template"]
    with open(path, "w") as f:
        json.dump(payload, f)
    return load_preprocessed_metadata(str(output_dir))


def test_the_bake_records_the_template_it_rendered_with(tmp_path):
    metadata, text = _bake_chat(tmp_path / "override", chat_template=OVERRIDE)
    assert OVERRIDE_RENDER in text, f"premise: the rows render with the override, got {text!r}"
    assert metadata.rendered_chat_template == OVERRIDE

    stock, _ = _bake_chat(tmp_path / "stock")
    assert stock.rendered_chat_template == load_cached_tokenizer(QWEN3_0_6B).chat_template


def test_a_stated_override_without_force_is_refused(tmp_path):
    """Every stated knob matches the artifact, yet the run keeps the checkpoint's template and would export it."""
    metadata, _ = _bake_chat(tmp_path, chat_template=OVERRIDE)
    args = SFTScriptArguments(chat_template=OVERRIDE)
    tokenizer = _run_tokenizer(args)
    assert tokenizer.chat_template != OVERRIDE, "premise: the checkpoint's own template survives the run's setup"

    with pytest.raises(ValueError, match="force_chat_template") as raised:
        _verify(metadata, args, tokenizer)
    assert "--chat_template override" in str(raised.value)
    assert f"the template {QWEN3_0_6B} ships" in str(raised.value)


def test_a_forced_override_trains_and_exports_the_rows_template(tmp_path):
    metadata, _ = _bake_chat(tmp_path, chat_template=OVERRIDE)
    args = SFTScriptArguments(chat_template=OVERRIDE, force_chat_template=True)
    tokenizer = _run_tokenizer(args)
    assert tokenizer.chat_template == OVERRIDE

    _verify(metadata, args, tokenizer)


def test_an_unstated_template_against_an_override_bake_is_refused(tmp_path, caplog):
    """The warning's remedy names the force flag, since stating chat_template alone changes nothing."""
    metadata, _ = _bake_chat(tmp_path, chat_template=OVERRIDE)
    args = SFTScriptArguments()

    with caplog.at_level(logging.WARNING), pytest.raises(ValueError, match="force_chat_template"):
        _verify(metadata, args, _run_tokenizer(args))
    unstated = [record.message for record in caplog.records if "does not state" in record.message]
    assert unstated and "force_chat_template: true" in unstated[0], unstated


def test_a_stock_bake_holds_the_run_to_the_template_its_rows_saw(tmp_path):
    metadata, _ = _bake_chat(tmp_path)
    args = SFTScriptArguments()
    _verify(metadata, args, _run_tokenizer(args))

    with pytest.raises(ValueError, match=f"the template {QWEN3_0_6B} ships"):
        _verify(metadata, args, _run_tokenizer(args, shipped_template=OTHER_CHECKPOINT_TEMPLATE))


def test_a_stamp_predating_the_recorded_template(tmp_path):
    """Without the field, only a recorded override says what rendered; a stock bake records nothing to compare."""
    _bake_chat(tmp_path / "stock")
    stock = _as_predating_the_recorded_template(tmp_path / "stock")
    args = SFTScriptArguments()
    _verify(stock, args, _run_tokenizer(args, shipped_template=OTHER_CHECKPOINT_TEMPLATE))

    _bake_chat(tmp_path / "override", chat_template=OVERRIDE)
    override = _as_predating_the_recorded_template(tmp_path / "override")
    unforced = SFTScriptArguments(chat_template=OVERRIDE)
    with pytest.raises(ValueError, match="force_chat_template"):
        _verify(override, unforced, _run_tokenizer(unforced))
    forced = SFTScriptArguments(chat_template=OVERRIDE, force_chat_template=True)
    _verify(override, forced, _run_tokenizer(forced))


def test_a_text_artifact_renders_no_template_to_hold(tmp_path):
    tokenizer = load_cached_tokenizer(QWEN3_0_6B)
    metadata, _ = _bake(
        tmp_path, {"text": ["Two plus two is four."]}, tokenizer, model_name_or_path=QWEN3_0_6B, mode="text"
    )
    assert metadata.rendered_chat_template is None

    args = SFTScriptArguments()
    _verify(metadata, args, _run_tokenizer(args, shipped_template=OTHER_CHECKPOINT_TEMPLATE))


def _vlm_run(args: SFTScriptArguments):
    """The processing class a VLM run hands its trainer, after the tokenizer seam."""
    processor = load_cached_processor(QWEN3_5_9B)
    tokenizer = configure_run_tokenizer(
        args, None, processor.tokenizer, MAX_LENGTH, embeddings_sharded=is_input_embedding_tp_sharded
    )
    return install_resolved_tokenizer(processor, tokenizer)


def test_a_vlm_run_is_held_to_the_template_its_processor_rendered_with(tmp_path):
    """A processor renders and exports with a template of its own, so both sides read the processor's."""
    load_cached_processor(QWEN3_5_9B)  # skips where the processor is not cached
    config = PreprocessingConfig(model_name_or_path=QWEN3_5_9B, is_vlm=True, chat_template=VLM_OVERRIDE)
    metadata, _ = _bake(
        tmp_path,
        {"prompt": [TURNS]},
        load_vlm_processor(config),
        model_name_or_path=QWEN3_5_9B,
        is_vlm=True,
        chat_template=VLM_OVERRIDE,
    )
    assert metadata.rendered_chat_template == VLM_OVERRIDE

    unforced = SFTScriptArguments(chat_template=VLM_OVERRIDE)
    with pytest.raises(ValueError, match="force_chat_template"):
        _verify(metadata, unforced, _vlm_run(unforced), QWEN3_5_9B)
    forced = SFTScriptArguments(chat_template=VLM_OVERRIDE, force_chat_template=True)
    _verify(metadata, forced, _vlm_run(forced), QWEN3_5_9B)


def test_a_vlm_stamp_predating_the_recorded_template_is_not_held_to_its_override(tmp_path):
    """Such a stamp's processor may have rendered with its own template despite the recorded override, so the
    override proves nothing about the rows and a run keeping the processor's template is not refused."""
    load_cached_processor(QWEN3_5_9B)
    config = PreprocessingConfig(model_name_or_path=QWEN3_5_9B, is_vlm=True, chat_template=VLM_OVERRIDE)
    _bake(
        tmp_path,
        {"prompt": [TURNS]},
        load_vlm_processor(config),
        model_name_or_path=QWEN3_5_9B,
        is_vlm=True,
        chat_template=VLM_OVERRIDE,
    )
    legacy = _as_predating_the_recorded_template(tmp_path)

    unforced = SFTScriptArguments(chat_template=VLM_OVERRIDE)
    _verify(legacy, unforced, _vlm_run(unforced), QWEN3_5_9B)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
