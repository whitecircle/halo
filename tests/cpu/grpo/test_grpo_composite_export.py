#!/usr/bin/env python
"""The real offline GRPO entry script on a tiny composite, on CPU, with only the loader's GPU placement replaced: its
export directory must hold what a rollout engine builds the composite's processor from, beside the run's tokenizer, with
the served context the tokenizer shipped and not the run's window."""

import json
import sys
from pathlib import Path

import pytest
import yaml
from transformers import AutoProcessor, PreTrainedTokenizerBase
from transformers.utils import CONFIG_NAME, PROCESSOR_NAME

import halo.model_loading.parallel.loader as model_loader
import scripts.training.grpo.offline as offline_grpo_script
from tests.common.gloo import run_gloo_ranks
from tests.common.tiny_models import (
    TINY_DENSE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    TINY_UNWRAPPED_COMPOSITE_FAMILIES,
    build_tiny_family_checkpoint,
)
from tests.common.tiny_processors import TINY_PROCESSOR_MODEL_TYPES, tiny_processor

_FAMILIES = {**TINY_MOE_VLM_FAMILIES, **TINY_UNWRAPPED_COMPOSITE_FAMILIES}
_WINDOW_HALF = 32
_TOKENIZER_FILE = "tokenizer.json"


def _message(role: str, words: str) -> dict:
    return {"role": role, "content": [{"type": "text", "text": words}]}


def _rows(path: Path) -> None:
    row = {
        "prompt": [_message("user", "Describe the image .")],
        "completions": [[_message("assistant", "A square of noise .")], [_message("assistant", "A square .")]],
        "rewards": [1.0, -1.0],
    }
    path.write_text("".join(f"{json.dumps(row)}\n" for _ in range(8)))


def _config(tmp_path: Path, checkpoint: Path, **overrides) -> Path:
    fields = {
        "model_name_or_path": str(checkpoint),
        "attn_implementation": "sdpa",
        "moe_balancing": "none",
        "dataset": [str(tmp_path / "rows.jsonl")],
        "test_size": 0.25,
        "loss_type": "grpo",
        "kl_beta": 0.0,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "max_steps": 1,
        "gradient_checkpointing": False,
        "max_prompt_length": _WINDOW_HALF,
        "max_completion_length": _WINDOW_HALF,
        "output_dir": str(tmp_path / "run"),
        "save_strategy": "steps",
        "save_steps": 1,
        "save_only_model": True,
        "eval_strategy": "no",
        "report_to": "none",
        "use_cpu": True,
        "bf16": False,
        "use_liger_kernel": False,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
        "dataloader_drop_last": True,
        "generate_eval_examples": False,
    }
    path = tmp_path / "offline.yaml"
    path.write_text(yaml.safe_dump(fields | overrides))
    return path


def _run_the_script(_rank: int, config: str) -> None:
    load_on_the_gpu = model_loader.load_whole_model
    model_loader.load_whole_model = lambda *args, **kwargs: load_on_the_gpu(*args, **{**kwargs, "device_map": "cpu"})
    sys.argv = ["offline.py", config]
    offline_grpo_script.main()


def _export(tmp_path: Path, checkpoint: Path, **overrides) -> Path:
    _rows(tmp_path / "rows.jsonl")
    run_gloo_ranks(_run_the_script, 1, str(_config(tmp_path, checkpoint, **overrides)))
    return tmp_path / "run" / "checkpoint-1"


@pytest.mark.parametrize("family", TINY_PROCESSOR_MODEL_TYPES)
def test_a_composite_export_holds_what_an_engine_builds_its_processor_from(tmp_path, family):
    processor = tiny_processor(family)
    checkpoint = tmp_path / "base"
    build_tiny_family_checkpoint(_FAMILIES[family], str(checkpoint), processor.tokenizer)
    processor.save_pretrained(checkpoint)

    export = _export(tmp_path, checkpoint)

    assert (export / CONFIG_NAME).is_file() and (export / _TOKENIZER_FILE).is_file()
    reloaded = AutoProcessor.from_pretrained(export, local_files_only=True)
    assert type(reloaded) is type(processor)
    assert type(reloaded.image_processor) is type(processor.image_processor)
    assert reloaded.tokenizer.model_max_length != 2 * _WINDOW_HALF, "the run's window is not the served context"

    (export / PROCESSOR_NAME).unlink()
    with pytest.raises(OSError, match="image processor"):
        AutoProcessor.from_pretrained(export, local_files_only=True)


def test_a_composite_shipping_no_processor_exports_its_tokenizer(tmp_path):
    processor = tiny_processor("qwen3_5_moe")
    checkpoint = tmp_path / "base"
    build_tiny_family_checkpoint(TINY_MOE_VLM_FAMILIES["qwen3_5_moe"], str(checkpoint), processor.tokenizer)
    processor.tokenizer.chat_template = processor.chat_template
    processor.tokenizer.save_pretrained(checkpoint)

    export = _export(tmp_path, checkpoint)

    assert (export / _TOKENIZER_FILE).is_file()
    assert not (export / PROCESSOR_NAME).exists()


def test_a_text_checkpoint_export_holds_its_tokenizer_alone(tmp_path):
    processor = tiny_processor("qwen3_5_moe")
    checkpoint = tmp_path / "base"
    build_tiny_family_checkpoint(TINY_DENSE_FAMILIES["qwen3"], str(checkpoint), processor.tokenizer)
    processor.tokenizer.chat_template = processor.chat_template
    processor.tokenizer.save_pretrained(checkpoint)

    export = _export(tmp_path, checkpoint)

    assert (export / _TOKENIZER_FILE).is_file()
    assert not (export / PROCESSOR_NAME).exists()
    assert isinstance(AutoProcessor.from_pretrained(export, local_files_only=True), PreTrainedTokenizerBase)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
