#!/usr/bin/env python
"""A bf16 conversion of a training checkpoint is still that run's resume source.

``convert_to_bf16.py`` changes no state a resume reads, so the converted directory must resume
exactly as its source does. For a ``merge_expert_lora_on_save`` checkpoint that means the marker,
which keeps the resume on base plus ``resume_adapter/``, and that adapter directory arriving whole
behind it; for a precompute DPO/KTO run, ``reference_logps.pt``; for every run, ``scheduler.pt``.
The adapter directory must also stay out of the conversion's own load and write: its
``adapter_config.json`` sits one level down, where ``from_pretrained`` never looks.

    python tests/cpu/checkpoint/test_convert_to_bf16_resume_state.py
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from scripts.after_training.convert_to_bf16 import convert_to_bf16
from src.checkpoint.adapters import EXPERT_LORA_PEFT_TYPE
from src.checkpoint.format import (
    ADAPTER_CONFIG_FILE,
    ADAPTER_SAFETENSORS_FILE,
    REFERENCE_LOGPS_FILE,
    RESUME_ADAPTER_DIR,
    RESUME_ADAPTER_MARKER_FILE,
    SCHEDULER_STATE_FILE,
    read_checkpoint_key_set,
    write_resume_adapter_marker,
)
from src.training.environment import _classify_resume_checkpoint
from tests.common.models import TINY_QWEN3_CONFIG

_RESUME_ENTRIES = (
    RESUME_ADAPTER_MARKER_FILE,
    f"{RESUME_ADAPTER_DIR}/{ADAPTER_SAFETENSORS_FILE}",
    f"{RESUME_ADAPTER_DIR}/{ADAPTER_CONFIG_FILE}",
    REFERENCE_LOGPS_FILE,
    SCHEDULER_STATE_FILE,
)


def _merged_training_checkpoint(path: Path) -> Path:
    """A merge-on-save checkpoint as the trainer leaves it: merged weights at the root, the resume
    adapter beside them under its marker, and the run's resume sidecars."""
    torch.manual_seed(0)
    Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG)).save_pretrained(path)
    adapter_dir = path / RESUME_ADAPTER_DIR
    adapter_dir.mkdir()
    save_file(
        {"model.layers.0.mlp.experts.down_proj.lora_A": torch.ones(2, 4)}, str(adapter_dir / ADAPTER_SAFETENSORS_FILE)
    )
    (adapter_dir / ADAPTER_CONFIG_FILE).write_text(json.dumps({"peft_type": EXPERT_LORA_PEFT_TYPE}))
    write_resume_adapter_marker(str(path))
    (path / REFERENCE_LOGPS_FILE).write_bytes(REFERENCE_LOGPS_FILE.encode())
    (path / SCHEDULER_STATE_FILE).write_bytes(SCHEDULER_STATE_FILE.encode())
    return path


def test_a_converted_merged_checkpoint_resumes_like_its_source(tmp_path):
    source = _merged_training_checkpoint(tmp_path / "checkpoint-3")
    out = tmp_path / "bf16"

    convert_to_bf16(str(source), str(out), "causal_lm")

    assert _classify_resume_checkpoint(str(out)) == "merged_adapter", (
        "without its marker a resume refuses the converted directory as a merged checkpoint missing its adapter"
    )
    for entry in _RESUME_ENTRIES:
        assert (out / entry).is_file(), f"{entry} was dropped by the conversion"
        assert (out / entry).read_bytes() == (source / entry).read_bytes(), f"{entry} changed in the conversion"
    assert read_checkpoint_key_set(str(out)) == read_checkpoint_key_set(str(source)), (
        "the resume adapter's tensors must stay out of the converted weights"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
