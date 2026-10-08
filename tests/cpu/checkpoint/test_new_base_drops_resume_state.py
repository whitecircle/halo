#!/usr/bin/env python
"""A tool that writes a new base model carries none of its source run's resume state.

``patch_vocab.py`` grows a vocabulary and ``merge_adapter_into_base`` folds an adapter into its base:
each output is a new model, a starting point for the next run rather than a checkpoint of the one it
came from. Pointed at a training checkpoint, their aux-file copy would otherwise carry that run's
``trainer_state.json``, ``scheduler.pt``, ``rng_state_*``, ``reference_logps.pt`` and a merge-on-save checkpoint's
``resume_adapter/`` with its marker. The marker is the worst of them: a resume from the output would
build the policy from the ORIGINAL base plus that resume adapter and silently drop the new weights
(the grown vocabulary, the folded adapter).

    python tests/cpu/checkpoint/test_new_base_drops_resume_state.py
"""

import sys

import pytest
import torch
from accelerate import PartialState
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
from transformers.trainer import TRAINER_STATE_NAME

from scripts.after_training.merge_peft_adapters import merge_peft_adapter
from src.checkpoint.format import (
    ADAPTER_SAFETENSORS_FILE,
    REFERENCE_LOGPS_FILE,
    RESUME_ADAPTER_DIR,
    RESUME_ADAPTER_MARKER_FILE,
    SCHEDULER_STATE_FILE,
    write_resume_adapter_marker,
)
from src.training.environment import _classify_resume_checkpoint
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.utils import load_script_module

PartialState()

patch_vocab = load_script_module("scripts/before_training/patch_vocab.py")

RESUME_STATE = (
    TRAINER_STATE_NAME,
    SCHEDULER_STATE_FILE,
    "rng_state_0.pth",
    REFERENCE_LOGPS_FILE,
    RESUME_ADAPTER_MARKER_FILE,
    RESUME_ADAPTER_DIR,
)
VOCAB = {"<unk>": 0, "<eos>": 1, "hello": 2, "world": 3}


def _training_checkpoint(path) -> str:
    """A tiny merge-on-save training checkpoint: weights, tokenizer and every resume file."""
    backend = Tokenizer(models.WordLevel(VOCAB, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>"
    )
    torch.manual_seed(0)
    Qwen3ForCausalLM(Qwen3Config(**{**TINY_QWEN3_CONFIG, "vocab_size": len(VOCAB)})).save_pretrained(path)
    tokenizer.save_pretrained(path)
    (path / RESUME_ADAPTER_DIR).mkdir()
    save_file(
        {"x.experts.down_proj.lora_A": torch.ones(2, 2)}, str(path / RESUME_ADAPTER_DIR / ADAPTER_SAFETENSORS_FILE)
    )
    write_resume_adapter_marker(str(path))
    for name in (TRAINER_STATE_NAME, SCHEDULER_STATE_FILE, "rng_state_0.pth", REFERENCE_LOGPS_FILE):
        (path / name).write_bytes(name.encode())
    assert _classify_resume_checkpoint(str(path)) == "merged_adapter", "premise: the source resumes from its adapter"
    return str(path)


def _assert_no_resume_state(out) -> None:
    carried = [name for name in RESUME_STATE if (out / name).exists()]
    assert not carried, f"the new base carries its source run's resume state: {carried}"
    assert _classify_resume_checkpoint(str(out)) == "full", (
        "a resume from the new base must build from its own weights"
    )


def test_a_vocab_patch_of_a_training_checkpoint_carries_no_resume_state(tmp_path, monkeypatch):
    source = _training_checkpoint(tmp_path / "checkpoint-3")
    out = tmp_path / "patched"
    monkeypatch.setattr(
        sys,
        "argv",
        ["patch_vocab.py", "--model_id", source, "--output_dir", str(out), "--patterns", '["hello world"]'],
    )

    patch_vocab.main()

    assert Qwen3ForCausalLM.from_pretrained(out).get_input_embeddings().weight.shape[0] > len(VOCAB), (
        "premise: the patch grew the vocabulary"
    )
    _assert_no_resume_state(out)


def test_an_adapter_merged_into_a_training_checkpoint_carries_none_of_its_resume_state(tmp_path):
    base = _training_checkpoint(tmp_path / "checkpoint-3")
    adapter = tmp_path / "adapter"
    peft_model = get_peft_model(Qwen3ForCausalLM.from_pretrained(base), LoraConfig(r=4, target_modules=["q_proj"]))
    peft_model.peft_config["default"].base_model_name_or_path = base
    peft_model.save_pretrained(adapter)
    out = tmp_path / "merged"

    merge_peft_adapter(adapter_dir=str(adapter), output_dir=str(out), verbose=False)

    _assert_no_resume_state(out)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
