#!/usr/bin/env python
"""An adapter resume of a tied-embedding model takes a tied tensor's value under any of its names.

A run that grows a tied model's vocabulary (``added_special_tokens``) saves the embedding with its adapters (PEFT's
``save_embedding_layers="auto"``). The gathered save, which every FSDP2 and mixed EP run writes, names that tensor
once, while the live selection a resume checks against names it under both ``embed_tokens`` and ``lm_head``; without
the alias the strict restore refuses a complete file. Pinned on a tiny tied Qwen3 with a resized vocabulary, its
adapter written as the gathered branch writes it: the restore takes it bit-equal, the alias-blind check refuses it,
and a genuinely missing tensor of an untied model is still refused.

    python tests/cpu/peft/test_adapter_restore_tied_alias.py
"""

import os

import pytest
import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from halo.checkpoint.adapters import write_adapter_file
from halo.checkpoint.format import ADAPTER_SAFETENSORS_FILE
from halo.checkpoint.parallel.adapters import (
    _attention_adapter_mismatch,
    gather_adapter_state,
    restore_adapters,
    saved_adapter_tensors,
)

VOCAB, GROWN_VOCAB = 128, 136
EMBED_KEY = "base_model.model.model.embed_tokens.weight"
HEAD_KEY = "base_model.model.lm_head.weight"


def _checkpoint(tmp_path_factory, *, tied: bool) -> str:
    """A tiny Qwen3 checkpoint on disk: PEFT reads its config to tell that a run grew the vocabulary."""
    directory = str(tmp_path_factory.mktemp("tied" if tied else "untied"))
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=tied,
    )
    Qwen3ForCausalLM(config).save_pretrained(directory)
    return directory


@pytest.fixture(scope="module")
def tied_dir(tmp_path_factory) -> str:
    return _checkpoint(tmp_path_factory, tied=True)


@pytest.fixture(scope="module")
def untied_dir(tmp_path_factory) -> str:
    return _checkpoint(tmp_path_factory, tied=False)


def _resized_peft_model(base_dir: str, *, seed: int):
    """The checkpoint at ``base_dir`` with its vocabulary grown and LoRA on ``q_proj``/``v_proj``, every tensor its
    adapter saves (the embedding, and an untied head, among them) drawn from ``seed``."""
    model = Qwen3ForCausalLM.from_pretrained(base_dir)
    model.resize_token_embeddings(GROWN_VOCAB)
    peft_model = get_peft_model(model, LoraConfig(r=4, target_modules=["q_proj", "v_proj"]))
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for tensor in saved_adapter_tensors(peft_model).values():
            tensor.copy_(torch.randn(tensor.shape, generator=generator))
    return peft_model


def _gathered_save(peft_model, directory: str) -> dict[str, torch.Tensor]:
    """The adapter file and config the gathered branch of ``save_peft_adapters`` writes; returns the tensors."""
    peft_model.peft_config["default"].save_pretrained(directory)
    write_adapter_file(gather_adapter_state(peft_model), directory)
    return load_file(os.path.join(directory, ADAPTER_SAFETENSORS_FILE))


def test_a_tied_resized_embedding_restores_under_both_names(tied_dir, tmp_path):
    trained = _resized_peft_model(tied_dir, seed=1)
    saved = _gathered_save(trained, str(tmp_path))
    assert EMBED_KEY in saved and HEAD_KEY not in saved, "premise: the gathered save names the tied tensor once"
    fresh = _resized_peft_model(tied_dir, seed=2)
    live = saved_adapter_tensors(fresh)
    assert live[HEAD_KEY] is live[EMBED_KEY], "premise: the live selection names the tied tensor twice"
    assert _attention_adapter_mismatch(live, saved, "ckpt"), "premise: the alias-blind check refuses the file"

    assert restore_adapters(str(tmp_path), fresh, is_cp_mode=False) is not None

    expected = saved_adapter_tensors(trained)
    restored = saved_adapter_tensors(fresh)
    assert sorted(restored) == sorted(expected)
    for key, tensor in restored.items():
        assert torch.equal(tensor, expected[key]), key
    assert fresh.get_output_embeddings().weight is fresh.get_input_embeddings().weight


def test_a_genuinely_missing_tensor_of_an_untied_model_is_still_refused(untied_dir, tmp_path):
    """Anti-vacuity: the alias reads tensor identity, so an untied head the file lacks is not taken from the
    embedding."""
    trained = _resized_peft_model(untied_dir, seed=1)
    saved = _gathered_save(trained, str(tmp_path))
    assert {EMBED_KEY, HEAD_KEY} <= set(saved), "premise: an untied model saves both"
    del saved[HEAD_KEY]
    write_adapter_file(saved, str(tmp_path))
    fresh = _resized_peft_model(untied_dir, seed=2)
    before = fresh.get_output_embeddings().weight.detach().clone()

    with pytest.raises(RuntimeError, match="lm_head"):
        restore_adapters(str(tmp_path), fresh, is_cp_mode=False)
    assert torch.equal(fresh.get_output_embeddings().weight, before)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
