#!/usr/bin/env python
"""An adapter on one half of a tied embedding pair trains a policy whose input embedding and output head differ. The
in-memory merge writes it untied and the EP gathered save writes it untied or refuses it; the streaming merge, which
keeps the base's keys and config, refuses it. An adapter elsewhere leaves the pair tied."""

import json
import os

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from halo.checkpoint.parallel.ep import save_ep_gathered_checkpoint
from halo.checkpoint.tools.adapter_merge import _untie_adapted_embeddings, merge_peft_adapter
from tests.common.adapters import randomize_adapters
from tests.common.checkpoint_io import save_context
from tests.common.grown_vocab import GROWING_TOKENS, build_word_level_checkpoint, grow_vocabulary

HEAD_KEY = "lm_head.weight"
EMBEDDING_KEY = "model.embed_tokens.weight"

# Each splits the pair: training computes the input embedding and the output head with different weights.
SPLITTING = {
    "lora_head": {"target_modules": ["lm_head"]},
    "lora_embedding": {"target_modules": ["embed_tokens"]},
    "lora_both": {"target_modules": ["embed_tokens", "lm_head"]},
    "full_head": {"target_modules": ["q_proj"], "modules_to_save": ["lm_head"]},
    "full_embedding": {"target_modules": ["q_proj"], "modules_to_save": ["embed_tokens"]},
    "full_both": {"target_modules": ["q_proj"], "modules_to_save": ["embed_tokens", "lm_head"]},
}
ATTENTION_ONLY = {"target_modules": ["q_proj", "v_proj"]}
# PEFT's ensure_weight_tying hands the head the embedding's full copy: both halves still compute with one weight.
TIED_COPY = {"target_modules": ["q_proj"], "modules_to_save": ["embed_tokens"], "ensure_weight_tying": True}


@pytest.fixture(scope="module", params=[True, False], ids=["tied", "untied"])
def base_dir(request, tmp_path_factory) -> str:
    directory = str(tmp_path_factory.mktemp("tied" if request.param else "untied"))
    build_word_level_checkpoint(directory, tied=request.param)
    return directory


@pytest.fixture(scope="module")
def tied_base_dir(tmp_path_factory) -> str:
    directory = str(tmp_path_factory.mktemp("tied_only"))
    build_word_level_checkpoint(directory, tied=True)
    return directory


def _trained(base_dir: str, adapter: dict) -> PeftModel:
    """Adapters and full copies moved off their init, so every half they touch computes with its own weight."""
    model = get_peft_model(
        AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32), LoraConfig(r=4, lora_alpha=8, **adapter)
    )
    randomize_adapters(model, std=0.5)
    generator = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "modules_to_save" in name:
                param.add_(torch.randn(param.shape, generator=generator))
    return model


def _ids() -> torch.Tensor:
    return torch.randint(2, 100, (1, 12), generator=torch.Generator().manual_seed(1))


def _logits(model: torch.nn.Module, ids: torch.Tensor | None = None) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=_ids() if ids is None else ids).logits


def _saved_adapter(base_dir: str, adapter: dict, adapter_dir: str) -> PeftModel:
    trained = _trained(base_dir, adapter)
    trained.save_pretrained(adapter_dir)
    AutoTokenizer.from_pretrained(base_dir).save_pretrained(adapter_dir)
    return trained


def _embedding_keys(checkpoint: str) -> set[str]:
    keys = set()
    for name in os.listdir(checkpoint):
        if name.endswith(".safetensors"):
            with safe_open(os.path.join(checkpoint, name), "pt") as handle:
                keys |= set(handle.keys()) & {HEAD_KEY, EMBEDDING_KEY}
    return keys


def _ties(checkpoint: str) -> bool:
    with open(os.path.join(checkpoint, "config.json")) as handle:
        return json.load(handle)["tie_word_embeddings"]


def _assert_untied_policy(checkpoint: str, trained: PeftModel, ids: torch.Tensor | None = None) -> None:
    assert not _ties(checkpoint), "the checkpoint holds two embedding tensors but says they are tied"
    assert _embedding_keys(checkpoint) == {HEAD_KEY, EMBEDDING_KEY}
    reloaded = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32)
    assert reloaded.get_input_embeddings().weight is not reloaded.get_output_embeddings().weight
    torch.testing.assert_close(_logits(reloaded, ids), _logits(trained, ids), rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("adapter", sorted(SPLITTING))
def test_the_merge_writes_the_trained_policy_untied(base_dir, tmp_path, adapter):
    adapter_dir, merged_dir = str(tmp_path / "adapter"), str(tmp_path / "merged")
    trained = _saved_adapter(base_dir, SPLITTING[adapter], adapter_dir)

    merge_peft_adapter(adapter_dir, merged_dir, dtype=torch.float32, verbose=False)

    _assert_untied_policy(merged_dir, trained)


def test_the_merge_keeps_each_half_off_the_other_halfs_adapter(tied_base_dir, tmp_path):
    """A head adapter leaves the input embedding at the base's weight, as training computed it."""
    adapter_dir, merged_dir = str(tmp_path / "adapter"), str(tmp_path / "merged")
    _saved_adapter(tied_base_dir, SPLITTING["lora_head"], adapter_dir)
    base = AutoModelForCausalLM.from_pretrained(tied_base_dir, dtype=torch.float32)

    merge_peft_adapter(adapter_dir, merged_dir, dtype=torch.float32, verbose=False)

    merged = AutoModelForCausalLM.from_pretrained(merged_dir, dtype=torch.float32)
    assert torch.equal(merged.get_input_embeddings().weight, base.get_input_embeddings().weight)
    assert not torch.equal(merged.get_output_embeddings().weight, base.get_output_embeddings().weight)


def test_a_grown_vocabulary_merges_untied_with_the_rows_both_halves_trained_on(tied_base_dir, tmp_path):
    """The adapter's saved rows land in the shared tensor before the untie, so neither half falls back to rebuilt
    rows."""
    adapter_dir, merged_dir = str(tmp_path / "adapter"), str(tmp_path / "merged")
    model = AutoModelForCausalLM.from_pretrained(tied_base_dir, dtype=torch.float32)
    rows = model.get_input_embeddings().num_embeddings
    tokenizer = grow_vocabulary(model, AutoTokenizer.from_pretrained(tied_base_dir), GROWING_TOKENS)
    trained = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, **SPLITTING["lora_head"]))
    randomize_adapters(trained, std=0.5)
    with torch.no_grad():
        added = trained.get_input_embeddings().weight[rows:]
        added.copy_(torch.randn(added.shape, generator=torch.Generator().manual_seed(3)))
    trained.save_pretrained(adapter_dir, save_embedding_layers=True)
    tokenizer.save_pretrained(adapter_dir)
    assert len(tokenizer) > rows, "premise: the run grew the vocabulary"

    merge_peft_adapter(adapter_dir, merged_dir, dtype=torch.float32, verbose=False)

    _assert_untied_policy(merged_dir, trained, torch.arange(rows - 4, len(tokenizer))[None])


def test_the_merge_refuses_to_untie_an_offloaded_pair(tied_base_dir):
    """A weight ``--device_map`` offloads sits on the meta device, so a copy of it would hold no values."""
    trained = _trained(tied_base_dir, SPLITTING["lora_head"])
    base = trained.get_base_model()
    offloaded = torch.nn.Parameter(
        torch.empty_like(base.lm_head.base_layer.weight, device="meta"), requires_grad=False
    )
    base.lm_head.base_layer.weight = base.model.embed_tokens.weight = offloaded

    with pytest.raises(ValueError, match="--device_map cpu"):
        _untie_adapted_embeddings(trained, log=lambda message: None)


@pytest.mark.parametrize("adapter", [ATTENTION_ONLY, TIED_COPY], ids=["attention", "one_copy_for_both"])
def test_an_adapter_keeping_the_pair_one_weight_merges_tied(tied_base_dir, tmp_path, adapter):
    adapter_dir, merged_dir = str(tmp_path / "adapter"), str(tmp_path / "merged")
    trained = _saved_adapter(tied_base_dir, adapter, adapter_dir)

    merge_peft_adapter(adapter_dir, merged_dir, dtype=torch.float32, verbose=False)

    assert _ties(merged_dir) and _embedding_keys(merged_dir) == {EMBEDDING_KEY}
    merged = AutoModelForCausalLM.from_pretrained(merged_dir, dtype=torch.float32)
    torch.testing.assert_close(_logits(merged), _logits(trained), rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("adapter", ["lora_head", "full_head", "full_embedding"])
def test_the_streaming_merge_refuses_an_adapter_splitting_a_tied_pair(tied_base_dir, tmp_path, adapter):
    """It writes the base's key set under the base's config, which holds one tensor for both halves."""
    adapter_dir, merged_dir = str(tmp_path / "adapter"), str(tmp_path / "merged")
    _saved_adapter(tied_base_dir, SPLITTING[adapter], adapter_dir)

    with pytest.raises(ValueError, match="ties to another tensor"):
        merge_peft_adapter(adapter_dir, merged_dir, streaming=True, verbose=False)
    assert not os.path.exists(merged_dir), "a refusal must leave no output"


def test_the_streaming_merge_keeps_an_adapter_off_the_pair_tied(tied_base_dir, tmp_path):
    adapter_dir, merged_dir = str(tmp_path / "adapter"), str(tmp_path / "merged")
    _saved_adapter(tied_base_dir, ATTENTION_ONLY, adapter_dir)

    merge_peft_adapter(adapter_dir, merged_dir, streaming=True, verbose=False)

    assert _ties(merged_dir) and _embedding_keys(merged_dir) == {EMBEDDING_KEY}


@pytest.mark.parametrize("adapter", ["lora_embedding", "lora_both"])
def test_the_gathered_save_refuses_a_lora_on_a_tied_pair(tied_base_dir, tmp_path, adapter):
    """It folds each tensor once under its first name, so a LoRA on either half would reach both."""
    with pytest.raises(ValueError, match="tied embedding pair"):
        save_ep_gathered_checkpoint(save_context(_trained(tied_base_dir, SPLITTING[adapter])), str(tmp_path))


@pytest.mark.parametrize("adapter", ["full_head", "full_both"])
def test_the_gathered_save_writes_full_copies_on_a_tied_pair_untied(tied_base_dir, tmp_path, adapter):
    trained = _trained(tied_base_dir, SPLITTING[adapter])

    save_ep_gathered_checkpoint(save_context(trained, training_checkpoint=True), str(tmp_path))

    _assert_untied_policy(str(tmp_path), trained)


def test_the_gathered_save_keeps_an_adapter_off_the_pair_tied(tied_base_dir, tmp_path):
    trained = _trained(tied_base_dir, ATTENTION_ONLY)

    save_ep_gathered_checkpoint(save_context(trained, training_checkpoint=True), str(tmp_path))

    assert _ties(str(tmp_path)) and _embedding_keys(str(tmp_path)) == {EMBEDDING_KEY}
    reloaded = AutoModelForCausalLM.from_pretrained(str(tmp_path), dtype=torch.float32)
    torch.testing.assert_close(_logits(reloaded), _logits(trained), rtol=1e-5, atol=1e-4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
