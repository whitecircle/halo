#!/usr/bin/env python
"""A rollout server launched from a checkpoint that ties its embedding pair holds one tensor for both halves, so the
sync gate refuses, at trainer construction, any PEFT adapter that makes training compute the two halves apart: a LoRA
or a ``modules_to_save`` copy on either half. An untied pair, or an adapter off the pair, syncs."""

import pytest
import torch
from peft import LoraConfig, get_peft_model

from halo.trainers.grpo.rollout.weight_push import reject_unsyncable_model
from tests.common.tiny_models import TINY_DENSE_FAMILY, tiny_family_model

SPLITTING = {
    "lora_head": {"target_modules": ["lm_head"]},
    "lora_embedding": {"target_modules": ["embed_tokens"]},
    "lora_both": {"target_modules": ["embed_tokens", "lm_head"]},
    "full_head": {"target_modules": ["q_proj"], "modules_to_save": ["lm_head"]},
    "full_embedding": {"target_modules": ["q_proj"], "modules_to_save": ["embed_tokens"]},
    "full_both": {"target_modules": ["q_proj"], "modules_to_save": ["embed_tokens", "lm_head"]},
}
TIED_COPY = {"target_modules": ["q_proj"], "modules_to_save": ["embed_tokens"], "ensure_weight_tying": True}


def _policy(tied: bool, adapter: dict):
    torch.manual_seed(0)
    base = tiny_family_model(TINY_DENSE_FAMILY, overrides={"tie_word_embeddings": tied})
    assert (base.get_input_embeddings().weight is base.get_output_embeddings().weight) == tied, "premise"
    return get_peft_model(base, LoraConfig(r=4, lora_alpha=8, **adapter))


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("adapter", sorted(SPLITTING))
def test_an_adapter_splitting_a_tied_pair_is_refused(adapter, backend):
    with pytest.raises(ValueError, match="holds them as one tensor"):
        reject_unsyncable_model(_policy(True, SPLITTING[adapter]), backend)


@pytest.mark.parametrize("adapter", sorted(SPLITTING))
def test_the_same_adapter_on_an_untied_pair_syncs(adapter):
    reject_unsyncable_model(_policy(False, SPLITTING[adapter]), "vllm")


def test_an_adapter_off_a_tied_pair_syncs():
    reject_unsyncable_model(_policy(True, {"target_modules": ["q_proj", "v_proj"]}), "vllm")


def test_one_full_copy_trained_for_both_halves_syncs():
    """PEFT's ``ensure_weight_tying`` hands the head the embedding's copy, so both halves compute with one weight."""
    model = _policy(True, TIED_COPY)
    head, embedding = model.get_output_embeddings(), model.get_input_embeddings()
    assert head.modules_to_save["default"].weight is embedding.modules_to_save["default"].weight, "premise"
    reject_unsyncable_model(model, "vllm")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
