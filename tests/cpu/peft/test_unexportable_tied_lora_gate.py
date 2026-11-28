#!/usr/bin/env python
"""``reject_unexportable_tied_embedding_lora``: native expert LoRA beside a LoRA on either half of a tied embedding
pair saves an adapter no tool exports, so construction refuses it. A full copy of the half, an adapter off the pair, an
untied base and a run with no expert LoRA pass this gate."""

import pytest
import torch
from peft import LoraConfig, get_peft_model

from halo.distributed.expert_parallel.moe_layer.expert_lora import EXPERT_LORA_RANK_MULTIPLE, model_has_expert_lora
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from halo.trainers.parallelism.adapter_rules import reject_unexportable_tied_embedding_lora
from tests.common.parallelism import single_process_ep_config
from tests.common.tiny_models import TINY_DENSE_FAMILY, TINY_MOE_FAMILIES, tiny_family_model

SPEC = ExpertLoraSpec(r=EXPERT_LORA_RANK_MULTIPLE, alpha=2 * EXPERT_LORA_RANK_MULTIPLE)
FAMILIES = ("lfm2_moe", "gemma4_text")
LORA_ON_A_HALF = {"head": {"target_modules": ["q_proj", "lm_head"]}, "embedding": {"target_modules": ["embed_tokens"]}}
FULL_HEAD_COPY = {"target_modules": ["q_proj"], "modules_to_save": ["lm_head"]}
ATTENTION_ONLY = {"target_modules": ["q_proj"]}


def _adapted(family: str, *, tied: bool, adapter: dict) -> torch.nn.Module:
    """``family``'s tiny model with native expert LoRA under one-rank EP and ``adapter`` wrapped over it."""
    torch.manual_seed(0)
    model = tiny_family_model(TINY_MOE_FAMILIES[family], overrides={"tie_word_embeddings": tied})
    shared = model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert shared is tied, f"premise: the pair is {'tied' if tied else 'untied'}"
    model = apply_ep_layers(model, single_process_ep_config(expert_lora=SPEC))
    peft_model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, task_type="CAUSAL_LM", **adapter))
    assert model_has_expert_lora(peft_model), "premise: the run trains native expert LoRA"
    return peft_model


@pytest.mark.parametrize("half", sorted(LORA_ON_A_HALF))
@pytest.mark.parametrize("family", FAMILIES)
def test_a_lora_on_a_tied_half_beside_expert_lora_is_refused(family, half):
    model = _adapted(family, tied=True, adapter=LORA_ON_A_HALF[half])

    with pytest.raises(ValueError, match="no tool can export") as refusal:
        reject_unexportable_tied_embedding_lora(model)
    assert "lora_modules_to_save with merge_expert_lora_on_save: true" in str(refusal.value), refusal.value


@pytest.mark.parametrize(
    ("tied", "adapter"),
    [
        (False, LORA_ON_A_HALF["head"]),
        (False, LORA_ON_A_HALF["embedding"]),
        (True, FULL_HEAD_COPY),
        (True, ATTENTION_ONLY),
    ],
    ids=["untied-head-lora", "untied-embedding-lora", "tied-full-head-copy", "tied-attention-only"],
)
@pytest.mark.parametrize("family", FAMILIES)
def test_an_exportable_expert_lora_run_passes(family, tied, adapter):
    reject_unexportable_tied_embedding_lora(_adapted(family, tied=tied, adapter=adapter))


def test_a_tied_head_lora_without_expert_lora_is_left_to_the_other_gates():
    torch.manual_seed(0)
    model = tiny_family_model(TINY_DENSE_FAMILY, overrides={"tie_word_embeddings": True})
    peft_model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["lm_head"]))
    base = peft_model.get_base_model()
    assert base.lm_head.base_layer.weight is base.model.embed_tokens.weight, "premise: the head is tied"

    reject_unexportable_tied_embedding_lora(peft_model)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
