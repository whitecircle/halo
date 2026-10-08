#!/usr/bin/env python
"""The EP lazy loader reads a multimodal checkpoint's per-expert keys into its text-only model.

A ``text_only_model`` run loads a VLM checkpoint into the text-only class, whose names drop the
wrapper's ``language_model.`` segment. Its experts are fused in memory, so a per-expert checkpoint key
(a Qwen3.5/3.6 checkpoint ``unfuse_moe_experts.py`` wrote, or a wrapper-less save of a model built from
its config) is never a model key itself: the fuser finds the fused parameter it feeds from the key's
namespace, which must therefore have the segment dropped too. Driven through the real
:func:`~src.distributed.expert_parallel.lazy_loader.load_ep_model_lazy` (EP patching stubbed — it needs
DeepEP and process groups) against ``from_pretrained`` of the same text-only class.

    python tests/cpu/parallelism/test_lazy_load_text_only_per_expert.py
"""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration

import src.distributed.expert_parallel.lazy_loader as ep_lazy_loader
from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.lazy_loader import load_ep_model_lazy
from src.models.loading.lazy_safetensors.weights import resolve_safetensors_index
from tests.common.models import TINY_QWEN35_MOE_CONFIG

RUN_DTYPE = torch.bfloat16
FUSED_EXPERT_SUFFIXES = ("experts.gate_up_proj", "experts.down_proj")
VISION = {"depth": 1, "hidden_size": 16, "intermediate_size": 16, "num_heads": 2}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> str:
    """A multimodal Qwen3.5-MoE checkpoint with its experts one tensor each under ``model.language_model.``."""
    path = tmp_path_factory.mktemp("qwen3_5_moe_per_expert")
    torch.manual_seed(0)
    config = Qwen3_5MoeConfig(
        text_config=dict(TINY_QWEN35_MOE_CONFIG),
        vision_config={**VISION, "out_hidden_size": TINY_QWEN35_MOE_CONFIG["hidden_size"]},
    )
    Qwen3_5MoeForConditionalGeneration(config).to(RUN_DTYPE).save_pretrained(path)
    weight_map, _ = resolve_safetensors_index(str(path))
    assert any(".language_model." in key and ".experts.0." in key for key in weight_map), (
        "premise: per-expert keys under the multimodal prefix"
    )
    return str(path)


@pytest.fixture(autouse=True)
def _stub_ep_patching(monkeypatch):
    monkeypatch.setattr(ep_lazy_loader, "patch_moe_model_for_ep", lambda model, *a, **k: model)
    monkeypatch.setattr(ep_lazy_loader, "create_ep_buffers", lambda *a, **k: None)


@pytest.mark.parametrize("ep_size", [1, 2])
def test_a_text_only_lazy_load_reads_every_expert(checkpoint, ep_size):
    reference = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=RUN_DTYPE).state_dict()
    ep_config = EPConfig(ep_size=ep_size, world_size=ep_size, gpus_per_node=ep_size)

    lazy = load_ep_model_lazy(
        checkpoint,
        ep_config,
        AutoConfig.from_pretrained(checkpoint),
        dtype=RUN_DTYPE,
        trust_remote_code=False,
        model_class=AutoModelForCausalLM,
    ).state_dict()

    assert lazy.keys() == reference.keys()
    local = slice(0, TINY_QWEN35_MOE_CONFIG["num_experts"] // ep_size)
    expected = {key: t[local] if key.endswith(FUSED_EXPERT_SUFFIXES) else t for key, t in reference.items()}
    stale = [key for key, tensor in lazy.items() if not torch.equal(tensor, expected[key])]
    assert not stale, f"{len(stale)} tensors differ from from_pretrained: {stale[:5]}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
