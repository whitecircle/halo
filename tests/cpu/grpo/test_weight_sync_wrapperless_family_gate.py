#!/usr/bin/env python
"""Wrapper-less runs must carry the same weight-sync contract as EP-wrapped ones.

``ep_size == 1`` with ``use_grouped_gemm: false`` leaves the stock HF module tree — no
``EPMoELayerBase`` instance exists, so a gate that walks live modules silently admits exactly the
families it exists to refuse (Inkling, GLM-5 Next among them). The family gate resolves the class
from ``config.model_type`` through the registry, and the engine gate reads the same spellings off
the config; these tests pin both resolution paths.

    python tests/cpu/grpo/test_weight_sync_wrapperless_family_gate.py
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import CONFIG_MAPPING, PretrainedConfig

from src.distributed.expert_parallel.layers.bailing import EPBailingMoELayer
from src.trainers.grpo.rollout.weight_sync import validate_weight_sync_support


class _StockModel(nn.Module):
    """A model with a config but no EP wrapper — the use_grouped_gemm: false module tree."""

    def __init__(self, model_type: str):
        super().__init__()
        if model_type in CONFIG_MAPPING:
            self.config = CONFIG_MAPPING[model_type]()
        else:  # remote-code spellings (Bailing family) have no in-library config class
            self.config = PretrainedConfig()
            self.config.model_type = model_type
        self.weight = nn.Parameter(torch.zeros(1))


_NO_EP = SimpleNamespace(is_ep_mode=False, ep_size=1, expert_tp_size=1, ep_group_size=1)


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_flag_false_family_is_refused_without_wrappers(backend):
    with pytest.raises(ValueError, match="does not support weight sync"):
        validate_weight_sync_support(_StockModel("inkling_mm_model"), backend)


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_unservable_model_type_is_refused_without_wrappers(backend):
    with pytest.raises(ValueError, match="cannot serve model_type"):
        validate_weight_sync_support(_StockModel("bailing_hybrid"), backend)


def test_unservable_model_type_is_refused_with_a_live_wrapper():
    model = _StockModel("bailing_hybrid")
    layer = object.__new__(EPBailingMoELayer)
    nn.Module.__init__(layer)
    model.moe = layer
    with pytest.raises(ValueError, match="cannot serve model_type"):
        validate_weight_sync_support(model, "vllm")


def test_unservability_is_per_engine():
    """One engine's loader gap must not refuse the family on the other: Laguna's SGLang loader
    refuses any partial update while vLLM's layerwise reload takes it."""
    with pytest.raises(ValueError, match="cannot serve model_type"):
        validate_weight_sync_support(_StockModel("laguna"), "sglang")
    validate_weight_sync_support(_StockModel("laguna"), "vllm")


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_servable_sibling_and_dense_models_pass(backend):
    validate_weight_sync_support(_StockModel("bailing_moe"), backend)  # Ling 2.0: both engines register it
    validate_weight_sync_support(_StockModel("qwen3"), backend)  # dense: no EP family resolves at all


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
