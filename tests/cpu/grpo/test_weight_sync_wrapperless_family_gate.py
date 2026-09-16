#!/usr/bin/env python
"""Wrapper-less runs must carry the same weight-sync contract as EP-wrapped ones — and, for any MoE
family, be refused outright.

``ep_size == 1`` with ``use_grouped_gemm: false`` leaves the stock HF module tree — no
``EPMoELayerBase`` instance exists, so a gate that walks live modules silently admits exactly the
families it exists to refuse (Inkling, GLM-5 Next among them). The family gate resolves the class
from ``config.model_type`` through the registry, and the engine gate reads the same spellings off
the config. A servable family without a wrapper is refused too: the sync ships experts in the layout
the wrapper's gather emits, and the dense walk would forward the stock tree's fused expert tensors
under module names, which the engine's loader drops with no error.

    python tests/cpu/grpo/test_weight_sync_wrapperless_family_gate.py
"""

import pytest
import torch.nn as nn

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.layers.bailing import EPBailingMoELayer
from src.distributed.expert_parallel.layers.glm4 import EPGlm4MoELayer
from src.distributed.expert_parallel.layers.laguna import EPLagunaMoELayer
from src.distributed.expert_parallel.layers.qwen3 import EPQwen3MoELayer
from src.distributed.expert_parallel.layers.qwen3_5 import EPQwen3_5MoELayer
from src.trainers.grpo.rollout.weight_sync import validate_weight_sync_support
from tests.common.weight_sync import StockModel


def _wrapped(model: nn.Module, layer_cls: type[EPMoELayerBase]) -> nn.Module:
    """``model`` with one live EP wrapper of ``layer_cls`` — the ``nn.Module`` half only, as the base
    ``__init__`` needs a process group; the gate reads the family off the instance's type."""
    layer = object.__new__(layer_cls)
    nn.Module.__init__(layer)
    model.moe = layer
    return model


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_flag_false_family_is_refused_without_wrappers(backend):
    with pytest.raises(ValueError, match="does not support weight sync"):
        validate_weight_sync_support(StockModel("inkling_mm_model"), backend)


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_unservable_model_type_is_refused_without_wrappers(backend):
    with pytest.raises(ValueError, match="cannot serve model_type"):
        validate_weight_sync_support(StockModel("bailing_hybrid"), backend)


def test_unservable_model_type_is_refused_with_a_live_wrapper():
    with pytest.raises(ValueError, match="cannot serve model_type"):
        validate_weight_sync_support(_wrapped(StockModel("bailing_hybrid"), EPBailingMoELayer), "vllm")


def test_unservability_is_per_engine():
    """One engine's loader gap must not refuse the family on the other: Laguna's SGLang loader
    refuses any partial update while vLLM's layerwise reload takes it."""
    with pytest.raises(ValueError, match="cannot serve model_type"):
        validate_weight_sync_support(_wrapped(StockModel("laguna"), EPLagunaMoELayer), "sglang")
    validate_weight_sync_support(_wrapped(StockModel("laguna"), EPLagunaMoELayer), "vllm")


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_servable_wrapped_sibling_and_dense_models_pass(backend):
    validate_weight_sync_support(_wrapped(StockModel("bailing_moe"), EPBailingMoELayer), backend)  # Ling 2.0
    validate_weight_sync_support(StockModel("qwen3"), backend)  # dense: no EP family resolves at all


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize(
    ("model_type", "layer_cls"),
    [("qwen3_moe", EPQwen3MoELayer), ("glm4_moe_lite", EPGlm4MoELayer), ("qwen3_5_moe", EPQwen3_5MoELayer)],
)
def test_servable_moe_without_a_live_wrapper_is_refused_with_the_remedy(backend, model_type, layer_cls):
    """The stock tree forwards fused expert tensors under module names; the refusal names the knob that
    installs the wrappers, and the same model with a wrapper passes (the refusal is about the
    wrapper, not the family). Per-expert and fused-native hub layouts alike: the validated layout is
    the gather's, whichever it is. GptOss is the sink-gate test's wrapped stub."""
    with pytest.raises(ValueError, match="no live EP wrapper") as refused:
        validate_weight_sync_support(StockModel(model_type), backend)
    assert "use_grouped_gemm: true" in str(refused.value)
    assert layer_cls.__name__ in str(refused.value)
    validate_weight_sync_support(_wrapped(StockModel(model_type), layer_cls), backend)


def test_wrapperless_refusal_yields_to_the_sharper_family_and_engine_facts():
    """A family no engine serves, or one engine cannot take, is refused for THAT reason even without a
    wrapper: the loader fact names the remedy (another engine, or none), the wrapper hint would not."""
    with pytest.raises(ValueError, match="does not support weight sync"):
        validate_weight_sync_support(StockModel("glm5_next"), "vllm")
    with pytest.raises(ValueError, match="cannot serve model_type"):
        validate_weight_sync_support(StockModel("zaya"), "sglang")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
