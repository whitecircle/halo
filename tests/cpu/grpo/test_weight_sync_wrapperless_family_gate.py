#!/usr/bin/env python
"""Wrapper-less runs carry the same weight-sync contract as EP-wrapped ones and, for any MoE family, are refused.
The gate reads ``config.model_type``, since ``ep_size == 1`` + ``use_grouped_gemm: false`` leaves no EP layer."""

import pytest
import torch.nn as nn

from halo.distributed.expert_parallel.families.bailing import EPBailingMoELayer
from halo.distributed.expert_parallel.families.glm4 import EPGlm4MoELayer
from halo.distributed.expert_parallel.families.laguna import EPLagunaMoELayer
from halo.distributed.expert_parallel.families.qwen3 import EPQwen3MoELayer
from halo.distributed.expert_parallel.families.qwen3_5 import EPQwen3_5MoELayer
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.trainers.grpo.rollout.weight_push import reject_unsyncable_model
from tests.common.weight_sync import StockModel


def _wrapped(model: nn.Module, layer_cls: type[EPMoELayerBase]) -> nn.Module:
    """``model`` with one live EP layer of ``layer_cls``, the ``nn.Module`` half only, as the base ``__init__`` needs a
    process group."""
    layer = object.__new__(layer_cls)
    nn.Module.__init__(layer)
    model.moe = layer
    return model


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_flag_false_family_is_refused_without_wrappers(backend):
    with pytest.raises(ValueError, match="does not support weight sync"):
        reject_unsyncable_model(StockModel("inkling_mm_model"), backend)


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_unservable_model_type_is_refused_without_wrappers(backend):
    with pytest.raises(ValueError, match="cannot serve model_type"):
        reject_unsyncable_model(StockModel("bailing_hybrid"), backend)


def test_unservable_model_type_is_refused_with_a_live_wrapper():
    with pytest.raises(ValueError, match="cannot serve model_type"):
        reject_unsyncable_model(_wrapped(StockModel("bailing_hybrid"), EPBailingMoELayer), "vllm")


def test_unservability_is_per_engine():
    """Laguna's SGLang loader refuses any partial update while vLLM's layerwise reload takes it."""
    with pytest.raises(ValueError, match="cannot serve model_type"):
        reject_unsyncable_model(_wrapped(StockModel("laguna"), EPLagunaMoELayer), "sglang")
    reject_unsyncable_model(_wrapped(StockModel("laguna"), EPLagunaMoELayer), "vllm")


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_servable_wrapped_sibling_and_dense_models_pass(backend):
    reject_unsyncable_model(_wrapped(StockModel("bailing_moe"), EPBailingMoELayer), backend)  # Ling 2.0
    reject_unsyncable_model(StockModel("qwen3"), backend)  # dense: no EP family resolves at all


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize(
    ("model_type", "layer_cls"),
    [("qwen3_moe", EPQwen3MoELayer), ("glm4_moe_lite", EPGlm4MoELayer), ("qwen3_5_moe", EPQwen3_5MoELayer)],
)
def test_servable_moe_without_a_live_wrapper_is_refused_with_the_remedy(backend, model_type, layer_cls):
    """The stock tree forwards fused expert tensors under module names; the refusal names the knob that installs the EP
    layers. GptOss is the sink-gate test's wrapped stub."""
    with pytest.raises(ValueError, match="no live EP layer") as refused:
        reject_unsyncable_model(StockModel(model_type), backend)
    assert "use_grouped_gemm: true" in str(refused.value)
    assert layer_cls.__name__ in str(refused.value)
    reject_unsyncable_model(_wrapped(StockModel(model_type), layer_cls), backend)


def test_wrapperless_refusal_yields_to_the_sharper_family_and_engine_facts():
    """The loader fact names the remedy (another engine, or none); the wrapper hint would not."""
    with pytest.raises(ValueError, match="does not support weight sync"):
        reject_unsyncable_model(StockModel("glm5_next"), "vllm")
    with pytest.raises(ValueError, match="cannot serve model_type"):
        reject_unsyncable_model(StockModel("zaya"), "sglang")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
