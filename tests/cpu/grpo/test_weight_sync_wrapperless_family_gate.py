#!/usr/bin/env python
"""Wrapper-less runs must carry the same weight-sync contract as EP-wrapped ones.

``ep_size == 1`` with ``use_grouped_gemm: false`` leaves the stock HF module tree — no
``EPMoELayerBase`` instance exists, so a gate that walks live modules silently admits exactly the
families it exists to refuse (Mistral4, DeepSeek-V4 among them).
The gates resolve the family's class from ``config.model_type`` through the registry; these
tests pin both resolution paths and the per-model-type engine roster
(``_WEIGHT_SYNC_UNSUPPORTED_MODEL_TYPES``).

    python tests/cpu/grpo/test_weight_sync_wrapperless_family_gate.py
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import CONFIG_MAPPING, PretrainedConfig

from src.distributed.expert_parallel.expert_weights import ep_layer_class_by_model_type
from src.distributed.expert_parallel.layers.bailing import EPBailingMoELayer
from src.trainers.grpo.rollout.weight_sync import validate_backend_parallelism, validate_weight_sync_support


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


def _model_type_without_fused_gather() -> str:
    """A ``model_type`` whose family inherits the empty ``gather_fused_expert_state_dict`` default.

    Read off the live registry rather than pinned: a named family that later gained a fused gather
    would leave this test passing on the wrong rejection — or failing while the gate works — instead
    of exercising the wrapper-less resolution path it exists for. Which families implement the fused
    layout is pinned separately, in ``tests/cpu/grpo/test_rollout_backend_selection.py``.
    """
    missing = sorted(
        model_type
        for model_type, cls in ep_layer_class_by_model_type().items()
        if not cls.implements_fused_expert_layout()
    )
    assert missing, "every EP family now implements the fused gather — this gate is dead code, delete it"
    return missing[0]


def test_flag_false_family_is_refused_without_wrappers():
    with pytest.raises(ValueError, match="does not support vLLM weight sync"):
        validate_weight_sync_support(_StockModel("deepseek_v4"))


def test_unservable_model_type_is_refused_without_wrappers():
    with pytest.raises(ValueError, match="refuses weight sync for model_type"):
        validate_weight_sync_support(_StockModel("bailing_hybrid"))


def test_unservable_model_type_is_refused_with_a_live_wrapper():
    model = _StockModel("bailing_hybrid")
    layer = object.__new__(EPBailingMoELayer)
    nn.Module.__init__(layer)
    model.moe = layer
    with pytest.raises(ValueError, match="refuses weight sync for model_type"):
        validate_weight_sync_support(model)


def test_fused_layout_engine_is_refused_without_wrappers():
    model_type = _model_type_without_fused_gather()
    with pytest.raises(ValueError, match="does not implement"):
        validate_backend_parallelism("sglang", _NO_EP, _StockModel(model_type))


def test_servable_sibling_and_dense_models_pass():
    validate_weight_sync_support(_StockModel("bailing_moe"))  # Ling 2.0: vLLM registers the class
    validate_weight_sync_support(_StockModel("qwen3"))  # dense: no EP family resolves at all
    # Anti-vacuity for the rejection above: vLLM takes the per-expert layout the same family gathers,
    # so the engine's expert layout is what makes SGLang refuse it — not the model_type resolving.
    validate_backend_parallelism("vllm", _NO_EP, _StockModel(_model_type_without_fused_gather()))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
