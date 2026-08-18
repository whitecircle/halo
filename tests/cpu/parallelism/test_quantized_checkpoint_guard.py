#!/usr/bin/env python
"""EP must refuse a natively-quantized checkpoint on every family, not just the one leaf class.

The EP path materializes experts as plain ``nn.Parameter``s, so an MXFP4/int checkpoint cannot be
trained. ``patch_moe_model_for_ep`` refuses one up front — but the scan runs on the class named in
``MOE_LAYER_MAP``, and for ten of the eleven families that class is the MoE *block*, whose expert
tensors hang off a child module (``GptOssMLP.experts.gate_up_proj``). Only ``Gemma4TextExperts`` is
itself the expert leaf. A shallow ``recurse=False`` scan therefore sees nothing on gpt-oss — the
family the error message names — and lets the run continue to an unrelated failure at first forward.

Both tests drive the real ``patch_moe_model_for_ep`` on the real upstream ``GptOssMLP`` layout, so
they cover the guard's depth AND a transformers relayout in one place: the refusal test fails with
``recurse=False``, and the acceptance test fails if the guard starts firing on bf16.

Run: pytest tests/cpu/parallelism/test_quantized_checkpoint_guard.py
"""

import sys

import pytest
import torch
import torch.nn as nn
from transformers import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP

from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.patching import MOE_LAYER_MAP, patch_moe_model_for_ep


def _model_with_gptoss_mlp() -> nn.Module:
    config = GptOssConfig(
        hidden_size=32,
        intermediate_size=32,
        num_local_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=1,
        vocab_size=64,
    )
    block = GptOssMLP(config)
    assert type(block).__name__ in MOE_LAYER_MAP, "gpt-oss must still be the mapped MoE block"
    model = nn.Module()
    model.mlp = block
    return model


def _ep_config() -> EPConfig:
    # ep_size=1 keeps wrapper construction process-group-free; the quantization gate itself runs
    # before construction and is identical at every ep_size.
    return EPConfig(ep_size=1, world_size=1, gpus_per_node=1, use_grouped_gemm=False)


def test_quantized_experts_are_refused_at_patch_time():
    """The real guard must see the nested uint8 expert tensor and refuse the model."""
    model = _model_with_gptoss_mlp()
    with torch.no_grad():
        model.mlp.experts.gate_up_proj = nn.Parameter(
            torch.zeros_like(model.mlp.experts.gate_up_proj, dtype=torch.uint8), requires_grad=False
        )
    with pytest.raises(ValueError, match="de-quantized"):
        patch_moe_model_for_ep(model, _ep_config())


def test_float_experts_are_not_rejected():
    """The guard must not fire on a normal bf16 model — patching proceeds and wraps the block."""
    model = _model_with_gptoss_mlp().to(torch.bfloat16)
    returned = patch_moe_model_for_ep(model, _ep_config())
    assert type(returned.mlp).__name__ != "GptOssMLP", "the MoE block was not wrapped"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
