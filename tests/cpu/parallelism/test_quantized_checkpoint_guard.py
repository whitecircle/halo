#!/usr/bin/env python
"""EP refuses a natively-quantized checkpoint on every family, since a shallow scan of ``MOE_LAYER_MAP``'s class
misses nested expert tensors. The gate runs before construction at every ``ep_size``."""

import pytest
import torch
import torch.nn as nn
from transformers import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP

from halo.distributed.expert_parallel.families.roster import MOE_LAYER_MAP
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from tests.common.parallelism import single_process_ep_config


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


def test_quantized_experts_are_refused_at_patch_time():
    model = _model_with_gptoss_mlp()
    with torch.no_grad():
        model.mlp.experts.gate_up_proj = nn.Parameter(
            torch.zeros_like(model.mlp.experts.gate_up_proj, dtype=torch.uint8), requires_grad=False
        )
    with pytest.raises(ValueError, match="de-quantized"):
        apply_ep_layers(model, single_process_ep_config())


def test_float_experts_are_not_rejected():
    model = _model_with_gptoss_mlp().to(torch.bfloat16)
    returned = apply_ep_layers(model, single_process_ep_config())
    assert type(returned.mlp).__name__ != "GptOssMLP", "the MoE block was not wrapped"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
