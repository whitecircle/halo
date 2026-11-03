#!/usr/bin/env python
"""Weight-sync construction gate for GptOss sinks removed by the FA2 reset. The sync sends named parameters only, so
the engine would keep the pretrained sinks: a permanent log-prob offset no sync heals and no error names."""

import pytest
import torch
import torch.nn as nn
from transformers import CONFIG_MAPPING

from halo.distributed.expert_parallel.families.gpt_oss import EPGptOssMoELayer
from halo.trainers.grpo.rollout.weight_push import reject_unsyncable_model

NUM_HEADS = 4


class _Attention(nn.Module):
    def __init__(self, sinks: torch.Tensor | None):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.sinks = nn.Parameter(sinks) if sinks is not None else None


def _ep_wrapped_moe() -> nn.Module:
    """A live GptOss EP layer (the ``nn.Module`` half; the base ``__init__`` needs a process group), since a
    wrapper-less MoE is refused before the sink verdict matters."""
    layer = object.__new__(EPGptOssMoELayer)
    nn.Module.__init__(layer)
    return layer


class _GptOssStub(nn.Module):
    """GptOss-shaped tree: real family config, one decoder layer with a sinks slot and an EP-wrapped MoE block."""

    def __init__(self, sinks: torch.Tensor | None):
        super().__init__()
        self.config = CONFIG_MAPPING["gpt_oss"](
            num_hidden_layers=1, num_attention_heads=NUM_HEADS, num_local_experts=4, num_experts_per_tok=2
        )
        layer = nn.Module()
        layer.self_attn = _Attention(sinks)
        layer.mlp = _ep_wrapped_moe()
        backbone = nn.Module()
        backbone.layers = nn.ModuleList([layer])
        self.model = backbone


def test_sync_refuses_fa2_removed_sinks():
    with pytest.raises(ValueError, match="reset_sinks"):
        reject_unsyncable_model(_GptOssStub(sinks=None), "vllm")


def test_sync_accepts_live_sinks():
    reject_unsyncable_model(_GptOssStub(sinks=torch.zeros(NUM_HEADS)), "vllm")


def test_sync_accepts_in_place_neutralized_sinks():
    """The non-FA2 reset fills dtype.min in place, so the Parameter syncs and the engine serves the same neutralized
    sinks the trainer computes with."""
    reject_unsyncable_model(_GptOssStub(sinks=torch.full((NUM_HEADS,), torch.finfo(torch.bfloat16).min)), "vllm")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
