#!/usr/bin/env python
"""Weight-sync construction gate for GptOss sinks removed by the flash_attention_2 reset.

Under ``reset_sinks: true`` (the default) with FA2, the sink Parameter is set to ``None`` and leaves
``named_parameters()``. The RL weight-sync sends named parameters only, so nothing is ever pushed
for those slots — the rollout engine keeps generating with the PRETRAINED sinks while the trainer
runs without them, a permanent trainer↔generator log-prob offset no sync heals and no error names.
``validate_weight_sync_support`` must refuse that shape at construction; sink layouts the sync CAN
represent (live sinks, or the in-place ``dtype.min`` reset every non-FA2 implementation applies)
must keep passing.

    python tests/cpu/grpo/test_weight_sync_sink_gate.py
"""

import pytest
import torch
import torch.nn as nn
from transformers import CONFIG_MAPPING

from src.trainers.grpo.rollout.weight_sync import validate_weight_sync_support

NUM_HEADS = 4


class _Attention(nn.Module):
    def __init__(self, sinks: torch.Tensor | None):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.sinks = nn.Parameter(sinks) if sinks is not None else None


class _GptOssStub(nn.Module):
    """Minimal GptOss-shaped tree: real family config + one decoder layer with a sinks slot."""

    def __init__(self, sinks: torch.Tensor | None):
        super().__init__()
        self.config = CONFIG_MAPPING["gpt_oss"](
            num_hidden_layers=1, num_attention_heads=NUM_HEADS, num_local_experts=4, num_experts_per_tok=2
        )
        layer = nn.Module()
        layer.self_attn = _Attention(sinks)
        backbone = nn.Module()
        backbone.layers = nn.ModuleList([layer])
        self.model = backbone


def test_sync_refuses_fa2_removed_sinks():
    with pytest.raises(ValueError, match="reset_sinks"):
        validate_weight_sync_support(_GptOssStub(sinks=None))


def test_sync_accepts_live_sinks():
    validate_weight_sync_support(_GptOssStub(sinks=torch.zeros(NUM_HEADS)))


def test_sync_accepts_in_place_neutralized_sinks():
    """The non-FA2 reset fills dtype.min IN PLACE — the Parameter survives, syncs, and the engine
    serves the same neutralized sinks the trainer computes with. Consistent, so allowed."""
    validate_weight_sync_support(_GptOssStub(sinks=torch.full((NUM_HEADS,), torch.finfo(torch.bfloat16).min)))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
