#!/usr/bin/env python
"""The Qwen-style EP gate weights take their top-k renorm rule from the family, never from a default.

``_softmax_gate_weights_at`` (routing replay, balancing-bias reweighting) renormalizes the selected
softmax weights or not. Qwen3's router carries the rule as a configurable, default-off
``norm_topk_prob``; Qwen3.5's router has no such attribute and always renormalizes. A default for a
missing attribute would turn an upstream rename on Qwen3 into silently renormalized weights, so the
Qwen3 layer reads the attribute outright and a missing one raises.

    python tests/cpu/parallelism/test_softmax_gate_renorm_rule.py
"""

import pytest
import torch
from torch import nn
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter

from src.distributed.expert_parallel.layers.qwen3 import EPQwen3MoELayer

TOKENS, EXPERTS, TOP_K, HIDDEN = 6, 8, 2, 16


def _router(norm_topk_prob: bool) -> Qwen3MoeTopKRouter:
    config = type(
        "_Cfg",
        (),
        {
            "num_experts_per_tok": TOP_K,
            "num_experts": EXPERTS,
            "hidden_size": HIDDEN,
            "norm_topk_prob": norm_topk_prob,
        },
    )
    router = Qwen3MoeTopKRouter(config)
    torch.nn.init.normal_(router.weight, generator=torch.Generator().manual_seed(0))
    return router


def _layer(gate: nn.Module) -> EPQwen3MoELayer:
    layer = EPQwen3MoELayer.__new__(EPQwen3MoELayer)
    nn.Module.__init__(layer)
    layer.gate = gate
    return layer


def test_qwen3_unnormalized_router_stays_unnormalized():
    router = _router(norm_topk_prob=False)
    logits, scores, indices = router(torch.randn(TOKENS, HIDDEN, generator=torch.Generator().manual_seed(1)))
    replayed = _layer(router)._gate_weights_at(logits, indices)
    assert torch.equal(replayed, scores)
    assert not torch.allclose(replayed.sum(-1), torch.ones(TOKENS)), "fixture weights already sum to 1"


def test_qwen3_router_without_norm_topk_prob_raises():
    router = _router(norm_topk_prob=False)
    logits, _, indices = router(torch.randn(TOKENS, HIDDEN, generator=torch.Generator().manual_seed(2)))
    del router.norm_topk_prob
    with pytest.raises(AttributeError, match="norm_topk_prob"):
        _layer(router)._gate_weights_at(logits, indices)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
