#!/usr/bin/env python
"""CPU test: the rotate-half CP families must ride the shared base rotary, not private copies.

``UlyssesAttentionBase._apply_rotary_core`` IS the rotate-half convention (via
``_apply_partial_rotary``), and Qwen3 / Qwen3.5-3.6 / Bailing inherit it. Qwen3 / Qwen3-MoE /
Qwen3-VL-text ship full-width ``cos``, where the helper's full-rotary branch must be bit-identical
to the rotate-half expression a private copy would spell; a drift there corrupts every CP run of
those families with no error, and a re-forked override is how the drift gets in.

Run: ``python tests/cpu/parallelism/test_cp_qwen3_rotary.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from src.distributed.context_parallel.base_layer import UlyssesAttentionBase
from src.distributed.context_parallel.layers.bailing import BailingMoeV2UlyssesAttention
from src.distributed.context_parallel.layers.qwen3 import Qwen3MoeUlyssesAttention
from src.distributed.context_parallel.layers.qwen3_5 import Qwen3_5MoeUlyssesAttention

# The families whose rotary IS the base default — each one that re-declares the hook is a fork.
ROTATE_HALF_FAMILIES = (Qwen3MoeUlyssesAttention, Qwen3_5MoeUlyssesAttention, BailingMoeV2UlyssesAttention)

HIDDEN = 32
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 8


class _FakeConfig:
    model_type = "qwen3_moe"
    num_attention_heads = NUM_Q_HEADS
    num_key_value_heads = NUM_KV_HEADS


class _FakeQwen3Attention(nn.Module):
    """Minimal stand-in for a Qwen3-style attention module (post-projection Q/K RMSNorm)."""

    def __init__(self):
        super().__init__()
        self.config = _FakeConfig()
        self.layer_idx = 0
        self.head_dim = HEAD_DIM
        self.scaling = HEAD_DIM**-0.5
        self.attention_dropout = 0.0
        self.is_causal = True

        torch.manual_seed(0)
        self.q_proj = nn.Linear(HIDDEN, NUM_Q_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.q_norm = nn.RMSNorm(HEAD_DIM)
        self.k_norm = nn.RMSNorm(HEAD_DIM)
        self.o_proj = nn.Linear(NUM_Q_HEADS * HEAD_DIM, HIDDEN, bias=False)


def _wrapper() -> Qwen3MoeUlyssesAttention:
    return Qwen3MoeUlyssesAttention(_FakeQwen3Attention(), cp_group=None, cp_size=2)


def _rope_inputs(seed: int, dtype=torch.float32):
    """Q/K in the optimized ``[B, S, H, D]`` layout with cos/sin already unsqueezed as the base does."""
    torch.manual_seed(seed)
    batch, seq = 2, 3
    q = torch.randn(batch, seq, NUM_Q_HEADS, HEAD_DIM, dtype=dtype)
    k = torch.randn(batch, seq, NUM_KV_HEADS, HEAD_DIM, dtype=dtype)
    cos = torch.randn(batch, seq, 1, HEAD_DIM, dtype=dtype)
    sin = torch.randn(batch, seq, 1, HEAD_DIM, dtype=dtype)
    return q, k, cos, sin


@pytest.mark.parametrize("cls", ROTATE_HALF_FAMILIES)
def test_rotate_half_families_do_not_fork_the_base_rotary(cls):
    """The hook lives on the base; a family copy is silent drift the next base fix would miss."""
    for method in ("_apply_rotary_core", "_apply_partial_rotary", "_rotate_half"):
        assert method not in vars(cls), (
            f"{cls.__name__} re-declares {method}; the rotate-half rotary lives on "
            f"UlyssesAttentionBase — delete the fork or justify a genuinely different rotary."
        )
        assert getattr(cls, method) is getattr(UlyssesAttentionBase, method)


def test_qwen3_rotary_delegates_to_the_shared_helper():
    """The inherited ``_apply_rotary_core`` must route both Q and K through the partial helper."""
    wrapper = _wrapper()
    q, k, cos, sin = _rope_inputs(seed=1)
    shared = UlyssesAttentionBase._apply_partial_rotary
    with patch.object(UlyssesAttentionBase, "_apply_partial_rotary", autospec=True, side_effect=shared) as helper:
        wrapper._apply_rotary_core(q, k, cos, sin)

    assert helper.call_count == 2, "_apply_rotary_core must route both Q and K through the base helper"


def test_full_width_rotary_matches_the_rotate_half_reference():
    """Full-width ``cos`` (every shipped Qwen3 variant) -> exactly ``(x*cos) + (rotate_half(x)*sin)``."""
    wrapper = _wrapper()
    q, k, cos, sin = _rope_inputs(seed=2, dtype=torch.float64)

    q_out, k_out = wrapper._apply_rotary_core(q, k, cos, sin)

    rotate_half = UlyssesAttentionBase._rotate_half
    torch.testing.assert_close(q_out, (q * cos) + (rotate_half(q) * sin), rtol=0, atol=0)
    torch.testing.assert_close(k_out, (k * cos) + (rotate_half(k) * sin), rtol=0, atol=0)


def test_rotary_broadcasts_one_cos_row_over_every_head():
    """Layout-agnostic: ``[B, S, H, D]`` in, same shape out, each head rotated by the same row."""
    wrapper = _wrapper()
    q, k, cos, sin = _rope_inputs(seed=3)

    q_out, k_out = wrapper._apply_rotary_core(q, k, cos, sin)

    assert q_out.shape == q.shape and k_out.shape == k.shape
    rotate_half = UlyssesAttentionBase._rotate_half
    for head in range(NUM_KV_HEADS):
        expected = (k[:, :, head] * cos[:, :, 0]) + (rotate_half(k[:, :, head]) * sin[:, :, 0])
        torch.testing.assert_close(k_out[:, :, head], expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
