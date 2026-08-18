#!/usr/bin/env python
"""CPU tests: Mistral4's llama-4 attention scale under CP — applied, from upstream, never dropped.

``Mistral4Attention.forward`` multiplies Q by ``get_llama_4_attn_scale(position_ids, beta,
original_max_position_embeddings)`` unconditionally. The CP wrapper must do the same or a CP run
trains a different attention than every non-CP run of the same checkpoint — silently, since the
scale is a smooth ramp. Two ways it can be lost, pinned here:

* the wrapper returned Q UNSCALED when ``rope_parameters`` lacked either key (the config injects
  them only when ``rope_parameters`` is None, so a checkpoint shipping its own dict can lack them);
* the scale is indexed by GLOBAL position and the legacy path applies it after the all-to-all, so
  this rank's chunk positions are the wrong tensor — the CP wrapper publishes the full ones.

Run: ``python tests/cpu/parallelism/test_cp_llama4_attn_scale.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.distributed.context_parallel.layers import mistral4 as cp_mistral4
from src.distributed.context_parallel.layers.mistral4 import Mistral4UlyssesAttention

HIDDEN = 32
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
QK_NOPE = 6
QK_ROPE = 4
QK_HEAD_DIM = QK_NOPE + QK_ROPE
V_HEAD_DIM = 8
KV_LORA_RANK = 12

BETA = 0.1
ORIGINAL_MAX_PE = 8


class _FakeConfig:
    num_attention_heads = NUM_Q_HEADS
    num_key_value_heads = NUM_KV_HEADS
    rope_interleave = False

    def __init__(self, rope_parameters):
        self.rope_parameters = rope_parameters


class _FakeMistral4Attention(nn.Module):
    """Minimal stand-in for ``Mistral4Attention`` (MLA geometry + a rope_parameters dict)."""

    def __init__(self, rope_parameters):
        super().__init__()
        self.config = _FakeConfig(rope_parameters)
        self.layer_idx = 0
        self.qk_head_dim = QK_HEAD_DIM
        self.qk_nope_head_dim = QK_NOPE
        self.qk_rope_head_dim = QK_ROPE
        self.v_head_dim = V_HEAD_DIM
        self.kv_lora_rank = KV_LORA_RANK
        self.scaling = QK_HEAD_DIM**-0.5
        self.attention_dropout = 0.0
        self.is_causal = True

        torch.manual_seed(0)
        self.q_proj = nn.Linear(HIDDEN, NUM_Q_HEADS * QK_HEAD_DIM, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(HIDDEN, KV_LORA_RANK + QK_ROPE, bias=False)
        self.kv_a_layernorm = nn.LayerNorm(KV_LORA_RANK)
        self.kv_b_proj = nn.Linear(KV_LORA_RANK, NUM_KV_HEADS * (QK_NOPE + V_HEAD_DIM), bias=False)


def _rope_parameters(**overrides):
    params = {"llama_4_scaling_beta": BETA, "original_max_position_embeddings": ORIGINAL_MAX_PE}
    params.update(overrides)
    return {key: value for key, value in params.items() if value is not None}


def _wrapper(rope_parameters=None) -> Mistral4UlyssesAttention:
    # The wrapper refuses to build without upstream's scale function, so skip where the family is
    # absent rather than reporting its named ImportError as a failure.
    pytest.importorskip("transformers.models.mistral4.modeling_mistral4")
    attention = _FakeMistral4Attention(_rope_parameters() if rope_parameters is None else rope_parameters)
    return Mistral4UlyssesAttention(attention, cp_group=None, cp_size=2)


def _qk(seq_len: int):
    torch.manual_seed(1)
    return (
        torch.randn(2, NUM_Q_HEADS, seq_len, QK_HEAD_DIM),
        torch.randn(2, NUM_KV_HEADS, seq_len, QK_HEAD_DIM),
    )


def test_scale_is_upstreams_own_function_applied_to_q_only():
    """No private copy of the formula: an upstream change would otherwise leave CP runs of Mistral4
    scaled differently from every other run of the same checkpoint."""
    upstream = pytest.importorskip("transformers.models.mistral4.modeling_mistral4").get_llama_4_attn_scale
    assert cp_mistral4.get_llama_4_attn_scale is upstream

    wrapper = _wrapper()
    seq_len = 4 * ORIGINAL_MAX_PE
    q, k = _qk(seq_len)
    position_ids = torch.arange(seq_len, dtype=torch.long).expand(2, seq_len).contiguous()

    q_out, k_out = wrapper._post_rope(q, k, position_ids)

    expected = upstream(position_ids, BETA, ORIGINAL_MAX_PE)
    assert expected.shape == (2, 1, seq_len, 1), "the scale must broadcast over [B, H, S, D]"
    torch.testing.assert_close(q_out, q * expected.to(q.dtype), rtol=0, atol=0)
    torch.testing.assert_close(k_out, k, rtol=0, atol=0)
    # The ramp is the whole point: inside the original window the scale is exactly 1.0.
    torch.testing.assert_close(q_out[:, :, :ORIGINAL_MAX_PE], q[:, :, :ORIGINAL_MAX_PE], rtol=0, atol=0)
    assert not torch.allclose(q_out[:, :, ORIGINAL_MAX_PE:], q[:, :, ORIGINAL_MAX_PE:])


@pytest.mark.parametrize("missing", ["llama_4_scaling_beta", "original_max_position_embeddings"])
def test_unresolvable_scale_raises_instead_of_training_unscaled(missing):
    """A rope_parameters dict without the key must raise: skipping the scale trains a different
    attention than the family defines, with nothing in the log."""
    with pytest.raises(ValueError, match="llama-4 attention scale"):
        _wrapper(_rope_parameters(**{missing: None}))


def test_missing_rope_parameters_entirely_also_raises():
    with pytest.raises(ValueError, match="llama-4 attention scale"):
        _wrapper({})


def test_absent_global_positions_raise_rather_than_silently_skipping_the_scale():
    """The legacy path applies the scale AFTER the all-to-all, where Q spans the whole sequence, so
    it takes the wrapper-published global positions; ``None`` means nothing published them."""
    wrapper = _wrapper()
    q, k = _qk(ORIGINAL_MAX_PE)

    with pytest.raises(ValueError, match="FULL sequence's position_ids"):
        wrapper._post_rope(q, k, None)


def test_no_per_layer_position_gather_survives():
    """The positions are threaded down once by the wrapper: a per-layer all-gather here was one
    extra collective per attention layer per forward."""
    assert not hasattr(Mistral4UlyssesAttention, "_gather_position_ids")
    assert not hasattr(Mistral4UlyssesAttention, "_compute_llama_4_attn_scale")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
