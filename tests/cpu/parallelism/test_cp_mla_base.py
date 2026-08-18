#!/usr/bin/env python
"""CPU tests for the shared MLA Ulysses base (``MLAUlyssesAttentionBase``).

GLM4-MoE-Lite and Mistral4 are both DeepSeek-V3-style MLA, and the whole shared layer — the
``head_dim`` back-fill, the geometry read off the wrapped module, ``_project_qkv``,
``_apply_rotary_pos_emb``, the legacy-path flag — lives once on the base. These tests fail if a
family re-forks a private copy (a hand-written geometry block drifts on ``use_qkv_lora``,
``rope_interleave`` and the ``qk_rope_head_dim`` fallback) or if the hoisted MLA compute stops
matching the compressed-KV reference.

Run: ``python tests/cpu/parallelism/test_cp_mla_base.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from src.distributed.context_parallel.base_layer import MLAUlyssesAttentionBase
from src.distributed.context_parallel.layers.glm4 import Glm4MoeLiteUlyssesAttention
from src.distributed.context_parallel.layers.mistral4 import Mistral4UlyssesAttention

HIDDEN = 32
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2
QK_NOPE = 6
QK_ROPE = 4
QK_HEAD_DIM = QK_NOPE + QK_ROPE
V_HEAD_DIM = 8
KV_LORA_RANK = 12


class _FakeConfig:
    num_attention_heads = NUM_Q_HEADS
    num_key_value_heads = NUM_KV_HEADS
    rope_interleave = False
    # Mistral4's wrapper refuses a config whose rope_parameters cannot resolve the llama-4 attention
    # scale; that guard is pinned in test_cp_llama4_attn_scale.py, and the shared-MLA tests here
    # construct the family for real, so the fake has to satisfy it.
    rope_parameters = {"llama_4_scaling_beta": 0.1, "original_max_position_embeddings": 8192}


class _RopelessConfig:
    """``_FakeConfig`` minus ``rope_interleave`` — the shape a defensive ``getattr`` default hides.

    Not a subclass: the attribute would be inherited and the omission would not be one.
    """

    num_attention_heads = NUM_Q_HEADS
    num_key_value_heads = NUM_KV_HEADS
    rope_parameters = _FakeConfig.rope_parameters


class _FakeMLAAttention(nn.Module):
    """Minimal stand-in for a DeepSeek-V3-style MLA attention module (no Q compression)."""

    def __init__(self):
        super().__init__()
        self.config = _FakeConfig()
        self.layer_idx = 0
        self.qk_head_dim = QK_HEAD_DIM
        self.qk_nope_head_dim = QK_NOPE
        self.qk_rope_head_dim = QK_ROPE
        self.v_head_dim = V_HEAD_DIM
        self.kv_lora_rank = KV_LORA_RANK
        self.num_key_value_groups = NUM_Q_HEADS // NUM_KV_HEADS
        self.scaling = QK_HEAD_DIM**-0.5
        self.attention_dropout = 0.0
        self.is_causal = True

        torch.manual_seed(0)
        self.q_proj = nn.Linear(HIDDEN, NUM_Q_HEADS * QK_HEAD_DIM, bias=False)
        # transformers 5.16 layout: the unused Q-compression branch is present but None.
        self.q_a_proj = None
        self.q_a_layernorm = None
        self.q_b_proj = None
        self.kv_a_proj_with_mqa = nn.Linear(HIDDEN, KV_LORA_RANK + QK_ROPE, bias=False)
        self.kv_a_layernorm = nn.LayerNorm(KV_LORA_RANK)
        self.kv_b_proj = nn.Linear(KV_LORA_RANK, NUM_KV_HEADS * (QK_NOPE + V_HEAD_DIM), bias=False)


class _FakeCompressedQAttention(_FakeMLAAttention):
    """The same module with Q on the LoRA-style compression (``q_a_proj`` → norm → ``q_b_proj``)."""

    def __init__(self):
        super().__init__()
        self.q_proj = None
        self.q_lora_rank = 16
        self.q_a_proj = nn.Linear(HIDDEN, self.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.LayerNorm(self.q_lora_rank)
        self.q_b_proj = nn.Linear(self.q_lora_rank, NUM_Q_HEADS * QK_HEAD_DIM, bias=False)


def _make_wrapper(cls, attn=None):
    return cls(attn if attn is not None else _FakeMLAAttention(), cp_group=None, cp_size=2)


@pytest.mark.parametrize("cls", [Glm4MoeLiteUlyssesAttention, Mistral4UlyssesAttention])
def test_mla_families_do_not_fork_the_shared_compute(cls):
    """Neither family may re-declare the hoisted MLA methods — a private copy is silent drift."""
    for method in ("_project_qkv", "_apply_rotary_pos_emb", "_flash_attention", "_resolve_num_kv_heads"):
        assert method not in vars(cls), (
            f"{cls.__name__} re-declares {method}; the MLA implementation lives on "
            f"MLAUlyssesAttentionBase — delete the fork or justify a genuinely different layout."
        )
        assert getattr(cls, method) is getattr(MLAUlyssesAttentionBase, method)


@pytest.mark.parametrize("cls", [Glm4MoeLiteUlyssesAttention, Mistral4UlyssesAttention])
def test_legacy_path_flag_is_declared_once_on_the_mla_base(cls):
    """MLA geometry (qk vs v head dims) IS the reason for the legacy path — one declaration, not
    one per family, or a new MLA family silently lands on flash-attn's native GQA."""
    assert "_optimize_attention" not in vars(cls)
    assert cls._optimize_attention is False


@pytest.mark.parametrize("cls", [Glm4MoeLiteUlyssesAttention, Mistral4UlyssesAttention])
def test_head_dim_backfill_runs_on_the_base_not_in_the_family(cls):
    """``UlyssesAttentionBase.__init__`` reads ``original_attention.head_dim``, which MLA modules
    spell ``qk_head_dim``. The back-fill that bridges the two belongs to the MLA base — a family
    copy is the same fork this file exists to catch."""
    attn = _FakeMLAAttention()
    assert not hasattr(attn, "head_dim"), "the fake must reproduce an MLA module's naming"

    backfilled_before_base = None
    base_init = MLAUlyssesAttentionBase.__init__

    def spy(self, original_attention, cp_group, cp_size):
        nonlocal backfilled_before_base
        backfilled_before_base = hasattr(original_attention, "head_dim")
        base_init(self, original_attention, cp_group, cp_size)

    with patch.object(MLAUlyssesAttentionBase, "__init__", spy):
        wrapper = cls(attn, cp_group=None, cp_size=2)

    assert backfilled_before_base is False, (
        f"{cls.__name__}.__init__ back-fills head_dim itself; MLAUlyssesAttentionBase owns that."
    )
    assert attn.head_dim == QK_HEAD_DIM
    assert wrapper.head_dim == QK_HEAD_DIM


def test_backfill_never_overwrites_a_head_dim_the_family_carries():
    """``if not hasattr`` guard: a module that already declares ``head_dim`` keeps its own."""
    attn = _FakeMLAAttention()
    attn.head_dim = QK_HEAD_DIM + 1

    wrapper = Glm4MoeLiteUlyssesAttention(attn, cp_group=None, cp_size=2)

    assert attn.head_dim == QK_HEAD_DIM + 1
    assert wrapper.head_dim == QK_HEAD_DIM + 1


@pytest.mark.parametrize("cls", [Glm4MoeLiteUlyssesAttention, Mistral4UlyssesAttention])
def test_mla_geometry_is_read_off_the_wrapped_module_by_the_base(cls):
    """The base compute reads these attributes off the module; every family must get them, with the
    SAME answer — the hand-copied blocks this replaced disagreed on three of them."""
    attn = _FakeMLAAttention()
    wrapper = _make_wrapper(cls, attn)

    assert wrapper.qk_head_dim == QK_HEAD_DIM
    assert wrapper.qk_nope_head_dim == QK_NOPE
    assert wrapper.qk_rope_head_dim == QK_ROPE
    assert wrapper.v_head_dim == V_HEAD_DIM
    assert wrapper.kv_lora_rank == KV_LORA_RANK
    assert wrapper.use_qkv_lora is False  # the fake module has a plain q_proj
    assert wrapper.rope_interleave is False  # read off the config, not defaulted


def test_glm4_carries_no_layer_of_its_own():
    """GLM4-MoE-Lite is pure MLA: it declares its HF class names and nothing else. A re-added
    ``__init__`` here is the hand-copied geometry block coming back."""
    assert "__init__" not in vars(Glm4MoeLiteUlyssesAttention)


@pytest.mark.parametrize("cls", [Glm4MoeLiteUlyssesAttention, Mistral4UlyssesAttention])
def test_use_qkv_lora_follows_the_modules_project_qkv_actually_calls(cls):
    """A compressed-Q module must take the ``q_a_proj``→norm→``q_b_proj`` branch, and only it — both
    families decide it from the same fact (the module's own ``q_a_proj``, not ``q_lora_rank``)."""
    wrapper = _make_wrapper(cls, _FakeCompressedQAttention())
    assert wrapper.use_qkv_lora is True

    batch, seq = 2, 3
    hidden = torch.randn(batch, seq, HIDDEN)
    attn = wrapper.original_attention
    q, _, _ = wrapper._project_qkv(hidden, batch, seq)

    expected = attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(hidden))).view(batch, seq, NUM_Q_HEADS, QK_HEAD_DIM)
    torch.testing.assert_close(q, expected)


@pytest.mark.parametrize("cls", [Glm4MoeLiteUlyssesAttention, Mistral4UlyssesAttention])
def test_rope_interleave_follows_the_config_instead_of_a_defensive_default(cls):
    """Both families' configs default it to True and their forwards read it unconditionally; a
    ``getattr(config, "rope_interleave", False)`` would invert the family's own rotary."""
    attn = _FakeMLAAttention()
    attn.config.rope_interleave = True
    assert _make_wrapper(cls, attn).rope_interleave is True

    # And a config that never declares it must fail loud: a ``getattr(..., False)`` default picks the
    # OPPOSITE of both families' own default and silently trains a different rotary.
    bare = _FakeMLAAttention()
    bare.config = _RopelessConfig()
    with pytest.raises(AttributeError, match="rope_interleave"):
        _make_wrapper(cls, bare)


@pytest.mark.parametrize("cls", [Glm4MoeLiteUlyssesAttention, Mistral4UlyssesAttention])
def test_kv_head_count_comes_from_kv_b_proj_not_the_config(cls):
    """MLA's K/V are whatever ``kv_b_proj`` emits (upstream: one head per QUERY head), so a config
    ``num_key_value_heads`` that disagrees must not decide the reshape or the GQA repeat factor."""
    attn = _FakeMLAAttention()
    attn.config.num_key_value_heads = NUM_Q_HEADS  # config disagrees with the projection's width

    wrapper = _make_wrapper(cls, attn)

    assert wrapper.num_kv_heads == NUM_KV_HEADS
    assert wrapper.local_num_key_value_groups == (NUM_Q_HEADS // 2) // (NUM_KV_HEADS // 2)
    batch, seq = 2, 3
    _, k, v = wrapper._project_qkv(torch.randn(batch, seq, HIDDEN), batch, seq)
    assert k.shape == (batch, seq, NUM_KV_HEADS, QK_HEAD_DIM)
    assert v.shape == (batch, seq, NUM_KV_HEADS, V_HEAD_DIM)


@pytest.mark.parametrize("cls", [Glm4MoeLiteUlyssesAttention, Mistral4UlyssesAttention])
def test_project_qkv_matches_compressed_kv_reference(cls):
    """Shapes and values of the hoisted projection vs an inline MLA reference."""
    wrapper = _make_wrapper(cls)
    attn = wrapper.original_attention
    batch, seq = 2, 3
    hidden = torch.randn(batch, seq, HIDDEN)

    q, k, v = wrapper._project_qkv(hidden, batch, seq)

    assert q.shape == (batch, seq, NUM_Q_HEADS, QK_HEAD_DIM)
    assert k.shape == (batch, seq, NUM_KV_HEADS, QK_HEAD_DIM)
    assert v.shape == (batch, seq, NUM_KV_HEADS, V_HEAD_DIM)

    ref_q = attn.q_proj(hidden).view(batch, seq, NUM_Q_HEADS, QK_HEAD_DIM)
    compressed = attn.kv_a_proj_with_mqa(hidden)
    k_pass, k_rot_shared = torch.split(compressed, [KV_LORA_RANK, QK_ROPE], dim=-1)
    kv = attn.kv_b_proj(attn.kv_a_layernorm(k_pass)).view(batch, seq, NUM_KV_HEADS, QK_NOPE + V_HEAD_DIM)
    ref_v = kv[..., QK_NOPE:]

    torch.testing.assert_close(q, ref_q)
    torch.testing.assert_close(v, ref_v)
    torch.testing.assert_close(k[..., :QK_NOPE], kv[..., :QK_NOPE])
    # The shared rotary K is broadcast identically to every KV head.
    for head in range(NUM_KV_HEADS):
        torch.testing.assert_close(k[:, :, head, QK_NOPE:], k_rot_shared)


@pytest.mark.parametrize("cls", [Glm4MoeLiteUlyssesAttention, Mistral4UlyssesAttention])
@pytest.mark.parametrize("interleave", [False, True])
def test_apply_rotary_pos_emb_touches_only_the_rope_half(cls, interleave):
    wrapper = _make_wrapper(cls)
    wrapper.rope_interleave = interleave
    batch, heads, seq = 2, NUM_Q_HEADS, 3
    q = torch.randn(batch, heads, seq, QK_HEAD_DIM)
    k = torch.randn(batch, NUM_KV_HEADS, seq, QK_HEAD_DIM)
    cos = torch.randn(batch, seq, QK_ROPE)
    sin = torch.randn(batch, seq, QK_ROPE)

    q_out, k_out = wrapper._apply_rotary_pos_emb(q, k, cos, sin)

    assert q_out.shape == q.shape and k_out.shape == k.shape
    # nope half passes through untouched; rope half is rotated.
    torch.testing.assert_close(q_out[..., :QK_NOPE], q[..., :QK_NOPE])
    torch.testing.assert_close(k_out[..., :QK_NOPE], k[..., :QK_NOPE])
    assert not torch.allclose(q_out[..., QK_NOPE:], q[..., QK_NOPE:])

    rotary = wrapper._apply_rotary_emb_interleaved if interleave else wrapper._apply_rotary_emb
    torch.testing.assert_close(q_out[..., QK_NOPE:], rotary(q[..., QK_NOPE:], cos.unsqueeze(1), sin.unsqueeze(1)))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
