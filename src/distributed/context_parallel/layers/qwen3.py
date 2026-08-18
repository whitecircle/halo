"""Ulysses attention wrapper for Qwen3 / Qwen3-MoE / Qwen3-VL.

Model-specific features handled here:

- Q/K RMSNorm applied after projection.
- Standard rotate-half RoPE, which is the base's default ``_apply_rotary_core``.
"""

from __future__ import annotations

import torch

from src.distributed.context_parallel.base_layer import UlyssesAttentionBase


class Qwen3MoeUlyssesAttention(UlyssesAttentionBase):
    """Ulysses-wrapped attention for Qwen3 MoE / dense / Qwen3-VL text models."""

    HF_MODULE_NAMES = ("Qwen3MoeAttention", "Qwen3Attention", "Qwen3VLTextAttention")

    def _project_qkv(
        self, hidden_states: torch.Tensor, batch_size: int, local_seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden states to Q, K, V with Qwen3's post-projection RMSNorm."""
        attn = self.original_attention
        hidden_shape = (batch_size, local_seq_len, -1, self.head_dim)

        query_states = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape))
        key_states = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape))
        value_states = attn.v_proj(hidden_states).view(hidden_shape)

        return query_states, key_states, value_states
