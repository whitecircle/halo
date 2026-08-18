"""Ulysses attention wrapper for Qwen3.5 / Qwen3.6 full-attention layers.

``q_proj`` is double-width (``num_q_heads * head_dim * 2``): output splits into query +
sigmoid gate, with ``attn_output * sigmoid(gate)`` before ``o_proj``. The gate is computed
on the local shard and applied after the all-to-all, so no extra comm. The family's
linear-attention layers (``Qwen3_5MoeGatedDeltaNet``) are rejected by validate_model_for_ulysses.
"""

from __future__ import annotations

import torch

from src.distributed.context_parallel.base_layer import UlyssesAttentionBase


class Qwen3_5MoeUlyssesAttention(UlyssesAttentionBase):
    """Ulysses-wrapped Qwen3.5 / Qwen3.6 full-attention block (see module docstring)."""

    HF_MODULE_NAMES = ("Qwen3_5MoeAttention", "Qwen3_5Attention")

    def _project_qkv(
        self, hidden_states: torch.Tensor, batch_size: int, local_seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project to Q/K/V, returning the sigmoid gate from the double-width q_proj as a 4th value.

        The base ``forward`` threads the extra through as a local and hands it back to
        :meth:`_post_attention` — no instance stash, so gradient-checkpoint recompute is re-entrant.
        """
        attn = self.original_attention
        hidden_shape = (batch_size, local_seq_len, -1, self.head_dim)

        q_out = attn.q_proj(hidden_states).view(batch_size, local_seq_len, -1, self.head_dim * 2)
        query_states, gate = torch.chunk(q_out, 2, dim=-1)
        gate = gate.reshape(batch_size, local_seq_len, -1)

        query_states = attn.q_norm(query_states)
        key_states = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape))
        value_states = attn.v_proj(hidden_states).view(hidden_shape)

        return query_states, key_states, value_states, gate

    def _post_attention(self, attn_output: torch.Tensor, *extras: torch.Tensor) -> torch.Tensor:
        """Sigmoid gate before o_proj; both in local-sequence layout after the all-to-all."""
        (gate,) = extras
        return attn_output * torch.sigmoid(gate)
