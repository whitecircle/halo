"""Ulysses attention wrapper for Cohere2 MoE (Command A+) models.

Model-specific features handled here:

- RoPE only on sliding-window layers (full-attention layers are NoPE), plus the dense-prefix
  ``force_rope`` override — mirrors ``Cohere2MoeAttention.forward``.
- Interleaved (GPT-J-style) rotary applied in fp32.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.distributed.context_parallel.base_layer import UlyssesAttentionBase


class Cohere2MoeUlyssesAttention(UlyssesAttentionBase):
    """Ulysses-wrapped attention for Cohere2 MoE models."""

    HF_MODULE_NAMES = ("Cohere2MoeAttention",)

    def _configure(self, original_attention: nn.Module) -> None:
        self._use_rope = self.sliding_window is not None or getattr(original_attention, "force_rope", False)

    def debug_fields(self) -> dict[str, object]:
        return super().debug_fields() | {"rope": self._use_rope}

    def _project_qkv(
        self, hidden_states: torch.Tensor, batch_size: int, local_seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._project_qkv_plain(hidden_states, batch_size, local_seq_len)

    def _apply_rotary_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-copy RoPE in ``[B, S, H, D]`` layout; identity on NoPE layers."""
        if not self._use_rope:
            return q, k
        return self._rotary_emb(q, cos, sin), self._rotary_emb(k, cos, sin)

    @staticmethod
    def _rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Interleaved rotary in fp32 (matches ``modeling_cohere2_moe.apply_rotary_pos_emb``);
        layout-agnostic (operates only on the last dim)."""
        dtype = x.dtype
        x = x.float()
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
        return (x * cos + rotated * sin).to(dtype)
