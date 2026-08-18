"""Ulysses attention wrapper for GLM4 MoE Lite (DeepSeek-V3-style MLA).

Behaviour comes entirely from :class:`MLAUlyssesAttentionBase`: compressed KV with a shared rotary
``k_rot`` broadcast to every KV head, the legacy ``[B, H, S, D]`` path with V padded to
``qk_head_dim``, and RoPE on the rope half only. The family adds no hook of its own.
"""

from __future__ import annotations

from src.distributed.context_parallel.base_layer import MLAUlyssesAttentionBase


class Glm4MoeLiteUlyssesAttention(MLAUlyssesAttentionBase):
    """Ulysses-wrapped attention for GLM4 MoE Lite models (legacy path only)."""

    HF_MODULE_NAMES = ("Glm4MoeLiteAttention",)
