"""Ulysses attention wrapper for GLM4 MoE Lite (DeepSeek-V3-style MLA).

Everything here is the shared MLA layer (:class:`MLAUlyssesAttentionBase`): compressed KV splits
into ``k_pass`` and a shared rotary ``k_rot`` broadcast to every KV head before the all-to-all; Q/K
carry ``qk_head_dim`` against a separate ``v_head_dim``, so the legacy path runs and the MLA flash
helper pads V; RoPE applies to the rope half only. No llama-4 scaling, so this family adds no hook.
"""

from __future__ import annotations

from src.distributed.context_parallel.base_layer import MLAUlyssesAttentionBase


class Glm4MoeLiteUlyssesAttention(MLAUlyssesAttentionBase):
    """Ulysses-wrapped attention for GLM4 MoE Lite models (legacy path only)."""

    HF_MODULE_NAMES = ("Glm4MoeLiteAttention",)
