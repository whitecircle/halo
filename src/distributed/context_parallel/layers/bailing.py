"""Ulysses attention wrapper for Bailing MoE / Ling 2.0.

Standard softmax GQA in a transformers-v4-era remote-code shell; the wrapper carries the family's
spellings rather than new attention math:

- fused ``query_key_value`` projection split into Q/K/V, and an output projection named ``dense``;
- Q/K RMSNorm over the head dim (``use_qk_norm``) applied before RoPE;
- partial rotary (``partial_rotary_factor``), rotating the leading ``cos.shape[-1]`` channels only;
- the softmax scale computed inline in the family's forward, so the module carries no ``scaling``;
- a three-value attention return, where transformers-v5 decoder layers take two.

``ATTENTION_CLASSES`` picks the eager / sdpa / FA2 subclass from ``_attn_implementation``; all three
are claimed, since they hold the same weights and CP replaces their forward either way. Ling 3.0
(``BailingMoeV3*``) and the Lightning-Attention-2 sibling are separate architectures; the latter
shares these class names and is rejected in :mod:`~src.distributed.context_parallel.validation`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.distributed.context_parallel.base_layer import UlyssesAttentionBase


class BailingMoeV2UlyssesAttention(UlyssesAttentionBase):
    """Ulysses-wrapped attention for Bailing MoE / Ling 2.0 (see module docstring)."""

    HF_MODULE_NAMES = (
        "BailingMoeV2Attention",
        "BailingMoeV2FlashAttention2",
        "BailingMoeV2SdpaAttention",
    )

    # The remote code declares only the v4 ``_supports_flash_attn_2``, which transformers v5 ignores
    # in favour of ``_supports_flash_attn``, so the model will not build under a flash label and
    # every Bailing run is labelled sdpa. That label describes the family's own forward, which this
    # wrapper replaces, and the geometry (uniform head_dim 128 GQA, no sinks, no additive bias) is
    # what CP's flash kernel needs.
    REQUIRES_FLASH_ATTN_LABEL = False

    def _configure(self, original_attention: nn.Module) -> None:
        self.use_qk_norm = bool(getattr(self.config, "use_qk_norm", False))

    def debug_fields(self) -> dict[str, object]:
        return super().debug_fields() | {"use_qk_norm": self.use_qk_norm}

    def _resolve_scaling(self) -> float:
        """Bailing scales by ``1/sqrt(head_dim)`` inside its own forward, carrying no ``scaling``."""
        return self.head_dim**-0.5

    @property
    def _output_projection(self) -> nn.Module:
        return self.original_attention.dense

    def _project_qkv(
        self, hidden_states: torch.Tensor, batch_size: int, local_seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the fused ``query_key_value`` projection, then the optional Q/K RMSNorm."""
        attn = self.original_attention

        qkv = attn.query_key_value(hidden_states).view(
            batch_size, local_seq_len, self.num_q_heads + 2 * self.num_kv_heads, self.head_dim
        )
        query_states, key_states, value_states = qkv.split(
            [self.num_q_heads, self.num_kv_heads, self.num_kv_heads], dim=-2
        )

        if self.use_qk_norm:
            query_states = attn.query_layernorm(query_states)
            key_states = attn.key_layernorm(key_states)

        return query_states, key_states, value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, object]:
        """Base forward, re-shaped to the v4 return contract.

        Bailing's decoder (and MTP) layers unpack ``(hidden_states, attn_weights, present_key_value)``
        from attention, so the base's two-value return would raise on the first forward. The cache is
        handed back as the family's own forward does; CP never writes one.
        """
        attn_output, attn_weights = super().forward(hidden_states, position_embeddings, **kwargs)
        return attn_output, attn_weights, kwargs.get("past_key_value")
