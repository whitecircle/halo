"""Ulysses attention wrapper for Mistral4 (text backbone of mistral3 VLMs).

DeepSeek-V3-style MLA on the shared :class:`MLAUlyssesAttentionBase`; the one family-specific piece
is the position-dependent llama-4 factor Q is scaled by after RoPE. That scale is indexed by global
position and the legacy path applies it after the all-to-all, so it reads the full-sequence
``position_ids`` the CP wrapper publishes rather than this rank's chunk.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers.models.mistral4.modeling_mistral4 import get_llama_4_attn_scale

from src.distributed.context_parallel.base_layer import MLAUlyssesAttentionBase

# The two ``rope_parameters`` entries the llama-4 scale is computed from. ``Mistral4Config`` injects
# them only when ``rope_parameters`` is None, so a checkpoint shipping its own dict can lack them.
LLAMA_4_SCALE_ROPE_KEYS = ("llama_4_scaling_beta", "original_max_position_embeddings")


class Mistral4UlyssesAttention(MLAUlyssesAttentionBase):
    """Ulysses-wrapped attention for Mistral4 (MLA + YARN + llama-4 scaling)."""

    HF_MODULE_NAMES = ("Mistral4Attention",)

    def __init__(
        self,
        original_attention: nn.Module,
        cp_group: dist.ProcessGroup,
        cp_size: int,
    ):
        super().__init__(original_attention, cp_group, cp_size)

        # Mistral4Attention.forward applies this scale unconditionally, so a CP run that skipped it
        # would train a different attention than the family defines, and than a non-CP run of the
        # same checkpoint.
        rope_params = getattr(self.config, "rope_parameters", None) or {}
        missing = [key for key in LLAMA_4_SCALE_ROPE_KEYS if rope_params.get(key) is None]
        if missing:
            raise ValueError(
                f"Mistral4 under Context Parallelism cannot resolve the llama-4 attention scale: "
                f"config.rope_parameters is missing {missing}. Add them to rope_parameters "
                f"(model_init_kwargs), or train this checkpoint without CP."
            )
        self._llama_4_scaling_beta = rope_params["llama_4_scaling_beta"]
        self._llama_4_original_max_pe = rope_params["original_max_position_embeddings"]

    def _post_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.LongTensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scale Q by the position-dependent llama-4 factor, in the legacy ``[B, H, S, D]`` layout."""
        if position_ids is None:
            raise ValueError(
                "Mistral4UlyssesAttention needs the FULL sequence's position_ids for the llama-4 "
                "attention scale. UlyssesCPModelWrapper publishes them on every patched attention "
                "layer once per forward — run Mistral4 CP through it (load_model_for_cp / "
                "load_model_for_ep_cp)."
            )
        scale = get_llama_4_attn_scale(position_ids, self._llama_4_scaling_beta, self._llama_4_original_max_pe)
        return q * scale.to(q.dtype), k
