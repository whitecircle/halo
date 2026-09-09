"""EP wrapper for Mistral4 MoE (fused experts + group-routed softmax + shared expert).

Mistral4 (text backbone of ``mistral3`` VLMs) ships a DeepSeek-V3-style MoE block. Routing matches
GLM-4 MoE Lite except that the score function is softmax rather than sigmoid, with no
``e_score_correction_bias``; the fused weight layout matches GLM-4 / Qwen3.5, so it reuses the base
fused-GLU helpers.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.distributed.expert_parallel.base_layer import EPGroupLimitedMoELayerBase


class EPMistral4MoELayer(EPGroupLimitedMoELayerBase):
    """EP wrapper for Mistral4 MoE (``Mistral4MoE``) using DeepEP."""

    HF_MODULE_NAMES = ("Mistral4MoE",)  # text backbone of mistral3 VLMs
    # Both spellings: the published Mistral-Small-4 checkpoints are composite (top-level
    # ``model_type: mistral3`` wrapping a ``mistral4`` text tower), and the merge/lazy-load resolvers
    # key on the top-level type. Same dual declaration as gemma4/qwen3_5/inkling.
    HF_MODEL_TYPES = ("mistral4", "mistral3")

    _supports_bias_balancing = True

    _NUM_EXPERTS_ATTR_PATHS = ("n_routed_experts", "experts.num_experts")

    _SHARED_EXPERT_ATTRS = ("shared_experts",)

    def _init_summary_extras(self, original_layer: nn.Module) -> tuple[str, ...]:
        return (f"n_group={self.n_group}, topk_group={self.topk_group}, top_k={self.top_k}",)

    def _routing_scores(self, router_logits: torch.Tensor) -> torch.Tensor:
        """``Mistral4TopkRouter`` scores with a softmax over all experts, where the rest of the
        group-limited families use a per-expert sigmoid."""
        return router_logits.softmax(dim=-1)
