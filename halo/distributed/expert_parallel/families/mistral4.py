"""EP wrapper for Mistral4 MoE: GLM-4 MoE Lite's group-limited routing with a softmax score and no
correction bias, over fused experts and a shared expert."""

from __future__ import annotations

import torch
import torch.nn as nn

from halo.distributed.expert_parallel.moe_layer.shared_experts_layer import EPGroupLimitedMoELayerBase
from halo.models.moe.family_specs import MISTRAL4


class EPMistral4MoELayer(EPGroupLimitedMoELayerBase):
    """EP wrapper for Mistral4 MoE (``Mistral4MoE``) using DeepEP."""

    HF_MODULE_NAMES = ("Mistral4MoE",)
    FAMILY = MISTRAL4

    SUPPORTS_BIAS_BALANCING = True

    _NUM_EXPERTS_ATTR_PATHS = ("n_routed_experts", "experts.num_experts")

    def _init_summary_extras(self, original_layer: nn.Module) -> tuple[str, ...]:
        return (f"n_group={self.n_group}, topk_group={self.topk_group}, top_k={self.top_k}",)

    def _routing_scores(self, router_logits: torch.Tensor) -> torch.Tensor:
        """``Mistral4TopkRouter`` scores with a softmax over all experts, where the rest of the
        group-limited families use a per-expert sigmoid."""
        return router_logits.softmax(dim=-1)
