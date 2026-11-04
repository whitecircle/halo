"""EP layer for Mistral4 MoE: group-limited routing with a softmax score and no correction bias."""

from __future__ import annotations

import torch

from halo.distributed.expert_parallel.moe_layer.shared_experts_layer import EPGroupLimitedMoELayerBase
from halo.models.moe.family_specs import MISTRAL4


class EPMistral4MoELayer(EPGroupLimitedMoELayerBase):
    HF_MODULE_NAMES = ("Mistral4MoE",)
    FAMILY = MISTRAL4

    _SHARED_EXPERT_REQUIRED = True

    def _routing_scores(self, router_logits: torch.Tensor) -> torch.Tensor:
        """``Mistral4TopkRouter`` scores with a softmax over all experts."""
        return router_logits.softmax(dim=-1)
