"""EP layer for Qwen3 MoE (pre-fused experts)."""

from __future__ import annotations

import torch

from halo.distributed.expert_parallel.moe_layer.shared_experts_layer import EPSharedExpertsMoELayerBase
from halo.models.moe.family_specs import QWEN3


class EPQwen3MoELayer(EPSharedExpertsMoELayerBase):
    """``Qwen3MoeExperts`` fuses ``gate_up_proj``/``down_proj``; gate_up splits into matmul-convention halves."""

    HF_MODULE_NAMES = ("Qwen3MoeSparseMoeBlock",)
    FAMILY = QWEN3
    SUPPORTS_MOK = True

    # ``Qwen3MoeTopKRouter`` renormalizes with no floor.
    _TOPK_WEIGHT_NORM_EPS = 0.0

    def _gate_weights_at(self, router_logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Renormalized per the router's configurable (default-off) ``norm_topk_prob``, read live so routing replay
        matches the router the forward just called."""
        return self._softmax_gate_weights_at(router_logits, indices, normalize=self.gate.norm_topk_prob)

    def _route_hidden_states(self, flat: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        return self._route_router_choice_or_biased(flat)
