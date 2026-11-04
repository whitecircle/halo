"""EP layer for Qwen3.5 / Qwen3.6 MoE (fused experts + sigmoid-gated shared expert)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from halo.distributed.expert_parallel.moe_layer.shared_experts_layer import EPSharedExpertsMoELayerBase
from halo.models.moe.family_specs import QWEN3_5


class EPQwen3_5MoELayer(EPSharedExpertsMoELayerBase):
    """EP layer for ``Qwen3_5MoeSparseMoeBlock``. The shared expert and its sigmoid gate are not distributed."""

    HF_MODULE_NAMES = ("Qwen3_5MoeSparseMoeBlock",)
    FAMILY = QWEN3_5

    # ``SEVERS_AUX_LOSS`` stays False: the EP layer calls the real ``Qwen3_5MoeTopKRouter``, so the recorder still
    # fires for the text-only sibling.

    # ``Qwen3_5MoeTopKRouter`` renormalizes with no floor.
    _TOPK_WEIGHT_NORM_EPS = 0.0

    _SHARED_EXPERT_REQUIRED = True

    def _adopt_shared_experts(self, original_layer: nn.Module) -> None:
        super()._adopt_shared_experts(original_layer)
        self.shared_expert_gate = original_layer.shared_expert_gate

    def _gate_weights_at(self, router_logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """The shared softmax gating; this router has no ``norm_topk_prob`` and renormalizes unconditionally."""
        return self._softmax_gate_weights_at(router_logits, indices, normalize=True)

    def _route_hidden_states(self, flat: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        return self._route_router_choice_or_biased(flat)

    def _biased_route(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The unbiased softmax over the selected logits: ``Qwen3_5MoeTopKRouter``'s arithmetic up to its bf16 cast."""
        indices = self._biased_topk(router_logits)
        return indices, F.softmax(torch.gather(router_logits.float(), -1, indices), dim=-1)

    def _shared_expert_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return super()._shared_expert_output(hidden_states) * torch.sigmoid(self.shared_expert_gate(hidden_states))

    def replicated_named_params(self) -> list:
        return super().replicated_named_params() + self._submodule_named_params(
            [("shared_expert_gate", self.shared_expert_gate)]
        )
