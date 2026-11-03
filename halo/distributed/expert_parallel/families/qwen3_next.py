"""EP layer for Qwen3-Next (Qwen3.5-MoE's block under a router with a configurable renormalization)."""

from __future__ import annotations

import torch

from halo.distributed.expert_parallel.families.qwen3_5 import EPQwen3_5MoELayer
from halo.models.moe.family_specs import QWEN3_NEXT


class EPQwen3NextMoELayer(EPQwen3_5MoELayer):
    """Qwen3.5-MoE's EP contract, except ``Qwen3NextTopKRouter`` renormalizes only under its ``norm_topk_prob``, so
    every gate weight the layer re-derives reads that flag live."""

    HF_MODULE_NAMES = ("Qwen3NextSparseMoeBlock",)
    FAMILY = QWEN3_NEXT

    def _gate_weights_at(self, router_logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return self._softmax_gate_weights_at(router_logits, indices, normalize=self.gate.norm_topk_prob)

    def _biased_route(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The biased top-k gated by :meth:`_gate_weights_at`, since the unconditional renormalization of
        ``_deepseek_biased_route`` would mis-gate a ``norm_topk_prob: false`` router."""
        topk_indices = self._biased_topk(router_logits)
        return topk_indices, self._gate_weights_at(router_logits, topk_indices)
