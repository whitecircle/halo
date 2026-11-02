"""EP layer for Qwen3-Next (Qwen3.5-MoE's block under a router with a configurable renormalization)."""

from __future__ import annotations

import torch

from halo.distributed.expert_parallel.families.qwen3_5 import EPQwen3_5MoELayer
from halo.models.moe.family_specs import QWEN3_NEXT


class EPQwen3NextMoELayer(EPQwen3_5MoELayer):
    """EP layer for ``Qwen3NextSparseMoeBlock``, sharing Qwen3.5-MoE's whole EP contract.

    Same fused ``gate_up_proj``/``down_proj`` experts and sigmoid-gated shared expert; ``Qwen3NextTopKRouter``
    renormalizes the top-k weights only under its ``norm_topk_prob``, so every gate weight the layer re-derives
    (routing replay, the balancing-biased route) reads that flag live.
    """

    HF_MODULE_NAMES = ("Qwen3NextSparseMoeBlock",)
    FAMILY = QWEN3_NEXT

    def _gate_weights_at(self, router_logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """``Qwen3NextTopKRouter`` weights at ``indices``: the shared softmax gating, renormalized per the
        router's own ``norm_topk_prob``."""
        return self._softmax_gate_weights_at(router_logits, indices, normalize=self.gate.norm_topk_prob)

    def _biased_route(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The biased top-k gated by :meth:`_gate_weights_at`: the unconditional renormalization of
        ``_deepseek_biased_route`` would mis-gate a ``norm_topk_prob: false`` router. ``_biased_topk`` applies the
        routing replay itself."""
        topk_indices = self._biased_topk(router_logits)
        return topk_indices, self._gate_weights_at(router_logits, topk_indices)
