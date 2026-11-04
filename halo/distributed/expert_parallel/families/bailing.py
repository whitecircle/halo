"""EP layer for Bailing MoE (individual experts fused at init + shared experts)."""

from __future__ import annotations

import torch
import torch.nn as nn

from halo.distributed.expert_parallel.moe_layer.shared_experts_layer import EPSharedExpertsMoELayerBase
from halo.models.moe.family_specs import BAILING


class EPBailingMoELayer(EPSharedExpertsMoELayerBase):
    """EP layer for BailingMoeV2/V3SparseMoeBlock. Per-expert ``nn.ModuleList`` experts, stacked into 3D at init. The
    gate returns ``(topk_idx, topk_weight, logits)`` with all routing internal; bias-update balancing adopts its
    ``expert_bias``. V3 shares V2's layout and gate arithmetic."""

    HF_MODULE_NAMES = ("BailingMoeV2SparseMoeBlock", "BailingMoeV3SparseMoeBlock")
    FAMILY = BAILING

    # Upstream builds the shared expert only under a config flag, so an absent one is a configuration.
    _SHARED_EXPERT_REQUIRED = False

    # Selection happens inside the gate (the forward never consults ``_transient_balancing_bias``), so the
    # base's transient fallback would register a buffer nothing reads.
    _supports_transient_balancing_bias = False
    # The EP layer returns a bare tensor where the V3 block returns ``(hidden, router_logits)``, so
    # ``outputs.router_logits`` stays empty; no Bailing modeling computes an aux loss regardless.
    SEVERS_AUX_LOSS = True

    def _adopt_expert_compute(self, original_layer: nn.Module, experts: nn.ModuleList) -> None:
        """Resolve the activation off the first expert this rank fuses; the ``nn.ModuleList`` container holds none."""
        self._resolve_activation(experts[self.expert_start], original_layer=original_layer)

    def _adopt_expert_params(self, experts: nn.ModuleList, weights_already_sharded: bool = False):
        """Stack per-expert weights into 3D matmul-convention tensors. The lazy loader materializes this rank's experts
        at their global indices, so the global slice applies either way."""
        start, end = self.expert_start, self.expert_end
        local_experts = [experts[i] for i in range(start, end)]

        # [E_local, M, H] for gate/up, [E_local, H, M] for down
        gate_stacked = torch.stack([e.gate_proj.weight.data for e in local_experts])
        up_stacked = torch.stack([e.up_proj.weight.data for e in local_experts])
        down_stacked = torch.stack([e.down_proj.weight.data for e in local_experts])
        self._store_separate_glu_params(gate_stacked, up_stacked, down_stacked)

    def _gate_weights_at(self, router_logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """``BailingMoeV2Gate`` weights at ``indices``; its ``expert_bias`` perturbs selection only."""
        scores = torch.sigmoid(router_logits.float()).type_as(router_logits)
        scores = torch.gather(scores, dim=1, index=indices).type_as(router_logits)
        weights = self._renormalize_topk_weights(scores) if self.top_k > 1 else scores
        return weights * float(self.gate.routed_scaling_factor)

    def _route_hidden_states(self, flat: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.amp.autocast("cuda", enabled=False):
            topk_idx, topk_weight, router_logits = self.gate(self._router_input(flat))
        return self._replay_router_choice(router_logits, topk_idx, topk_weight)
