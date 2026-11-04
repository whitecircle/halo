"""EP layer for LFM-2 MoE (fused experts with sigmoid routing + expert bias)."""

from __future__ import annotations

import torch
import torch.nn as nn

from halo.distributed.expert_parallel.moe_layer.shared_experts_layer import EPSharedExpertsMoELayerBase
from halo.models.moe.family_specs import LFM2


class EPLfm2MoELayer(EPSharedExpertsMoELayerBase):
    """EP layer for LFM2 MoE: sigmoid routing with optional ``expert_bias`` correction. The block carries no shared
    expert, so the declared slot registers ``None`` and the shared leg is skipped."""

    HF_MODULE_NAMES = ("Lfm2MoeSparseMoeBlock",)
    FAMILY = LFM2

    # ``Lfm2MoeTopKRouter``'s renormalization floor.
    _TOPK_WEIGHT_NORM_EPS = 1e-6

    def _adopt_routing(self, original_layer: nn.Module) -> None:
        """Top-k plus the sigmoid-routing knobs off the gate, without defaults: they change the routing math, so a
        moved attribute must raise."""
        super()._adopt_routing(original_layer)
        self.norm_topk_prob = self.gate.norm_topk_prob
        self.routed_scaling_factor = self.gate.routed_scaling_factor
        self.use_expert_bias = self.gate.use_expert_bias

        if not self.use_expert_bias:
            return
        if not hasattr(original_layer, "expert_bias"):
            # An upstream move, not a configuration: flipping the flag off would route unbiased for the rest of the run
            # with no shape, dtype or key changing.
            raise AttributeError(
                f"EPLfm2MoELayer: the router declares use_expert_bias=True but "
                f"{type(original_layer).__name__} carries no 'expert_bias' — the family moved or "
                f"renamed the buffer upstream. Routing on would silently drop the pretrained "
                f"selection bias."
            )
        eb = original_layer.expert_bias
        if eb.is_meta:
            # A zero substitute would change routing and, under bias-update adoption, become the
            # exported bias. Every load path materializes the block before patching.
            raise RuntimeError(
                "EPLfm2MoELayer: original_layer.expert_bias is still on meta — the loader never "
                "materialized it. Fix the load path rather than routing on a zeroed bias."
            )
        self.register_buffer("expert_bias", eb.clone())

    def _native_slot_absence_is_legal(self) -> bool:
        """``use_expert_bias: false`` registers no buffer, which is a configuration and not a rename."""
        return not self.use_expert_bias

    def _materialize_native_balancing_slot(self) -> bool:
        """Create the architecture's own ``expert_bias`` on a ``use_expert_bias: false`` checkpoint. Zero-initialized,
        a no-op at creation; the strategy layer mirrors the flag into ``model.config``."""
        self.register_buffer(
            "expert_bias", torch.zeros(self.num_experts, dtype=torch.float32, device=self._resident_device())
        )
        self.use_expert_bias = True
        return True

    def _router_logits(self, flat: torch.Tensor) -> torch.Tensor:
        """``Lfm2MoeTopKRouter.forward`` needs the block's ``expert_bias`` and runs its own top-k, while selection must
        stay with the EP layer (balancing bias and routing replay recompute it)."""
        return self._linear_router_logits(flat)

    def route_tokens_to_experts(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        routing_weights = router_logits.sigmoid()

        # Both biases perturb selection only; the gate weights gathered below stay unbiased.
        selected_experts = self._apply_routing_replay(
            torch.topk(self._selection_scores(routing_weights), k=self.top_k, dim=-1)[1]
        )
        topk_weights = routing_weights.gather(1, selected_experts)

        if self.norm_topk_prob:
            topk_weights = self._renormalize_topk_weights(topk_weights)

        topk_weights = topk_weights * self.routed_scaling_factor

        return selected_experts, topk_weights
