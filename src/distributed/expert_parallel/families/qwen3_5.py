"""EP wrapper for Qwen3.5 / Qwen3.6 MoE (fused experts + sigmoid-gated shared expert)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.distributed.expert_parallel.moe_layer.shared_experts_layer import EPSharedExpertsMoELayerBase
from src.models.moe.family_specs import QWEN3_5


class EPQwen3_5MoELayer(EPSharedExpertsMoELayerBase):
    """EP wrapper for Qwen3.5 MoE (Qwen3_5MoeSparseMoeBlock) using DeepEP.

    Fused gate_up_proj/down_proj routed experts (GLM4 layout, base fused-GLU helpers).
    ``Qwen3_5MoeTopKRouter`` returns ``(router_logits, router_scores, router_indices)``. The shared
    expert (``Qwen3_5MoeMLP``) and its sigmoid gate (``Linear(hidden, 1)``) are not distributed.
    """

    HF_MODULE_NAMES = ("Qwen3_5MoeSparseMoeBlock",)
    FAMILY = QWEN3_5

    # The only balancing under PP, which rejects ``aux_loss``. ``SEVERS_AUX_LOSS`` stays False: the wrapper
    # calls the real ``Qwen3_5MoeTopKRouter``, so the recorder still fires for the text-only sibling.
    SUPPORTS_BIAS_BALANCING = True

    # ``Qwen3_5MoeTopKRouter`` renormalizes with no floor.
    _TOPK_WEIGHT_NORM_EPS = 0.0

    # ``gate.weight`` is ``[num_experts, hidden]``, so its length is the expert count.
    _NUM_EXPERTS_ATTR_PATHS = ("num_experts", "experts.num_experts", "experts", "gate.weight")

    def _adopt_shared_experts(self, original_layer: nn.Module) -> None:
        super()._adopt_shared_experts(original_layer)
        self.shared_expert_gate = getattr(original_layer, "shared_expert_gate", None)

    def _init_summary_extras(self, original_layer: nn.Module) -> tuple[str, ...]:
        return (f"shared_expert_gate={'yes' if self.shared_expert_gate else 'no'}",)

    def _gate_weights_at(self, router_logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """``Qwen3_5MoeTopKRouter`` weights at ``indices``: the shared softmax gating. This router
        carries no ``norm_topk_prob`` and renormalizes unconditionally."""
        return self._softmax_gate_weights_at(router_logits, indices, normalize=True)

    def _route_hidden_states(self, flat: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """The router's own choice, or with a balancing bias the DeepSeek-V3 biased route over its logits."""
        with torch.amp.autocast("cuda", enabled=False):
            router_output = self.gate(self._router_input(flat))
        if not (isinstance(router_output, tuple) and len(router_output) == 3):
            raise RuntimeError(
                f"Qwen3.5 router returned {type(router_output)} (expected a 3-tuple of "
                f"logits/scores/indices) — transformers changed the router contract; update EPQwen3_5MoELayer."
            )
        router_logits, topk_weights, topk_indices = router_output
        if self._balancing_bias(router_logits) is None:
            return self._replay_router_choice(router_logits, topk_indices, topk_weights)
        # Top-k on the bias-adjusted probabilities, gated by the unbiased renormalized softmax at those indices:
        # Qwen3_5MoeTopKRouter's arithmetic, so a zero bias selects and gates identically up to the router's bf16
        # cast (this path stays fp32). _deepseek_biased_route applies the routing replay itself.
        topk_weights, topk_indices = self._deepseek_biased_route(router_logits)
        self._record_expert_load(topk_indices)
        return topk_indices, topk_weights

    def _shared_expert_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The shared expert, sigmoid-gated where the block carries the gate."""
        shared_output = self.shared_expert(hidden_states)
        if self.shared_expert_gate is not None:
            shared_output = shared_output * torch.sigmoid(self.shared_expert_gate(hidden_states))
        return shared_output

    def replicated_named_params(self) -> list:
        return self._submodule_named_params(
            [
                ("shared_expert", self.shared_expert),
                ("shared_expert_gate", self.shared_expert_gate),
            ]
        )
