"""EP layer for Qwen3 MoE (pre-fused experts)."""

from __future__ import annotations

import torch
import torch.nn as nn

from halo.distributed.expert_parallel.moe_layer.shared_experts_layer import EPSharedExpertsMoELayerBase
from halo.models.moe.family_specs import QWEN3


class EPQwen3MoELayer(EPSharedExpertsMoELayerBase):
    """EP layer for Qwen3 MoE with pre-fused expert tensors.

    ``Qwen3MoeExperts`` provides fused ``gate_up_proj``/``down_proj``; gate_up is split here and the
    halves stored in matmul convention.
    """

    HF_MODULE_NAMES = ("Qwen3MoeSparseMoeBlock",)
    FAMILY = QWEN3

    # The gate has no slot to export a bias, so this admits only the trainer-only bias_update_transient.
    SUPPORTS_BIAS_BALANCING = True

    # ``Qwen3MoeTopKRouter`` renormalizes with no floor.
    _TOPK_WEIGHT_NORM_EPS = 0.0

    _NUM_EXPERTS_ATTR_PATHS = ("num_experts", "experts.num_experts", "experts")

    def _adopt_expert_params(self, experts: nn.Module, weights_already_sharded: bool = False):
        """Split pre-fused gate_up_proj into separate matmul-convention tensors (ETP shards dim M;
        weights_already_sharded skips dim-0 slicing)."""
        self._require_fused_experts(experts)
        start, end = self._local_expert_range(weights_already_sharded)

        local_gate_up = experts.gate_up_proj.data[start:end]  # [E_local, 2*M, H]
        gate, up = local_gate_up.chunk(2, dim=1)  # each [E_local, M, H]
        local_down = experts.down_proj.data[start:end]  # [E_local, H, M]
        self._store_separate_glu_params(gate, up, local_down)

    def _gate_weights_at(self, router_logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """``Qwen3MoeTopKRouter`` weights at ``indices``: the shared softmax gating, renormalized per the
        router's own configurable (default-off) ``norm_topk_prob``, read live so routing replay matches
        the router the forward just called."""
        return self._softmax_gate_weights_at(router_logits, indices, normalize=self.gate.norm_topk_prob)

    def _route_hidden_states(self, flat: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """The router's own choice, or with a balancing bias the biased top-k gated by :meth:`_gate_weights_at`."""
        with torch.amp.autocast("cuda", enabled=False):
            router_logits, weights, experts = self.gate(self._router_input(flat))
        if self._balancing_bias(router_logits) is None:
            return self._replay_router_choice(router_logits, experts, weights)
        # Balancing shifts selection only. Weights come from _gate_weights_at, the same function routing replay
        # uses, so they honour this family's configurable norm_topk_prob; the unconditional renormalization in
        # _deepseek_biased_route would not, and a zero bias must reproduce the unbalanced route exactly.
        # _biased_topk applies the routing replay itself.
        experts = self._biased_topk(router_logits)
        weights = self._gate_weights_at(router_logits, experts)
        self._record_expert_load(experts)
        return experts, weights
