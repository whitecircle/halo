"""EP wrapper for Qwen3 MoE (pre-fused experts)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.distributed.expert_parallel.base_layer import EPSeparateGluMoELayerBase


class EPQwen3MoELayer(EPSeparateGluMoELayerBase):
    """EP wrapper for Qwen3 MoE with pre-fused expert tensors.

    ``Qwen3MoeExperts`` provides fused ``gate_up_proj``/``down_proj``; gate_up is split here and the
    halves stored in matmul convention.
    """

    HF_MODULE_NAMES = ("Qwen3MoeSparseMoeBlock",)
    HF_MODEL_TYPES = ("qwen3_moe",)

    # The wrapper holds the router's logits, so it can bias top-k itself. Without this the family has
    # no balancing under pipeline parallelism, which rejects aux_loss.
    _supports_bias_balancing = True

    _NUM_EXPERTS_ATTR_PATHS = ("num_experts", "experts.num_experts", "experts")

    # Hub checkpoint / vLLM loader layout: ``experts.{i}.{gate,up,down}_proj.weight`` (per expert).
    # transformers fuses them into ``experts.gate_up_proj`` on load and reverts on save.
    _HUB_PER_EXPERT_KEYS = ("gate_proj", "up_proj", "down_proj")

    def _init_expert_params(self, experts: nn.Module, weights_already_sharded: bool = False):
        """Split pre-fused gate_up_proj into separate matmul-convention tensors (ETP shards dim M;
        weights_already_sharded skips dim-0 slicing)."""
        self._require_fused_experts(experts)
        if weights_already_sharded:
            start, end = 0, self.experts_per_rank
        else:
            start, end = self.expert_start, self.expert_end

        local_gate_up = experts.gate_up_proj.data[start:end]  # [E_local, 2*M, H]
        gate, up = local_gate_up.chunk(2, dim=1)  # each [E_local, M, H]
        local_down = experts.down_proj.data[start:end]  # [E_local, H, M]
        self._store_separate_glu_params(gate, up, local_down)

    # No gather_fused_expert_state_dict here: SGLang 0.5.17's qwen3_moe loader maps per-expert names
    # only (the ``*_fused`` variant is gpt_oss-only), so fused keys are dropped without error and a
    # synced step would serve a trained router over launch-weight experts. The absent override keeps
    # ``implements_fused_expert_layout()`` False, so the construction gate rejects SGLang here; add
    # it only against a loader that consumes the fused pair.

    def _gate_weights_at(self, router_logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """``Qwen3MoeTopKRouter`` weights at ``indices``: the shared renormalized-softmax gating,
        whose ``norm_topk_prob`` this family carries as a configurable (default-off) router
        attribute."""
        return self._softmax_gate_weights_at(router_logits, indices)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        B, S, H = hidden_states.shape
        input_dtype = hidden_states.dtype
        flat = hidden_states.view(-1, H)

        with torch.amp.autocast("cuda", enabled=False):
            _logits, weights, experts = self.gate(flat.float() if self._fp32_router_input else flat)

        if self._balancing_bias(_logits) is not None:
            # Balancing shifts selection only. Weights come from _gate_weights_at, the same function
            # routing replay uses, so they honour this family's configurable norm_topk_prob; the
            # unconditional renormalization in _deepseek_biased_route would not, and a zero bias must
            # reproduce the unbalanced route exactly. _biased_topk applies the routing replay itself.
            experts = self._biased_topk(_logits)
            weights = self._gate_weights_at(_logits, experts)
            self._record_expert_load(experts)
        elif self._forced_topk_indices is not None:
            experts = self._maybe_replace_selection(experts.long())
            weights = self._gate_weights_at(_logits, experts)

        result = self._dispatch_compute_combine(flat, experts, weights.float(), input_dtype)
        return result.view(B, S, H)
