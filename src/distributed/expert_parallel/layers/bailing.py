"""EP wrapper for Bailing MoE (individual experts fused at init + shared experts)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.distributed.expert_parallel.base_layer import EPSeparateGluMoELayerBase


class EPBailingMoELayer(EPSeparateGluMoELayerBase):
    """EP wrapper for Bailing MoE (BailingMoeV2/V3SparseMoeBlock) using DeepEP.

    Experts are a per-expert ``nn.ModuleList`` (not pre-fused 3D tensors), fused into 3D here at
    init. The gate returns ``(topk_idx, topk_weight, logits)`` with all routing (sigmoid,
    group-limited top-k, expert bias, normalization, scaling) internal. ``shared_experts`` runs on
    all tokens and is not distributed. V3 (Ling 3.0) shares V2's expert layout and gate arithmetic;
    its block returns ``(hidden, router_logits)`` where V2 returns a tensor, which its decoder layer
    handles by testing the return for a tuple.

    Bias-update balancing uses the gate's own persistent ``expert_bias`` buffer, added to the sigmoid
    scores for selection only (``topk_method: noaux_tc``), instead of the base's transient
    side-buffer. That buffer is part of the checkpoint, so a gathered save exports the trained bias.
    """

    HF_MODULE_NAMES = ("BailingMoeV2SparseMoeBlock", "BailingMoeV3SparseMoeBlock")
    # Ling-mini-2.0 / Ring-mini-linear-2.0 / Ling-3.0 (hybrid KDA+MLA attention, same MoE block)
    HF_MODEL_TYPES = ("bailing_moe", "bailing_moe_linear", "bailing_hybrid")

    # Only Ling 2.0 is servable: vLLM 0.26.0 registers ``BailingMoeV2ForCausalLM`` alone. Ling 3.0's
    # ``bailing_hybrid`` has no model class in vLLM or SGLang, and Ring's checkpoints declare
    # ``BailingMoeLinearV2ForCausalLM`` where both engines register ``BailingMoeV2_5ForCausalLM``.
    _WEIGHT_SYNC_UNSUPPORTED_MODEL_TYPES = ("bailing_hybrid", "bailing_moe_linear")

    _supports_bias_balancing = True
    _NATIVE_BALANCING_BIAS_ATTR = "gate.expert_bias"
    # Selection happens inside the gate (the forward never consults ``_balancing_bias``), so the
    # base's transient fallback would register a buffer nothing reads.
    _supports_transient_balancing_bias = False
    # The wrapper returns a bare tensor where the hub block returns ``(hidden, router_logits)``, so
    # the decoder's tuple test collects nothing and ``outputs.router_logits`` stays empty. No Bailing
    # modeling computes an aux loss regardless; the family balances through ``expert_bias``.
    _ep_severs_aux_loss = True

    # Hub checkpoint / vLLM loader layout: ``experts.{i}.{gate,up,down}_proj.weight`` (per expert).
    _HUB_PER_EXPERT_KEYS = ("gate_proj", "up_proj", "down_proj")

    _NUM_EXPERTS_ATTR_PATHS = ("experts", "gate.num_experts")

    _SHARED_EXPERT_ATTRS = ("shared_experts",)

    def _init_expert_compute(self, original_layer: nn.Module, experts: nn.ModuleList) -> None:
        """Resolve the activation off the first expert this rank fuses: each ``BailingMoeV2MLP``
        holds ``ACT2FN[config.hidden_act]``, the plain ``nn.ModuleList`` container holds none."""
        self._resolve_activation(experts[self.expert_start], original_layer=original_layer)

    def _init_expert_params(self, experts: nn.ModuleList, weights_already_sharded: bool = False):
        """Stack per-expert weights (nn.Linear convention) into 3D matmul-convention tensors.

        The lazy loader materializes only this rank's experts, at their global indices (remote ones
        stay on meta), so the global slice ``[expert_start, expert_end)`` applies in both cases;
        ``weights_already_sharded`` does not re-index here.
        """
        start, end = self.expert_start, self.expert_end
        local_experts = [experts[i] for i in range(start, end)]

        # [E_local, M, H] for gate/up, [E_local, H, M] for down
        gate_stacked = torch.stack([e.gate_proj.weight.data for e in local_experts])
        up_stacked = torch.stack([e.up_proj.weight.data for e in local_experts])
        down_stacked = torch.stack([e.down_proj.weight.data for e in local_experts])
        self._store_separate_glu_params(gate_stacked, up_stacked, down_stacked)

    def _gate_weights_at(self, router_logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """``BailingMoeV2Gate`` weights at ``indices``: sigmoid scores, gather, top-k renorm (+1e-20),
        then ``routed_scaling_factor``. The gate's ``expert_bias`` perturbs selection only, so it does
        not appear here."""
        scores = torch.sigmoid(router_logits.float()).type_as(router_logits)
        scores = torch.gather(scores, dim=1, index=indices).type_as(router_logits)
        weights = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if self.top_k > 1 else scores
        return weights * float(self.gate.routed_scaling_factor)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        orig_shape = hidden_states.shape
        B, S, H = orig_shape
        input_dtype = hidden_states.dtype

        residuals = hidden_states  # for shared experts
        flat = hidden_states.view(-1, H)

        with torch.amp.autocast("cuda", enabled=False):
            topk_idx, topk_weight, _logits = self.gate(flat.float() if self._fp32_router_input else flat)

        if self._forced_topk_indices is not None:
            topk_idx = self._maybe_replace_selection(topk_idx.long())
            topk_weight = self._gate_weights_at(_logits, topk_idx)

        self._record_expert_load(topk_idx)

        shared_fn = (lambda: self.shared_experts(residuals)) if self.shared_experts is not None else None
        return self._dispatch_compute_combine_shared(
            flat, topk_idx.long(), topk_weight.float(), input_dtype, orig_shape, shared_fn
        )
