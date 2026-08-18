"""EP wrapper for Bailing MoE (individual experts fused at init + shared experts)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.distributed.expert_parallel.base_layer import EPSeparateGluMoELayerBase


class EPBailingMoELayer(EPSeparateGluMoELayerBase):
    """EP wrapper for Bailing MoE (BailingMoeV2/V3SparseMoeBlock) using DeepEP.

    Experts are a per-expert ``nn.ModuleList`` (NOT pre-fused 3D tensors); fused into 3D here at init.
    The gate returns ``(topk_idx, topk_weight, logits)`` with all routing (sigmoid, group-limited
    top-k, expert bias, normalization, scaling) internal. ``shared_experts`` runs on all tokens and is
    not distributed. V3 (Ling 3.0) shares V2's expert layout and gate arithmetic exactly; its block
    returns ``(hidden, router_logits)`` where V2 returns a tensor, which its decoder layer already
    handles by testing the return for a tuple.

    Bias-update balancing rides the gate's NATIVE mechanism: the hub gate adds its persistent
    ``expert_bias`` buffer to the sigmoid scores for selection only (``topk_method: noaux_tc``), so
    instead of the base's transient side-buffer this wrapper hands that buffer itself to the
    balancing callback. Sign-updates therefore steer routing through the exact arithmetic the model
    was pretrained with, and — because the buffer is part of the checkpoint — a gathered save exports
    the final bias, so a served checkpoint routes as training did.
    """

    HF_MODULE_NAMES = ("BailingMoeV2SparseMoeBlock", "BailingMoeV3SparseMoeBlock")
    # Ling-mini-2.0 / Ring-mini-linear-2.0 / Ling-3.0 (hybrid KDA+MLA attention, same MoE block)
    HF_MODEL_TYPES = ("bailing_moe", "bailing_moe_linear", "bailing_hybrid")

    # Only Ling 2.0 is servable: vLLM 0.26.0 registers ``BailingMoeV2ForCausalLM`` alone — Ling 3.0's
    # ``bailing_hybrid`` has no model class in vLLM or SGLang, and Ring's checkpoints declare
    # ``BailingMoeLinearV2ForCausalLM`` where both engines register ``BailingMoeV2_5ForCausalLM``.
    # RL weight sync for those two would stream into an engine that cannot even load the model.
    _WEIGHT_SYNC_UNSUPPORTED_MODEL_TYPES = ("bailing_hybrid", "bailing_moe_linear")

    _supports_bias_balancing = True
    # Bias-update balancing adopts the hub gate's persistent ``expert_bias`` buffer (the model's own
    # ``noaux_tc`` mechanism — the wrapper could not add a side-buffer itself, routing lives inside the
    # hub gate), so the final bias exports with every checkpoint. Machinery lives in the base.
    _NATIVE_BALANCING_BIAS_ATTR = "gate.expert_bias"
    # Selection is the gate's alone (forward never consults ``_balancing_bias``), so the base's
    # transient fallback would be a buffer nothing reads — refuse it instead.
    _supports_transient_balancing_bias = False
    # The wrapper CALLS the hub gate but returns a bare tensor where the hub block returns
    # ``(hidden, router_logits)``, so the decoder's tuple test never collects router logits and
    # ``outputs.router_logits`` stays empty. (No Bailing modeling — V2 or V3 — computes an aux loss
    # anyway: the family is aux-loss-free by design, balanced through ``expert_bias``.)
    _ep_severs_aux_loss = True

    # Hub checkpoint / vLLM loader layout: ``experts.{i}.{gate,up,down}_proj.weight`` — the same
    # ``nn.ModuleList`` of MLPs the wrapper fuses at init.
    _HUB_PER_EXPERT_KEYS = ("gate_proj", "up_proj", "down_proj")

    _NUM_EXPERTS_ATTR_PATHS = ("experts", "gate.num_experts")

    _SHARED_EXPERT_ATTRS = ("shared_experts",)

    def _init_expert_compute(self, original_layer: nn.Module, experts: nn.ModuleList) -> None:
        """Each ``BailingMoeV2MLP`` resolves ACT2FN[config.hidden_act]; read it off the first expert
        this rank fuses — the container is a plain ``nn.ModuleList``, which owns no activation."""
        self._resolve_activation(experts[self.expert_start], original_layer=original_layer)

    def _init_expert_params(self, experts: nn.ModuleList, weights_already_sharded: bool = False):
        """Stack per-expert weights (nn.Linear convention) into 3D matmul-convention tensors.

        The lazy loader materializes only this rank's experts at their GLOBAL indices (remote stay on
        meta), so read the global slice ``[expert_start, expert_end)`` in both cases —
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
        ``routed_scaling_factor`` (routing replay re-derives weights from live logits). The gate's
        ``expert_bias`` perturbs selection only, so it does not appear here."""
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
