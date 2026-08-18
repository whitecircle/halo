"""EP wrapper for Qwen3.5 / Qwen3.6 MoE (fused experts + sigmoid-gated shared expert)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.distributed.expert_parallel.base_layer import EPMoELayerBase


class EPQwen3_5MoELayer(EPMoELayerBase):
    """EP wrapper for Qwen3.5 MoE (Qwen3_5MoeSparseMoeBlock) using DeepEP.

    Fused gate_up_proj/down_proj routed experts (GLM4 layout, base fused-GLU helpers).
    ``Qwen3_5MoeTopKRouter`` returns ``(router_logits, router_scores, router_indices)``. The shared
    expert (``Qwen3_5MoeMLP``) and its sigmoid gate (``Linear(hidden, 1)``) are not distributed.
    """

    HF_MODULE_NAMES = ("Qwen3_5MoeSparseMoeBlock",)
    # Qwen3.6 ships under these same types (Qwen/Qwen3.6-35B-A3B declares ``qwen3_5_moe`` with a
    # ``qwen3_5_moe_text`` tower), so there is no 3.6 spelling to list. ``*_text`` is the text tower.
    HF_MODEL_TYPES = ("qwen3_5_moe", "qwen3_5_moe_text")

    # The wrapper re-derives selection from the router's logits, so it can bias top-k itself. That is
    # the only balancing available under PP, which rejects ``aux_loss``. ``_ep_severs_aux_loss`` stays
    # False: the wrapper calls the real ``Qwen3_5MoeTopKRouter`` module, so the HF OutputRecorder
    # still fires, and the recorder path is live for the text-only ``ForCausalLM`` sibling.
    _supports_bias_balancing = True

    # Hub layout: ``experts.{i}.{gate,up,down}_proj.weight`` per expert; transformers registers this
    # family against the ``qwen2_moe`` converter, which fuses those into ``experts.gate_up_proj`` on
    # load and reverts on save. The gather writes the fused tensor, so ``unfuse_moe_experts`` repairs
    # a save that bypassed the revert under these names.
    _HUB_PER_EXPERT_KEYS = ("gate_proj", "up_proj", "down_proj")

    # ``gate.weight`` is ``[num_experts, hidden]``, so its length is the expert count.
    _NUM_EXPERTS_ATTR_PATHS = ("num_experts", "experts.num_experts", "experts", "gate.weight")

    # Singular, unlike the other families; the sigmoid gate below is a second replicated module,
    # which is why ``replicated_named_params`` is overridden.
    _SHARED_EXPERT_ATTRS = ("shared_expert",)

    def _init_shared_experts(self, original_layer: nn.Module) -> None:
        super()._init_shared_experts(original_layer)
        self.shared_expert_gate = getattr(original_layer, "shared_expert_gate", None)

    def _init_summary_extras(self, original_layer: nn.Module) -> tuple[str, ...]:
        return (f"shared_expert_gate={'yes' if self.shared_expert_gate else 'no'}",)

    def replicated_named_params(self) -> list:
        return self._submodule_named_params(
            [
                ("shared_expert", self.shared_expert),
                ("shared_expert_gate", self.shared_expert_gate),
            ]
        )

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        orig_shape = hidden_states.shape
        B, S, H = orig_shape
        input_dtype = hidden_states.dtype

        residuals = hidden_states  # for shared experts
        flat = hidden_states.view(-1, H)

        with torch.amp.autocast("cuda", enabled=False):
            router_output = self.gate(flat.float() if self._fp32_router_input else flat)

        if not (isinstance(router_output, tuple) and len(router_output) == 3):
            raise RuntimeError(
                f"Qwen3.5 router returned {type(router_output)} (expected a 3-tuple of "
                f"logits/scores/indices) — transformers changed the router contract; update EPQwen3_5MoELayer."
            )
        _router_logits, topk_weights, topk_indices = router_output

        if self._balancing_bias(_router_logits) is not None:
            # Bias-update balancing re-derives selection from the router's own logits: top-k on the
            # bias-adjusted probabilities, gate on the unbiased renormalized softmax at those
            # indices. That matches Qwen3_5MoeTopKRouter, so a zero bias selects the same experts and
            # gates them identically up to the router's bf16 cast (this path stays fp32).
            # _deepseek_biased_route applies the routing replay itself.
            topk_weights, topk_indices = self._deepseek_biased_route(_router_logits)
            self._record_expert_load(topk_indices)
        elif self._forced_topk_indices is not None:
            topk_indices = self._maybe_replace_selection(topk_indices.long())
            topk_weights = self._gate_weights_at(_router_logits, topk_indices)

        shared_fn = (lambda: self._shared_forward(residuals)) if self.shared_expert is not None else None
        return self._dispatch_compute_combine_shared(
            flat, topk_indices.long(), topk_weights.float(), input_dtype, orig_shape, shared_fn
        )

    def _gate_weights_at(self, router_logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """``Qwen3_5MoeTopKRouter`` weights at ``indices``: the shared renormalized-softmax gating.
        This router carries no ``norm_topk_prob`` attribute and renormalizes unconditionally, which
        is the helper's default."""
        return self._softmax_gate_weights_at(router_logits, indices)

    def _shared_forward(self, residuals: torch.Tensor) -> torch.Tensor:
        """Always-active shared-expert FFN (sigmoid-gated), run on all tokens locally."""
        shared_output = self.shared_expert(residuals)
        if self.shared_expert_gate is not None:
            shared_output = shared_output * torch.sigmoid(self.shared_expert_gate(residuals))
        return shared_output
