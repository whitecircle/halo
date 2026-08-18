"""EP wrapper for LFM-2 MoE (fused experts with sigmoid routing + expert bias)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.distributed.expert_parallel.base_layer import EPSharedExpertsMoELayerBase
from src.distributed.runtime import current_device


class EPLfm2MoELayer(EPSharedExpertsMoELayerBase):
    """EP wrapper for LFM2 MoE fused experts using DeepEP.

    Fused gate_up_proj/down_proj experts, SwiGLU, sigmoid routing with optional
    ``expert_bias`` correction. No shared experts (the block carries none, so the declared slot
    registers ``None`` and the shared base forward skips that leg), no group routing.
    """

    HF_MODULE_NAMES = ("Lfm2MoeSparseMoeBlock",)
    # Both spellings, like every composite family: ``Lfm2VlConfig.model_type = "lfm2_vl"`` wraps an
    # lfm2_moe text tower, and the model-type-keyed consumers (sharded-EP save, merge_ep_shards,
    # unfuse/quantize) resolve the TOP-LEVEL type — without the wrapper spelling they refuse a
    # checkpoint the toolkit itself trained.
    HF_MODEL_TYPES = ("lfm2_moe", "lfm2_vl")

    _supports_bias_balancing = True
    # ``route_tokens_to_experts`` adds this wrapper-level buffer (re-registered off the hub block, so
    # it exports at the block's own key) to the selection score itself; bias-update adoption rides it
    # and the trained bias ships with every checkpoint. When ``use_expert_bias`` is false the slot is
    # materialized at enable time instead (same sigmoid-score space, zero-initialized — a semantic
    # no-op at creation), and the config flag below flips so serving engines load and apply it.
    _NATIVE_BALANCING_BIAS_ATTR = "expert_bias"
    _NATIVE_BALANCING_CONFIG_FLAG = "use_expert_bias"
    # Hub checkpoint / vLLM loader layout: ``experts.{i}.w{1,3,2}.weight`` — Llama-style names.
    _PER_EXPERT_UNFUSED_KEYS = ("w1", "w3", "w2")  # w1 = gate, w3 = up, w2 = down

    _NUM_EXPERTS_ATTR_PATHS = ("experts.num_experts", "gate.num_experts")

    _SHARED_EXPERT_ATTRS = ("shared_experts",)

    def _init_routing(self, original_layer: nn.Module) -> None:
        """Top-k plus the sigmoid-routing knobs. Read off ``Lfm2MoeTopKRouter`` (the gate), not the
        block, and without defaults: these change the routing math, so a moved attribute must fail
        here rather than silently route with a hand-picked constant."""
        super()._init_routing(original_layer)
        self.norm_topk_prob = self.gate.norm_topk_prob
        self.routed_scaling_factor = self.gate.routed_scaling_factor
        self.use_expert_bias = self.gate.use_expert_bias

        if not self.use_expert_bias:
            return
        if not hasattr(original_layer, "expert_bias"):
            # The router asks for a bias the block does not carry: an upstream MOVE, not a
            # configuration. Flipping the flag off here (as this once did) routes unbiased for the
            # rest of the run with no shape, dtype or key moving.
            raise AttributeError(
                f"EPLfm2MoELayer: the router declares use_expert_bias=True but "
                f"{type(original_layer).__name__} carries no 'expert_bias' — the family moved or "
                f"renamed the buffer upstream. Routing on would silently drop the pretrained "
                f"selection bias."
            )
        eb = original_layer.expert_bias
        if eb.is_meta:
            # A zero substitute would silently change routing — and, under bias-update adoption,
            # become the exported bias. Every load path materializes the block before patching.
            raise RuntimeError(
                "EPLfm2MoELayer: original_layer.expert_bias is still on meta — the loader never "
                "materialized it. Fix the load path rather than routing on a zeroed bias."
            )
        self.register_buffer("expert_bias", eb.clone())

    def _native_slot_absence_is_legal(self) -> bool:
        """``use_expert_bias: false`` registers no buffer at all — a configuration, not a rename.

        Read off the wrapper's own flag: it is what this family's routing and slot materialization
        both key on, and :meth:`_init_routing` already refuses the one state that would make it lie
        (a router asking for a bias its block does not carry). Reaching into the router here instead
        would put a construction-time attribute on the selection path, which the bias-injection
        callers exercise without one.
        """
        return not self.use_expert_bias

    def _materialize_native_balancing_slot(self) -> bool:
        """Create the architecture's own ``expert_bias`` on a ``use_expert_bias: false`` checkpoint.

        LFM-2's slot is config-gated, not structural: the zero-initialized buffer is a semantic no-op
        at creation and lives in the exact sigmoid-score space the transient side-buffer would use, so
        materializing it changes nothing about training — it only makes the trained bias exportable.
        ``use_expert_bias`` flips here for the wrapper's own routing; the strategy layer mirrors it
        into ``model.config`` (``_NATIVE_BALANCING_CONFIG_FLAG``) so engines load and apply the tensor.
        """
        device = next((p.device for p in self.parameters()), None)
        if device is None or device.type == "meta":
            device = current_device()
        self.register_buffer("expert_bias", torch.zeros(self.num_experts, dtype=torch.float32, device=device))
        self.use_expert_bias = True
        return True

    def _router_logits(self, flat: torch.Tensor) -> torch.Tensor:
        """Bare linear over the router's weight — ``Lfm2MoeTopKRouter.forward`` cannot be called
        here: it requires the block's ``expert_bias`` and runs its own top-k, and the selection
        must stay with the wrapper (balancing bias and routing replay recompute it)."""
        with torch.amp.autocast("cuda", enabled=False):
            return nn.functional.linear(flat.float() if self._fp32_router_input else flat, self.gate.weight)

    def route_tokens_to_experts(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """LFM2 routing with sigmoid activation and optional expert bias."""
        routing_weights = router_logits.sigmoid()

        # Both biases perturb selection only; the gate weights gathered below stay unbiased. The
        # declared native slot IS ``expert_bias``, registered exactly when ``use_expert_bias`` holds.
        scores_for_routing = self._selection_scores(routing_weights)

        # Identity, not equality: an unbiased selection is the same OBJECT, which is what selects the
        # cheaper single-topk path below (it reads values and indices in one call).
        if scores_for_routing is routing_weights:
            topk_weights, selected_experts = torch.topk(routing_weights, k=self.top_k, dim=-1)
        else:
            _, selected_experts = torch.topk(scores_for_routing, k=self.top_k, dim=-1)
            topk_weights = torch.gather(routing_weights, dim=1, index=selected_experts)

        if self._forced_topk_indices is not None:
            # Replay re-gathers weights from the live unbiased scores, keeping norm/scaling native.
            selected_experts = self._maybe_replace_selection(selected_experts)
            topk_weights = torch.gather(routing_weights, dim=1, index=selected_experts)

        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-6
            topk_weights = topk_weights / denominator

        topk_weights = topk_weights * self.routed_scaling_factor

        self._record_expert_load(selected_experts)
        return selected_experts, topk_weights
