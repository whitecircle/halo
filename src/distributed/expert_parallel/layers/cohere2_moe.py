"""EP wrapper for Cohere2 MoE / Command A+ (fused experts, top-k-then-activate router)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.distributed.expert_parallel.base_layer import EPSharedExpertsMoELayerBase


class EPCohere2MoELayer(EPSharedExpertsMoELayerBase):
    """EP wrapper for Cohere2 MoE (Cohere2MoeSparseMoeBlock) using DeepEP.

    Fused gate_up_proj/down_proj routed experts (base fused-GLU helpers). ``Cohere2MoeTopKRouter``
    selects top-k on the raw logits and activates only the selected scores: softmax over the k
    scores, or sigmoid with optional top-k renorm. The shared expert is one fused MLP on every
    token, combined with the routed output by ``sum`` or ``average`` (``(routed + shared) / 2``).
    Layers whose ``mlp_layer_types`` entry is ``dense`` instantiate ``Cohere2MoeMLP`` instead of the
    sparse block, so they are never wrapped.
    """

    HF_MODULE_NAMES = ("Cohere2MoeSparseMoeBlock",)
    # The text backbone's spelling plus the composite VLM spelling in the Command A+ checkpoint's
    # config.json; checkpoint-keyed tooling (sharded merge, unfuse, lazy-load routing) resolves
    # either config to this class.
    HF_MODEL_TYPES = ("cohere2_moe", "cohere2_vision")

    # The wrapper re-derives selection from the router's logits, so it can bias top-k itself. With no
    # aux-loss wiring and no exported bias slot, ``moe_balancing: auto`` resolves to ``none`` and
    # ``bias_update_transient`` is the opt-in for a trainer-only side-buffer.
    _supports_bias_balancing = True

    # No end-to-end sync has been validated against the pinned vLLM 0.26.0 server, including
    # layerwise-reload skip coverage for its FusedMoE experts.
    _supports_weight_sync = False
    _WEIGHT_SYNC_REFUSAL_REASON = (
        "no end-to-end weight sync has been validated for the Cohere2 MoE family against the "
        "pinned vLLM 0.26.0 server (agent-docs/models/cohere2-moe.md)"
    )

    # The Command A+ checkpoint index spells the vision tower ``model.vision_tower.vision_model.*``
    # while the module tree drops the ``vision_model`` segment (a from_pretrained-only conversion),
    # so the lazy loader would read all 437 tower tensors as absent. Route to ``from_pretrained``.
    _supports_lazy_loading = False

    # Hub layout: ``experts.{i}.{gate,up,down}_proj.weight`` per expert; transformers fuses those
    # into ``experts.gate_up_proj`` on load and reverts on save. The gather writes the fused tensor,
    # so ``unfuse_moe_experts`` repairs a save that bypassed the revert under these names.
    _HUB_PER_EXPERT_KEYS = ("gate_proj", "up_proj", "down_proj")

    # ``gate.weight`` is ``[num_experts, hidden]``, so its length is the expert count.
    _NUM_EXPERTS_ATTR_PATHS = ("num_experts", "experts.num_experts", "experts", "gate.weight")

    _SHARED_EXPERT_ATTRS = ("shared_experts",)

    def _init_routing(self, original_layer: nn.Module) -> None:
        """Top-k plus the router's activation choice: ``Cohere2MoeTopKRouter`` activates only the
        selected raw scores, so the activation and the renorm flag determine every gate weight."""
        super()._init_routing(original_layer)
        self.expert_selection_fn = getattr(self.gate, "expert_selection_fn", "softmax")
        if self.expert_selection_fn not in ("softmax", "sigmoid"):
            raise ValueError(
                f"{type(self).__name__}: expert_selection_fn must be 'softmax' or 'sigmoid', "
                f"got {self.expert_selection_fn!r} — transformers changed the router contract."
            )
        self.norm_topk_prob = bool(getattr(self.gate, "norm_topk_prob", True))

    def _init_shared_experts(self, original_layer: nn.Module) -> None:
        """The shared expert plus the block's combination strategy, which scales the whole output:
        defaulting ``average`` to ``sum`` would emit 2x every token."""
        super()._init_shared_experts(original_layer)
        strategy = getattr(original_layer, "shared_expert_combination_strategy", "sum")
        if strategy not in ("sum", "average"):
            raise ValueError(
                f"{type(self).__name__}: shared_expert_combination_strategy must be 'sum' or "
                f"'average', got {strategy!r} — transformers changed the block contract."
            )
        self._output_scale = 0.5 if (self.shared_experts is not None and strategy == "average") else 1.0

    def _init_summary_extras(self, original_layer: nn.Module) -> tuple[str, ...]:
        return (
            f"selection={self.expert_selection_fn}",
            f"output_scale={self._output_scale}",
        )

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        output = super().forward(hidden_states, **kwargs)
        return output if self._output_scale == 1.0 else output * self._output_scale

    def route_tokens_to_experts(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k on the raw logits (bias-perturbable), gate weights from the unbiased selected scores.

        Without balancing, ``_biased_topk`` selects on the raw logits as the HF router does (and
        applies routing replay); a transient bias perturbs selection only, leaving the weights below
        as the family's own gating computes them.
        """
        topk_indices = self._biased_topk(router_logits)
        topk_weights = self._gate_weights_at(router_logits, topk_indices)
        self._record_expert_load(topk_indices)
        return topk_indices, topk_weights

    def _gate_weights_at(self, router_logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """``Cohere2MoeTopKRouter`` weights at ``indices``: activate only the selected raw scores,
        softmax over the k scores or sigmoid with optional top-k renorm."""
        scores = router_logits.gather(-1, indices)
        if self.expert_selection_fn == "softmax":
            return F.softmax(scores, dim=-1, dtype=torch.float)
        weights = torch.sigmoid(scores)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights
