"""EP layer for Cohere2 MoE / Command A+ (fused experts, top-k-then-activate router)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from halo.distributed.expert_parallel.moe_layer.shared_experts_layer import EPSharedExpertsMoELayerBase
from halo.models.moe.family_specs import COHERE2_MOE


class EPCohere2MoELayer(EPSharedExpertsMoELayerBase):
    """EP layer for Cohere2 MoE (Cohere2MoeSparseMoeBlock) using DeepEP.

    Fused gate_up_proj/down_proj routed experts (base fused-GLU helpers). ``Cohere2MoeTopKRouter``
    selects top-k on the raw logits and activates only the selected scores: softmax over the k
    scores, or sigmoid with optional top-k renorm. The shared expert is one fused MLP on every
    token, combined with the routed output by ``sum`` or ``average`` (``(routed + shared) / 2``).
    Layers whose ``mlp_layer_types`` entry is ``dense`` instantiate ``Cohere2MoeMLP`` instead of the
    sparse block, so they are never wrapped.
    """

    HF_MODULE_NAMES = ("Cohere2MoeSparseMoeBlock",)
    FAMILY = COHERE2_MOE

    # No exported bias slot: ``moe_balancing: auto`` resolves to ``none``, ``bias_update_transient`` opts in.
    SUPPORTS_BIAS_BALANCING = True

    # The sigmoid top-k renorm adds no floor.
    _TOPK_WEIGHT_NORM_EPS = 0.0

    # ``gate.weight`` is ``[num_experts, hidden]``, so its length is the expert count.
    _NUM_EXPERTS_ATTR_PATHS = ("num_experts", "experts.num_experts", "experts", "gate.weight")

    def _adopt_routing(self, original_layer: nn.Module) -> None:
        """Top-k plus the router's activation choice: ``Cohere2MoeTopKRouter`` activates only the
        selected raw scores, so the activation and the renorm flag determine every gate weight."""
        super()._adopt_routing(original_layer)
        self.expert_selection_fn = getattr(self.gate, "expert_selection_fn", "softmax")
        if self.expert_selection_fn not in ("softmax", "sigmoid"):
            raise ValueError(
                f"{type(self).__name__}: expert_selection_fn must be 'softmax' or 'sigmoid', "
                f"got {self.expert_selection_fn!r} — transformers changed the router contract."
            )
        self.norm_topk_prob = bool(getattr(self.gate, "norm_topk_prob", True))

    def _adopt_shared_experts(self, original_layer: nn.Module) -> None:
        """The shared expert plus the block's combination strategy, which scales the whole output:
        defaulting ``average`` to ``sum`` would emit 2x every token."""
        super()._adopt_shared_experts(original_layer)
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
        return self._renormalize_topk_weights(weights) if self.norm_topk_prob else weights

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        output = super().forward(hidden_states, **kwargs)
        return output if self._output_scale == 1.0 else output * self._output_scale
