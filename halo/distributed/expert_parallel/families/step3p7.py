"""EP layer for Step-3.7 MoE (fused experts, per-layer post-activation clamp, sigmoid routing)."""

from __future__ import annotations

import math

import torch.nn as nn

from halo.distributed.expert_parallel.moe_layer.shared_experts_layer import EPGroupLimitedMoELayerBase
from halo.kernels.fused_glu import resolve_clamped_glu_mul
from halo.models.moe.family_specs import STEP3P7


class EPStep3p7MoELayer(EPGroupLimitedMoELayerBase):
    """EP layer for ``Step3p7SparseMoeBlock`` (Step-3.7-Flash). A per-layer post-activation SwiGLU clamp (most layers
    carry none), sigmoid selection-only ``e_score_correction_bias`` routing as plain top-k, always-renormalized weights
    with the routed scaling folded in, and one always-on shared expert carrying its own clamp."""

    HF_MODULE_NAMES = ("Step3p7SparseMoeBlock",)
    FAMILY = STEP3P7

    # ``Step3p7TopKRouter`` renormalizes unconditionally with no floor, and nothing declares ``norm_topk_prob`` or a
    # group knob, so the shared body runs as plain top-k. ``routed_scaling_factor``, which upstream applies to the
    # routed-expert sum, resolves into the per-token weights instead: equivalent, since the combine is linear.
    # Required: at 1.0 every routed weight misses the model's own factor.
    _TOPK_WEIGHT_NORM_EPS = 0.0  # the value its router divides with
    _OPTIONAL_ROUTING_KNOBS = ("n_group", "topk_group", "norm_topk_prob")

    _SHARED_EXPERT_REQUIRED = True

    def _adopt_expert_compute(self, original_layer: nn.Module, experts: nn.Module) -> None:
        """This layer's clamp bound, ``inf`` meaning unclamped (Step-3.7-Flash clamps layers 43-44 at 7.0); a clamped
        layer latches its own combine. Read as an attribute, since a substituted default un-clamps those layers."""
        super()._adopt_expert_compute(original_layer, experts)
        self.swiglu_limit = float(experts.limit)
        if not math.isinf(self.swiglu_limit):
            self._fused_glu_mul = resolve_clamped_glu_mul(self.act_fn, self.swiglu_limit, clamp_activated=True)

    def _init_summary_extras(self, original_layer: nn.Module) -> tuple[str, ...]:
        return (*super()._init_summary_extras(original_layer), f"swiglu_limit={self.swiglu_limit}")
