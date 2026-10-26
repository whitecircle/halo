"""EP layer for Step-3.7 MoE (fused experts, per-layer post-activation clamp, sigmoid routing)."""

from __future__ import annotations

import math

import torch.nn as nn

from halo.distributed.expert_parallel.moe_layer.shared_experts_layer import EPGroupLimitedMoELayerBase
from halo.kernels.fused_glu import resolve_clamped_glu_mul
from halo.models.moe.family_specs import STEP3P7


class EPStep3p7MoELayer(EPGroupLimitedMoELayerBase):
    """EP layer for Step-3.7 MoE (``Step3p7SparseMoeBlock``, Step-3.7-Flash) using DeepEP.

    Fused routed experts with a per-layer post-activation SwiGLU clamp (most layers carry none), sigmoid
    selection-only ``e_score_correction_bias`` routing as plain top-k, always-renormalized gate weights with the
    block's routed scaling folded in, and one always-on shared expert carrying its own clamp.
    """

    HF_MODULE_NAMES = ("Step3p7SparseMoeBlock",)
    FAMILY = STEP3P7

    # The modeling has no aux-loss machinery, so ``moe_balancing: auto`` resolves to ``bias_update``.
    SUPPORTS_BIAS_BALANCING = True

    # ``Step3p7TopKRouter`` renormalizes the selected weights unconditionally and with no floor;
    # nothing in the block, the router or the config declares ``norm_topk_prob`` or a group knob, so
    # the shared body runs as plain top-k. The block's own ``routed_scaling_factor`` (config
    # ``moe_router_scaling_factor``), which upstream applies to the routed-expert sum, resolves into
    # the per-token weights instead: equivalent, since the combine is linear in the expert outputs.
    # Required rather than defaulted: at 1.0 every routed weight misses the model's own factor.
    _TOPK_WEIGHT_NORM_EPS = 0.0  # a floor here would change every routed weight the router emits
    _OPTIONAL_ROUTING_KNOBS = ("n_group", "topk_group", "norm_topk_prob")

    # ``gate.weight`` is ``[num_experts, hidden]``; the block itself carries no config reference,
    # so the count comes off the router or the experts container.
    _NUM_EXPERTS_ATTR_PATHS = ("gate.num_experts",)

    # The block always builds its shared expert, so an absent one is an upstream rename rather than a
    # configuration, and adopting nothing would drop it from every output.
    _SHARED_EXPERT_REQUIRED = True

    def _adopt_expert_compute(self, original_layer: nn.Module, experts: nn.Module) -> None:
        """The activation (the base latch serves the unclamped layers) plus this layer's own clamp bound, where
        ``inf`` means unclamped (most layers; Step-3.7-Flash clamps layers 43-44 at 7.0). A clamped layer latches
        its own combine. Read as an attribute: a substituted default un-clamps those layers."""
        super()._adopt_expert_compute(original_layer, experts)
        self.swiglu_limit = float(experts.limit)
        if not math.isinf(self.swiglu_limit):
            self._fused_glu_mul = resolve_clamped_glu_mul(self.act_fn, self.swiglu_limit, clamp_activated=True)

    def _init_summary_extras(self, original_layer: nn.Module) -> tuple[str, ...]:
        return (f"swiglu_limit={self.swiglu_limit}",)
