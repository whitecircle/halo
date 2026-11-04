"""EP layer for GLM-4 MoE Lite's fused sigmoid-routed MoE (Laguna subclasses it)."""

from __future__ import annotations

import torch
import torch.nn as nn

from halo.distributed.expert_parallel.moe_layer.shared_experts_layer import EPGroupLimitedMoELayerBase
from halo.models.moe.family_specs import GLM4


class EPGlm4MoELayer(EPGroupLimitedMoELayerBase):
    """EP layer for GLM-4 MoE Lite's fused sigmoid-routed MoE; Laguna subclasses it."""

    HF_MODULE_NAMES = ("Glm4MoeLiteMoE",)
    FAMILY = GLM4

    # Remote-code revisions declare no group limiting, so those two fall through to the neutral 1. ``norm_topk_prob``
    # and ``routed_scaling_factor`` (1.8 in-library) stay required: defaulting either rescales every routed weight.
    _OPTIONAL_ROUTING_KNOBS = ("n_group", "topk_group")

    def _adopt_routing(self, original_layer: nn.Module) -> None:
        """Also materialize the correction bias a lazy load can leave on meta: Laguna's pinned remote-code revision
        omits it from its checkpoint, so it would be read from meta storage on every selection."""
        correction_bias = getattr(self.gate, "e_score_correction_bias", None)
        if correction_bias is not None and correction_bias.is_meta:
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.zeros(correction_bias.shape, dtype=correction_bias.dtype),
                requires_grad=correction_bias.requires_grad,
            )
        super()._adopt_routing(original_layer)

    def _native_slot_absence_is_legal(self) -> bool:
        """True only for a remote-code router: those revisions ship no correction bias, while the in-library routers
        register it unconditionally, so an absent slot there is a rename."""
        return not type(self._live_router_module()).__module__.startswith("transformers.models.")
