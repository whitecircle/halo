"""EP layer for GLM-4 MoE Lite's fused sigmoid-routed MoE (Laguna subclasses it)."""

from __future__ import annotations

import torch
import torch.nn as nn

from halo.distributed.expert_parallel.moe_layer.shared_experts_layer import EPGroupLimitedMoELayerBase
from halo.models.moe.family_specs import GLM4


class EPGlm4MoELayer(EPGroupLimitedMoELayerBase):
    """EP layer for GLM-4 MoE Lite's fused sigmoid-routed MoE using DeepEP.

    Fused gate_up_proj/down_proj routed experts, group-based sigmoid routing, plus a shared expert on
    every token (not distributed). Laguna subclasses it.
    """

    HF_MODULE_NAMES = ("Glm4MoeLiteMoE",)
    FAMILY = GLM4

    SUPPORTS_BIAS_BALANCING = True

    # The remote-code revisions this EP layer also serves declare no group limiting, so those two fall
    # through to the neutral 1. ``norm_topk_prob`` and ``routed_scaling_factor`` (1.8 in-library) stay
    # required: defaulting either rescales every routed weight.
    _OPTIONAL_ROUTING_KNOBS = ("n_group", "topk_group")

    # The remote-code revisions spell the container ``routed_experts``, so it is probed first.
    _NUM_EXPERTS_ATTR_PATHS = ("routed_experts.num_experts", "config.num_local_experts")

    def _adopt_routing(self, original_layer: nn.Module) -> None:
        """The shared group-limited knobs, plus the correction bias a lazy load can leave on meta.

        Laguna's pinned remote-code revision omits the correction bias from its checkpoint (the
        in-library format always writes it), so the parameter survives the load unmaterialized and
        would be read from meta storage on every selection."""
        correction_bias = getattr(self.gate, "e_score_correction_bias", None)
        if correction_bias is not None and correction_bias.is_meta:
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.zeros(correction_bias.shape, dtype=correction_bias.dtype),
                requires_grad=correction_bias.requires_grad,
            )
        super()._adopt_routing(original_layer)

    def _init_summary_extras(self, original_layer: nn.Module) -> tuple[str, ...]:
        return (f"hf_layer={type(original_layer).__name__}",)

    def _native_slot_absence_is_legal(self) -> bool:
        """True only for a remote-code router: those revisions ship no correction bias, while the
        in-library ``Glm4MoeLiteTopkRouter`` (and Laguna's) registers it unconditionally, so an
        absent slot there is a rename."""
        router = getattr(self, self._ROUTER_ATTR, None) if self._ROUTER_ATTR else None
        return not type(router).__module__.startswith("transformers.models.")
