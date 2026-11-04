"""EP layer for Laguna's MoE block: GLM-4 MoE Lite's shape with a normalizing router."""

from __future__ import annotations

from halo.distributed.expert_parallel.families.glm4 import EPGlm4MoELayer
from halo.models.moe.family_specs import LAGUNA


class EPLagunaMoELayer(EPGlm4MoELayer):
    """Shares GLM-4 MoE Lite's whole EP contract but for the routing knobs it may omit: ``LagunaTopKRouter.forward``
    normalizes unconditionally and ``LagunaConfig`` declares no ``norm_topk_prob``, so the in-library block resolves to
    the normalizing default. Hub remote-code revisions set it on the router, and that wins."""

    HF_MODULE_NAMES = ("LagunaSparseMoeBlock",)
    FAMILY = LAGUNA

    # ``LagunaSparseMoeBlock`` declares only ``routed_scaling_factor``, which stays required.
    _OPTIONAL_ROUTING_KNOBS = ("n_group", "topk_group", "norm_topk_prob")

    # ``LagunaTopKRouter`` and both pinned remote revisions renormalize with no floor.
    _TOPK_WEIGHT_NORM_EPS = 0.0
