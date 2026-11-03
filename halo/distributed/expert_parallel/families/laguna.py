"""EP layer for Laguna's MoE block: GLM-4 MoE Lite's shape with a normalizing router."""

from __future__ import annotations

from halo.distributed.expert_parallel.families.glm4 import EPGlm4MoELayer
from halo.models.moe.family_specs import LAGUNA


class EPLagunaMoELayer(EPGlm4MoELayer):
    """Shares GLM-4 MoE Lite's whole EP contract but for the top-k weight default: ``LagunaTopKRouter.forward``
    normalizes unconditionally and ``LagunaConfig`` declares no ``norm_topk_prob``, so for the in-library block the
    family default is the only source. Hub remote-code revisions set it on the router, and that wins."""

    HF_MODULE_NAMES = ("LagunaSparseMoeBlock",)
    FAMILY = LAGUNA

    _NORM_TOPK_PROB_DEFAULT = True
    # ``LagunaSparseMoeBlock`` declares only ``routed_scaling_factor``, which stays required.
    _OPTIONAL_ROUTING_KNOBS = ("n_group", "topk_group", "norm_topk_prob")
