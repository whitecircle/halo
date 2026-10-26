"""EP layer for Laguna's MoE block: GLM-4 MoE Lite's shape with a normalizing router."""

from __future__ import annotations

from halo.distributed.expert_parallel.families.glm4 import EPGlm4MoELayer
from halo.models.moe.family_specs import LAGUNA


class EPLagunaMoELayer(EPGlm4MoELayer):
    """EP layer for ``LagunaSparseMoeBlock``, sharing GLM-4 MoE Lite's whole EP contract.

    Same sigmoid routing, fused ``gate_up_proj``/``down_proj`` storage, shared expert on every token
    and per-expert hub layout; the families differ only in the top-k weight default.
    ``LagunaTopKRouter.forward`` normalizes the gathered weights unconditionally and ``LagunaConfig``
    declares no ``norm_topk_prob`` (``modular_laguna`` deletes the inherited field), so for the
    in-library block the family default is the only source for the knob. Hub remote-code revisions do
    set it on the router, and the resolution chain lets that declaration win.
    """

    HF_MODULE_NAMES = ("LagunaSparseMoeBlock",)
    FAMILY = LAGUNA

    _NORM_TOPK_PROB_DEFAULT = True
    # ``LagunaSparseMoeBlock`` declares only ``routed_scaling_factor`` (which stays required): no
    # group limiting and, per the class docstring, no ``norm_topk_prob`` in the chain.
    _OPTIONAL_ROUTING_KNOBS = ("n_group", "topk_group", "norm_topk_prob")
