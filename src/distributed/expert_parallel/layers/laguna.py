"""EP wrapper for Laguna's MoE block — GLM-4 MoE Lite's shape with a normalizing router."""

from __future__ import annotations

from src.distributed.expert_parallel.layers.glm4 import EPGlm4MoELayer


class EPLagunaMoELayer(EPGlm4MoELayer):
    """EP wrapper for ``LagunaSparseMoeBlock``, sharing GLM-4 MoE Lite's whole EP contract.

    Same sigmoid routing, fused ``gate_up_proj``/``down_proj`` storage, shared expert on every token
    and per-expert hub layout; the families differ only in the top-k weight default.
    ``LagunaTopKRouter.forward`` normalizes the gathered weights UNCONDITIONALLY and
    ``LagunaConfig`` declares no ``norm_topk_prob`` at all (``modular_laguna`` deletes the field it
    would have inherited), so the in-library block carries the knob nowhere and the family default is
    the only thing that can supply it. Hub remote-code revisions do set it on the router, and the
    resolution chain still lets that declaration win.
    """

    HF_MODULE_NAMES = ("LagunaSparseMoeBlock",)
    HF_MODEL_TYPES = ("laguna",)

    _NORM_TOPK_PROB_DEFAULT = True
    # ``LagunaSparseMoeBlock`` declares only ``routed_scaling_factor`` (which stays required): no
    # group limiting and, per the class docstring, no ``norm_topk_prob`` anywhere in the chain.
    _OPTIONAL_ROUTING_KNOBS = ("n_group", "topk_group", "norm_topk_prob")

    # The one engine-side hole in the bias export contract: vLLM 0.26.0's Laguna loader registers
    # this tensor under an internal name and silently drops the exported hub key, so a copy served
    # THERE routes on the pretrained bias (a transformers reload routes as trained). Declared on the
    # class so the balancing enable path can surface it as a run warning; agent-docs/models/laguna.md
    # carries the detail.
    _SERVED_BALANCING_BIAS_DROPPED_BY = "vLLM 0.26.0"

    # Laguna is the one family on the roster whose hub spelling differs from its module spelling; the two
    # pairs mirror transformers' ``laguna`` ``WeightRenaming`` entries (pinned by
    # ``tests/cpu/parallelism/test_laguna_export_key_renames.py``).
    _EXPORT_KEY_RENAMES = (
        ("gate.e_score_correction_bias", "experts.e_score_correction_bias"),
        ("shared_experts.", "shared_expert."),
    )
