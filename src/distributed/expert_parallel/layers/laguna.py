"""EP wrapper for Laguna's MoE block: GLM-4 MoE Lite's shape with a normalizing router."""

from __future__ import annotations

from src.distributed.expert_parallel.layers.glm4 import EPGlm4MoELayer


class EPLagunaMoELayer(EPGlm4MoELayer):
    """EP wrapper for ``LagunaSparseMoeBlock``, sharing GLM-4 MoE Lite's whole EP contract.

    Same sigmoid routing, fused ``gate_up_proj``/``down_proj`` storage, shared expert on every token
    and per-expert hub layout; the families differ only in the top-k weight default.
    ``LagunaTopKRouter.forward`` normalizes the gathered weights unconditionally and ``LagunaConfig``
    declares no ``norm_topk_prob`` (``modular_laguna`` deletes the inherited field), so for the
    in-library block the family default is the only source for the knob. Hub remote-code revisions do
    set it on the router, and the resolution chain lets that declaration win.
    """

    HF_MODULE_NAMES = ("LagunaSparseMoeBlock",)
    HF_MODEL_TYPES = ("laguna",)

    _NORM_TOPK_PROB_DEFAULT = True
    # ``LagunaSparseMoeBlock`` declares only ``routed_scaling_factor`` (which stays required): no
    # group limiting and, per the class docstring, no ``norm_topk_prob`` in the chain.
    _OPTIONAL_ROUTING_KNOBS = ("n_group", "topk_group", "norm_topk_prob")

    # vLLM 0.26.0's Laguna loader registers this tensor under an internal name and drops the exported
    # hub key, so a copy served there routes on the pretrained bias while a transformers reload
    # routes as trained. Declared on the class so the balancing enable path surfaces it as a run
    # warning; agent-docs/models/laguna.md has the detail.
    _SERVED_BALANCING_BIAS_DROPPED_BY = "vLLM 0.26.0"

    # Laguna's hub spelling differs from its module spelling; the two pairs mirror transformers'
    # ``laguna`` ``WeightRenaming`` entries (pinned by
    # ``tests/cpu/parallelism/test_laguna_export_key_renames.py``).
    _EXPORT_KEY_RENAMES = (
        ("gate.e_score_correction_bias", "experts.e_score_correction_bias"),
        ("shared_experts.", "shared_expert."),
    )
