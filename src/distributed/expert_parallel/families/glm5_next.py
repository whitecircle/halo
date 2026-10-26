"""EP wrapper for GLM-5 Next MoE (fused experts, clamped SwiGLU, sigmoid noaux-tc routing)."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from src.distributed.expert_parallel.moe_layer.shared_experts_layer import EPGroupLimitedMoELayerBase
from src.kernels.fused_glu import resolve_clamped_glu_mul
from src.models.moe.family_specs import GLM5_NEXT


class EPGlm5NextMoELayer(EPGroupLimitedMoELayerBase):
    """EP wrapper for GLM-5 Next MoE (``Glm5NextTextMoE``, GLM-5.3-Flash) using DeepEP.

    Fused routed experts with the family's clamped SwiGLU, DeepSeek-V3-style sigmoid routing whose
    ``e_score_correction_bias`` perturbs selection only (the wrapper reuses the router's logits and re-derives the
    rest), and one always-on shared expert carrying its own clamp.
    """

    HF_MODULE_NAMES = ("Glm5NextTextMoE",)
    FAMILY = GLM5_NEXT

    # ``SEVERS_AUX_LOSS`` stays False: ``_router_logits`` calls the real ``Glm5NextTextTopkRouter``, so
    # ``outputs.router_logits`` still populates and the aux loss keeps working under EP.
    SUPPORTS_BIAS_BALANCING = True

    # ``gate.weight`` is ``[num_experts, hidden]``; the config spells the count
    # ``num_local_experts`` (attribute-mapped onto its stored ``n_routed_experts``).
    _NUM_EXPERTS_ATTR_PATHS = ("gate.num_experts", "config.num_local_experts")

    # The block always builds its shared expert, so an absent one is an upstream rename rather than a
    # configuration, and adopting nothing would drop it from every output.
    _SHARED_EXPERT_REQUIRED = True

    def _adopt_expert_compute(self, original_layer: nn.Module, experts: nn.Module) -> None:
        """``Glm5NextTextExperts._apply_gate`` implements clamp-then-SiLU with no act_fn on the module
        and no config branch, so the combine latched into the base seam is unconditional; only the
        clamp bound comes off the live experts module, read as an attribute so a rename raises."""
        self.swiglu_limit = float(experts.swiglu_limit)
        self._fused_glu_mul = resolve_clamped_glu_mul(F.silu, self.swiglu_limit, clamp_activated=False)

    def _init_summary_extras(self, original_layer: nn.Module) -> tuple[str, ...]:
        return (f"swiglu_limit={self.swiglu_limit}",)
