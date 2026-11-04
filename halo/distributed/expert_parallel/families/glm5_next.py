"""EP layer for GLM-5 Next MoE (fused experts, clamped SwiGLU, sigmoid noaux-tc routing)."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from halo.distributed.expert_parallel.moe_layer.shared_experts_layer import EPGroupLimitedMoELayerBase
from halo.kernels.fused_glu import resolve_clamped_glu_mul
from halo.models.moe.family_specs import GLM5_NEXT


class EPGlm5NextMoELayer(EPGroupLimitedMoELayerBase):
    """EP layer for ``Glm5NextTextMoE`` (GLM-5.3-Flash). Clamped SwiGLU experts, DeepSeek-V3-style sigmoid routing
    whose ``e_score_correction_bias`` perturbs selection only, one always-on shared expert with its own clamp."""

    HF_MODULE_NAMES = ("Glm5NextTextMoE",)
    FAMILY = GLM5_NEXT

    # ``SEVERS_AUX_LOSS`` stays False: ``_router_logits`` calls the real ``Glm5NextTextTopkRouter``, so
    # ``outputs.router_logits`` still populates and the aux loss keeps working under EP.

    def _adopt_expert_compute(self, original_layer: nn.Module, experts: nn.Module) -> None:
        """``Glm5NextTextExperts._apply_gate`` clamps then SiLUs with no act_fn and no config branch, so the latched
        combine is unconditional; the clamp bound is read as an attribute so a rename raises."""
        self.swiglu_limit = float(experts.swiglu_limit)
        self._fused_glu_mul = resolve_clamped_glu_mul(F.silu, self.swiglu_limit, clamp_activated=False)

    def _init_summary_extras(self, original_layer: nn.Module) -> tuple[str, ...]:
        return (*super()._init_summary_extras(original_layer), f"swiglu_limit={self.swiglu_limit}")
