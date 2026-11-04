"""EP layer for Gemma-4 routed experts (the router is in the parent decoder layer)."""

from __future__ import annotations

import torch
import torch.nn as nn

from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.models.moe.family_specs import GEMMA4


class EPGemma4MoELayer(EPMoELayerBase):
    """EP layer for ``Gemma4TextExperts``. Gemma4 inlines MoE compute in its decoder layer, so only the experts are
    replaced; the sibling router stays FSDP-managed and bf16 (fp32 would reach DeepEP's bf16-only combine)."""

    HF_MODULE_NAMES = ("Gemma4TextExperts",)
    FAMILY = GEMMA4

    # The router is a sibling module in the decoder layer, so only expert grad-sync hooks register.
    _ROUTER_ATTR = None

    @classmethod
    def _find_experts_container(cls, layer: nn.Module) -> nn.Module:
        """The wrapped module is itself the expert container."""
        return layer

    def _adopt_routing(self, original_layer: nn.Module) -> None:
        """No-op: the sibling ``Gemma4TextRouter`` passes the routing decision in already made."""

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Match ``Gemma4TextExperts.forward``: tokens are already flat ``[B*S, H]``."""
        return self._dispatch_compute_combine(
            hidden_states, top_k_index.long(), top_k_weights.float(), hidden_states.dtype
        )
