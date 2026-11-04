"""EP layer for Nemotron-H MoE: gateless experts on a latent, which is what travels between the ranks."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from halo.distributed.expert_parallel.moe_layer.shared_experts_layer import EPGroupLimitedMoELayerBase
from halo.models.moe.family_specs import NEMOTRON_H


class EPNemotronHMoELayer(EPGroupLimitedMoELayerBase):
    """EP layer for ``NemotronHMoE``. Routing reads the hidden state, while ``fc1_latent_proj`` and
    ``fc2_latent_proj`` stay replicated around the dispatch, so the wire and the experts run at the latent's width. The
    expert-TP sums sit inside that span: the FFN stays linear over its intermediate axis, whose relu2 is elementwise."""

    HF_MODULE_NAMES = ("NemotronHMoE",)
    FAMILY = NEMOTRON_H

    # The router's module never runs here, and the modeling has no aux loss to lose: the bias is the one balancing.
    SEVERS_AUX_LOSS = True

    _SHARED_EXPERT_REQUIRED = True

    @classmethod
    def _detect_hidden_dim(cls, router: nn.Module | None, experts: nn.Module) -> int:
        """The latent; the router's input width is the hidden state's, which never reaches the wire."""
        return experts.up_proj.shape[2]

    def _adopt_expert_compute(self, original_layer: nn.Module, experts: nn.Module) -> None:
        super()._adopt_expert_compute(original_layer, experts)
        self.fc1_latent_proj = original_layer.fc1_latent_proj
        self.fc2_latent_proj = original_layer.fc2_latent_proj

    def _adopt_expert_params(self, experts: nn.Module, weights_already_sharded: bool = False) -> None:
        start, end = self._local_expert_range(weights_already_sharded)
        up = experts.up_proj.data[start:end]  # [E, M, L]
        down = experts.down_proj.data[start:end]  # [E, L, M]
        if self.expert_tp_size > 1:
            up, down = self._etp_narrow(up, 1), self._etp_narrow(down, 2)
        self.up_proj = nn.Parameter(up.transpose(1, 2).contiguous())
        self.down_proj = nn.Parameter(down.transpose(1, 2).contiguous())

    def _expert_activation(self, tokens: torch.Tensor) -> torch.Tensor:
        """Autocast off: its relu2 squares, which autocast returns in fp32, and ``grouped_mm`` casts nothing back."""
        with torch.amp.autocast("cuda", enabled=False):
            return self.act_fn(tokens)

    def _glu_ffn(self, x: torch.Tensor, proj: Callable[[torch.Tensor, str], torch.Tensor]) -> torch.Tensor:
        return proj(self._expert_activation(proj(x, "up_proj")), "down_proj")

    def _glu_combine_name(self) -> str:
        return "none (gateless)"

    def _warm_expert_activation(self, gate_up: torch.Tensor) -> torch.Tensor:
        return self._expert_activation(gate_up)

    def _router_logits(self, flat: torch.Tensor) -> torch.Tensor:
        """In fp32 from the router's bf16 weight, as ``NemotronHTopkRouter`` computes them."""
        with torch.amp.autocast("cuda", enabled=False):
            return F.linear(flat.float(), self.gate.weight.float())

    def _dispatch_compute_combine(
        self, flat: torch.Tensor, experts: torch.Tensor, weights: torch.Tensor, input_dtype: torch.dtype
    ) -> torch.Tensor:
        routed = super()._dispatch_compute_combine(self.fc1_latent_proj(flat), experts, weights, input_dtype)
        return self.fc2_latent_proj(routed)

    def replicated_named_params(self) -> list:
        return super().replicated_named_params() + self._submodule_named_params(
            [("fc1_latent_proj", self.fc1_latent_proj), ("fc2_latent_proj", self.fc2_latent_proj)]
        )
