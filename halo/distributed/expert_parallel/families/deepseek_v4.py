"""EP layer for DeepSeek-V4 MoE (fused experts, clamped SwiGLU, hash + top-k routing, shared expert)."""

from __future__ import annotations

import torch
import torch.nn as nn

from halo.distributed.expert_parallel.moe_layer.shared_experts_layer import EPSharedExpertsMoELayerBase
from halo.kernels.fused_glu import resolve_clamped_glu_mul
from halo.models.moe.family_specs import DEEPSEEK_V4


class EPDeepseekV4MoELayer(EPSharedExpertsMoELayerBase):
    """EP layer for ``DeepseekV4SparseMoeBlock``. Two router variants adopted whole as ``gate``, so
    ``e_score_correction_bias`` / ``tid2eid`` stay checkpointable: top-k layers select on the biased scores and gate on
    the unbiased ones; hash layers select via the frozen ``tid2eid[input_ids]`` lookup."""

    HF_MODULE_NAMES = ("DeepseekV4SparseMoeBlock",)
    FAMILY = DEEPSEEK_V4

    SUPPORTS_BIAS_BALANCING = True
    # Routing is re-derived from ``gate.weight``, so ``outputs.router_logits`` stays empty under EP.
    SEVERS_AUX_LOSS = True

    _NUM_EXPERTS_ATTR_PATHS = ("experts.num_experts", "gate.num_experts")

    def _adopt_routing(self, original_layer: nn.Module) -> None:
        super()._adopt_routing(original_layer)
        self.is_hash = bool(getattr(original_layer, "is_hash", False))
        self.routed_scaling_factor = getattr(self.gate, "routed_scaling_factor", 1.0)
        self.score_fn = self.gate.score_fn

        if self.is_hash and self.gate.tid2eid.numel() > 0 and not self.gate.tid2eid.is_meta:
            # DeepEP dispatch asserts each token's top-k expert ids are distinct (device-side
            # `ptx::deduplicate`), so a duplicate row must fail here, not crash the CUDA context.
            sorted_rows = self.gate.tid2eid.sort(dim=-1).values
            if self.top_k > 1 and bool((sorted_rows[:, 1:] == sorted_rows[:, :-1]).any()):
                raise ValueError(
                    "EPDeepseekV4MoELayer: the hash router's tid2eid table assigns DUPLICATE expert "
                    "ids to at least one token id. DeepEP dispatch requires distinct experts per "
                    "token — fix the tid2eid table (each row must hold top_k distinct expert ids)."
                )

    def _adopt_expert_compute(self, original_layer: nn.Module, experts: nn.Module) -> None:
        """Attribute reads: a substituted default for ``act_fn`` or ``limit`` rescales the clamp on every expert."""
        self.act_fn = experts.act_fn
        self.limit = float(experts.limit)
        self._fused_glu_mul = resolve_clamped_glu_mul(self.act_fn, self.limit, clamp_activated=False)

    def _init_summary_extras(self, original_layer: nn.Module) -> tuple[str, ...]:
        return (f"is_hash={self.is_hash}",)

    def enable_bias_balancing(self) -> bool:
        """Top-k layers only: a balancing bias cannot shift a hash layer's frozen ``tid2eid`` selection."""
        if self.is_hash:
            return False
        return super().enable_bias_balancing()

    def _route_hidden_states(
        self, flat: torch.Tensor, input_ids: torch.Tensor | None = None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select by hash lookup or biased top-k; gate on the unbiased scores, always renormalized."""
        scores = self.score_fn(self._linear_router_logits(flat).float())
        if not self.is_hash:
            indices = self._select_topk_on_scores(scores)
        elif input_ids is None:
            raise RuntimeError(
                "EPDeepseekV4MoELayer: this is a hash_moe layer (frozen tid2eid routing) but the "
                "forward received no input_ids — the caller ran the model from inputs_embeds, "
                "which DeepSeek-V4 hash routing cannot support. Pass input_ids."
            )
        else:
            indices = self.gate.tid2eid[input_ids.reshape(-1)].long()
        # Replay stays in lockstep across EP layers, so hash layers consume their identity slice too.
        indices = self._apply_routing_replay(indices)
        weights = self._renormalize_topk_weights(scores.gather(1, indices)) * self.routed_scaling_factor
        self._record_expert_load(indices)
        return indices, weights
