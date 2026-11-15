"""Inkling 2's router at the run dtype: the upstream router matmuls fp32 hidden states against a weight transformers
pins to fp32, and the toolkit trains that weight at the run dtype (FSDP2 takes one dtype per shard group), so the
patched forward upcasts it at use and routes in fp32 as upstream does."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import transformers.models.inkling2.modeling_inkling2 as inkling2_modeling

from halo.models.upstream.upstream_guard import verify_upstream_parameters

# ``Inkling2TopKRouter``'s renormalization floor.
_TOPK_WEIGHT_NORM_EPS = 1e-20
# Kept for the parity test: this module is the only one that replaces it.
UPSTREAM_ROUTER_FORWARD = inkling2_modeling.Inkling2TopKRouter.forward


def _router_forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = hidden_states.float().reshape(-1, self.hidden_dim)
    logits = F.linear(flat, self.weight.float())
    scores = self.score_fn(logits)
    indices = torch.topk(scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + _TOPK_WEIGHT_NORM_EPS)
    return logits, weights * self.routed_scaling_factor, indices


def apply_inkling2_patches() -> None:
    router = inkling2_modeling.Inkling2TopKRouter
    if router.forward is _router_forward:
        return
    verify_upstream_parameters(router.forward, ("self", "hidden_states"), "_router_forward")
    router.forward = _router_forward
