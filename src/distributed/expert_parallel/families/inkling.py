"""EP wrapper for Inkling (Thinking Machines): fused experts, joint routed+shared normalisation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from src.models.moe.family_specs import INKLING


class EPInklingMoELayer(EPMoELayerBase):
    """EP wrapper for Inkling (``InklingMoE``) using DeepEP.

    Fused routed experts (GLM4 layout). One router projection emits ``n_routed_experts + n_shared_experts``
    logits, and the routed top-k and the shared experts are normalised jointly, so the shared FFN is scaled by its
    share (``gammas``) — why this layer cannot use :class:`EPSharedExpertsMoELayerBase`, whose shared-expert call
    takes no extra arguments. Selection runs on ``sigmoid(routed_logits) + e_score_correction_bias``; the weights
    stay derived from the unbiased logits.
    """

    HF_MODULE_NAMES = ("InklingMoE",)
    FAMILY = INKLING

    SUPPORTS_BIAS_BALANCING = True
    # The causal-LM loss never reads router_logits, so there is no aux-loss path either way.
    SEVERS_AUX_LOSS = True

    # ``gate.weight`` is ``[n_routed_experts + n_shared_experts, hidden]``, so it over-counts by
    # ``n_shared_experts`` and cannot be used to infer the routed-expert count.
    _NUM_EXPERTS_ATTR_PATHS = ("experts.num_experts",)

    def _adopt_routing(self, original_layer: nn.Module) -> None:
        """Top-k plus the two knobs of the joint routed+shared normalisation, both on the router."""
        super()._adopt_routing(original_layer)
        self.n_shared_experts = self.gate.n_shared_experts
        self.route_scale = self.gate.route_scale
        if self.n_shared_experts < 1:
            # ``_route_joint`` slices the last ``n_shared_experts`` logit columns; at 0 the ``[..., :-0]``
            # slice would drop every routed logit instead.
            raise ValueError(f"EPInklingMoELayer requires n_shared_experts >= 1, got {self.n_shared_experts}")

    def _init_summary_extras(self, original_layer: nn.Module) -> tuple[str, ...]:
        return (f"top_k={self.top_k}, n_shared={self.n_shared_experts}, route_scale={self.route_scale}",)

    def _route_joint(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reproduce ``InklingTopkRouter``: biased-sigmoid selection, joint routed+shared softmax.

        Returns ``(topk_indices, topk_weights, shared_gammas)``.
        """
        routed_logits = router_logits[..., : -self.n_shared_experts]
        shared_logits = router_logits[..., -self.n_shared_experts :]

        topk_indices = self._apply_routing_replay(self._select_topk_on_scores(routed_logits.sigmoid()))

        # Routed and shared logits are normalised together, so a shared expert takes mass away from
        # the routed ones; splitting this into two softmaxes would change the weights.
        topk_logits = torch.cat([routed_logits.gather(-1, topk_indices), shared_logits], dim=-1)
        log_probs = F.logsigmoid(topk_logits)
        weights = torch.exp(log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True))
        weights = weights * self.route_scale * self.gate.global_scale

        self._record_expert_load(topk_indices)
        return (
            topk_indices,
            weights[..., : self.top_k].contiguous(),
            weights[..., -self.n_shared_experts :].contiguous(),
        )

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        orig_shape = hidden_states.shape
        _B, _S, H = orig_shape
        input_dtype = hidden_states.dtype

        residuals = hidden_states  # shared experts read the layer input, not the routed output
        flat = hidden_states.view(-1, H)

        # The router projection is re-run here rather than calling ``self.gate``: the balancing bias
        # enters between the logits and the top-k, and the shared logits are needed for the joint
        # normalisation that produces ``gammas``.
        topk_indices, topk_weights, shared_gammas = self._route_joint(self._linear_router_logits(flat).float())

        # Routing runs in fp32, but the shared FFN is a bf16 bmm: an fp32 gamma promotes its
        # activations and mismatches ``down_proj``. Stock HF's gammas carry the input dtype.
        gammas = shared_gammas.to(input_dtype)
        shared_fn = (
            (lambda: self.shared_experts(residuals, gammas=gammas)) if self.shared_experts is not None else None
        )

        return self._dispatch_compute_combine_shared(
            flat, topk_indices.long(), topk_weights.float(), input_dtype, orig_shape, shared_fn
        )
