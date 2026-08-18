"""EP wrapper for Inkling (Thinking Machines): fused experts, joint routed+shared normalisation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.distributed.expert_parallel.base_layer import EPMoELayerBase


class EPInklingMoELayer(EPMoELayerBase):
    """EP wrapper for Inkling (``InklingMoE``) using DeepEP.

    Fused ``gate_up_proj``/``down_proj`` routed experts (GLM4 layout, base fused-GLU helpers).

    ``InklingTopkRouter`` differs from every other family here: one projection emits
    ``n_routed_experts + n_shared_experts`` logits, and the routed top-k and the shared experts are
    normalised **jointly** (``logsumexp`` over ``top_k + n_shared`` logits) so the shared experts
    compete for probability mass — the "shared-expert sink". The shared FFN is then scaled by its
    share (``gammas``), which is why it cannot use :class:`EPSharedExpertsMoELayerBase`, whose
    shared-expert call takes no extra arguments.

    Selection runs on ``sigmoid(routed_logits) + e_score_correction_bias`` — a learned routing bias
    the family ships — so ``moe_balancing: bias_update`` adds its balancing bias on top of it, while
    the returned weights stay derived from the unbiased logits.
    """

    HF_MODULE_NAMES = ("InklingMoE",)
    # The composite (multimodal) config spells its model_type "inkling_mm_model" — there is no
    # bare "inkling" spelling anywhere in transformers (verified through 5.16) or on the hub checkpoint.
    HF_MODEL_TYPES = ("inkling_mm_model", "inkling_text")

    _supports_bias_balancing = True
    # ``_route`` adds this exported native slot to the selection score itself (selection-only, like
    # upstream), so bias-update adoption rides it and the trained bias ships with every checkpoint.
    _NATIVE_BALANCING_BIAS_ATTR = "gate.e_score_correction_bias"
    # The causal-LM loss never reads router_logits — there is no aux-loss path to sever or keep, so
    # ``moe_balancing: auto`` must resolve to bias_update.
    _ep_severs_aux_loss = True

    # The hub checkpoint keeps Thinking Machines' original namespace (``model.llm.*``,
    # ``wq_du``/``wk_dv``, interleaved ``w13_weight``); the lazy loaders translate it through the
    # transformers conversion entry declared here (renames + de-interleave, ``hub_conversion.py``).
    # The RL weight sync stays refused: it sends live module-tree names, which a server loading hub
    # names would silently skip.
    _HUB_CONVERSION_KEYS = ("inkling_mm_model",)
    _supports_weight_sync = False

    # ``gate.weight`` is ``[n_routed + n_shared, hidden]``, so it over-counts by ``n_shared`` and
    # must NOT be used to infer the routed-expert count.
    _NUM_EXPERTS_ATTR_PATHS = ("experts.num_experts",)

    _SHARED_EXPERT_ATTRS = ("shared_experts",)

    def _init_routing(self, original_layer: nn.Module) -> None:
        """Top-k plus the two knobs of the joint routed+shared normalisation, both on the router."""
        super()._init_routing(original_layer)
        self.n_shared_experts = self.gate.n_shared_experts
        self.route_scale = self.gate.route_scale
        if self.n_shared_experts < 1:
            # ``_route`` slices the last ``n_shared_experts`` logit columns; ``[..., :-0]`` would
            # silently drop every routed logit instead.
            raise ValueError(f"EPInklingMoELayer requires n_shared_experts >= 1, got {self.n_shared_experts}")

    def _init_summary_extras(self, original_layer: nn.Module) -> tuple[str, ...]:
        return (f"top_k={self.top_k}, n_shared={self.n_shared_experts}, route_scale={self.route_scale}",)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        orig_shape = hidden_states.shape
        _B, _S, H = orig_shape
        input_dtype = hidden_states.dtype

        residuals = hidden_states  # shared experts read the layer input, not the routed output
        flat = hidden_states.view(-1, H)

        # The router projection is re-run here rather than calling ``self.gate``: the balancing bias
        # has to enter between the logits and the top-k, and the shared logits are needed for the
        # joint normalisation that produces ``gammas``.
        with torch.amp.autocast("cuda", enabled=False):
            router_logits = F.linear(flat.float() if self._fp32_router_input else flat, self.gate.weight)

        topk_indices, topk_weights, shared_gammas = self._route(router_logits.float())

        # Routing runs in fp32, but the shared FFN is a bf16 bmm: an fp32 gamma promotes its
        # activations and mismatches ``down_proj``. Stock HF's gammas carry the input dtype.
        gammas = shared_gammas.to(input_dtype)
        shared_fn = (
            (lambda: self.shared_experts(residuals, gammas=gammas)) if self.shared_experts is not None else None
        )

        return self._dispatch_compute_combine_shared(
            flat, topk_indices.long(), topk_weights.float(), input_dtype, orig_shape, shared_fn
        )

    def _route(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reproduce ``InklingTopkRouter``: biased-sigmoid selection, joint routed+shared softmax.

        Returns ``(topk_indices, topk_weights, shared_gammas)``.
        """
        routed_logits = router_logits[..., : -self.n_shared_experts]
        shared_logits = router_logits[..., -self.n_shared_experts :]

        scores_for_choice = self._selection_scores(routed_logits.sigmoid())
        topk_indices = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)[1]
        topk_indices = self._maybe_replace_selection(topk_indices)

        # Routed and shared logits are normalised together, so a shared expert can take mass away
        # from the routed ones; splitting this into two softmaxes would change the weights.
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
