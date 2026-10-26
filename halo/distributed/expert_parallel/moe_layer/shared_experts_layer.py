"""EP MoE layer bases for the standard routed + shared-expert forward and DeepSeek-V3 group-limited routing."""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn

from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase


class EPSharedExpertsMoELayerBase(EPMoELayerBase):
    """EP MoE layer whose ``forward`` is the routed template: route the flattened tokens, then the routed experts
    plus the shared expert, which reads the layer input rather than the routed output.

    A subclass supplies ``route_tokens_to_experts`` over the fp32 router logits, or overrides
    :meth:`_route_hidden_states` where its router routes the tokens itself; :meth:`_router_logits` where its
    router cannot be called bare (LFM-2), :meth:`_shared_expert_output` where the shared leg is gated (Qwen3.5).
    """

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        orig_shape = hidden_states.shape
        flat = hidden_states.view(-1, orig_shape[-1])
        topk_indices, topk_weights = self._route_hidden_states(flat, **kwargs)
        shared = getattr(self, self._shared_expert_attr, None) is not None
        shared_fn = partial(self._shared_expert_output, hidden_states) if shared else None
        return self._dispatch_compute_combine_shared(
            flat, topk_indices.long(), topk_weights.float(), hidden_states.dtype, orig_shape, shared_fn
        )

    def _route_hidden_states(self, flat: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """``(topk_indices, topk_weights)`` for the flattened tokens: ``route_tokens_to_experts`` over the fp32
        router logits."""
        return self.route_tokens_to_experts(self._router_logits(flat).float())

    def _router_logits(self, flat: torch.Tensor) -> torch.Tensor:
        """The gate's logits for the flattened tokens; of a ``(logits, weights, indices)`` router output only the
        family-native logits are kept, since routing replay and balancing re-derive the rest."""
        # Disable autocast so the gate matmul runs at the input dtype (upcast when fp32_router).
        with torch.amp.autocast("cuda", enabled=False):
            router_output = self.gate(self._router_input(flat))
        return router_output[0] if isinstance(router_output, tuple) else router_output

    def _shared_expert_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The shared expert on the layer input."""
        return getattr(self, self._shared_expert_attr)(hidden_states)


class EPGroupLimitedMoELayerBase(EPSharedExpertsMoELayerBase):
    """EP MoE layer routing DeepSeek-V3 style: score -> selection-only bias -> group-limited top-k.

    The families sharing it (GLM-4 MoE Lite, Laguna, GLM-5 Next, Mistral4, Step-3.7) differ only in the score
    function, the ``norm_topk_prob`` fallback and the renormalization floor, so a mis-ordered bias add cannot
    land in one of them alone. ``n_group: 1`` is plain top-k.
    """

    # Fallback when neither the block, its router nor its config declares ``norm_topk_prob`` — GLM-4
    # MoE Lite leaves the gate weights un-normalized, the others normalize.
    _NORM_TOPK_PROB_DEFAULT: bool = True

    # Routing knobs (any spelling) this family's block, router and config genuinely do not declare —
    # Laguna and Step-3.7 carry no group limiting and no ``norm_topk_prob``. Every other knob is
    # required: absence means an upstream rename, and the neutral default would then be a wrong live
    # value (routed scaling 1.0 rather than the block's 2.5, under-scaling every routed weight), so the
    # resolution below raises instead of defaulting.
    _OPTIONAL_ROUTING_KNOBS: tuple[str, ...] = ()

    def _adopt_routing(self, original_layer: nn.Module) -> None:
        """Resolve the knobs :meth:`_group_limited_topk` reads off the block, its router, then its config; a knob
        none declares raises unless listed in :attr:`_OPTIONAL_ROUTING_KNOBS`."""
        super()._adopt_routing(original_layer)  # top_k
        sources = (original_layer, self.gate, getattr(original_layer, "config", None))

        def _resolve(names: str | tuple[str, ...], default, *, required: bool = True):
            spellings = (names,) if isinstance(names, str) else names
            for name in spellings:
                for source in sources:
                    if source is not None and hasattr(source, name):
                        return getattr(source, name)
            # Any spelling opts the knob out: a family whose router only carries the alias must be
            # able to declare it, and the two names are one knob.
            if required and not set(spellings) & set(self._OPTIONAL_ROUTING_KNOBS):
                raise AttributeError(
                    f"{type(self).__name__} resolved none of {spellings} off "
                    f"{type(original_layer).__name__}, its router or its config — the family renamed "
                    f"the knob upstream. Defaulting to {default!r} would silently change every routed "
                    f"weight this layer emits. Declare it in _OPTIONAL_ROUTING_KNOBS if this family "
                    f"genuinely has no such knob."
                )
            return default

        # Not required anywhere: its fallback is read off the live expert container rather than a
        # constant, so a rename cannot turn it into a wrong number.
        self.n_routed_experts = _resolve("n_routed_experts", self.num_experts, required=False)
        # The ``*TopkRouter`` modules spell it ``num_group``, configs ``n_group``; missing it disables
        # group-limited routing.
        self.n_group = _resolve(("n_group", "num_group"), 1)
        self.topk_group = _resolve("topk_group", 1)
        self.norm_topk_prob = _resolve("norm_topk_prob", self._NORM_TOPK_PROB_DEFAULT)
        self.routed_scaling_factor = float(_resolve("routed_scaling_factor", 1.0))

    def _routing_scores(self, router_logits: torch.Tensor) -> torch.Tensor:
        """The scores both selection and the gate weights read: sigmoid, the DeepSeek-V3 form (Mistral4 uses
        softmax)."""
        return router_logits.sigmoid()

    def route_tokens_to_experts(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(topk_indices, topk_weights)`` — selection on the biased scores, weights on the unbiased."""
        scores = self._routing_scores(router_logits)
        topk_indices, topk_weights = self._group_limited_topk(self._selection_scores(scores), scores)
        self._record_expert_load(topk_indices)
        return topk_indices, topk_weights

    def _group_limited_topk(
        self, scores_for_select: torch.Tensor, gate_scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(topk_indices, topk_weights)``: groups and indices picked on the biased ``scores_for_select``, weights
        gathered from the unbiased ``gate_scores``."""
        if self.n_group > 1:
            experts_per_group = self.n_routed_experts // self.n_group
            group_scores = scores_for_select.view(-1, self.n_group, experts_per_group).topk(2, dim=-1)[0].sum(dim=-1)
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1).expand(-1, self.n_group, experts_per_group).reshape(-1, self.n_routed_experts)
            )
            # -inf, not 0.0: scores are sigmoid(logits)+bias (positive), so a 0.0 fill leaves a group beatable.
            scores_for_select = scores_for_select.masked_fill(~score_mask.bool(), float("-inf"))

        topk_indices = torch.topk(scores_for_select, k=self.top_k, dim=-1, sorted=False)[1]
        topk_indices = self._apply_routing_replay(topk_indices)
        topk_weights = gate_scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights = self._renormalize_topk_weights(topk_weights)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights
