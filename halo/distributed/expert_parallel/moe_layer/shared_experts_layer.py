"""EP MoE layer bases for the standard routed + shared-expert forward and DeepSeek-V3 group-limited routing."""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn

from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase


class EPSharedExpertsMoELayerBase(EPMoELayerBase):
    """EP MoE layer whose ``forward`` routes the flattened tokens, then adds the shared expert, which reads the layer
    input. A subclass supplies ``route_tokens_to_experts`` over the fp32 router logits, or overrides
    :meth:`_route_hidden_states`, :meth:`_router_logits` (LFM-2) or :meth:`_shared_expert_output` (Qwen3.5)."""

    SUPPORTS_BIAS_BALANCING = True

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
        return self.route_tokens_to_experts(self._router_logits(flat).float())

    def _router_logits(self, flat: torch.Tensor) -> torch.Tensor:
        """Of a ``(logits, weights, indices)`` router output only the family-native logits are kept, since routing
        replay and balancing re-derive the rest."""
        # Disable autocast so the gate matmul runs at the input dtype (upcast when fp32_router).
        with torch.amp.autocast("cuda", enabled=False):
            router_output = self.gate(self._router_input(flat))
        return router_output[0] if isinstance(router_output, tuple) else router_output

    def _shared_expert_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return getattr(self, self._shared_expert_attr)(hidden_states)

    def _route_router_choice_or_biased(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The ``(logits, weights, indices)`` router's own choice, or with a balancing bias :meth:`_biased_route`."""
        with torch.amp.autocast("cuda", enabled=False):
            router_output = self.gate(self._router_input(flat))
        if not (isinstance(router_output, tuple) and len(router_output) == 3):
            raise RuntimeError(
                f"{type(self.gate).__name__} returned {type(router_output)} (expected a 3-tuple of "
                f"logits/scores/indices) — transformers changed the router contract; update {type(self).__name__}."
            )
        router_logits, topk_weights, topk_indices = router_output
        if self._transient_balancing_bias(router_logits) is None:
            return self._replay_router_choice(router_logits, topk_indices, topk_weights)
        return self._biased_route(router_logits)


class EPGroupLimitedMoELayerBase(EPSharedExpertsMoELayerBase):
    """EP MoE layer routing DeepSeek-V3 style: score -> selection-only bias -> group-limited top-k. The sharing
    families differ only in the score function, optional routing knobs and renormalization floor, so a mis-ordered
    bias add cannot land in one alone. ``n_group: 1`` is plain top-k."""

    # Routing knobs this family genuinely does not declare (Laguna and Step-3.7 carry no group limiting or
    # ``norm_topk_prob``). Any other absence is an upstream rename, where the neutral default would be a wrong live
    # value (routed scaling 1.0 instead of the block's 2.5), so resolution raises.
    _OPTIONAL_ROUTING_KNOBS: tuple[str, ...] = ()

    def _adopt_routing(self, original_layer: nn.Module) -> None:
        """Resolve the knobs :meth:`_group_limited_topk` reads off the block, its router, then its config; a knob
        none declares raises unless listed in :attr:`_OPTIONAL_ROUTING_KNOBS`."""
        super()._adopt_routing(original_layer)
        sources = (original_layer, self.gate, getattr(original_layer, "config", None))

        def _resolve(names: str | tuple[str, ...], default):
            spellings = (names,) if isinstance(names, str) else names
            for name in spellings:
                for source in sources:
                    if source is not None and hasattr(source, name):
                        return getattr(source, name)
            # Any spelling opts the knob out: a family whose router only carries the alias must be
            # able to declare it, and the two names are one knob.
            if not set(spellings) & set(self._OPTIONAL_ROUTING_KNOBS):
                raise AttributeError(
                    f"{type(self).__name__} resolved none of {spellings} off "
                    f"{type(original_layer).__name__}, its router or its config — the family renamed "
                    f"the knob upstream. Defaulting to {default!r} would silently change every routed "
                    f"weight this layer emits. Declare it in _OPTIONAL_ROUTING_KNOBS if this family "
                    f"genuinely has no such knob."
                )
            return default

        # The ``*TopkRouter`` modules spell it ``num_group``, configs ``n_group``; missing it disables
        # group-limited routing.
        self.n_group = _resolve(("n_group", "num_group"), 1)
        self.topk_group = _resolve("topk_group", 1)
        self.norm_topk_prob = _resolve("norm_topk_prob", True)
        self.routed_scaling_factor = float(_resolve("routed_scaling_factor", 1.0))

    def _init_summary_extras(self, original_layer: nn.Module) -> tuple[str, ...]:
        return (
            *super()._init_summary_extras(original_layer),
            f"n_group={self.n_group}",
            f"topk_group={self.topk_group}",
            f"norm_topk_prob={self.norm_topk_prob}",
            f"routed_scaling_factor={self.routed_scaling_factor}",
        )

    def _routing_scores(self, router_logits: torch.Tensor) -> torch.Tensor:
        """Scores selection and gate weights both read: sigmoid, the DeepSeek-V3 form (Mistral4 uses softmax)."""
        return router_logits.sigmoid()

    def route_tokens_to_experts(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select on the biased scores; weights read the unbiased ones."""
        scores = self._routing_scores(router_logits)
        return self._group_limited_topk(self._selection_scores(scores), scores)

    def _group_limited_topk(
        self, scores_for_select: torch.Tensor, gate_scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.n_group > 1:
            experts_per_group = self.num_experts // self.n_group
            group_scores = scores_for_select.view(-1, self.n_group, experts_per_group).topk(2, dim=-1)[0].sum(dim=-1)
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1).expand(-1, self.n_group, experts_per_group).reshape(-1, self.num_experts)
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
