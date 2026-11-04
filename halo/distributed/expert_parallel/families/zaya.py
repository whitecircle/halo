"""EP layer for Zaya routed experts (the router is part of the wrapped ZayaSparseMoeBlock)."""

from __future__ import annotations

import torch

from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.models.moe.family_specs import ZAYA


class EPZayaMoELayer(EPMoELayerBase):
    """EP layer for ``ZayaSparseMoeBlock``. The gate's forward runs unchanged: it holds the cross-layer EDA state and
    is the balancing router in every mode (``patch_zaya_router_load_recording``). It returns flattened ``[T, top_k]``
    probabilities/indices with its learned discard slot masked to expert 0 at weight 0, which EP dispatches as ``-1``
    so they never reach expert 0's rank."""

    HF_MODULE_NAMES = ("ZayaSparseMoeBlock",)
    FAMILY = ZAYA

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        prev_router_hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        router_states = (
            self._router_input(prev_router_hidden_states) if prev_router_hidden_states is not None else None
        )
        _router_logits, router_probs, router_indices, prev_router_hidden_states = self.gate(
            self._router_input(hidden_states), router_states=router_states
        )

        batch_size, seq_length, emb_dim = hidden_states.shape
        hidden_states_flat = hidden_states.reshape(batch_size * seq_length, emb_dim)

        experts = router_indices.long()
        if self.ep_size > 1:
            # A masked discard would otherwise ride the all-to-all to expert 0's rank in every layer, loading it with
            # rows that contribute nothing. At ep1 nothing is dispatched and the grouped path takes only local ids.
            experts = experts.masked_fill(router_probs == 0, -1)
        expert_output = self._dispatch_compute_combine(
            hidden_states_flat,
            experts,
            router_probs.float(),
            hidden_states.dtype,
        )
        expert_output = expert_output.reshape(batch_size, seq_length, emb_dim)

        # The EDA state stays attached: it is the only path by which this layer's routing loss reaches the previous
        # gate's ``down_proj``/``router_states_scale``.
        return expert_output, prev_router_hidden_states
