"""EP wrapper for Zaya routed experts (the router is part of the wrapped ZayaSparseMoeBlock)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.distributed.expert_parallel.base_layer import EPMoELayerBase


class EPZayaMoELayer(EPMoELayerBase):
    """EP wrapper for ``ZayaSparseMoeBlock`` (gate + experts) using DeepEP.

    The wrapper replaces the whole block, keeps the gate's forward unchanged (it holds the
    cross-layer EDA state threaded through ``prev_router_hidden_states``; its params sync like any
    adopted router: FSDP only in the ``experts_fsdp_managed`` regime, the router hook or deferred
    sweep otherwise), and routes experts through DeepEP. The gate returns already-flattened
    ``[T, top_k]`` probabilities/indices with its learned "discard" slot masked to expert 0 at weight
    0; under EP those picks are dispatched as ``-1`` (no expert), so they never reach expert 0's rank.

    Bias-update balancing uses the gate's persistent ``balancing_biases`` buffer, which the gate adds
    to the softmax scores for selection only and which is part of the checkpoint. The gate is the
    balancing router in every mode (EP and plain FSDP alike): ``patch_zaya_router_load_recording``
    declares the ``expert_load_counter`` slot and records loads inside the gate's own forward, which
    this wrapper calls unchanged.

    Expert TP is supported via the base fused-GLU helper; gradient checkpointing on top of EP is not.
    """

    HF_MODULE_NAMES = ("ZayaSparseMoeBlock",)
    HF_MODEL_TYPES = ("zaya",)

    # The gate's EDA cross-layer state leaves replay semantics undefined.
    _supports_routing_replay = False

    # Per-layer GC re-wraps the cross-layer EDA state with a fresh grad_fn → polynomial backward.
    _supports_gradient_checkpointing = False

    _NUM_EXPERTS_ATTR_PATHS = ("experts.num_experts",)

    def _init_summary_extras(self, original_layer: nn.Module) -> tuple[str, ...]:
        return ("native balancing_biases gate",)

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        prev_router_hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self._fp32_router_input:
            router_input = hidden_states.float()
            router_states = prev_router_hidden_states.float() if prev_router_hidden_states is not None else None
        else:
            router_input = hidden_states
            router_states = prev_router_hidden_states
        _router_logits, router_probs, router_indices, prev_router_hidden_states = self.gate(
            router_input, router_states=router_states
        )

        batch_size, seq_length, emb_dim = hidden_states.shape
        hidden_states_flat = hidden_states.reshape(batch_size * seq_length, emb_dim)

        experts = router_indices.long()
        if self.ep_size > 1:
            # A masked discard would otherwise ride the all-to-all to expert 0's rank in every layer,
            # loading it with rows that contribute nothing. Zero weight is the same test the gate's
            # load recording uses. At ep1 nothing is dispatched and the grouped path takes only real
            # local ids, so the upstream spelling stays.
            experts = experts.masked_fill(router_probs == 0, -1)
        expert_output = self._dispatch_compute_combine(
            hidden_states_flat,
            experts,
            router_probs.float(),
            hidden_states.dtype,
        )
        expert_output = expert_output.reshape(batch_size, seq_length, emb_dim)

        # The EDA state stays attached, as ``ZayaSparseMoeBlock`` returns it: it is the only path by
        # which this layer's routing loss reaches the previous gate's
        # ``down_proj``/``router_states_scale``.
        return expert_output, prev_router_hidden_states
