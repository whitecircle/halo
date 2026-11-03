"""Routing-replay stand-ins shared by the CPU suites: a bare EP layer and the two engine wire formats."""

import base64
import io

import numpy as np
from torch import nn

from halo.distributed.expert_parallel.moe_layer.shared_experts_layer import EPGroupLimitedMoELayerBase
from halo.models.moe.family_specs import MoEFamilySpec


class BareEPLayer(EPGroupLimitedMoELayerBase):
    """Minimal EP layer for the selection helpers: no dispatcher, rooted at the class owning group-limited routing, so
    helpers are what families inherit."""

    FAMILY = MoEFamilySpec(model_types=())

    def __init__(self, *, top_k: int, num_experts: int, **attrs):
        nn.Module.__init__(self)
        self.top_k = top_k
        self.num_experts = num_experts
        self.n_routed_experts = num_experts
        self.n_group = 1
        self.topk_group = 1
        self.norm_topk_prob = True
        self.routed_scaling_factor = 1.0
        self._forced_topk_indices = None
        self._forced_cursor = 0
        self._forced_consumed_total = 0
        self._capture_routing = False
        self._captured_routing_chunks = []
        self._replay_flip_counts = None
        for key, value in attrs.items():
            setattr(self, key, value)

    def forward(self, hidden_states, **kwargs):  # pragma: no cover — never called
        raise NotImplementedError


def npy_routing_payload(array) -> str:
    """vLLM's wire form: a base64 ``.npy`` (the header carries the shape)."""
    buf = io.BytesIO()
    np.save(buf, np.asarray(array))
    return base64.b64encode(buf.getvalue()).decode()


def raw_routing_payload(array) -> str:
    """SGLang's wire form: base64 raw little-endian int32 rows, the shape implied."""
    return base64.b64encode(np.asarray(array, dtype=np.int32).tobytes()).decode()
