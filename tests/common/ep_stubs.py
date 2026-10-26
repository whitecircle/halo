"""Concrete ``EPMoELayerBase`` subclasses for CPU tests: the boilerplate, not the payload.

``EPMoELayerBase.__init__`` builds an ``EPConfig`` and DeepEP buffers, which a CPU test cannot call,
so every EP unit test would otherwise restate the same bypass, the no-expert-LoRA default and a dummy
``forward``. Keeping them here means a new abstract method on the real base surfaces in one place,
and the export contract each test is about (its parameters, its ``expert_named_params``) stays
visible in the test file.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from src.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from src.models.moe.family_specs import MoEFamilySpec


class StubEPLayerBase(EPMoELayerBase):
    """An EP MoE layer constructible without live EP process groups. Its spec claims no ``model_type``, so a
    stub never joins the family roster."""

    FAMILY = MoEFamilySpec(model_types=())

    def __init__(self):
        nn.Module.__init__(self)  # EPMoELayerBase.__init__ needs live EP groups
        self._expert_lora_attrs = frozenset()

    def forward(self, hidden_states, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} is a state/export stub — its forward is never exercised")


def bare_ep_layer(cls: type[EPMoELayerBase], *, ep_size: int = 1, expert_tp_size: int = 1, **expert_params):
    """``cls`` without its constructor, holding the state the export reads on one process and ``expert_params``
    (``name=tensor``) as its expert parameters: its real gather runs, every collective at world size 1 unless a
    test fakes the transport."""
    layer = object.__new__(cls)
    nn.Module.__init__(layer)
    layer._expert_lora_attrs = frozenset()
    layer.expert_tp_size = expert_tp_size
    layer.expert_tp_group = "expert-tp-group"
    layer.ep_config = SimpleNamespace(ep_size=ep_size, dispatch_ep_group="dispatch-ep-group")
    for name, tensor in expert_params.items():
        setattr(layer, name, nn.Parameter(torch.as_tensor(tensor)))
    return layer
