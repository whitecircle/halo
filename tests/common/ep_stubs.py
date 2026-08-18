"""Concrete ``EPMoELayerBase`` subclasses for CPU tests: the boilerplate, not the payload.

``EPMoELayerBase.__init__`` builds an ``EPConfig`` and DeepEP buffers, which a CPU test cannot call,
so every EP unit test would otherwise restate the same bypass, the no-expert-LoRA default and a dummy
``forward``. Keeping them here means a new abstract method on the real base surfaces in one place,
and the export contract each test is about (its parameters, its ``expert_named_params``, its
``gather_expert_state_dict``) stays visible in the test file.
"""

from __future__ import annotations

from torch import nn

from src.distributed.expert_parallel.base_layer import EPMoELayerBase


class StubEPLayerBase(EPMoELayerBase):
    """An EP MoE layer constructible without live EP process groups.

    A subclass that overrides ``gather_expert_state_dict`` must also override
    ``merge_shards_to_hf``: ``EPExpertGatherMixin.__init_subclass__`` enforces that pair, which is why
    neither is stubbed here.
    """

    def __init__(self):
        nn.Module.__init__(self)  # EPMoELayerBase.__init__ needs live EP groups
        self._expert_lora_attrs = frozenset()

    def forward(self, hidden_states, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} is a state/export stub — its forward is never exercised")
