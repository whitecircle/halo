"""Concrete ``EPMoELayerBase`` subclasses for CPU tests, and a rank's real EP layer on stubbed process groups. A new
abstract method on the real base surfaces here, and each test's export contract stays visible in its own file."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from halo.distributed.expert_parallel import config as ep_config_module
from halo.distributed.expert_parallel.moe_layer import layer as layer_module
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.models.moe.family_specs import MoEFamilySpec
from tests.common.parallelism import stubbed_rank_ep_config


class StubEPLayerBase(EPMoELayerBase):
    """An EP MoE layer constructible with no live EP groups; its spec claims no ``model_type``, so it never joins the
    roster."""

    FAMILY = MoEFamilySpec(model_types=())

    def __init__(self):
        nn.Module.__init__(self)  # EPMoELayerBase.__init__ needs live EP groups.
        self._expert_lora_attrs = frozenset()

    def forward(self, hidden_states, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} is a state/export stub — its forward is never exercised")


def bare_ep_layer(cls: type[EPMoELayerBase], **expert_params):
    """``cls`` without its constructor, holding what the export reads on one process and ``expert_params``; its real
    gather runs at world size 1."""
    layer = object.__new__(cls)
    nn.Module.__init__(layer)
    layer._expert_lora_attrs = frozenset()
    layer.expert_tp_size = 1
    layer.expert_tp_group = "expert-tp-group"
    layer.ep_config = SimpleNamespace(ep_size=1, dispatch_ep_group="dispatch-ep-group")
    for name, tensor in expert_params.items():
        setattr(layer, name, nn.Parameter(torch.as_tensor(tensor)))
    return layer


class _NoDispatcher:
    """DeepEP's stand-in for a layer whose expert compute a test calls on a dispatched view."""

    def __init__(self, *args, **kwargs):
        pass


def stubbed_rank_ep_layer(
    layer_cls: type[EPMoELayerBase],
    block: nn.Module,
    world_size: int,
    ep_size: int,
    rank: int,
    *,
    grouped: bool,
    **config_fields,
) -> EPMoELayerBase:
    """Rank ``rank``'s ``layer_cls`` around ``block`` at ``ep_size``-way dispatch and quotient expert TP, with no
    dispatcher. ``grouped`` makes ``grouped_gemm_enabled`` hold, as on SM90+. Built outside the stubbed groups, so no
    grad-sync hook registers."""
    config = stubbed_rank_ep_config(world_size, ep_size, rank, layer_cls.detect_num_experts(block), **config_fields)
    with (
        patch.object(layer_module, "DeepEPDispatcher", _NoDispatcher),
        patch.object(ep_config_module, "supports_grouped_mm", return_value=grouped),
    ):
        return layer_cls(block, config)
