#!/usr/bin/env python
"""The EP whole-layer gather's retain gate and the sharded-save expert-LoRA guard. Every rank enters the same
collectives while only the save rank assembles; off the writer that assembly is tens of GB per layer at 397B."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from halo.checkpoint.parallel.ep import reject_unmergeable_ep_sharded_save
from halo.distributed.expert_parallel.moe_layer import gather as expert_gather
from halo.models.moe.expert_layout import ExpertLayout, ExpertLayoutKind
from halo.models.moe.expert_lora_spec import EXPERT_LORA_SUFFIXES
from halo.models.moe.family_specs import MoEFamilySpec
from halo.runtime import collectives
from tests.common.ep_stubs import StubEPLayerBase


class _StubEPLayer(StubEPLayerBase):
    """Records every gather call so the collective count can be asserted."""

    def __init__(self, *, expert_lora: bool = False):
        super().__init__()
        self._expert_lora_attrs = frozenset({"gate_up_proj"}) if expert_lora else frozenset()
        self.gate_up_proj = nn.Parameter(torch.zeros(2, 4, 6))
        self.router = nn.Linear(6, 2, bias=False)
        self.register_buffer("expert_bias", torch.ones(2), persistent=True)
        self.register_buffer("rotary_cache", torch.ones(3), persistent=False)
        self.gather_calls: list[str] = []

    def expert_named_params(self):
        base = [("gate_up_proj", self.gate_up_proj)]
        if self._expert_lora_attrs:
            base += [
                ("gate_up_proj_lora_A", nn.Parameter(torch.zeros(2, 4, 2))),
                ("gate_up_proj_lora_B", nn.Parameter(torch.zeros(2, 2, 6))),
            ]
        return base

    def gather_expert_state_dict(self, *, device="cpu", merge_lora: bool = False, retain: bool = True) -> dict:
        self.gather_calls.append(retain)
        return {"experts.gate_up_proj": torch.zeros(4, 4, 6, device=device)} if retain else {}


def test_retain_true_returns_the_whole_layer():
    layer = _StubEPLayer()
    gathered = layer.gather_layer_state_dict("model.layers.0.mlp", retain=True)
    assert set(gathered) == {
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.router.weight",
        "model.layers.0.mlp.expert_bias",
    }


def test_retain_false_runs_the_gather_but_keeps_nothing():
    """Non-save ranks still enter the collective yet assemble nothing."""
    layer = _StubEPLayer()
    gathered = layer.gather_layer_state_dict("model.layers.0.mlp", retain=False)
    assert gathered == {}
    assert layer.gather_calls == [False], (
        "the expert gather is a collective — it must run on every rank, and the flag must reach it"
    )


class _ReverseOrderedSet(frozenset):
    """Iterates in the one order a sorted reader cannot produce, where a real frozenset's order is per-process."""

    def __iter__(self):
        return iter(sorted(frozenset.__iter__(self), reverse=True))


def test_expert_adapters_list_in_one_order_on_every_rank():
    """The readers' order reaches any collective a later consumer runs over them, and a rank-dependent one hangs it."""
    layer = _StubEPLayer()
    attrs = ("down_proj", "gate_up_proj")
    layer._expert_lora_attrs = _ReverseOrderedSet(attrs)
    for attr in attrs:
        for suffix in EXPERT_LORA_SUFFIXES:
            setattr(layer, f"{attr}{suffix}", nn.Parameter(torch.zeros(2, 4, 4)))

    assert [name for name, _ in layer.expert_lora_named_params()] == [
        "down_proj_lora_A",
        "down_proj_lora_B",
        "gate_up_proj_lora_A",
        "gate_up_proj_lora_B",
    ]


def test_gather_exports_persistent_buffers_only():
    """Exporting a rotary cache here would break merged-from-sharded == gathered."""
    layer = _StubEPLayer()
    gathered = layer.gather_layer_state_dict("model.layers.0.mlp", retain=True)
    assert "model.layers.0.mlp.expert_bias" in gathered
    assert "model.layers.0.mlp.rotary_cache" not in gathered


EP_SIZE = 4
LOCAL_EXPERTS, HIDDEN, INTER = 2, 6, 4
EP_GROUP = "fake-dispatch-ep-group"


class _FusedGatherLayer(StubEPLayerBase):
    def __init__(self):
        super().__init__()
        self.expert_tp_size = 1
        self.expert_tp_group = None
        self.ep_config = SimpleNamespace(ep_size=EP_SIZE, dispatch_ep_group=EP_GROUP)
        self.gate_up_proj = nn.Parameter(torch.randn(LOCAL_EXPERTS, HIDDEN, 2 * INTER))
        self.down_proj = nn.Parameter(torch.randn(LOCAL_EXPERTS, INTER, HIDDEN))


class _PerExpertGatherLayer(_FusedGatherLayer):
    FAMILY = MoEFamilySpec(
        model_types=(),
        expert_layout=ExpertLayout(ExpertLayoutKind.FUSED_SPLIT_PER_EXPERT, ("gate_proj", "up_proj", "down_proj")),
    )


@pytest.fixture
def ep_group(monkeypatch):
    """Emulates the ``EP_SIZE``-rank expert-axis all-gather."""
    calls: list[tuple[int, ...]] = []

    def fake_all_gather_into_tensor(output, tensor, group=None):
        assert group == EP_GROUP
        calls.append(tuple(tensor.shape))
        rows = tensor.shape[0]
        for rank in range(EP_SIZE):
            output[rank * rows : (rank + 1) * rows].copy_(tensor)

    monkeypatch.setattr(collectives.dist, "all_gather_into_tensor", fake_all_gather_into_tensor)
    return calls


def test_family_gather_assembles_only_on_the_retaining_rank(ep_group):
    layer = _FusedGatherLayer()
    retained = layer.gather_expert_state_dict(device="cpu", retain=True)
    assert {key: tuple(t.shape) for key, t in retained.items()} == {
        "experts.gate_up_proj": (LOCAL_EXPERTS * EP_SIZE, 2 * INTER, HIDDEN),
        "experts.down_proj": (LOCAL_EXPERTS * EP_SIZE, HIDDEN, INTER),
    }
    assert layer.gather_expert_state_dict(device="cpu", retain=False) == {}


def test_non_retaining_rank_enters_every_expert_collective(ep_group):
    """A rank that skipped one would hang the peers still in it."""
    layer = _FusedGatherLayer()
    layer.gather_expert_state_dict(device="cpu", retain=True)
    retaining = list(ep_group)
    ep_group.clear()
    layer.gather_expert_state_dict(device="cpu", retain=False)
    assert ep_group == retaining, "same collectives, same shapes, in the same order"


def test_per_expert_split_never_runs_off_the_writer(ep_group, monkeypatch):
    """The per-expert split is the allocation: one fresh contiguous tensor per expert per projection."""
    splits = []
    real_layout = expert_gather.to_hub_expert_tensors

    def counting_layout(layout, tensors, *, device):
        splits.append(len(tensors))
        return real_layout(layout, tensors, device=device)

    monkeypatch.setattr(expert_gather, "to_hub_expert_tensors", counting_layout)
    layer = _PerExpertGatherLayer()

    assert layer.gather_expert_state_dict(device="cpu", retain=False) == {}
    assert splits == [], "a non-writing rank must not build the per-expert tensors"

    retained = layer.gather_expert_state_dict(device="cpu", retain=True)
    assert splits == [2]
    assert len(retained) == LOCAL_EXPERTS * EP_SIZE * 3


def _model_with(layer) -> nn.Module:
    layer.ep_config = SimpleNamespace(expert_tp_size=1, ep_group_size=8, num_ep_groups=1)
    root = nn.Module()
    root.config = SimpleNamespace(model_type="gpt_oss", auto_map=None)
    root.mlp = layer
    return root


def test_sharded_save_rejects_native_expert_lora(monkeypatch):
    """``merge_ep_shards.py`` never reads the adapter shard keys, so the merged experts would be the frozen base."""
    monkeypatch.setenv("DIST_SHARED_FILESYSTEM", "1")
    layer = _StubEPLayer(expert_lora=True)
    with pytest.raises(ValueError, match="expert LoRA"):
        reject_unmergeable_ep_sharded_save(_model_with(layer), world_size=8)


def test_sharded_save_allowed_without_expert_lora(monkeypatch):
    monkeypatch.setenv("DIST_SHARED_FILESYSTEM", "1")
    layer = _StubEPLayer(expert_lora=False)
    reject_unmergeable_ep_sharded_save(_model_with(layer), world_size=8)


def test_sharded_save_rejects_multiple_ep_groups(monkeypatch):
    """EP groups are DP replicas holding the same experts, so the shards merge duplicated."""
    monkeypatch.setenv("DIST_SHARED_FILESYSTEM", "1")
    layer = _StubEPLayer(expert_lora=False)
    model = _model_with(layer)
    layer.ep_config.ep_group_size = 4
    layer.ep_config.num_ep_groups = 2
    with pytest.raises(ValueError, match="ep_group_size"):
        reject_unmergeable_ep_sharded_save(model, world_size=8)


@pytest.mark.parametrize("field", ("ep_group_size", "expert_tp_size"))
def test_topology_guards_do_not_pass_on_a_config_that_never_declared_the_field(monkeypatch, field):
    """``getattr(ep_cfg, "ep_group_size", world_size)`` substitutes the value the comparison accepts, clearing both
    checks vacuously."""
    monkeypatch.setenv("DIST_SHARED_FILESYSTEM", "1")
    layer = _StubEPLayer(expert_lora=False)
    model = _model_with(layer)
    delattr(layer.ep_config, field)
    with pytest.raises(AttributeError, match=field):
        reject_unmergeable_ep_sharded_save(model, world_size=8)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
