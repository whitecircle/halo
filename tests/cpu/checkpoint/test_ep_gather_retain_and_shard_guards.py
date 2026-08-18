#!/usr/bin/env python
"""CPU tests for the EP whole-layer gather's retain gate and for the sharded-save expert-LoRA guard.

- ``gather_ep_layer_weights(..., retain=False)``: every rank must enter the SAME collectives (the
  family gather, and ``full_tensor()`` on each replicated DTensor) while only the save rank keeps the
  result. Without the gate every rank materializes a full host copy of the layer, taking per-node CPU
  peak to ``local_ranks`` x one gathered layer.
- ``retain`` reaches INSIDE the family gather: the expert all-gathers stay on every rank, but the
  layout assembly that follows them — the per-expert split, the transpose + ``contiguous``, the host
  copy — must not run off the writer. That assembly is the tens of GB per layer at 397B, paid on 511
  of 512 ranks per checkpoint. A family that ignores the flag is rejected rather than obeyed.
- The gather exports PERSISTENT buffers only: the sharded save writes persistent buffers alone, so a
  non-persistent buffer here breaks the "merged-from-sharded == gathered" invariant the merge
  transforms are built around.
- ``save_sharded_ep=True`` + native expert LoRA: the shard key set comes from ``expert_named_params()``,
  which includes ``<attr>_lora_A/_lora_B``, so those are written as ``.shard_N`` keys that
  ``merge_ep_shards.py``'s base-root pattern never matches — they pass through dead while the merged
  experts are the frozen base. Must be rejected up front.

Run: pytest tests/cpu/checkpoint/test_ep_gather_retain_and_shard_guards.py
"""

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.distributed.expert_parallel import expert_gather
from src.distributed.expert_parallel.expert_weights import gather_ep_layer_weights
from src.distributed.expert_parallel.saving import _check_ep_sharded_save_supported
from tests.common.ep_stubs import StubEPLayerBase


class _StubEPLayer(StubEPLayerBase):
    """Minimal EP layer: one gathered expert tensor, one replicated router param, one persistent and
    one non-persistent buffer. Records every gather call so the collective count can be asserted."""

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

    def gather_expert_state_dict(self, device: str = "cpu", merge_lora: bool = False, retain: bool = True) -> dict:
        self.gather_calls.append(retain)
        return {"experts.gate_up_proj": torch.zeros(4, 4, 6, device=device)} if retain else {}

    @classmethod
    def merge_shards_to_hf(cls, prefix: str, params: dict) -> dict:  # pragma: no cover - paired override
        return {}


def test_retain_true_returns_the_whole_layer():
    layer = _StubEPLayer()
    gathered = gather_ep_layer_weights("model.layers.0.mlp", layer, retain=True)
    assert set(gathered) == {
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.router.weight",
        "model.layers.0.mlp.expert_bias",
    }


def test_retain_false_runs_the_gather_but_keeps_nothing():
    """Non-save ranks must still enter the family gather (a collective) yet assemble nothing."""
    layer = _StubEPLayer()
    gathered = gather_ep_layer_weights("model.layers.0.mlp", layer, retain=False)
    assert gathered == {}
    assert layer.gather_calls == [False], (
        "the expert gather is a collective — it must run on every rank, and the flag must reach it"
    )


def test_gather_exports_persistent_buffers_only():
    """The sharded save writes persistent buffers only; exporting a rotary cache here would break the
    merged-from-sharded == gathered invariant."""
    layer = _StubEPLayer()
    gathered = gather_ep_layer_weights("model.layers.0.mlp", layer, retain=True)
    assert "model.layers.0.mlp.expert_bias" in gathered
    assert "model.layers.0.mlp.rotary_cache" not in gathered


EP_SIZE = 4
LOCAL_EXPERTS, HIDDEN, INTER = 2, 6, 4
EP_GROUP = "fake-dispatch-ep-group"


class _FusedGatherLayer(StubEPLayerBase):
    """Fused-GLU expert shards over a ``EP_SIZE``-rank dispatch group, gathered by the REAL base
    methods — the path every family without a gather override takes."""

    def __init__(self):
        super().__init__()
        self.expert_tp_size = 1
        self.expert_tp_group = None
        self.ep_config = SimpleNamespace(ep_size=EP_SIZE, dispatch_ep_group=EP_GROUP)
        self.gate_up_proj = nn.Parameter(torch.randn(LOCAL_EXPERTS, HIDDEN, 2 * INTER))
        self.down_proj = nn.Parameter(torch.randn(LOCAL_EXPERTS, INTER, HIDDEN))


class _PerExpertGatherLayer(_FusedGatherLayer):
    """Same shards, split into the per-expert hub layout by the base gather (GLM-4 / LFM-2 shape)."""

    _PER_EXPERT_UNFUSED_KEYS = ("gate_proj", "up_proj", "down_proj")


class _IgnoresRetainLayer(_FusedGatherLayer):
    """A family whose gather assembles regardless of the flag — the regression the caller must catch."""

    def gather_expert_state_dict(self, device: str = "cpu", merge_lora: bool = False, retain: bool = True) -> dict:
        return super().gather_expert_state_dict(device, merge_lora=merge_lora, retain=True)

    @classmethod
    def merge_shards_to_hf(cls, prefix: str, params: dict) -> dict:  # pragma: no cover - paired override
        return {}


@pytest.fixture
def ep_group(monkeypatch):
    """Emulate the ``EP_SIZE``-rank expert-axis all-gather, counting every entry."""
    calls: list[tuple[int, ...]] = []

    def fake_all_gather_into_tensor(output, tensor, group=None):
        assert group == EP_GROUP
        calls.append(tuple(tensor.shape))
        rows = tensor.shape[0]
        for rank in range(EP_SIZE):
            output[rank * rows : (rank + 1) * rows].copy_(tensor)

    monkeypatch.setattr(expert_gather.dist, "all_gather_into_tensor", fake_all_gather_into_tensor)
    return calls


def test_family_gather_assembles_only_on_the_retaining_rank(ep_group):
    """The writer gets the full expert tensors; every other rank gets nothing to hold."""
    layer = _FusedGatherLayer()
    retained = layer.gather_expert_state_dict(device="cpu", retain=True)
    assert {key: tuple(t.shape) for key, t in retained.items()} == {
        "experts.gate_up_proj": (LOCAL_EXPERTS * EP_SIZE, 2 * INTER, HIDDEN),
        "experts.down_proj": (LOCAL_EXPERTS * EP_SIZE, HIDDEN, INTER),
    }
    assert layer.gather_expert_state_dict(device="cpu", retain=False) == {}


def test_non_retaining_rank_enters_every_expert_collective(ep_group):
    """The all-gathers are group-wide: a rank that skipped one would hang the peers still in it."""
    layer = _FusedGatherLayer()
    layer.gather_expert_state_dict(device="cpu", retain=True)
    retaining = list(ep_group)
    ep_group.clear()
    layer.gather_expert_state_dict(device="cpu", retain=False)
    assert ep_group == retaining, "same collectives, same shapes, in the same order"


def test_per_expert_split_never_runs_off_the_writer(ep_group, monkeypatch):
    """The per-expert split is the allocation — one fresh contiguous tensor per expert per projection.
    Skipping it is the whole point of the gate, so it must not run when nothing is retained."""
    splits = []
    real_split = expert_gather.EPExpertGatherMixin._unfuse_fused_to_per_expert.__func__

    def counting_split(cls, fused):
        splits.append(len(fused))
        return real_split(cls, fused)

    monkeypatch.setattr(expert_gather.EPExpertGatherMixin, "_unfuse_fused_to_per_expert", classmethod(counting_split))
    layer = _PerExpertGatherLayer()

    assert layer.gather_expert_state_dict(device="cpu", retain=False) == {}
    assert splits == [], "a non-writing rank must not build the per-expert tensors"

    retained = layer.gather_expert_state_dict(device="cpu", retain=True)
    assert splits == [2]
    assert len(retained) == LOCAL_EXPERTS * EP_SIZE * 3


def test_fused_gather_refusal_is_explicit_not_an_empty_result():
    """``retain=False`` returns ``{}`` from every gather, so the "no fused layout" answer cannot also
    be an empty dict — a non-sending rank would be indistinguishable from a family SGLang cannot load.
    The base raises, and the pre-flight gate reads the override off the class."""
    layer = _FusedGatherLayer()
    assert not type(layer).implements_fused_expert_layout()
    with pytest.raises(NotImplementedError, match="no fused expert layout"):
        layer.gather_fused_expert_state_dict()


def test_layer_gather_rejects_a_family_that_ignores_retain(ep_group):
    """Silently keeping the tensors would put a whole gathered layer on every non-writing rank —
    the memory the flag exists to avoid, with nothing in the logs."""
    with pytest.raises(RuntimeError, match="ignored retain=False"):
        gather_ep_layer_weights("model.layers.0.mlp", _IgnoresRetainLayer(), retain=False)


def _ep_layers(layer):
    ep_cfg = SimpleNamespace(expert_tp_size=1, ep_group_size=8, num_ep_groups=1)
    layer.ep_config = ep_cfg
    return [("model.layers.0.mlp", layer)]


def _model_with(layer) -> nn.Module:
    root = nn.Module()
    root.config = SimpleNamespace(model_type="gpt_oss", auto_map=None)
    root.mlp = layer
    return root


def test_sharded_save_rejects_native_expert_lora(monkeypatch):
    """The adapter shards are written under keys merge_ep_shards.py never reads, so the merged experts
    would be the frozen base — reject before training instead of after."""
    monkeypatch.setenv("DIST_SHARED_FILESYSTEM", "1")
    layer = _StubEPLayer(expert_lora=True)
    with pytest.raises(ValueError, match="expert LoRA"):
        _check_ep_sharded_save_supported(_model_with(layer), _ep_layers(layer), world_size=8)


def test_sharded_save_allowed_without_expert_lora(monkeypatch):
    monkeypatch.setenv("DIST_SHARED_FILESYSTEM", "1")
    layer = _StubEPLayer(expert_lora=False)
    _check_ep_sharded_save_supported(_model_with(layer), _ep_layers(layer), world_size=8)


def test_sharded_save_rejects_multiple_ep_groups(monkeypatch):
    """Several EP groups = DP replicas, each holding the SAME experts, so the shards merge with
    every expert duplicated. Only a single group spanning the world is mergeable."""
    monkeypatch.setenv("DIST_SHARED_FILESYSTEM", "1")
    layer = _StubEPLayer(expert_lora=False)
    ep_layers = _ep_layers(layer)
    layer.ep_config.ep_group_size = 4
    layer.ep_config.num_ep_groups = 2
    with pytest.raises(ValueError, match="ep_group_size"):
        _check_ep_sharded_save_supported(_model_with(layer), ep_layers, world_size=8)


@pytest.mark.parametrize("field", ("ep_group_size", "expert_tp_size"))
def test_topology_guards_do_not_pass_on_a_config_that_never_declared_the_field(monkeypatch, field):
    """Neither topology guard may fall back to the value that makes it pass.

    ``getattr(ep_cfg, "ep_group_size", world_size)`` / ``getattr(ep_cfg, "expert_tp_size", 1)``
    substitute exactly the value the very next comparison accepts, so anything that is not a
    finalized ``EPConfig`` clears both checks VACUOUSLY — the sharded save would then run on a
    topology nothing validated and merge duplicated or expert-TP-split experts. ``EPConfig`` sets
    both fields unconditionally, so this is unreachable through the production path today; the
    point is that the guards must not be able to reach a pass verdict without reading the topology.
    """
    monkeypatch.setenv("DIST_SHARED_FILESYSTEM", "1")
    layer = _StubEPLayer(expert_lora=False)
    ep_layers = _ep_layers(layer)
    delattr(layer.ep_config, field)
    with pytest.raises(AttributeError, match=field):
        _check_ep_sharded_save_supported(_model_with(layer), ep_layers, world_size=8)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
