"""``reject_unmergeable_ep_sharded_save`` fires before training, not at the first save."""

from types import SimpleNamespace

import pytest
import torch.nn as nn

import halo.checkpoint.parallel.ep as ep_saving_mod
from halo.checkpoint.parallel.ep import reject_unmergeable_ep_sharded_save
from halo.distributed.expert_parallel.families.bailing import EPBailingMoELayer
from halo.distributed.expert_parallel.families.glm4 import EPGlm4MoELayer
from halo.distributed.expert_parallel.families.qwen3_5 import EPQwen3_5MoELayer
from halo.distributed.expert_parallel.families.zaya import EPZayaMoELayer
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.models.moe.family_specs import find_moe_family_spec
from tests.common.distributed import reset_shared_filesystem_consensus


def _fake_model(model_type):
    model = nn.Module()
    model.config = SimpleNamespace(model_type=model_type, auto_map=None)
    return model


def _fake_ep_model(model_type, world_size=1):
    model = _fake_model(model_type)
    model.mlp = _FakeEPLayer(world_size)
    return model


def _ep_config(world_size, expert_tp_size=1):
    """A single global EP group, no expert TP."""
    return SimpleNamespace(expert_tp_size=expert_tp_size, ep_group_size=world_size, num_ep_groups=1)


class _FakeEPLayer(EPMoELayerBase):
    """The validator reads only ``ep_config`` and the expert-LoRA predicate, so the heavy base ``__init__`` is
    skipped."""

    def __init__(self, world_size, expert_tp_size=1):
        nn.Module.__init__(self)
        self.ep_config = _ep_config(world_size, expert_tp_size)
        self._expert_lora_attrs = ()

    def forward(self, hidden_states, **kwargs):
        raise NotImplementedError("stand-in: never invoked")


def _non_shared_two_node_fs(monkeypatch, ranks_per_node=8):
    """16 ranks / 8 per node, consensus memo cleared."""
    reset_shared_filesystem_consensus()
    monkeypatch.setenv("DIST_SHARED_FILESYSTEM", "0")
    monkeypatch.setattr(ep_saving_mod, "get_local_world_size", lambda: ranks_per_node)


def test_ep_sharded_rejected_on_non_shared_multi_node_fs(monkeypatch):
    """Shards scatter across nodes while each index names all of them, so merge_ep_shards.py never sees a complete
    set."""
    _non_shared_two_node_fs(monkeypatch)
    model = _fake_model("gpt_oss")
    model.mlp = _FakeEPLayer(world_size=16)
    with pytest.raises(ValueError, match="shared output filesystem"):
        reject_unmergeable_ep_sharded_save(model, world_size=16)


def test_ep_sharded_allowed_on_non_shared_single_node(monkeypatch):
    """One node holds every shard plus the index."""
    _non_shared_two_node_fs(monkeypatch)
    model = _fake_model("gpt_oss")
    model.mlp = _FakeEPLayer(world_size=8)
    reject_unmergeable_ep_sharded_save(model, world_size=8)


def test_ep_sharded_rejects_merge_expert_lora_at_construction():
    """Gathered-only."""
    with pytest.raises(ValueError, match="gathered EP save"):
        reject_unmergeable_ep_sharded_save(_fake_model("gpt_oss"), world_size=1, merge_expert_lora_on_save=True)


def test_ep_sharded_rejected_without_ep_layers():
    """Every save falls through to the gathered writer while the planned merge waits for shards that never exist."""
    with pytest.raises(ValueError, match="no EP MoE layers"):
        reject_unmergeable_ep_sharded_save(_fake_model("gpt_oss"), world_size=1)


def test_ep_sharded_allowed_with_ep_layers(monkeypatch):
    """Otherwise a validator rejecting every run satisfies the guard above."""
    monkeypatch.setenv("DIST_SHARED_FILESYSTEM", "1")
    model = _fake_model("gpt_oss")
    model.mlp = _FakeEPLayer(world_size=8)
    reject_unmergeable_ep_sharded_save(model, world_size=8)


@pytest.mark.parametrize("model_type", ["gemma4_moe", ""])
def test_ep_sharded_rejects_unclaimed_model_type(model_type):
    with pytest.raises(ValueError, match="no MoE family spec claims"):
        reject_unmergeable_ep_sharded_save(_fake_ep_model(model_type))


@pytest.mark.parametrize(
    "model_type",
    ["gpt_oss", "qwen3_moe", "qwen3_5_moe", "bailing_moe_linear", "glm4_moe_lite", "lfm2_moe", "zaya", "mistral4"],
)
def test_ep_sharded_accepts_mergeable_family(model_type):
    reject_unmergeable_ep_sharded_save(_fake_ep_model(model_type))


def test_ep_sharded_family_gate_runs_on_a_valid_topology(monkeypatch):
    """The family gate is not shadowed by the topology checks."""
    monkeypatch.setenv("DIST_SHARED_FILESYSTEM", "1")
    with pytest.raises(ValueError, match="no MoE family spec claims"):
        reject_unmergeable_ep_sharded_save(_fake_ep_model("nope", world_size=8), world_size=8)
    reject_unmergeable_ep_sharded_save(_fake_ep_model("gpt_oss", world_size=8), world_size=8)


def test_model_type_resolves_to_the_declaring_family_spec():
    assert find_moe_family_spec("glm4_moe_lite") is EPGlm4MoELayer.FAMILY
    assert find_moe_family_spec("bailing_moe_linear") is EPBailingMoELayer.FAMILY
    assert find_moe_family_spec("bailing_moe") is EPBailingMoELayer.FAMILY
    # Qwen3.6 resolves through the 3.5 types it ships under.
    assert find_moe_family_spec("qwen3_5_moe") is EPQwen3_5MoELayer.FAMILY
    assert find_moe_family_spec("qwen3_5_moe_text") is EPQwen3_5MoELayer.FAMILY
    assert find_moe_family_spec("zaya") is EPZayaMoELayer.FAMILY


def test_laguna_shares_the_glm4_layout_and_is_merge_supported():
    """Its own wrapper (the router normalizes top-k weights), with GLM-4's ``ExpertLayout``."""
    laguna = find_moe_family_spec("laguna")
    assert laguna is not EPGlm4MoELayer.FAMILY
    assert laguna.expert_layout == EPGlm4MoELayer.FAMILY.expert_layout
    reject_unmergeable_ep_sharded_save(_fake_ep_model("laguna"))


def test_unclaimed_model_type_is_rejected_not_guessed():
    """No spelling heuristic."""
    assert find_moe_family_spec("qwen3_next_moe") is None
    with pytest.raises(ValueError, match="no MoE family spec claims"):
        reject_unmergeable_ep_sharded_save(_fake_ep_model("qwen3_next_moe"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
