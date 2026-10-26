"""CPU tests for the sharded-save guard.

``reject_unmergeable_ep_sharded_save`` must fire BEFORE training, not at (or after) the first save:
(a) a ``model_type`` no registered EP layer class claims can never merge its per-rank shards into a
loadable checkpoint; (b) ``merge_expert_lora_on_save=True`` is incompatible with the sharded save;
(c) a run with no EP layers reads the flag nowhere — a refusal, not a no-op; (d) a multi-node job on
per-node local disks scatters the shards where the merge never sees a complete set.
"""

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
    """A model of ``model_type`` holding one EP layer on a single EP group spanning ``world_size``."""
    model = _fake_model(model_type)
    model.mlp = _FakeEPLayer(world_size)
    return model


def _ep_config(world_size, expert_tp_size=1):
    """ep_config satisfying the topology checks: a single global EP group, no expert TP."""
    return SimpleNamespace(expert_tp_size=expert_tp_size, ep_group_size=world_size, num_ep_groups=1)


class _FakeEPLayer(EPMoELayerBase):
    """A real ``EPMoELayerBase`` instance, which is what ``named_ep_layers`` keys on.

    The validator reads only ``ep_config`` and the expert-LoRA predicate, so the heavy base
    ``__init__`` (process groups, DeepEP buffers) is deliberately skipped.
    """

    def __init__(self, world_size, expert_tp_size=1):
        nn.Module.__init__(self)
        self.ep_config = _ep_config(world_size, expert_tp_size)
        self._expert_lora_attrs = ()

    def forward(self, hidden_states, **kwargs):
        raise NotImplementedError("stand-in: never invoked")


def _non_shared_two_node_fs(monkeypatch, ranks_per_node=8):
    """Non-shared output FS on a 2-node shape (16 ranks / 8 per node), consensus memo cleared."""
    reset_shared_filesystem_consensus()
    monkeypatch.setenv("DIST_SHARED_FILESYSTEM", "0")
    monkeypatch.setattr(ep_saving_mod, "get_local_world_size", lambda: ranks_per_node)


def test_ep_sharded_rejected_on_non_shared_multi_node_fs(monkeypatch):
    """Per-node local disks: the shards are keyed by GLOBAL rank and scatter across the nodes while
    each node's index names all of them, so merge_ep_shards.py never sees a complete set."""
    _non_shared_two_node_fs(monkeypatch)
    model = _fake_model("gpt_oss")
    model.mlp = _FakeEPLayer(world_size=16)
    with pytest.raises(ValueError, match="shared output filesystem"):
        reject_unmergeable_ep_sharded_save(model, world_size=16)


def test_ep_sharded_allowed_on_non_shared_single_node(monkeypatch):
    """Anti-over-rejection: one node holds every shard plus the index, so a local disk is complete."""
    _non_shared_two_node_fs(monkeypatch)
    model = _fake_model("gpt_oss")
    model.mlp = _FakeEPLayer(world_size=8)
    reject_unmergeable_ep_sharded_save(model, world_size=8)


def test_ep_sharded_rejects_merge_expert_lora_at_construction():
    """``merge_expert_lora_on_save`` is gathered-only, and must fail at trainer construction."""
    with pytest.raises(ValueError, match="gathered EP save"):
        reject_unmergeable_ep_sharded_save(_fake_model("gpt_oss"), world_size=1, merge_expert_lora_on_save=True)


def test_ep_sharded_rejected_without_ep_layers():
    """``save_sharded_ep=True`` on a run with no EP layers must not no-op: every save then silently
    falls through to the gathered writer while the merge step the user planned waits for shards that
    never exist. A set value that nothing reads is a config that does not do what it says."""
    with pytest.raises(ValueError, match="no EP MoE layers"):
        reject_unmergeable_ep_sharded_save(_fake_model("gpt_oss"), world_size=1)


def test_ep_sharded_allowed_with_ep_layers(monkeypatch):
    """Anti-over-rejection twin: the real EP shape the flag exists for must pass the same validator.
    Without it the guard above is satisfied by a validator that rejects EVERY run."""
    monkeypatch.setenv("DIST_SHARED_FILESYSTEM", "1")
    model = _fake_model("gpt_oss")
    model.mlp = _FakeEPLayer(world_size=8)
    reject_unmergeable_ep_sharded_save(model, world_size=8)


@pytest.mark.parametrize("model_type", ["gemma4_moe", ""])
def test_ep_sharded_rejects_unclaimed_model_type(model_type):
    """A model_type no EP layer class claims would train into unmergeable shards — must raise."""
    with pytest.raises(ValueError, match="no MoE family spec claims"):
        reject_unmergeable_ep_sharded_save(_fake_ep_model(model_type))


@pytest.mark.parametrize(
    "model_type",
    ["gpt_oss", "qwen3_moe", "qwen3_5_moe", "bailing_moe_linear", "glm4_moe_lite", "lfm2_moe", "zaya", "mistral4"],
)
def test_ep_sharded_accepts_mergeable_family(model_type):
    reject_unmergeable_ep_sharded_save(_fake_ep_model(model_type))


def test_ep_sharded_family_gate_runs_on_a_valid_topology(monkeypatch):
    """The family gate is not shadowed by the topology checks: an unclaimed family on a valid
    single-group topology is refused, and a supported one on the same topology passes."""
    monkeypatch.setenv("DIST_SHARED_FILESYSTEM", "1")
    with pytest.raises(ValueError, match="no MoE family spec claims"):
        reject_unmergeable_ep_sharded_save(_fake_ep_model("nope", world_size=8), world_size=8)
    reject_unmergeable_ep_sharded_save(_fake_ep_model("gpt_oss", world_size=8), world_size=8)


def test_model_type_resolves_to_the_declaring_family_spec():
    assert find_moe_family_spec("glm4_moe_lite") is EPGlm4MoELayer.FAMILY
    assert find_moe_family_spec("bailing_moe_linear") is EPBailingMoELayer.FAMILY
    assert find_moe_family_spec("bailing_moe") is EPBailingMoELayer.FAMILY
    # Qwen3.6 resolves through the 3.5 types it actually ships under, not a 3.6 spelling.
    assert find_moe_family_spec("qwen3_5_moe") is EPQwen3_5MoELayer.FAMILY
    assert find_moe_family_spec("qwen3_5_moe_text") is EPQwen3_5MoELayer.FAMILY
    assert find_moe_family_spec("zaya") is EPZayaMoELayer.FAMILY


def test_laguna_shares_the_glm4_layout_and_is_merge_supported():
    """Laguna has its OWN wrapper (its router normalizes the top-k weights, GLM-4's does not) but the
    identical expert layout, so its spec declares GLM-4's ``ExpertLayout`` and save_sharded_ep=True is
    valid for it."""
    laguna = find_moe_family_spec("laguna")
    assert laguna is not EPGlm4MoELayer.FAMILY
    assert laguna.expert_layout == EPGlm4MoELayer.FAMILY.expert_layout
    reject_unmergeable_ep_sharded_save(_fake_ep_model("laguna"))


def test_unclaimed_model_type_is_rejected_not_guessed():
    """No spelling heuristic: an unclaimed model_type must fail the gate rather than be routed onto
    some other family's expert layout."""
    assert find_moe_family_spec("qwen3_next_moe") is None
    with pytest.raises(ValueError, match="no MoE family spec claims"):
        reject_unmergeable_ep_sharded_save(_fake_ep_model("qwen3_next_moe"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
