"""CPU tests for the construction-time sharded-save guard.

``validate_ep_sharded_save`` must fire BEFORE training, not at (or after) the first save:
(a) a ``model_type`` no registered EP layer class claims can never merge its per-rank shards into a
loadable checkpoint; (b) ``merge_expert_lora_on_save=True`` is incompatible with the sharded save;
(c) a run with no EP layers reads the flag nowhere — a refusal, not a no-op; (d) a multi-node job on
per-node local disks scatters the shards where the merge never sees a complete set.
"""

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import src.distributed.expert_parallel.saving as ep_saving_mod
from src.distributed import runtime
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.expert_weights import (
    ep_layer_classes,
    resolve_ep_merge_layer_class,
)
from src.distributed.expert_parallel.layers.bailing import EPBailingMoELayer
from src.distributed.expert_parallel.layers.glm4 import EPGlm4MoELayer
from src.distributed.expert_parallel.layers.qwen3_5 import EPQwen3_5MoELayer
from src.distributed.expert_parallel.layers.zaya import EPZayaMoELayer
from src.distributed.expert_parallel.saving import (
    _check_ep_merge_family_supported,
    _check_ep_sharded_save_supported,
    validate_ep_sharded_save,
)


def _fake_model(model_type):
    model = nn.Module()
    model.config = SimpleNamespace(model_type=model_type, auto_map=None)
    return model


def _ep_config(world_size, expert_tp_size=1):
    """ep_config satisfying the topology checks: a single global EP group, no expert TP."""
    return SimpleNamespace(expert_tp_size=expert_tp_size, ep_group_size=world_size, num_ep_groups=1)


def _fake_ep_layers(world_size, expert_tp_size=1):
    """One fake EP layer for the inner checker, which takes the layer list directly."""
    return [("model.layers.0.mlp", SimpleNamespace(ep_config=_ep_config(world_size, expert_tp_size)))]


class _FakeEPLayer(EPMoELayerBase):
    """A real ``EPMoELayerBase`` instance, which is what ``find_ep_layers`` keys on.

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
    monkeypatch.setattr(runtime, "_SHARED_FILESYSTEM_CONSENSUS", None)
    monkeypatch.setenv("DIST_SHARED_FILESYSTEM", "0")
    monkeypatch.setattr(ep_saving_mod, "get_local_world_size", lambda: ranks_per_node)


def test_ep_sharded_rejected_on_non_shared_multi_node_fs(monkeypatch):
    """Per-node local disks: the shards are keyed by GLOBAL rank and scatter across the nodes while
    each node's index names all of them, so merge_ep_shards.py never sees a complete set."""
    _non_shared_two_node_fs(monkeypatch)
    model = _fake_model("gpt_oss")
    model.mlp = _FakeEPLayer(world_size=16)
    with pytest.raises(ValueError, match="shared output filesystem"):
        validate_ep_sharded_save(model, world_size=16)


def test_ep_sharded_allowed_on_non_shared_single_node(monkeypatch):
    """Anti-over-rejection: one node holds every shard plus the index, so a local disk is complete."""
    _non_shared_two_node_fs(monkeypatch)
    model = _fake_model("gpt_oss")
    model.mlp = _FakeEPLayer(world_size=8)
    validate_ep_sharded_save(model, world_size=8)


def test_ep_sharded_rejects_merge_expert_lora_at_construction():
    """``merge_expert_lora_on_save`` is gathered-only, and must fail at trainer construction."""
    with pytest.raises(ValueError, match="gathered EP save"):
        validate_ep_sharded_save(_fake_model("gpt_oss"), world_size=1, merge_expert_lora_on_save=True)


def test_ep_sharded_rejected_without_ep_layers():
    """``save_sharded_ep=True`` on a run with no EP layers must not no-op: every save then silently
    falls through to the gathered writer while the merge step the user planned waits for shards that
    never exist. A set value that nothing reads is a config that does not do what it says."""
    with pytest.raises(ValueError, match="no EP MoE layers"):
        validate_ep_sharded_save(_fake_model("gpt_oss"), world_size=1)


def test_ep_sharded_allowed_with_ep_layers(monkeypatch):
    """Anti-over-rejection twin: the real EP shape the flag exists for must pass the same validator.
    Without it the guard above is satisfied by a validator that rejects EVERY run."""
    monkeypatch.setenv("DIST_SHARED_FILESYSTEM", "1")
    model = _fake_model("gpt_oss")
    model.mlp = _FakeEPLayer(world_size=8)
    validate_ep_sharded_save(model, world_size=8)


@pytest.mark.parametrize("model_type", ["gemma4_moe", ""])
def test_ep_sharded_rejects_unclaimed_model_type(model_type):
    """A model_type no EP layer class claims would train into unmergeable shards — must raise."""
    with pytest.raises(ValueError, match="no registered EP layer class"):
        _check_ep_merge_family_supported(_fake_model(model_type))


@pytest.mark.parametrize(
    "model_type",
    ["gpt_oss", "qwen3_moe", "qwen3_5_moe", "bailing_moe_linear", "glm4_moe_lite", "lfm2_moe", "zaya", "mistral4"],
)
def test_ep_sharded_accepts_mergeable_family(model_type):
    _check_ep_merge_family_supported(_fake_model(model_type))


_HUB_NAMESPACE_MODEL_TYPES = sorted(
    model_type for cls in ep_layer_classes() if cls._EXPORTS_HUB_NAMESPACE for model_type in cls.HF_MODEL_TYPES
)
assert _HUB_NAMESPACE_MODEL_TYPES, "no family declares _EXPORTS_HUB_NAMESPACE — the refusal below would be vacuous"


@pytest.mark.parametrize("model_type", _HUB_NAMESPACE_MODEL_TYPES)
def test_ep_sharded_rejects_hub_namespace_family(model_type):
    """A family whose gathered save writes the hub namespace through transformers' save-side revert
    cannot be merged from per-rank shards (the merge streams key by key), so the sharded save must be
    refused up front — under EVERY model_type spelling the family claims — rather than producing
    shards that merge into the module-tree spelling no serving engine reads."""
    with pytest.raises(ValueError, match="hub checkpoint namespace"):
        _check_ep_merge_family_supported(_fake_model(model_type))


def test_ep_sharded_family_gate_wired_into_topology_check(monkeypatch):
    """The shared checker (used by both the validator and _save_ep_sharded) rejects the family."""
    monkeypatch.setenv("DIST_SHARED_FILESYSTEM", "1")
    with pytest.raises(ValueError, match="no registered EP layer class"):
        _check_ep_sharded_save_supported(_fake_model("nope"), _fake_ep_layers(world_size=8), world_size=8)
    # A supported family with a valid single-group topology passes the same path.
    _check_ep_sharded_save_supported(_fake_model("gpt_oss"), _fake_ep_layers(world_size=8), world_size=8)


def test_ep_family_gate_is_skipped_when_the_layer_list_is_empty():
    """The INNER checker takes the EP layer list as an argument and is also called from the save
    path, where an empty list means "nothing to shard" — it must not judge the family there. The
    construction-time ``validate_ep_sharded_save`` above owns the "no EP layers at all" refusal."""
    _check_ep_sharded_save_supported(_fake_model("nope"), [], world_size=8)


def test_model_type_resolves_to_the_declaring_layer_class():
    assert resolve_ep_merge_layer_class("glm4_moe_lite") is EPGlm4MoELayer
    assert resolve_ep_merge_layer_class("bailing_moe_linear") is EPBailingMoELayer
    assert resolve_ep_merge_layer_class("bailing_moe") is EPBailingMoELayer
    # Qwen3.6 resolves through the 3.5 types it actually ships under, not a 3.6 spelling.
    assert resolve_ep_merge_layer_class("qwen3_5_moe") is EPQwen3_5MoELayer
    assert resolve_ep_merge_layer_class("qwen3_5_moe_text") is EPQwen3_5MoELayer
    assert resolve_ep_merge_layer_class("zaya") is EPZayaMoELayer


def test_laguna_inherits_the_glm4_transform_and_is_merge_supported():
    """Laguna has its OWN wrapper (its router normalizes the top-k weights, GLM-4's does not) but the
    identical expert layout, so it must inherit GLM-4's gather/merge rather than restate it — and
    keying the merge on the class must still make save_sharded_ep=True valid for it."""
    laguna_cls = resolve_ep_merge_layer_class("laguna")
    assert issubclass(laguna_cls, EPGlm4MoELayer) and laguna_cls is not EPGlm4MoELayer
    # Inherited, not overridden: the transform pair and the layout declaration are GLM-4's objects.
    assert laguna_cls.merge_shards_to_hf.__func__ is EPGlm4MoELayer.merge_shards_to_hf.__func__
    assert laguna_cls.gather_expert_state_dict is EPGlm4MoELayer.gather_expert_state_dict
    assert laguna_cls._PER_EXPERT_UNFUSED_KEYS == EPGlm4MoELayer._PER_EXPERT_UNFUSED_KEYS

    params = {"gate_up_proj": torch.randn(4, 8, 10), "down_proj": torch.randn(4, 5, 8)}
    laguna_merged = laguna_cls.merge_shards_to_hf("model.layers.0.mlp", params)
    glm4_merged = EPGlm4MoELayer.merge_shards_to_hf("model.layers.0.mlp", params)
    assert set(laguna_merged) == set(glm4_merged)
    for key, tensor in glm4_merged.items():
        assert torch.equal(laguna_merged[key], tensor), key

    _check_ep_merge_family_supported(_fake_model("laguna"))


def test_unclaimed_model_type_is_rejected_not_guessed():
    """No spelling heuristic: an unclaimed model_type must fail the gate rather than be routed onto
    some other family's expert layout."""
    assert resolve_ep_merge_layer_class("qwen3_next_moe") is None
    with pytest.raises(ValueError, match="no registered EP layer class"):
        _check_ep_merge_family_supported(_fake_model("qwen3_next_moe"))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
