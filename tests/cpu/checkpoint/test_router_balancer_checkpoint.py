#!/usr/bin/env python
"""Router balancing-bias checkpoint save/restore round-trip.

The DeepSeek-V3 bias-update balancer (``RouterBiasBalancingCallback``) accumulates a per-expert routing
correction that is a plain attribute on each EP/router module — invisible to FSDP2 and therefore absent
from the model checkpoint. ``DistributedTrainerMixin._persist_router_balancing_biases`` /
``_restore_router_balancing_biases`` carry it across a resume so a preempted MoE run keeps its balance
instead of re-imbalancing from zero. This exercises that round-trip on a fake module tree (no GPU/dist:
the rank helpers resolve to the single-process "save rank" / "main process").

    python tests/cpu/checkpoint/test_router_balancer_checkpoint.py
"""

import os
import tempfile
import types

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate import PartialState

import src.distributed.checkpoint.coordination as coordination
from src.models.moe_balancing import apply_router_balancing_sidecar
from src.trainers.mixins.base import DistributedTrainerMixin

# The mixin logs through accelerate's logger, which requires an initialized state.
PartialState()

_persist = DistributedTrainerMixin._persist_router_balancing_biases
_restore = DistributedTrainerMixin._restore_router_balancing_biases


class _Router(nn.Module):
    def __init__(self, num_experts: int):
        super().__init__()
        # Plain attribute, exactly like the real balancer (NOT a registered buffer/param).
        self.balancing_biases = torch.zeros(num_experts, dtype=torch.float32)


class _MoEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Router(4), _Router(4), _Router(4)])
        self.dense = nn.Linear(2, 2)  # a non-router module — must be ignored


def _fake_self(model):
    # is_pp_mode=False keeps the non-PP write path; the PP branch is collective (GPU-tested).
    me = types.SimpleNamespace(model=model, parallelism_config=types.SimpleNamespace(is_pp_mode=False))
    # Borrowed, not reimplemented: a stub unwrap could pass while the real one is broken.
    me._top_level_model = types.MethodType(DistributedTrainerMixin._top_level_model, me)
    return me


def test_persist_restore_roundtrip():
    model = _MoEModel()
    me = _fake_self(model)
    for i, router in enumerate(model.layers):
        router.balancing_biases.copy_(torch.arange(4, dtype=torch.float32) + i)
    saved = [r.balancing_biases.clone() for r in model.layers]

    with tempfile.TemporaryDirectory() as d:
        _persist(me, d)
        assert os.path.isfile(os.path.join(d, "router_balancing_biases.pt")), "biases file not written"

        # Zero first (the fresh model on resume), or the restore assert below is vacuous.
        for router in model.layers:
            router.balancing_biases.zero_()

        _restore(me, d)

    for router, want in zip(model.layers, saved, strict=True):
        assert torch.equal(router.balancing_biases, want), "bias not restored to its saved value"


def test_persist_noop_without_routers():
    # A dense model has no balancing router → nothing written (the flag is a no-op there).
    me = _fake_self(nn.Linear(3, 3))
    with tempfile.TemporaryDirectory() as d:
        _persist(me, d)
        assert not os.path.isfile(os.path.join(d, "router_balancing_biases.pt"))


def test_restore_noop_when_file_absent():
    # Resuming a checkpoint that predates balancing (no file) must not raise and must leave biases as-is.
    model = _MoEModel()
    original = [r.balancing_biases.clone() for r in model.layers]
    with tempfile.TemporaryDirectory() as d:
        _restore(_fake_self(model), d)
    for router, want in zip(model.layers, original, strict=True):
        assert torch.equal(router.balancing_biases, want)


def test_restore_raises_uniformly_when_a_peer_is_missing_the_sidecar(monkeypatch):
    """Present on some ranks only is a torn save, and every rank must raise on it.

    The rank whose copy is absent still ENTERS the presence consensus: short-circuiting there would
    leave it warm-starting from zero-init biases while its peers restore — a permanent cross-node
    routing divergence — and strand the peers in the readability consensus behind it.
    """
    entered = []

    def _consensus(local_flag):
        entered.append(local_flag)
        return False, True  # a peer holds the file, this rank does not

    monkeypatch.setattr(coordination, "rank_consensus", _consensus)
    model = _MoEModel()
    with tempfile.TemporaryDirectory() as d, pytest.raises(RuntimeError, match="present on some ranks"):
        _restore(_fake_self(model), d)
    assert entered == [False], "the rank without the file skipped the presence consensus"


def test_restore_raises_when_a_peer_could_not_read_the_sidecar(monkeypatch):
    """A readable local copy is not enough: one torn copy anywhere fails the whole world."""
    monkeypatch.setattr(coordination, "all_ranks_ok", lambda ok: False)
    model = _MoEModel()
    with tempfile.TemporaryDirectory() as d:
        torch.save({"layers.0": torch.ones(4)}, os.path.join(d, "router_balancing_biases.pt"))
        with pytest.raises(RuntimeError, match="unreadable on at least one rank"):
            _restore(_fake_self(model), d)


def test_restore_skips_unknown_module_name():
    # A saved name absent from the current model (architecture drift) is skipped, not fatal.
    model = _MoEModel()
    me = _fake_self(model)
    for i, router in enumerate(model.layers):
        router.balancing_biases.copy_(torch.full((4,), float(i + 1)))
    with tempfile.TemporaryDirectory() as d:
        _persist(me, d)
        path = os.path.join(d, "router_balancing_biases.pt")
        blob = torch.load(path, weights_only=False)
        blob["layers.99.does_not_exist"] = torch.ones(4)
        torch.save(blob, path)
        for router in model.layers:
            router.balancing_biases.zero_()
        _restore(me, d)  # must not raise
    for i, router in enumerate(model.layers):
        assert torch.equal(router.balancing_biases, torch.full((4,), float(i + 1)))


def test_restore_warns_when_the_sidecar_matches_no_live_router(caplog):
    """A sidecar full of trained biases and a model carrying none is the silent-loss shape.

    Under weight-sync RL both bias modes are downgraded to ``none`` (the sync ships parameters only),
    so no module holds the balancing attribute: the restore loop matches nothing, ``restored`` and
    ``missing`` are both empty, and the trained biases are dropped with no log line at all. The RL
    leg's own saves then emit no sidecar, so resuming BACK into a bias_update run restarts balancing
    from zero.
    """
    with tempfile.TemporaryDirectory() as d:
        torch.save({"layers.0": torch.ones(4)}, os.path.join(d, "router_balancing_biases.pt"))
        with caplog.at_level("WARNING"):
            _restore(_fake_self(nn.Linear(3, 3)), d)  # a tree with no balancing router at all
    assert any("DROPPED" in record.message for record in caplog.records), (
        "a sidecar that matched no live router was dropped in total silence"
    )


def test_restore_raises_on_a_broadcastable_shape_mismatch():
    """``copy_`` BROADCASTS: a size-1 saved entry would fill an [E] bias with one value and route
    every token by it. Only a non-broadcastable shape fails on its own, so the shape must be checked."""
    model = _MoEModel()
    with tempfile.TemporaryDirectory() as d:
        torch.save(
            # Keyed by MODULE name, as the writer keys it.
            {f"layers.{i}": torch.tensor([0.5]) for i in range(3)},
            os.path.join(d, "router_balancing_biases.pt"),
        )
        with pytest.raises(RuntimeError, match="shape"):
            _restore(_fake_self(model), d)
    for router in model.layers:
        assert not torch.equal(router.balancing_biases, torch.full((4,), 0.5)), (
            "the size-1 saved bias was broadcast across the whole expert dimension"
        )


# apply_router_balancing_sidecar: an offline PEFT merge starts from BASE weights, so the trained
# bias must be copied from the sidecar into the merged HUB model's native slots.


class _HubGptOssRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(4))


class _HubGptOssBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = _HubGptOssRouter()


class _HubGptOssModel(nn.Module):
    """Hub-tree stand-in whose config resolves the GptOss EP class (native slot 'router.bias')."""

    def __init__(self):
        super().__init__()
        self.config = type("Cfg", (), {"model_type": "gpt_oss"})()
        self.layers = nn.ModuleList([_HubGptOssBlock(), _HubGptOssBlock()])


def test_sidecar_applies_into_native_slots_with_peft_prefix():
    model = _HubGptOssModel()
    sidecar = {
        "base_model.model.layers.0": torch.full((4,), 0.25),
        "base_model.model.layers.1": torch.full((4,), -0.5),
    }
    applied, skipped = apply_router_balancing_sidecar(model, sidecar)
    assert applied == ["layers.0", "layers.1"] and not skipped
    assert torch.equal(model.layers[0].router.bias.data, torch.full((4,), 0.25))
    assert torch.equal(model.layers[1].router.bias.data, torch.full((4,), -0.5))


class _HubGptOssCausalLM(nn.Module):
    """Hub tree at the usual ``model.layers.N`` depth — the shape a CP-wrapped run's sidecar names
    one further level down (``model.model.layers.N``), and two under PEFT+CP."""

    def __init__(self):
        super().__init__()
        self.config = type("Cfg", (), {"model_type": "gpt_oss"})()
        self.model = _HubGptOssModel()


def test_sidecar_applies_under_peft_plus_context_parallelism():
    """PEFT+CP names every module one level deeper again: the CP wrapper's inner-model attribute is
    kept by ``unwrap_framework_wrappers``, so the sidecar records
    ``base_model.model.model.model.layers.N``. Stripping only the PEFT prefix leaves
    ``model.model.layers.N``, which resolves against no hub tree — every PEFT+CP run trained a bias
    whose merge then died on a KeyError."""
    model = _HubGptOssCausalLM()
    sidecar = {
        "base_model.model.model.model.layers.0": torch.full((4,), 0.25),
        "base_model.model.model.model.layers.1": torch.full((4,), -0.5),
    }
    applied, skipped = apply_router_balancing_sidecar(model, sidecar)
    assert applied == ["model.layers.0", "model.layers.1"] and not skipped
    assert torch.equal(model.model.layers[0].router.bias.data, torch.full((4,), 0.25))
    assert torch.equal(model.model.layers[1].router.bias.data, torch.full((4,), -0.5))


def test_sidecar_applies_under_context_parallelism_without_peft():
    """CP alone adds the same extra ``model.`` level."""
    model = _HubGptOssCausalLM()
    applied, _ = apply_router_balancing_sidecar(model, {"model.model.layers.0": torch.full((4,), 1.5)})
    assert applied == ["model.layers.0"]
    assert torch.equal(model.model.layers[0].router.bias.data, torch.full((4,), 1.5))


def test_sidecar_applies_into_hub_native_buffer_without_ep_class():
    router = nn.Module()
    router.register_buffer("balancing_biases", torch.zeros(4))
    model = nn.Module()
    model.config = type("Cfg", (), {"model_type": "not_registered"})()
    model.router = router
    applied, skipped = apply_router_balancing_sidecar(model, {"router": torch.full((4,), 2.0)})
    assert applied == ["router"] and not skipped
    assert torch.equal(router.balancing_biases, torch.full((4,), 2.0))


class _HubLfm2Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4, 8))
        self.use_expert_bias = False


class _HubLfm2Block(nn.Module):
    """``use_expert_bias: false`` hub block: NO ``expert_bias`` buffer until materialized."""

    def __init__(self):
        super().__init__()
        self.gate = _HubLfm2Gate()


class _HubLfm2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Cfg", (), {"model_type": "lfm2_moe", "use_expert_bias": False})()
        self.layers = nn.ModuleList([_HubLfm2Block()])


def test_sidecar_materializes_config_gated_slot_on_flagless_base():
    """A PEFT run saves no config.json, so a merge rebuilds the base with LFM-2's slot gate still
    off — the apply must re-materialize the slot, flip the gate on tree + config, and copy the
    bias, or the merged model silently serves pretrained routing while the tool blames the
    architecture for having no slot."""
    model = _HubLfm2Model()
    bias = torch.full((4,), 0.75)
    applied, skipped = apply_router_balancing_sidecar(model, {"base_model.model.layers.0": bias})
    assert applied == ["layers.0"] and not skipped
    block = model.layers[0]
    assert torch.equal(block.expert_bias, bias)
    assert block.expert_bias.dtype == torch.float32
    assert "expert_bias" in block.state_dict(), "slot must be a PERSISTENT buffer so the save carries it"
    assert block.gate.use_expert_bias is True, "live routing must consult the materialized slot"
    assert model.config.use_expert_bias is True, "exported config must tell engines to load the slot"


def test_sidecar_skips_transient_only_modules():
    """A family with no native slot trained a transient bias no artifact can serve — reported as
    skipped so the merge tools warn instead of silently dropping it."""
    model = nn.Module()
    model.config = type("Cfg", (), {"model_type": "qwen3_moe"})()
    model.block = nn.Linear(2, 2)
    applied, skipped = apply_router_balancing_sidecar(model, {"block": torch.zeros(4)})
    assert not applied and skipped == ["block"]


def test_sidecar_shape_mismatch_raises():
    model = _HubGptOssModel()
    with pytest.raises(ValueError, match="shape"):
        apply_router_balancing_sidecar(model, {"layers.0": torch.zeros(8)})


def test_sidecar_unknown_module_raises():
    model = _HubGptOssModel()
    with pytest.raises(KeyError, match="different model tree"):
        apply_router_balancing_sidecar(model, {"layers.7": torch.zeros(4)})


class _StageModel(_MoEModel):
    """A PP stage: local router names, and the map back to the unsplit model's names."""

    def global_parameter_name(self, name: str) -> str:
        return f"model.{name}"


def test_pp_biases_are_gathered_to_one_rank_not_to_every_rank(tmp_path, monkeypatch):
    """Under PP the stages' biases are merged into ONE sidecar under unsplit names — by a gather to
    rank 0 plus a broadcast of the merge, never an all-gather.

    Only the FS-aware save rank(s) consume the merge, so an all-gather makes every rank materialize
    one unpickled bias dict per rank: 512 of them at world 512, each carrying every router's
    expert-count-sized tensor, for a write one rank per node performs.
    """
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    monkeypatch.setattr(
        dist, "all_gather_object", lambda *a, **k: pytest.fail("the bias merge must not all-gather its dicts")
    )
    try:
        model = _StageModel()
        me = _fake_self(model)
        me.parallelism_config = types.SimpleNamespace(is_pp_mode=True)
        for index, router in enumerate(model.layers):
            router.balancing_biases.copy_(torch.arange(4, dtype=torch.float32) + index)

        _persist(me, str(tmp_path))
        saved = torch.load(tmp_path / "router_balancing_biases.pt", weights_only=True)
    finally:
        dist.destroy_process_group()

    assert set(saved) == {f"model.layers.{i}" for i in range(len(model.layers))}, saved.keys()
    for index, router in enumerate(model.layers):
        assert torch.equal(saved[f"model.layers.{index}"], router.balancing_biases)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
