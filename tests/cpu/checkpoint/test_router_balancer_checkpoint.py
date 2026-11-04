#!/usr/bin/env python
"""Router balancing-bias save/restore: a plain attribute on each EP/router module, absent from the model checkpoint."""

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

import halo.checkpoint.parallel.reading as coordination
from halo.checkpoint.parallel.sidecars import restore_balancing_biases, save_balancing_biases
from halo.models.moe.balancing import apply_balancing_bias_sidecar
from halo.runtime.consensus import RankConsensus


class _Router(nn.Module):
    def __init__(self, num_experts: int):
        super().__init__()
        # A plain attribute, like the real balancer.
        self.balancing_biases = torch.zeros(num_experts, dtype=torch.float32)


class _MoEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Router(4), _Router(4), _Router(4)])
        self.dense = nn.Linear(2, 2)  # Must be ignored.


def test_persist_restore_roundtrip():
    model = _MoEModel()
    for i, router in enumerate(model.layers):
        router.balancing_biases.copy_(torch.arange(4, dtype=torch.float32) + i)
    saved = [r.balancing_biases.clone() for r in model.layers]

    with tempfile.TemporaryDirectory() as d:
        save_balancing_biases(model, d, is_pp_mode=False)
        assert os.path.isfile(os.path.join(d, "router_balancing_biases.pt")), "biases file not written"

        # Zero first, or the restore assert below is vacuous.
        for router in model.layers:
            router.balancing_biases.zero_()

        restore_balancing_biases(model, d, is_pp_mode=False)

    for router, want in zip(model.layers, saved, strict=True):
        assert torch.equal(router.balancing_biases, want), "bias not restored to its saved value"


def test_persist_noop_without_routers():
    # A dense model has no balancing router, so nothing is written.
    with tempfile.TemporaryDirectory() as d:
        save_balancing_biases(nn.Linear(3, 3), d, is_pp_mode=False)
        assert not os.path.isfile(os.path.join(d, "router_balancing_biases.pt"))


def test_restore_noop_when_file_absent():
    # A checkpoint predating balancing must not raise and must leave biases as-is.
    model = _MoEModel()
    original = [r.balancing_biases.clone() for r in model.layers]
    with tempfile.TemporaryDirectory() as d:
        restore_balancing_biases(model, d, is_pp_mode=False)
    for router, want in zip(model.layers, original, strict=True):
        assert torch.equal(router.balancing_biases, want)


def test_restore_raises_uniformly_when_a_peer_is_missing_the_sidecar(monkeypatch):
    """Short-circuiting the presence consensus leaves one rank on zero-init biases, a permanent routing divergence,
    and strands peers."""
    entered = []

    def _consensus(local_flag):
        entered.append(local_flag)
        return RankConsensus(False, True)  # A peer holds the file, this rank does not.

    monkeypatch.setattr(coordination, "rank_consensus", _consensus)
    model = _MoEModel()
    with tempfile.TemporaryDirectory() as d, pytest.raises(RuntimeError, match="present on some ranks"):
        restore_balancing_biases(model, d, is_pp_mode=False)
    assert entered == [False], "the rank without the file skipped the presence consensus"


def test_restore_raises_when_a_peer_could_not_read_the_sidecar(monkeypatch):
    # Present on every rank (the presence join), unreadable on a peer (the read join).
    verdicts = iter([RankConsensus(True, True), RankConsensus(False, True)])
    monkeypatch.setattr(coordination, "rank_consensus", lambda ok: next(verdicts))
    model = _MoEModel()
    with tempfile.TemporaryDirectory() as d:
        torch.save({"layers.0": torch.ones(4)}, os.path.join(d, "router_balancing_biases.pt"))
        with pytest.raises(RuntimeError, match="unreadable on at least one rank"):
            restore_balancing_biases(model, d, is_pp_mode=False)


def test_restore_skips_unknown_module_name():
    # Architecture drift is skipped, not fatal.
    model = _MoEModel()
    for i, router in enumerate(model.layers):
        router.balancing_biases.copy_(torch.full((4,), float(i + 1)))
    with tempfile.TemporaryDirectory() as d:
        save_balancing_biases(model, d, is_pp_mode=False)
        path = os.path.join(d, "router_balancing_biases.pt")
        blob = torch.load(path, weights_only=False)
        blob["layers.99.does_not_exist"] = torch.ones(4)
        torch.save(blob, path)
        for router in model.layers:
            router.balancing_biases.zero_()
        restore_balancing_biases(model, d, is_pp_mode=False)
    for i, router in enumerate(model.layers):
        assert torch.equal(router.balancing_biases, torch.full((4,), float(i + 1)))


def test_restore_warns_when_the_sidecar_matches_no_live_router(caplog):
    """Under weight-sync RL both bias modes downgrade to ``none``, so the loop matches nothing and trained biases
    drop with no log line."""
    with tempfile.TemporaryDirectory() as d:
        torch.save({"layers.0": torch.ones(4)}, os.path.join(d, "router_balancing_biases.pt"))
        with caplog.at_level("WARNING"):
            restore_balancing_biases(nn.Linear(3, 3), d, is_pp_mode=False)  # No balancing router at all.
    assert any("DROPPED" in record.message for record in caplog.records), (
        "a sidecar that matched no live router was dropped in total silence"
    )


def test_restore_raises_on_a_broadcastable_shape_mismatch():
    """``copy_`` broadcasts: a size-1 entry would fill an [E] bias with one value and route every token by it."""
    model = _MoEModel()
    with tempfile.TemporaryDirectory() as d:
        torch.save(
            # Keyed by module name, as the writer keys it.
            {f"layers.{i}": torch.tensor([0.5]) for i in range(3)},
            os.path.join(d, "router_balancing_biases.pt"),
        )
        with pytest.raises(RuntimeError, match="shape"):
            restore_balancing_biases(model, d, is_pp_mode=False)
    for router in model.layers:
        assert not torch.equal(router.balancing_biases, torch.full((4,), 0.5)), (
            "the size-1 saved bias was broadcast across the whole expert dimension"
        )


# An offline PEFT merge starts from base weights, so the bias is copied into the merged hub model's native slots.


class _HubGptOssRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(4))


class _HubGptOssBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = _HubGptOssRouter()


class _HubGptOssModel(nn.Module):
    """Its config resolves the GptOss EP class (native slot 'router.bias')."""

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
    applied, skipped = apply_balancing_bias_sidecar(model, sidecar)
    assert applied == ["layers.0", "layers.1"] and not skipped
    assert torch.equal(model.layers[0].router.bias.data, torch.full((4,), 0.25))
    assert torch.equal(model.layers[1].router.bias.data, torch.full((4,), -0.5))


class _HubGptOssCausalLM(nn.Module):
    """The shape a CP-wrapped run's sidecar names one level down, and two under PEFT+CP."""

    def __init__(self):
        super().__init__()
        self.config = type("Cfg", (), {"model_type": "gpt_oss"})()
        self.model = _HubGptOssModel()


def test_sidecar_applies_under_peft_plus_context_parallelism():
    """The sidecar records ``base_model.model.model.model.layers.N``; stripping only the PEFT prefix would die on a
    KeyError."""
    model = _HubGptOssCausalLM()
    sidecar = {
        "base_model.model.model.model.layers.0": torch.full((4,), 0.25),
        "base_model.model.model.model.layers.1": torch.full((4,), -0.5),
    }
    applied, skipped = apply_balancing_bias_sidecar(model, sidecar)
    assert applied == ["model.layers.0", "model.layers.1"] and not skipped
    assert torch.equal(model.model.layers[0].router.bias.data, torch.full((4,), 0.25))
    assert torch.equal(model.model.layers[1].router.bias.data, torch.full((4,), -0.5))


def test_sidecar_applies_under_context_parallelism_without_peft():
    model = _HubGptOssCausalLM()
    applied, _ = apply_balancing_bias_sidecar(model, {"model.model.layers.0": torch.full((4,), 1.5)})
    assert applied == ["model.layers.0"]
    assert torch.equal(model.model.layers[0].router.bias.data, torch.full((4,), 1.5))


def test_sidecar_applies_into_hub_native_buffer_without_ep_class():
    router = nn.Module()
    router.register_buffer("balancing_biases", torch.zeros(4))
    model = nn.Module()
    model.config = type("Cfg", (), {"model_type": "not_registered"})()
    model.router = router
    applied, skipped = apply_balancing_bias_sidecar(model, {"router": torch.full((4,), 2.0)})
    assert applied == ["router"] and not skipped
    assert torch.equal(router.balancing_biases, torch.full((4,), 2.0))


class _HubLfm2Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4, 8))
        self.use_expert_bias = False


class _HubLfm2Block(nn.Module):
    """No ``expert_bias`` buffer until materialized."""

    def __init__(self):
        super().__init__()
        self.gate = _HubLfm2Gate()


class _HubLfm2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Cfg", (), {"model_type": "lfm2_moe", "use_expert_bias": False})()
        self.layers = nn.ModuleList([_HubLfm2Block()])


def test_sidecar_materializes_config_gated_slot_on_flagless_base():
    """A PEFT run saves no config.json, so the merge rebuilds with the gate off; unapplied, the model serves
    pretrained routing."""
    model = _HubLfm2Model()
    bias = torch.full((4,), 0.75)
    applied, skipped = apply_balancing_bias_sidecar(model, {"base_model.model.layers.0": bias})
    assert applied == ["layers.0"] and not skipped
    block = model.layers[0]
    assert torch.equal(block.expert_bias, bias)
    assert block.expert_bias.dtype == torch.float32
    assert "expert_bias" in block.state_dict(), "slot must be a PERSISTENT buffer so the save carries it"
    assert block.gate.use_expert_bias is True, "live routing must consult the materialized slot"
    assert model.config.use_expert_bias is True, "exported config must tell engines to load the slot"


def test_sidecar_skips_transient_only_modules():
    """Reported as skipped so the merge tools warn instead of silently dropping it."""
    model = nn.Module()
    model.config = type("Cfg", (), {"model_type": "qwen3_moe"})()
    model.block = nn.Linear(2, 2)
    applied, skipped = apply_balancing_bias_sidecar(model, {"block": torch.zeros(4)})
    assert not applied and skipped == ["block"]


def test_sidecar_shape_mismatch_raises():
    model = _HubGptOssModel()
    with pytest.raises(ValueError, match="shape"):
        apply_balancing_bias_sidecar(model, {"layers.0": torch.zeros(8)})


def test_sidecar_unknown_module_raises():
    model = _HubGptOssModel()
    with pytest.raises(KeyError, match="different model tree"):
        apply_balancing_bias_sidecar(model, {"layers.7": torch.zeros(4)})


class _StageModel(_MoEModel):
    def global_parameter_name(self, name: str) -> str:
        return f"model.{name}"


def test_pp_biases_are_gathered_to_one_rank_not_to_every_rank(tmp_path, monkeypatch):
    """An all-gather makes every rank hold one unpickled bias dict per rank, 512 at world 512, for a write one rank
    per node performs."""
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    monkeypatch.setattr(
        dist, "all_gather_object", lambda *a, **k: pytest.fail("the bias merge must not all-gather its dicts")
    )
    try:
        model = _StageModel()
        for index, router in enumerate(model.layers):
            router.balancing_biases.copy_(torch.arange(4, dtype=torch.float32) + index)

        save_balancing_biases(model, str(tmp_path), is_pp_mode=True)
        saved = torch.load(tmp_path / "router_balancing_biases.pt", weights_only=True)
    finally:
        dist.destroy_process_group()

    assert set(saved) == {f"model.layers.{i}" for i in range(len(model.layers))}, saved.keys()
    for index, router in enumerate(model.layers):
        assert torch.equal(saved[f"model.layers.{index}"], router.balancing_biases)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
