"""The topology fingerprint and the OptimizerShardStore gate deciding restore vs warm restart vs torn raise.
Consensus outcomes that cannot occur single-process are simulated by monkeypatching ``rank_consensus``."""

import os

import pytest
import torch
import torch.nn as nn

import halo.checkpoint.parallel.optimizer as optimizer_mod
import halo.checkpoint.parallel.reading as coordination_mod
from halo.checkpoint.parallel.optimizer import OptimizerShardStore
from halo.checkpoint.parallel.optimizer_meta import OptimizerStateFingerprint
from halo.checkpoint.parallel.sidecars import restore_lr_scheduler
from halo.runtime.consensus import RankConsensus
from tests.common.checkpoint_io import load_context

# What a 1.1.0 save wrote to optimizer_meta.pt, verbatim: one rank, AdamW, no parallelism.
SAVED_BY_1_1_0 = {
    "num_ranks": 1,
    "fingerprint": {
        "world_size": 1,
        "ep_size": 1,
        "expert_tp_size": 1,
        "cp_size": 1,
        "tp_size": 1,
        "pp_size": 1,
        "fsdp_shard_ep1_experts": True,
        "optimizer_class": "AdamW",
        "ep_scope": "node",
        "use_grouped_gemm": True,
        "hsdp": False,
        "nvlink_domain_size": 1,
        "expert_replica_size": 1,
    },
}


def _stub_consensus(patcher, fake) -> None:
    """Stubs both modules that ask; stubbing one answers half the gates from a real single-process world."""
    patcher.setattr(optimizer_mod, "rank_consensus", fake)
    patcher.setattr(coordination_mod, "rank_consensus", fake)


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=False)


class _StubParallelismConfig:
    def __init__(
        self,
        ep_size=1,
        expert_tp_size=1,
        cp_size=1,
        tp_size=1,
        fsdp_shard_ep1_experts=True,
        ep_scope="node",
        use_grouped_gemm=True,
        use_hsdp=False,
        nvlink_domain_size=8,
    ):
        self.ep_size = ep_size
        self.expert_tp_size = expert_tp_size
        self.cp_size = cp_size
        self.tp_size = tp_size
        self.fsdp_shard_ep1_experts = fsdp_shard_ep1_experts
        self.ep_scope = ep_scope
        self.use_grouped_gemm = use_grouped_gemm
        self.use_hsdp = use_hsdp
        self.nvlink_domain_size = nvlink_domain_size
        self.is_pp_mode = False
        self.is_cp_mode = cp_size > 1
        self.is_tp_mode = tp_size > 1
        self.pp_size = 1
        self.max_concurrent_loading = None
        self.experts_fsdp_managed = fsdp_shard_ep1_experts and ep_size * expert_tp_size == 1
        self.num_ep_groups = 1


def _ctx(
    model,
    optimizer=None,
    *,
    lr_scheduler=None,
    parallelism_config=None,
    has_ep_layers=False,
    fsdp_wrapped=True,
    super_optim=None,
):
    return load_context(
        model,
        parallelism_config or _StubParallelismConfig(),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        has_ep_layers=has_ep_layers,
        fsdp_wrapped=fsdp_wrapped,
        super_load_optimizer_and_scheduler=super_optim or _Recorder(),
    )


def _write_checkpoint(tmp_path, *, shard=True, meta=True, fingerprint=None, num_ranks=1, osd=None):
    if shard:
        payload = osd if osd is not None else {"state": {}, "param_groups": []}
        torch.save(payload, os.path.join(tmp_path, "optimizer_shard_00000.pt"))
    if meta:
        payload = {"num_ranks": num_ranks}
        if fingerprint is not None:
            payload["fingerprint"] = fingerprint
        torch.save(payload, os.path.join(tmp_path, "optimizer_meta.pt"))


def _sgd(model):
    return torch.optim.SGD(model.parameters(), lr=0.1)


def _live_fp(optimizer, parallelism_config=None, world_size=1):
    return OptimizerStateFingerprint.capture(parallelism_config or _StubParallelismConfig(), optimizer, world_size)


def test_capture_maps_config_fields():
    pc = _StubParallelismConfig(ep_size=8, expert_tp_size=2, cp_size=1, tp_size=1, fsdp_shard_ep1_experts=False)
    fp = OptimizerStateFingerprint.capture(pc, torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1), 16)
    assert fp.world_size == 16
    assert fp.ep_size == 8
    assert fp.expert_tp_size == 2
    assert fp.fsdp_shard_ep1_experts is False
    assert fp.optimizer_class == "SGD"


def test_capture_raises_on_a_config_missing_a_field():
    """A default would land in the saved and the live fingerprint alike, so a renamed field would pass the gate."""
    pc = _StubParallelismConfig()
    del pc.ep_scope
    with pytest.raises(AttributeError, match="ep_scope"):
        OptimizerStateFingerprint.capture(pc, _sgd(_TinyModel()), 1)


def test_dict_round_trip():
    fp = _live_fp(_sgd(_TinyModel()), _StubParallelismConfig(ep_size=2, cp_size=2), world_size=4)
    assert fp.mismatches(fp.to_dict()) == []


def test_the_saved_fingerprint_keeps_the_keys_1_1_0_wrote():
    """A renamed key would warm-restart every existing shard set (``use_hsdp`` is saved as ``hsdp``)."""
    live = _live_fp(torch.optim.AdamW(_TinyModel().parameters()))
    assert list(live.to_dict()) == list(SAVED_BY_1_1_0["fingerprint"])


@pytest.mark.parametrize("key", [key for key in SAVED_BY_1_1_0["fingerprint"] if key != "expert_replica_size"])
def test_a_saved_fingerprint_missing_a_key_mismatches(key):
    """Any key but the replica size: it warm-restarts, or raises under PP, as a topology change does."""
    live = _live_fp(_sgd(_TinyModel()))
    saved = {name: value for name, value in live.to_dict().items() if name != key}
    assert live.mismatches(saved) == [f"{key}: saved='<absent>' current={live.to_dict()[key]!r}"]


def test_a_saved_field_the_live_run_lacks_mismatches():
    live = _live_fp(_sgd(_TinyModel()))
    assert live.mismatches({**live.to_dict(), "future_field": 123}) == ["future_field: saved=123 current='<absent>'"]


def test_shape_preserving_topology_knobs_are_fingerprinted():
    """They change slice ownership, param names or the mesh rank while every shape stays identical, so nothing
    downstream catches them."""
    sgd = _sgd(_TinyModel())
    live = _live_fp(sgd, _StubParallelismConfig(ep_size=8), world_size=16)
    for knob, other, key in (
        ("ep_scope", "global", "ep_scope"),
        ("use_grouped_gemm", False, "use_grouped_gemm"),
        ("use_hsdp", True, "hsdp"),
        ("nvlink_domain_size", 4, "nvlink_domain_size"),
    ):
        saved = _live_fp(sgd, _StubParallelismConfig(ep_size=8, **{knob: other}), world_size=16).to_dict()
        assert [m.split(":")[0] for m in live.mismatches(saved)] == [key], knob


def test_mismatches_names_fields_with_values():
    live = _live_fp(_sgd(_TinyModel()), _StubParallelismConfig(ep_size=1), world_size=2)
    mismatched = live.mismatches({**live.to_dict(), "ep_size": 2, "optimizer_class": "AdamWBF16"})
    assert len(mismatched) == 2
    assert any(m.startswith("ep_size:") and "saved=2" in m and "current=1" in m for m in mismatched)
    assert any(m.startswith("optimizer_class:") and "AdamWBF16" in m for m in mismatched)
    assert live.mismatches(live.to_dict()) == []


def test_matching_fingerprint_restores(tmp_path, monkeypatch):
    model = _TinyModel()
    optimizer = _sgd(model)
    _write_checkpoint(tmp_path, fingerprint=_live_fp(optimizer).to_dict())
    restore = _Recorder()
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", restore)

    OptimizerShardStore(_ctx(model, optimizer)).load(str(tmp_path))

    assert len(restore.calls) == 1


def test_a_meta_written_by_1_1_0_resumes_exactly(tmp_path, monkeypatch):
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters())
    _write_checkpoint(tmp_path, fingerprint=SAVED_BY_1_1_0["fingerprint"], num_ranks=SAVED_BY_1_1_0["num_ranks"])
    restore = _Recorder()
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", restore)

    OptimizerShardStore(_ctx(model, optimizer, parallelism_config=_StubParallelismConfig(nvlink_domain_size=1))).load(
        str(tmp_path)
    )

    assert len(restore.calls) == 1


def test_mismatched_fingerprint_warm_restarts_naming_fields(tmp_path, monkeypatch, caplog):
    model = _TinyModel()
    optimizer = _sgd(model)
    saved = {**_live_fp(optimizer).to_dict(), "ep_size": 2, "world_size": 2}
    _write_checkpoint(tmp_path, fingerprint=saved)
    restore = _Recorder()
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", restore)

    with caplog.at_level("WARNING", logger=optimizer_mod.logger.name):
        OptimizerShardStore(_ctx(model, optimizer)).load(str(tmp_path))

    assert restore.calls == []
    warning = "\n".join(r.getMessage() for r in caplog.records)
    assert "fingerprint mismatch" in warning
    assert "ep_size: saved=2 current=1" in warning
    assert "world_size: saved=2 current=1" in warning


def test_mismatched_optimizer_class_warm_restarts(tmp_path, monkeypatch, caplog):
    """A different optimizer class has a different state schema."""
    model = _TinyModel()
    optimizer = _sgd(model)
    _write_checkpoint(tmp_path, fingerprint={**_live_fp(optimizer).to_dict(), "optimizer_class": "AdamWBF16"})
    restore = _Recorder()
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", restore)

    with caplog.at_level("WARNING", logger=optimizer_mod.logger.name):
        OptimizerShardStore(_ctx(model, optimizer)).load(str(tmp_path))

    assert restore.calls == []
    assert "optimizer_class" in "\n".join(r.getMessage() for r in caplog.records)


def test_meta_without_fingerprint_refuses_the_resume(tmp_path, monkeypatch):
    """The rank-count gate passes for any save at the same world size, admitting the permuted restores the
    fingerprint exists to catch."""
    model = _TinyModel()
    optimizer = _sgd(model)
    _write_checkpoint(tmp_path, fingerprint=None)
    restore = _Recorder()
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", restore)

    with pytest.raises(RuntimeError, match="no topology fingerprint") as excinfo:
        OptimizerShardStore(_ctx(model, optimizer)).load(str(tmp_path))

    assert str(tmp_path) in str(excinfo.value)
    assert "optimizer_meta.pt" in str(excinfo.value)
    assert restore.calls == []


class _ModelWithGradFreeParam(nn.Module):
    """GptOss attention sinks under a sink-dropping kernel are the live instance."""

    def __init__(self, param_name: str):
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=False)
        setattr(self, param_name, nn.Parameter(torch.zeros(4)))


def _shard(*, stateful=("fc.weight",), tracked=("fc.weight",)):
    return {
        "state": {name: {"momentum_buffer": torch.zeros(1)} for name in stateful},
        "param_groups": [{"params": list(tracked), "lr": 0.1}],
    }


@pytest.mark.parametrize("param_name", ("sinks", "learned_scale"))
def test_a_param_the_shard_tracked_without_state_is_not_a_coverage_failure(tmp_path, monkeypatch, param_name):
    """Optimizers create state lazily, so a tracked-but-never-updated param is absent from ``state``; excluding by
    name would cover one family's spelling."""
    model = _ModelWithGradFreeParam(param_name)
    optimizer = _sgd(model)
    _write_checkpoint(
        tmp_path,
        osd=_shard(tracked=("fc.weight", param_name)),
        fingerprint=_live_fp(optimizer).to_dict(),
    )
    restore = _Recorder()
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", restore)

    OptimizerShardStore(_ctx(model, optimizer)).load(str(tmp_path))

    assert len(restore.calls) == 1


def test_a_param_the_shard_never_tracked_fails_the_coverage_gate(tmp_path, monkeypatch):
    """A param in neither state nor param_groups is real layout drift that ``strict=False`` would silently
    reinitialize."""
    model = _ModelWithGradFreeParam("sinks")
    optimizer = _sgd(model)
    _write_checkpoint(tmp_path, osd=_shard(), fingerprint=_live_fp(optimizer).to_dict())
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", _Recorder())

    with pytest.raises(RuntimeError, match="no saved optimizer state"):
        OptimizerShardStore(_ctx(model, optimizer)).load(str(tmp_path))


def test_no_shards_no_meta_warm_restarts(tmp_path, monkeypatch, caplog):
    model = _TinyModel()
    optimizer = _sgd(model)
    restore = _Recorder()
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", restore)

    with caplog.at_level("WARNING", logger=optimizer_mod.logger.name):
        OptimizerShardStore(_ctx(model, optimizer, has_ep_layers=True)).load(str(tmp_path))

    assert restore.calls == []
    assert "No optimizer shards" in "\n".join(r.getMessage() for r in caplog.records)


def test_partial_shards_with_matching_fingerprint_raise_torn(tmp_path, monkeypatch):
    """Restoring moments on some ranks only would silently diverge replicas."""
    model = _TinyModel()
    optimizer = _sgd(model)
    _write_checkpoint(tmp_path, shard=False, fingerprint=_live_fp(optimizer).to_dict())
    _stub_consensus(monkeypatch, lambda local: RankConsensus(local, True))

    with pytest.raises(RuntimeError, match="torn"):
        OptimizerShardStore(_ctx(model, optimizer)).load(str(tmp_path))


def test_partial_shards_without_a_meta_are_refused_as_torn(tmp_path, monkeypatch):
    """The save stopped between shards and meta. The torn gate runs ahead of the rank-count gate."""
    model = _TinyModel()
    optimizer = _sgd(model)
    _write_checkpoint(tmp_path, shard=False, meta=False)
    # Peer-has-a-shard only for the shard-presence probe; later consensus stays single-process.
    seen_negative = []

    def fake_consensus(local):
        if not local and not seen_negative:
            seen_negative.append(True)
            return RankConsensus(False, True)
        return RankConsensus(local, local)

    _stub_consensus(monkeypatch, fake_consensus)
    restore = _Recorder()
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", restore)

    with pytest.raises(RuntimeError, match="torn set of an interrupted save"):
        OptimizerShardStore(_ctx(model, optimizer)).load(str(tmp_path))

    assert restore.calls == []


def test_all_shards_missing_under_matched_meta_raises_permutation(tmp_path, monkeypatch):
    """The complete set is merely elsewhere (a restart permuted rank→node); warm-restarting would silently reset
    Adam moments."""
    model = _TinyModel()
    optimizer = _sgd(model)
    _write_checkpoint(tmp_path, shard=False, fingerprint=_live_fp(optimizer).to_dict())
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", _Recorder())

    with pytest.raises(RuntimeError, match="optimizer state WAS written"):
        OptimizerShardStore(_ctx(model, optimizer)).load(str(tmp_path))


def test_other_ranks_shard_files_raise_permutation_without_fingerprint(tmp_path, monkeypatch):
    """No rank holds its own shard, so the unfingerprinted gate stays quiet; another rank's file proves the state
    exists."""
    model = _TinyModel()
    optimizer = _sgd(model)
    _write_checkpoint(tmp_path, shard=False, fingerprint=None)
    torch.save({"state": {}, "param_groups": []}, os.path.join(tmp_path, "optimizer_shard_00003.pt"))
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", _Recorder())

    with pytest.raises(RuntimeError, match="rank→node placement"):
        OptimizerShardStore(_ctx(model, optimizer)).load(str(tmp_path))


@pytest.mark.parametrize("mode", ["ep", "cp"])
def test_ep_and_cp_modes_restore_optimizer_state(tmp_path, monkeypatch, mode):
    model = _TinyModel()
    optimizer = _sgd(model)
    pc = _StubParallelismConfig(ep_size=2 if mode == "ep" else 1, cp_size=2 if mode == "cp" else 1)
    _write_checkpoint(tmp_path, fingerprint=_live_fp(optimizer, pc).to_dict())
    restore = _Recorder()
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", restore)

    ctx = _ctx(model, optimizer, parallelism_config=pc, has_ep_layers=mode == "ep", fsdp_wrapped=True)
    OptimizerShardStore(ctx).load(str(tmp_path))

    assert len(restore.calls) == 1


def test_non_fsdp_ep_falls_through_to_base_trainer(tmp_path):
    """Optimizer state is replicated there, so the base Trainer's optimizer.pt path restores it."""
    model = _TinyModel()
    optimizer = _sgd(model)
    super_optim = _Recorder()
    ctx = _ctx(model, optimizer, has_ep_layers=True, fsdp_wrapped=False, super_optim=super_optim)

    OptimizerShardStore(ctx).load(str(tmp_path))

    assert len(super_optim.calls) == 1
    assert super_optim.calls[0][0][0] == str(tmp_path)


def test_save_optimizer_shards_writes_fingerprint_meta(tmp_path):
    model = _TinyModel()
    optimizer = _sgd(model)
    model.fc.weight.grad = torch.zeros_like(model.fc.weight)
    optimizer.step()
    pc = _StubParallelismConfig(ep_size=2, fsdp_shard_ep1_experts=False)

    OptimizerShardStore(_ctx(model, optimizer, parallelism_config=pc)).save(str(tmp_path))

    assert os.path.isfile(os.path.join(tmp_path, "optimizer_shard_00000.pt"))
    meta = torch.load(os.path.join(tmp_path, "optimizer_meta.pt"), weights_only=False)
    assert meta["fingerprint"] == OptimizerStateFingerprint.capture(pc, optimizer, 1).to_dict()
    assert meta["num_ranks"] == 1


def test_save_then_load_round_trip_gate(tmp_path, monkeypatch):
    model = _TinyModel()
    optimizer = _sgd(model)
    model.fc.weight.grad = torch.zeros_like(model.fc.weight)
    optimizer.step()
    pc_save = _StubParallelismConfig(ep_size=2)
    OptimizerShardStore(_ctx(model, optimizer, parallelism_config=pc_save)).save(str(tmp_path))

    restore = _Recorder()
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", restore)

    OptimizerShardStore(_ctx(model, optimizer, parallelism_config=pc_save)).load(str(tmp_path))
    assert len(restore.calls) == 1

    # EP=2 checkpoint resumed at EP=1 warm-restarts, so no second restore.
    pc_resume = _StubParallelismConfig(ep_size=1)
    OptimizerShardStore(_ctx(model, optimizer, parallelism_config=pc_resume)).load(str(tmp_path))
    assert len(restore.calls) == 1


def _stepped(model, optimizer):
    model.fc.weight.grad = torch.zeros_like(model.fc.weight)
    optimizer.step()


def test_low_host_ram_preflight_warns_but_never_blocks_the_save(tmp_path, monkeypatch, caplog):
    """Aborting a checkpoint over an estimate would be worse than the OOM. fc.weight is 16 fp32 params, so the
    two-moment bound is 128 bytes; 127 must warn."""
    model = _TinyModel()
    optimizer = _sgd(model)
    _stepped(model, optimizer)
    monkeypatch.setattr(optimizer_mod, "available_host_ram_bytes", lambda: 127)

    with caplog.at_level("WARNING", logger=optimizer_mod.logger.name):
        OptimizerShardStore(_ctx(model, optimizer)).save(str(tmp_path))

    assert os.path.isfile(os.path.join(tmp_path, "optimizer_shard_00000.pt"))
    assert "MemAvailable" in "\n".join(r.getMessage() for r in caplog.records)


def test_ample_host_ram_preflight_is_silent(tmp_path, monkeypatch, caplog):
    """Exactly the estimate available is not below it."""
    model = _TinyModel()
    optimizer = _sgd(model)
    _stepped(model, optimizer)
    monkeypatch.setattr(optimizer_mod, "available_host_ram_bytes", lambda: 128)

    with caplog.at_level("WARNING", logger=optimizer_mod.logger.name):
        OptimizerShardStore(_ctx(model, optimizer)).save(str(tmp_path))

    assert "MemAvailable" not in "\n".join(r.getMessage() for r in caplog.records)


def test_restore_preflight_warns_without_blocking_the_restore(tmp_path, monkeypatch, caplog):
    model = _TinyModel()
    optimizer = _sgd(model)
    _write_checkpoint(tmp_path, fingerprint=_live_fp(optimizer).to_dict())
    restore = _Recorder()
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", restore)
    monkeypatch.setattr(optimizer_mod, "available_host_ram_bytes", lambda: 1)

    with caplog.at_level("WARNING", logger=optimizer_mod.logger.name):
        OptimizerShardStore(_ctx(model, optimizer)).load(str(tmp_path))

    assert len(restore.calls) == 1
    assert "MemAvailable" in "\n".join(r.getMessage() for r in caplog.records)


def test_a_peer_without_the_fingerprint_raises_on_every_rank(tmp_path, monkeypatch):
    """``optimizer_meta.pt`` is written once per node; a peer that raises never reaches the collectives below,
    leaving this rank blocked until the NCCL watchdog."""
    model = _TinyModel()
    optimizer = _sgd(model)
    _write_checkpoint(tmp_path, fingerprint=_live_fp(optimizer).to_dict())
    asked: list[bool] = []

    def peer_lacks_the_fingerprint(local):
        # Calls 1-3 are the shard-presence probe and the meta's torn and rank-count gates; call 4 is the fingerprint
        # question, the only one the peer answers differently.
        asked.append(local)
        return RankConsensus(False, True) if len(asked) == 4 else RankConsensus(local, local)

    monkeypatch.setattr(optimizer_mod, "rank_consensus", peer_lacks_the_fingerprint)
    restore = _Recorder()
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", restore)

    with pytest.raises(RuntimeError, match="no topology fingerprint"):
        OptimizerShardStore(_ctx(model, optimizer)).load(str(tmp_path))

    assert asked[3] is True, "this rank's own meta carried a fingerprint; only the peer's did not"
    assert restore.calls == []


def test_a_rank_that_cannot_produce_optimizer_state_fails_the_save(tmp_path, monkeypatch):
    """The caller rotates the previous checkpoint away on return, so a skip trades it for one with no optimizer
    state at exit code 0; FlashAdamW on uneven DTensors takes this path."""
    model = _TinyModel()
    optimizer = _sgd(model)
    _stepped(model, optimizer)

    def refuse(*args, **kwargs):
        raise RuntimeError("FlashAdamW: unevenly-sharded DTensors have no state_dict")

    monkeypatch.setattr(optimizer_mod, "get_optimizer_state_dict", refuse)

    with pytest.raises(RuntimeError, match="save_only_model"):
        OptimizerShardStore(_ctx(model, optimizer, parallelism_config=_StubParallelismConfig())).save(str(tmp_path))

    # A half-written shard set is what the resume gate would call torn.
    assert not os.path.isfile(os.path.join(tmp_path, "optimizer_shard_00000.pt"))
    assert not os.path.isfile(os.path.join(tmp_path, "optimizer_meta.pt"))


def _scheduler(optimizer, *, gamma=0.5):
    """Warmup-free: the shape whose step-0 LR is the full base LR."""
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: gamma**step)


def test_resume_applies_the_restored_schedule_lr_to_the_optimizer(tmp_path):
    """``load_state_dict`` updates only the scheduler and HF steps the optimizer first, so ``param_groups`` stays on
    the step-0 LR for one step."""
    saved_model = _TinyModel()
    saved_optimizer = torch.optim.SGD(saved_model.parameters(), lr=0.1)
    saved_scheduler = _scheduler(saved_optimizer)
    for _ in range(3):
        saved_scheduler.step()
    trained_lr = saved_optimizer.param_groups[0]["lr"]
    assert trained_lr != pytest.approx(0.1), "premise: the schedule must have moved off its step-0 LR"
    torch.save(saved_scheduler.state_dict(), os.path.join(tmp_path, "scheduler.pt"))

    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = _scheduler(optimizer)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1), "premise: a fresh run starts at step-0 LR"

    assert restore_lr_scheduler(scheduler, optimizer, str(tmp_path))

    assert scheduler.last_epoch == 3
    assert optimizer.param_groups[0]["lr"] == pytest.approx(trained_lr), (
        "the resumed run's first step would use the schedule's step-0 learning rate"
    )


def test_restored_schedule_lr_is_applied_per_param_group(tmp_path):
    """An optimizer with a decoupled group would otherwise be flattened onto the first group's LR."""
    model = _TinyModel()
    groups = [{"params": [model.fc.weight], "lr": 0.1}, {"params": [torch.nn.Parameter(torch.zeros(2))], "lr": 0.01}]
    saved_optimizer = torch.optim.SGD(groups)
    saved_scheduler = _scheduler(saved_optimizer)
    for _ in range(2):
        saved_scheduler.step()
    trained = [g["lr"] for g in saved_optimizer.param_groups]
    torch.save(saved_scheduler.state_dict(), os.path.join(tmp_path, "scheduler.pt"))

    live_model = _TinyModel()
    live_groups = [
        {"params": [live_model.fc.weight], "lr": 0.1},
        {"params": [torch.nn.Parameter(torch.zeros(2))], "lr": 0.01},
    ]
    optimizer = torch.optim.SGD(live_groups)
    scheduler = _scheduler(optimizer)

    restore_lr_scheduler(scheduler, optimizer, str(tmp_path))

    assert [g["lr"] for g in optimizer.param_groups] == pytest.approx(trained)
    assert trained[0] != pytest.approx(trained[1]), "premise: the groups must carry distinct LRs"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
