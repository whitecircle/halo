"""CPU tests for optimizer-state continuity on resume: the topology fingerprint and the
OptimizerShardStore gate that decides restore vs warm-restart vs torn-raise.

Per-rank optimizer shards restore only into the exact topology that wrote them, so
``optimizer_meta.pt`` carries an :class:`OptimizerStateFingerprint` and the load path branches:

- fingerprint matches → full restore (EP/CP included);
- fingerprint mismatch → warm restart with a warning naming the mismatched fields;
- shards present with no fingerprint in the meta (a pre-fingerprint checkpoint) → raise;
- shards missing everywhere with no trace they were written → warm restart (save_only_model);
- shards missing everywhere while other ranks' shard files or a fingerprint-matched meta prove
  state WAS written (a wholesale rank→node permutation on a non-shared FS) → raise;
- shards missing on a subset under a matching fingerprint → torn checkpoint → raise;
- non-sharded modes (no mixin FSDP2, no pure TP — replicated optimizer state) → base Trainer path.

Consensus outcomes that cannot occur single-process are simulated by monkeypatching the module's
``rank_consensus`` / ``all_ranks_ok`` helpers.
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

import src.distributed.checkpoint.coordination as coordination_mod
import src.distributed.checkpoint.optimizer as optimizer_mod
from src.distributed.checkpoint.context import CheckpointLoadContext
from src.distributed.checkpoint.fingerprint import OptimizerStateFingerprint
from src.distributed.checkpoint.optimizer import OptimizerShardStore


def _stub_consensus(patcher, fake) -> None:
    """Stub the consensus the resume gates ask, in BOTH modules that ask it: the direct
    ``rank_consensus`` calls and the shared ``all_ranks_ok`` helper the gates go through. Stubbing
    one alone answers half the gates from a real single-process world and the rest from the stub."""
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
        hsdp=False,
        nvlink_domain_size=8,
    ):
        self.ep_size = ep_size
        self.expert_tp_size = expert_tp_size
        self.cp_size = cp_size
        self.tp_size = tp_size
        self.fsdp_shard_ep1_experts = fsdp_shard_ep1_experts
        self.ep_scope = ep_scope
        self.use_grouped_gemm = use_grouped_gemm
        self.use_hsdp = hsdp
        self.nvlink_domain_size = nvlink_domain_size


def _ctx(
    model,
    optimizer=None,
    *,
    lr_scheduler=None,
    parallelism_config=None,
    is_cp_mode=False,
    has_ep_layers=False,
    fsdp_wrapped=True,
    super_optim=None,
):
    return CheckpointLoadContext(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        parallelism_config=parallelism_config,
        is_pp_mode=False,
        is_cp_mode=is_cp_mode,
        is_tp_mode=False,
        has_ep_layers=has_ep_layers,
        fsdp_wrapped=fsdp_wrapped,
        tp_rank=0,
        tp_size=1,
        super_load_from_checkpoint=_Recorder(),
        super_load_optimizer_and_scheduler=super_optim or _Recorder(),
    )


def _write_checkpoint(tmp_path, *, shard=True, meta=True, fingerprint=None, num_ranks=1, osd=None):
    """Write a minimal optimizer checkpoint: rank-0 shard and/or meta (with optional fingerprint)."""
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
    return OptimizerStateFingerprint.capture(parallelism_config, optimizer, world_size)


def test_capture_maps_config_fields():
    pc = _StubParallelismConfig(ep_size=8, expert_tp_size=2, cp_size=1, tp_size=1, fsdp_shard_ep1_experts=False)
    fp = OptimizerStateFingerprint.capture(pc, torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1), 16)
    assert fp.world_size == 16
    assert fp.ep_size == 8
    assert fp.expert_tp_size == 2
    assert fp.fsdp_shard_ep1_experts is False
    assert fp.optimizer_class == "SGD"


def test_capture_without_parallelism_config_defaults():
    fp = OptimizerStateFingerprint.capture(None, torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1), 2)
    assert (fp.ep_size, fp.expert_tp_size, fp.cp_size, fp.tp_size) == (1, 1, 1, 1)
    assert fp.fsdp_shard_ep1_experts is True


def test_dict_round_trip():
    fp = _live_fp(_sgd(_TinyModel()), _StubParallelismConfig(ep_size=2, cp_size=2), world_size=4)
    assert OptimizerStateFingerprint.from_dict(fp.to_dict()) == fp


@pytest.mark.parametrize("payload", [None, "not-a-dict", 7, {}, {"world_size": 2}, {"ep_size": 1}])
def test_from_dict_rejects_incomplete_payloads(payload):
    """Anything short of a complete fingerprint dict is a pre-fingerprint/foreign meta → None, which
    the resume gate refuses."""
    assert OptimizerStateFingerprint.from_dict(payload) is None


def test_from_dict_ignores_unknown_keys():
    data = {**_live_fp(_sgd(_TinyModel())).to_dict(), "future_field": 123}
    assert OptimizerStateFingerprint.from_dict(data) is not None


def test_shape_preserving_topology_knobs_are_fingerprinted():
    """These four change WHICH rank owns which slice (ep_scope, nvlink_domain_size), what the expert
    params are NAMED (use_grouped_gemm), or the FSDP mesh rank (hsdp) while every tensor shape stays
    identical — so nothing downstream can catch them. Without them in the fingerprint the restore
    succeeds and reports success on permuted or silently reinitialized state."""
    sgd = _sgd(_TinyModel())
    live = _live_fp(sgd, _StubParallelismConfig(ep_size=8), world_size=16)
    for field, other in (
        ("ep_scope", "global"),
        ("use_grouped_gemm", False),
        ("hsdp", True),
        ("nvlink_domain_size", 4),
    ):
        saved = OptimizerStateFingerprint.from_dict({**live.to_dict(), field: other})
        assert saved is not None, field
        assert any(m.startswith(f"{field}:") for m in live.mismatches(saved)), field


def test_mismatches_names_fields_with_values():
    live = _live_fp(_sgd(_TinyModel()), _StubParallelismConfig(ep_size=1), world_size=2)
    saved = OptimizerStateFingerprint.from_dict({**live.to_dict(), "ep_size": 2, "optimizer_class": "AdamWBF16"})
    assert saved is not None
    mismatched = live.mismatches(saved)
    assert len(mismatched) == 2
    assert any(m.startswith("ep_size:") and "saved=2" in m and "current=1" in m for m in mismatched)
    assert any(m.startswith("optimizer_class:") and "AdamWBF16" in m for m in mismatched)
    assert live.mismatches(live) == []


def test_matching_fingerprint_restores(tmp_path, monkeypatch):
    model = _TinyModel()
    optimizer = _sgd(model)
    _write_checkpoint(tmp_path, fingerprint=_live_fp(optimizer).to_dict())
    restore = _Recorder()
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", restore)

    OptimizerShardStore(_ctx(model, optimizer)).load(str(tmp_path))

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
    """Shards written by a different optimizer class carry a different state schema — never load
    them into the live optimizer."""
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
    """A num_ranks-only meta (a pre-fingerprint checkpoint) must RAISE, never restore.

    The rank-count gate it would otherwise fall back to passes for any save at the same world size,
    whatever its ep_scope / use_grouped_gemm / optimizer class — i.e. it admits exactly the permuted
    and silently-reinitialized restores the fingerprint exists to catch. The message must name the
    checkpoint and the deletion opt-in, since there is no warm-restart arm to fall into.
    """
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
    """A trainable parameter that never receives a gradient, so no optimizer ever gives it state.

    GptOss attention sinks under a sink-dropping kernel are the live instance; the parameter is
    named through the constructor because nothing about the property is tied to that spelling.
    """

    def __init__(self, param_name: str):
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=False)
        setattr(self, param_name, nn.Parameter(torch.zeros(4)))


def _shard(*, stateful=("fc.weight",), tracked=("fc.weight",)):
    """An optimizer shard in ``get_optimizer_state_dict``'s shape: FQN-keyed state + FQN param_groups."""
    return {
        "state": {name: {"momentum_buffer": torch.zeros(1)} for name in stateful},
        "param_groups": [{"params": list(tracked), "lr": 0.1}],
    }


@pytest.mark.parametrize("param_name", ("sinks", "learned_scale"))
def test_a_param_the_shard_tracked_without_state_is_not_a_coverage_failure(tmp_path, monkeypatch, param_name):
    """The gate must read the shard's own param_groups, not a hardcoded family parameter name.

    An optimizer creates state lazily on the first gradient, so a tracked-but-never-updated
    parameter is simply absent from ``state`` — with nothing to restore and no drift to report.
    Excluding it by NAME puts per-family knowledge in a generic path and covers exactly one
    family's spelling: any other grad-free parameter aborts every resume with a false drift report.
    """
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
    """The other half: a parameter in NEITHER state nor param_groups is a real layout drift (a
    rename, a different stage split) that ``strict=False`` would silently reinitialize."""
    model = _ModelWithGradFreeParam("sinks")
    optimizer = _sgd(model)
    _write_checkpoint(tmp_path, osd=_shard(), fingerprint=_live_fp(optimizer).to_dict())
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", _Recorder())

    with pytest.raises(RuntimeError, match="no saved optimizer state"):
        OptimizerShardStore(_ctx(model, optimizer)).load(str(tmp_path))


def test_no_shards_no_meta_warm_restarts(tmp_path, monkeypatch, caplog):
    """A checkpoint carrying no optimizer state at all (save_only_model) resumes with a warm restart,
    not an error — no shard exists to be gated."""
    model = _TinyModel()
    optimizer = _sgd(model)
    restore = _Recorder()
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", restore)

    with caplog.at_level("WARNING", logger=optimizer_mod.logger.name):
        OptimizerShardStore(_ctx(model, optimizer, has_ep_layers=True)).load(str(tmp_path))

    assert restore.calls == []
    assert "No optimizer shards" in "\n".join(r.getMessage() for r in caplog.records)


def test_partial_shards_with_matching_fingerprint_raise_torn(tmp_path, monkeypatch):
    """Fingerprint match ⇒ a complete shard set was written; my shard missing while a peer has one
    is a torn copy — restoring moments on some ranks only would silently diverge replicas."""
    model = _TinyModel()
    optimizer = _sgd(model)
    _write_checkpoint(tmp_path, shard=False, fingerprint=_live_fp(optimizer).to_dict())
    # Peer reports its shard present: consensus (all_ok=local, any_ok=True).
    _stub_consensus(monkeypatch, lambda local: (local, True))

    with pytest.raises(RuntimeError, match="torn"):
        OptimizerShardStore(_ctx(model, optimizer)).load(str(tmp_path))


def test_partial_shards_without_a_meta_warm_restart(tmp_path, monkeypatch, caplog):
    """A peer holds a shard but there is no meta at all: the rank-count gate fails first, so this is
    a warm restart, not the unfingerprinted-shards raise (nothing here can be gated either way)."""
    model = _TinyModel()
    optimizer = _sgd(model)
    _write_checkpoint(tmp_path, shard=False, meta=False)
    # Peer-has-a-shard only for the shard-presence probe; later consensus stays single-process.
    seen_negative = []

    def fake_consensus(local):
        if not local and not seen_negative:
            seen_negative.append(True)
            return (False, True)
        return (local, local)

    _stub_consensus(monkeypatch, fake_consensus)
    restore = _Recorder()
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", restore)

    with caplog.at_level("WARNING", logger=optimizer_mod.logger.name):
        OptimizerShardStore(_ctx(model, optimizer)).load(str(tmp_path))

    assert restore.calls == []


def test_all_shards_missing_under_matched_meta_raises_permutation(tmp_path, monkeypatch):
    """No rank sees its own shard, but the meta's fingerprint MATCHED this run — the complete shard
    set was written and is merely elsewhere (on a non-shared FS: a restart permuted the rank→node
    placement wholesale). Warm-restarting here silently reset Adam moments while weights, step and
    LR schedule resumed."""
    model = _TinyModel()
    optimizer = _sgd(model)
    _write_checkpoint(tmp_path, shard=False, fingerprint=_live_fp(optimizer).to_dict())
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", _Recorder())

    with pytest.raises(RuntimeError, match="optimizer state WAS written"):
        OptimizerShardStore(_ctx(model, optimizer)).load(str(tmp_path))


def test_other_ranks_shard_files_raise_permutation_without_fingerprint(tmp_path, monkeypatch):
    """The glob signal alone: no rank holds its OWN shard, so the unfingerprinted-shards gate does
    not fire, but another rank's shard file sitting in this node's checkpoint dir proves the state
    exists — misplaced, not absent."""
    model = _TinyModel()
    optimizer = _sgd(model)
    _write_checkpoint(tmp_path, shard=False, fingerprint=None)
    torch.save({"state": {}, "param_groups": []}, os.path.join(tmp_path, "optimizer_shard_00003.pt"))
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", _Recorder())

    with pytest.raises(RuntimeError, match="rank→node placement"):
        OptimizerShardStore(_ctx(model, optimizer)).load(str(tmp_path))


@pytest.mark.parametrize("mode_kwargs", [{"has_ep_layers": True}, {"is_cp_mode": True}])
def test_ep_and_cp_modes_restore_optimizer_state(tmp_path, monkeypatch, mode_kwargs):
    model = _TinyModel()
    optimizer = _sgd(model)
    pc = _StubParallelismConfig(ep_size=2 if "has_ep_layers" in mode_kwargs else 1)
    _write_checkpoint(tmp_path, fingerprint=_live_fp(optimizer, pc).to_dict())
    restore = _Recorder()
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", restore)

    ctx = _ctx(model, optimizer, parallelism_config=pc, fsdp_wrapped=True, **mode_kwargs)
    OptimizerShardStore(ctx).load(str(tmp_path))

    assert len(restore.calls) == 1


def test_non_fsdp_ep_falls_through_to_base_trainer(tmp_path):
    """ep_size==1 grouped-GEMM MoE without mixin FSDP2 (single GPU / replicated DDP): optimizer
    state is replicated, so the base Trainer's optimizer.pt path restores it (the pre-continuity
    code skipped restore entirely)."""
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
    saved_fp = OptimizerStateFingerprint.from_dict(meta["fingerprint"])
    assert saved_fp == OptimizerStateFingerprint.capture(pc, optimizer, 1)
    assert meta["num_ranks"] == 1


def test_save_then_load_round_trip_gate(tmp_path, monkeypatch):
    """A shard set written by the shard save passes the resume gate of the same topology and
    is rejected (warm restart) by a different one."""
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

    # Different topology (EP=2 checkpoint resumed at EP=1) → warm restart, so no second restore.
    pc_resume = _StubParallelismConfig(ep_size=1)
    OptimizerShardStore(_ctx(model, optimizer, parallelism_config=pc_resume)).load(str(tmp_path))
    assert len(restore.calls) == 1


def _stepped(model, optimizer):
    """One zero-grad step so the optimizer holds state to save."""
    model.fc.weight.grad = torch.zeros_like(model.fc.weight)
    optimizer.step()


def test_low_host_ram_preflight_warns_but_never_blocks_the_save(tmp_path, monkeypatch, caplog):
    """The host copy (cpu_offload) can OOM-kill the run, so the preflight must warn — and only warn:
    aborting a checkpoint over an estimate would be worse than the OOM it predicts. fc.weight is 16
    fp32 params, so the two-fp32-moment upper bound is 128 bytes; 127 available must warn (a sizing
    shrunk below 8 B/param stops warning here and fails the test)."""
    model = _TinyModel()
    optimizer = _sgd(model)
    _stepped(model, optimizer)
    monkeypatch.setattr(optimizer_mod, "available_host_ram_bytes", lambda: 127)

    with caplog.at_level("WARNING", logger=optimizer_mod.logger.name):
        OptimizerShardStore(_ctx(model, optimizer)).save(str(tmp_path))

    assert os.path.isfile(os.path.join(tmp_path, "optimizer_shard_00000.pt"))
    assert "MemAvailable" in "\n".join(r.getMessage() for r in caplog.records)


def test_ample_host_ram_preflight_is_silent(tmp_path, monkeypatch, caplog):
    """Exactly the estimate available (128 bytes for the 16-param fp32 model) is not "below" — the
    warning must not cry wolf on every healthy save."""
    model = _TinyModel()
    optimizer = _sgd(model)
    _stepped(model, optimizer)
    monkeypatch.setattr(optimizer_mod, "available_host_ram_bytes", lambda: 128)

    with caplog.at_level("WARNING", logger=optimizer_mod.logger.name):
        OptimizerShardStore(_ctx(model, optimizer)).save(str(tmp_path))

    assert "MemAvailable" not in "\n".join(r.getMessage() for r in caplog.records)


def test_restore_preflight_warns_without_blocking_the_restore(tmp_path, monkeypatch, caplog):
    """The restore-side twin: ``torch.load`` of the shard materializes the same bytes in host RAM,
    and the warning must not turn a legitimate restore into a failure."""
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
    """The unfingerprinted-shards verdict is the ``all`` of the world, not a rank-local read.

    ``optimizer_meta.pt`` is written once per NODE, so a non-shared filesystem can hold a complete
    meta on one node and a pre-fingerprint one on another. This rank's own meta is complete and its
    shard is present — it still must raise, because the peer that raises never reaches the
    collectives below and would leave this rank blocked in one until the NCCL watchdog kills the job.
    """
    model = _TinyModel()
    optimizer = _sgd(model)
    _write_checkpoint(tmp_path, fingerprint=_live_fp(optimizer).to_dict())
    asked: list[bool] = []

    def peer_lacks_the_fingerprint(local):
        # Call 1 is the shard-presence probe; call 2 is the fingerprint question, the only one the
        # peer answers differently.
        asked.append(local)
        return (False, True) if len(asked) == 2 else (local, local)

    monkeypatch.setattr(optimizer_mod, "rank_consensus", peer_lacks_the_fingerprint)
    restore = _Recorder()
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", restore)

    with pytest.raises(RuntimeError, match="no topology fingerprint"):
        OptimizerShardStore(_ctx(model, optimizer)).load(str(tmp_path))

    assert asked[1] is True, "this rank's own meta carried a fingerprint; only the peer's did not"
    assert restore.calls == []


def test_a_rank_that_cannot_produce_optimizer_state_fails_the_save(tmp_path, monkeypatch):
    """An unproducible optimizer state must RAISE, not skip the write.

    The caller deletes the base Trainer's optimizer.pt and rotates the previous checkpoint away as
    soon as this returns, so a skip trades the last checkpoint that HAS optimizer state for one that
    has none — at exit code 0. Reachable in production: an optimizer whose ``state_dict`` refuses the
    live sharding (FlashAdamW on unevenly-sharded DTensors) takes this path on EVERY save.
    """
    model = _TinyModel()
    optimizer = _sgd(model)
    _stepped(model, optimizer)

    def refuse(*args, **kwargs):
        raise RuntimeError("FlashAdamW: unevenly-sharded DTensors have no state_dict")

    monkeypatch.setattr(optimizer_mod, "get_optimizer_state_dict", refuse)

    with pytest.raises(RuntimeError, match="save_only_model"):
        OptimizerShardStore(_ctx(model, optimizer, parallelism_config=_StubParallelismConfig())).save(str(tmp_path))

    # Nothing half-written either: a shard set the resume gate would have to call torn.
    assert not os.path.isfile(os.path.join(tmp_path, "optimizer_shard_00000.pt"))
    assert not os.path.isfile(os.path.join(tmp_path, "optimizer_meta.pt"))


def _scheduler(optimizer, *, gamma=0.5):
    """A warmup-FREE decaying schedule: the shape whose step-0 LR is the full base LR."""
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: gamma**step)


def test_resume_applies_the_restored_schedule_lr_to_the_optimizer(tmp_path):
    """The first resumed step must run at the schedule's LR, not at the step-0 one.

    ``LRScheduler.load_state_dict`` only updates the scheduler's own ``__dict__``, and HF steps the
    optimizer BEFORE the scheduler — so a restore that stops at the scheduler leaves ``param_groups``
    on the LR ``_initial_step()`` wrote there. Under a warmup that is ~0; on a warmup-free schedule
    resumed deep into decay it is the FULL base LR for one step.
    """
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

    assert OptimizerShardStore(_ctx(model, optimizer, lr_scheduler=scheduler)).restore_lr_scheduler(str(tmp_path))

    assert scheduler.last_epoch == 3
    assert optimizer.param_groups[0]["lr"] == pytest.approx(trained_lr), (
        "the resumed run's first step would use the schedule's step-0 learning rate"
    )


def test_restored_schedule_lr_is_applied_per_param_group(tmp_path):
    """Per group, in order: an optimizer with a decoupled group (no-decay params, a Muon scalar
    group) would otherwise have every group flattened onto the first one's LR."""
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

    OptimizerShardStore(_ctx(live_model, optimizer, lr_scheduler=scheduler)).restore_lr_scheduler(str(tmp_path))

    assert [g["lr"] for g in optimizer.param_groups] == pytest.approx(trained)
    assert trained[0] != pytest.approx(trained[1]), "premise: the groups must carry distinct LRs"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
