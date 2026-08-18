"""CPU tests for the rank-uniform resume decisions of CheckpointLoader and OptimizerShardStore.

Every file-readability decision that gates a collective (or a warm-restart) must be consensus'd —
a rank-local branch hangs peers in DTensor collectives or silently diverges optimizer state. These
tests exercise the single-process decision logic and, via monkeypatched consensus helpers, the
"a peer disagreed" outcomes that cannot occur single-process:

- Pure-TP weight resume: rank 0 probes readability and broadcasts the verdict; a negative verdict
  (torn file on rank 0) sends ALL ranks to the base-Trainer fallback — never a per-rank split.
- The constructed-from-checkpoint skip and the TP+DP refusal never consume the checkpoint, so
  they must be decided BEFORE the per-rank read — at 100B+ scale that read is a host OOM, not a
  slowdown. The genuine pure-TP reload distributes each tensor into the live DTensor placements.
- FSDP2 optimizer restore: the terminal ``set_optimizer_state_dict`` outcome is all-reduced; if ANY
  rank failed, ALL ranks warm-restart together.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from peft.utils import SAFETENSORS_WEIGHTS_NAME as ADAPTER_WEIGHTS_NAME
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

import src.distributed.checkpoint.loader as loader_mod
import src.distributed.checkpoint.optimizer as optimizer_mod
import src.distributed.checkpoint.peft as peft_mod
from src.distributed.checkpoint.context import CheckpointLoadContext
from src.distributed.checkpoint.fingerprint import OptimizerStateFingerprint
from src.distributed.checkpoint.loader import CheckpointLoader
from src.distributed.checkpoint.optimizer import OptimizerShardStore
from src.distributed.checkpoint.peft import restore_adapters


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=False)


class _HandSlicedModel(_TinyModel):
    """A GptOss-shaped TP model: one param sliced by hand (a per-rank head slice, no DTensor)."""

    _tp_sharded_non_dtensor = [("sinks", 0)]

    def __init__(self):
        super().__init__()
        self.sinks = nn.Parameter(torch.zeros(2))


def _ctx(
    model,
    optimizer=None,
    *,
    is_tp_mode=False,
    fsdp_wrapped=False,
    has_ep_layers=False,
    fallback=None,
    lr_scheduler=None,
    tp_rank=0,
):
    return CheckpointLoadContext(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        parallelism_config=None,
        is_pp_mode=False,
        is_cp_mode=False,
        is_tp_mode=is_tp_mode,
        has_ep_layers=has_ep_layers,
        fsdp_wrapped=fsdp_wrapped,
        tp_rank=tp_rank,
        tp_size=2 if is_tp_mode else 1,
        super_load_from_checkpoint=fallback or _Recorder(),
        super_load_optimizer_and_scheduler=_Recorder(),
    )


def test_tp_torn_checkpoint_falls_back_uniformly(tmp_path):
    """A torn model.safetensors must be caught by the rank-0 probe rather than raising out of the
    per-rank read, so every rank takes the base-Trainer fallback."""
    (tmp_path / "model.safetensors").write_bytes(b"not a safetensors file")
    model = _TinyModel()
    before = model.fc.weight.detach().clone()
    fallback = _Recorder()
    loader = CheckpointLoader(_ctx(model, is_tp_mode=True, fallback=fallback))

    loader.load_model(str(tmp_path), model)  # must not raise

    assert len(fallback.calls) == 1
    assert torch.equal(model.fc.weight, before)


def test_tp_negative_rank0_verdict_forces_fallback(tmp_path, monkeypatch):
    """Even with a locally-readable checkpoint, a negative rank-0 verdict (torn file on the
    broadcast source) must send THIS rank to the fallback too — without the consensus each rank
    decides on its own read and the world splits across branches."""
    model = _TinyModel()
    save_file({"fc.weight": torch.ones(4, 4)}, os.path.join(tmp_path, "model.safetensors"))
    monkeypatch.setattr(loader_mod, "broadcast_from_rank0", lambda local: False)
    fallback = _Recorder()
    loader = CheckpointLoader(_ctx(model, is_tp_mode=True, fallback=fallback))

    loader.load_model(str(tmp_path), model)

    assert len(fallback.calls) == 1
    assert not torch.equal(model.fc.weight, torch.ones(4, 4))


def test_tp_readable_checkpoint_loads(tmp_path):
    model = _TinyModel()
    save_file({"fc.weight": torch.ones(4, 4)}, os.path.join(tmp_path, "model.safetensors"))
    fallback = _Recorder()
    loader = CheckpointLoader(_ctx(model, is_tp_mode=True, fallback=fallback))

    loader.load_model(str(tmp_path), model)

    assert fallback.calls == []
    assert torch.equal(model.fc.weight, torch.ones(4, 4))


@pytest.fixture
def single_rank_gloo(tmp_path):
    """A 1-rank gloo group so the tests can hold REAL DTensor params (the TP+DP verdict keys on
    ``isinstance(p.data, DTensor)``, which no stub can satisfy)."""
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        yield
    finally:
        dist.destroy_process_group()


def _dtensor_model():
    """A TP shard as transformers' load leaves it: a ``Shard``-placed DTensor on the TP mesh."""
    mesh = init_device_mesh("cpu", (1,))
    model = _TinyModel()
    model.fc.weight = nn.Parameter(distribute_tensor(torch.zeros(4, 4), mesh, [Shard(0)]))
    return model


def _simulate_non_rank0(monkeypatch, decisions):
    """A non-zero rank's view: no local rank-0 work, rank 0's broadcast answers replayed in call
    order — (1) sharded-resume rejection, (2) readability probe, (3) constructed-from-checkpoint,
    and on the genuine load (4) rank 0's matched key count."""
    monkeypatch.setattr(loader_mod, "is_global_main_process", lambda: False)
    answers = iter(decisions)
    monkeypatch.setattr(loader_mod, "broadcast_from_rank0", lambda local: next(answers))


def test_tp_skip_verdict_reads_nothing_on_non_rank0(tmp_path, single_rank_gloo, monkeypatch):
    """TP resume of a model constructed FROM the checkpoint: the DTensor params already hold the
    weights, so a non-zero rank must decide the skip BEFORE reading the checkpoint — there is no
    file here at all, so a read would raise."""
    model = _dtensor_model()
    setattr(model, loader_mod.LOADED_WEIGHTS_FROM_ATTR, str(tmp_path))
    _simulate_non_rank0(monkeypatch, decisions=[False, True, True])
    loader = CheckpointLoader(_ctx(model, is_tp_mode=True))

    loader.load_model(str(tmp_path), model)

    assert torch.equal(model.fc.weight.to_local(), torch.zeros(4, 4))


def test_tp_dp_refuses_a_model_not_built_from_the_checkpoint_without_reading(tmp_path, single_rank_gloo, monkeypatch):
    """FSDP2 over TP is a 2-D placement ``distribute_tensor`` does not invert for packed projections,
    so TP+DP only ever resumes a model constructed from the checkpoint — refused before any read."""
    model = _dtensor_model()
    _simulate_non_rank0(monkeypatch, decisions=[False, True, False])
    loader = CheckpointLoader(_ctx(model, is_tp_mode=True, fsdp_wrapped=True))

    with pytest.raises(RuntimeError, match="constructed FROM the checkpoint"):
        loader.load_model(str(tmp_path), model)


def test_tp_dp_best_model_reload_is_refused_naming_the_export(tmp_path, single_rank_gloo):
    """The best-model reload can never be the constructed-from-checkpoint skip, so under TP+DP it is
    refused outright — the message must point at exporting the best checkpoint directly."""
    model = _dtensor_model()
    setattr(model, loader_mod.LOADED_WEIGHTS_FROM_ATTR, str(tmp_path))
    save_file({"fc.weight": torch.ones(4, 4)}, os.path.join(tmp_path, "model.safetensors"))
    loader = CheckpointLoader(_ctx(model, is_tp_mode=True, fsdp_wrapped=True))

    with pytest.raises(RuntimeError, match="load_best_model_at_end"):
        loader.load_model(str(tmp_path), model, for_best_model=True)


def test_tp_distributes_the_checkpoint_into_the_live_dtensor(tmp_path, single_rank_gloo, monkeypatch):
    """The genuine pure-TP load: a model NOT built from the checkpoint gets every tensor
    ``distribute_tensor``ed into its DTensor placement — the by-value proof the resume restores
    trained weights instead of skipping."""
    model = _dtensor_model()
    save_file({"fc.weight": torch.ones(4, 4)}, os.path.join(tmp_path, "model.safetensors"))
    _simulate_non_rank0(monkeypatch, decisions=[False, True, False, 1])
    loader = CheckpointLoader(_ctx(model, is_tp_mode=True))

    loader.load_model(str(tmp_path), model)

    assert torch.equal(model.fc.weight.to_local(), torch.ones(4, 4))


def test_tp_best_model_reload_reads_even_when_built_from_the_checkpoint(tmp_path, single_rank_gloo):
    """``load_best_model_at_end``: the live weights trained PAST the best checkpoint, so the
    constructed-from-checkpoint skip must not apply — or the export silently carries the LAST
    weights."""
    model = _dtensor_model()
    setattr(model, loader_mod.LOADED_WEIGHTS_FROM_ATTR, str(tmp_path))
    save_file({"fc.weight": torch.ones(4, 4)}, os.path.join(tmp_path, "model.safetensors"))
    loader = CheckpointLoader(_ctx(model, is_tp_mode=True))

    loader.load_model(str(tmp_path), model, for_best_model=True)

    assert torch.equal(model.fc.weight.to_local(), torch.ones(4, 4))


def test_tp_plain_non_rank0_still_reads_for_the_genuine_load(tmp_path, monkeypatch):
    """Anti-over-skipping: a replicated (plain) param is copied from the rank's own read, so a
    non-zero rank on the genuine load path must still read its node's copy."""
    model = _TinyModel()
    save_file({"fc.weight": torch.ones(4, 4)}, os.path.join(tmp_path, "model.safetensors"))
    _simulate_non_rank0(monkeypatch, decisions=[False, True, False, 1])
    loader = CheckpointLoader(_ctx(model, is_tp_mode=True))

    loader.load_model(str(tmp_path), model)

    assert torch.equal(model.fc.weight, torch.ones(4, 4))


def test_tp_hand_sliced_param_takes_its_rank_slice(tmp_path):
    """A param TP shards by hand (GptOss sinks) is no DTensor: the checkpoint holds every head and
    the live param one rank's slice, so the load must chunk it by ``tp_rank`` — a whole copy would
    shape-mismatch, and rank 0's slice on every rank would duplicate its heads."""
    model = _HandSlicedModel()
    save_file(
        {"fc.weight": torch.ones(4, 4), "sinks": torch.arange(4, dtype=torch.float32)},
        os.path.join(tmp_path, "model.safetensors"),
    )
    loader = CheckpointLoader(_ctx(model, is_tp_mode=True, tp_rank=1))

    loader.load_model(str(tmp_path), model)

    assert torch.equal(model.sinks, torch.tensor([2.0, 3.0]))
    assert torch.equal(model.fc.weight, torch.ones(4, 4))


def test_fsdp2_resume_with_no_matching_keys_raises(tmp_path):
    """``set_model_state_dict`` runs strict=False (GptOss sinks and other legitimately absent keys
    would abort the whole restore otherwise), and it reports a wholesale mismatch as `unexpected`
    with an EMPTY `missing_keys` — so without this gate the call is a silent no-op and training
    resumes from base weights while logging success."""
    model = _TinyModel()
    before = model.fc.weight.detach().clone()
    save_file({"totally.wrong.key": torch.ones(4, 4)}, os.path.join(tmp_path, "model.safetensors"))
    loader = CheckpointLoader(_ctx(model, fsdp_wrapped=True))

    with pytest.raises(RuntimeError, match="fewer than half of the"):
        loader.load_model(str(tmp_path), model)

    assert torch.equal(model.fc.weight, before)


class _ThreeLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4, bias=False)
        self.fc2 = nn.Linear(4, 4, bias=False)
        self.fc3 = nn.Linear(4, 4, bias=False)


def test_fsdp2_resume_below_half_coverage_raises(tmp_path):
    """A checkpoint matching SOME keys but fewer than half is a wrong-layout checkpoint, not a
    legitimate resume — strict=False would apply the matching fraction and silently keep base
    weights for the rest."""
    model = _ThreeLayerModel()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    save_file({"fc1.weight": torch.ones(4, 4)}, os.path.join(tmp_path, "model.safetensors"))
    loader = CheckpointLoader(_ctx(model, fsdp_wrapped=True))

    with pytest.raises(RuntimeError, match="fewer than half of the"):
        loader.load_model(str(tmp_path), model)

    for n, p in model.named_parameters():
        assert torch.equal(p, before[n])


def test_fsdp2_resume_majority_coverage_loads_and_keeps_absent(tmp_path):
    """Majority coverage loads: gaps like tied lm_head / task heads are real, so absent tensors
    keep their construction values (warned, not raised). Needs a (1-rank gloo) process group —
    unlike the raise paths, this one reaches the real ``set_model_state_dict``."""
    model = _ThreeLayerModel()
    before_fc3 = model.fc3.weight.detach().clone()
    save_file(
        {"fc1.weight": torch.ones(4, 4), "fc2.weight": torch.full((4, 4), 2.0)},
        os.path.join(tmp_path, "model.safetensors"),
    )
    loader = CheckpointLoader(_ctx(model, fsdp_wrapped=True))

    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path}/pg_init")
    try:
        loader.load_model(str(tmp_path), model)
    finally:
        dist.destroy_process_group()

    assert torch.equal(model.fc1.weight, torch.ones(4, 4))
    assert torch.equal(model.fc2.weight, torch.full((4, 4), 2.0))
    assert torch.equal(model.fc3.weight, before_fc3)


def test_fsdp2_resume_with_matching_keys_reaches_the_load(tmp_path, monkeypatch):
    """Anti-vacuity for the gate above: a checkpoint whose keys DO match must still reach
    ``set_model_state_dict`` with the full dict. (The real call needs an initialized process group —
    ``broadcast_from_rank0`` — so it is recorded rather than executed here.)"""
    model = _TinyModel()
    save_file({"fc.weight": torch.ones(4, 4)}, os.path.join(tmp_path, "model.safetensors"))
    applied = _Recorder()
    monkeypatch.setattr(loader_mod, "set_model_state_dict", applied)
    loader = CheckpointLoader(_ctx(model, fsdp_wrapped=True))

    loader.load_model(str(tmp_path), model)

    assert len(applied.calls) == 1
    assert set(applied.calls[0][0][1]) == {"fc.weight"}


def _optimizer_checkpoint(tmp_path, model, with_meta: bool = True):
    """A per-rank optimizer shard, with the topology meta a complete save always writes beside it."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    torch.save({"state": {}, "param_groups": []}, os.path.join(tmp_path, "optimizer_shard_00000.pt"))
    if with_meta:
        fingerprint = OptimizerStateFingerprint.capture(_ctx(model, optimizer).parallelism_config, optimizer, 1)
        torch.save(
            {"num_ranks": 1, "fingerprint": fingerprint.to_dict(), "pp_stage_partition": None},
            os.path.join(tmp_path, "optimizer_meta.pt"),
        )
    return optimizer


def test_optimizer_shards_without_meta_warm_restart(tmp_path, monkeypatch):
    """A shard with no ``optimizer_meta.pt`` cannot be gated — neither the rank-count nor the
    fingerprint check has anything to compare — so it must warm-restart, not load ungated.

    Reachable two ways: a kill in the window between the shard save's write and its
    meta write, and a user who follows "delete every optimizer_shard_*.pt AND optimizer_meta.pt" by
    halves. Loading ungated silently maps another topology's moments onto this run's params.
    """
    model = _TinyModel()
    optimizer = _optimizer_checkpoint(tmp_path, model, with_meta=False)

    restore = _Recorder()
    warm_restarts = _Recorder()
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", restore)
    monkeypatch.setattr(OptimizerShardStore, "_warm_restart", lambda self, ckpt, msg: warm_restarts(ckpt, msg))

    OptimizerShardStore(_ctx(model, optimizer, fsdp_wrapped=True)).load(str(tmp_path))

    assert len(restore.calls) == 0, "ungated shards were restored"
    assert len(warm_restarts.calls) == 1


def test_optimizer_peer_failure_forces_uniform_warm_restart(tmp_path, monkeypatch):
    """Local restore succeeds but a peer failed (consensus returns False): this rank must
    warm-restart too. Warm-restarting only on this rank's OWN exception leaves loaded moments here
    while the failing rank reinitializes — silent optimizer divergence."""
    model = _TinyModel()
    optimizer = _optimizer_checkpoint(tmp_path, model)

    consensus_calls = []

    def fake_all_ranks_ok(local_ok):
        # Calls 1-4 gate meta/fingerprint/readability/FQN coverage; the 5th is the restore outcome,
        # which a loader that fails to consensus it never makes.
        consensus_calls.append(local_ok)
        return local_ok if len(consensus_calls) < 5 else False

    restore = _Recorder()
    warm_restarts = _Recorder()
    monkeypatch.setattr(optimizer_mod, "all_ranks_ok", fake_all_ranks_ok)
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", restore)
    monkeypatch.setattr(OptimizerShardStore, "_warm_restart", lambda self, ckpt, msg: warm_restarts(ckpt, msg))

    OptimizerShardStore(_ctx(model, optimizer, fsdp_wrapped=True)).load(str(tmp_path))

    assert len(restore.calls) == 1
    assert len(consensus_calls) == 5  # the restore outcome WAS consensus'd
    assert len(warm_restarts.calls) == 1


def test_optimizer_local_failure_warm_restarts_without_raise(tmp_path, monkeypatch):
    model = _TinyModel()
    optimizer = _optimizer_checkpoint(tmp_path, model)

    def raising_restore(*args, **kwargs):
        raise RuntimeError("shard shape mismatch")

    warm_restarts = _Recorder()
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", raising_restore)
    monkeypatch.setattr(OptimizerShardStore, "_warm_restart", lambda self, ckpt, msg: warm_restarts(ckpt, msg))

    # Must not raise: the failure is caught, consensus'd (single-process → False), warm restart.
    OptimizerShardStore(_ctx(model, optimizer, fsdp_wrapped=True)).load(str(tmp_path))

    assert len(warm_restarts.calls) == 1
    assert "at least one rank" in warm_restarts.calls[0][0][1]


def test_a_non_sharded_resume_refuses_per_rank_optimizer_shards(tmp_path):
    """The base Trainer's loader recognizes only optimizer.pt, so routing a shard-carrying
    checkpoint to it restores NOTHING silently — weights, step and LR schedule resume while the
    Adam moments reset, with no warning anywhere. The gate must raise BEFORE the fallback runs."""
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    torch.save({"state": {}, "param_groups": []}, os.path.join(tmp_path, "optimizer_shard_00000.pt"))
    loader = OptimizerShardStore(_ctx(model, optimizer))  # fsdp_wrapped=False → the base-Trainer route

    with pytest.raises(RuntimeError, match="optimizer_shard"):
        loader.load(str(tmp_path))

    assert loader.ctx.super_load_optimizer_and_scheduler.calls == [], "the silent HF route must not be reached"


def test_a_meta_only_checkpoint_is_refused_too(tmp_path):
    """The load side treats meta-without-shards as a failed gate; the non-sharded route must refuse
    the same artifact rather than silently restore nothing from it."""
    (tmp_path / "optimizer_meta.pt").write_bytes(b"x")
    loader = OptimizerShardStore(_ctx(_TinyModel()))

    with pytest.raises(RuntimeError, match="optimizer_shard"):
        loader.load(str(tmp_path))


def test_a_shard_free_checkpoint_still_takes_the_base_route(tmp_path):
    """Anti-over-rejection: an ordinary optimizer.pt checkpoint (and a None resume) must keep
    falling through to the base Trainer untouched."""
    model = _TinyModel()
    torch.save({}, os.path.join(tmp_path, "optimizer.pt"))
    loader = OptimizerShardStore(_ctx(model, torch.optim.SGD(model.parameters(), lr=0.1)))

    loader.load(str(tmp_path))
    loader.load(None)

    assert len(loader.ctx.super_load_optimizer_and_scheduler.calls) == 2


def test_a_meta_write_failure_defers_to_the_collective(tmp_path, monkeypatch):
    """optimizer_meta.pt is written between two collectives. A raw raise on the save rank (ENOSPC)
    would leave its peers in the closing barrier with no diagnostic — the write must ride the same
    DeferredRankFailure as the shard write and surface uniformly with the real cause."""
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    real_save = torch.save

    def failing_meta_save(obj, f, *args, **kwargs):
        if str(f).endswith("optimizer_meta.pt"):
            raise OSError("No space left on device")
        return real_save(obj, f, *args, **kwargs)

    monkeypatch.setattr(optimizer_mod.torch, "save", failing_meta_save)
    loader = OptimizerShardStore(_ctx(model, optimizer, fsdp_wrapped=True))

    # RuntimeError (the guard's uniform re-raise carrying the recorded reason), never the raw
    # OSError a bare write would throw past the collectives.
    with pytest.raises(RuntimeError, match="No space left on device"):
        loader.save(str(tmp_path))

    assert os.path.exists(os.path.join(tmp_path, "optimizer_shard_00000.pt")), "the shard write itself succeeded"


def _scheduler(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (step + 1))


def test_scheduler_missing_file_warm_restarts(tmp_path):
    """No scheduler.pt anywhere is a legitimate warm restart (save_only_model runs) — False, no raise."""
    model = _TinyModel()
    loader = OptimizerShardStore(_ctx(model, lr_scheduler=_scheduler(model)))

    assert loader.restore_lr_scheduler(str(tmp_path)) is False


def test_scheduler_torn_file_raises(tmp_path):
    """A scheduler.pt that exists but is unreadable on every rank must RAISE, not return False with
    only per-rank warnings — that silently re-warms the LR schedule from step 0 while the dataloader
    skips ahead (effective-LR desync from the resumed step)."""
    (tmp_path / "scheduler.pt").write_bytes(b"not a torch file")
    model = _TinyModel()
    loader = OptimizerShardStore(_ctx(model, lr_scheduler=_scheduler(model)))

    with pytest.raises(RuntimeError, match="torn/corrupt"):
        loader.restore_lr_scheduler(str(tmp_path))


def test_scheduler_readable_file_restores(tmp_path):
    src_model = _TinyModel()
    src_sched = _scheduler(src_model)
    for _ in range(5):
        src_sched.optimizer.step()
        src_sched.step()
    torch.save(src_sched.state_dict(), os.path.join(tmp_path, "scheduler.pt"))

    model = _TinyModel()
    sched = _scheduler(model)
    loader = OptimizerShardStore(_ctx(model, lr_scheduler=sched))

    assert loader.restore_lr_scheduler(str(tmp_path)) is True
    assert sched.last_epoch == src_sched.last_epoch
    assert sched.get_last_lr() == src_sched.get_last_lr()


def test_scheduler_none_with_present_file_returns_false(tmp_path):
    """No live scheduler to restore into: even a present (garbage) file is not an error."""
    (tmp_path / "scheduler.pt").write_bytes(b"garbage")
    model = _TinyModel()
    loader = OptimizerShardStore(_ctx(model))

    assert loader.restore_lr_scheduler(str(tmp_path)) is False


def test_adapter_attn_state_without_peft_model_raises(tmp_path):
    """A checkpoint carrying attention/PEFT adapter tensors resumed into a model with no PEFT
    adapters must RAISE, not silently skip the apply and then log a successful restore — that
    continues the run from base weights."""
    save_file(
        {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(2, 4)},
        os.path.join(tmp_path, "adapter_model.safetensors"),
    )
    with pytest.raises(RuntimeError, match="no PEFT adapters"):
        restore_adapters(str(tmp_path), _TinyModel(), is_cp_mode=False)


def test_adapter_expert_only_state_without_peft_model_ok(tmp_path, monkeypatch):
    """Expert-LoRA-only checkpoints (native EP grouped adapters, no PeftModel) must keep working:
    the expert state is applied via apply_ep_lora_adapters and nothing raises."""
    save_file(
        {"model.layers.0.mlp.experts.gate_up_proj.lora_A": torch.zeros(2, 4)},
        os.path.join(tmp_path, "adapter_model.safetensors"),
    )
    applied = _Recorder()
    monkeypatch.setattr(peft_mod, "apply_ep_lora_adapters", applied)

    restore_adapters(str(tmp_path), _TinyModel(), is_cp_mode=False)  # must not raise

    assert len(applied.calls) == 1
    assert set(applied.calls[0][0][1]) == {"model.layers.0.mlp.experts.gate_up_proj.lora_A"}


class _NamedSourceModel(_TinyModel):
    def __init__(self, source: str):
        super().__init__()
        self.config = SimpleNamespace(_name_or_path=source)


def _adapter_only_checkpoint(tmp_path, monkeypatch):
    """An adapter-only checkpoint (LoRA / expert-LoRA + save_only_model) plus a recorder standing in
    for the adapter restore, whose own PEFT-shape guards are a separate concern from the base-weight
    decision under test."""
    save_file({"base_model.model.fc.lora_A.weight": torch.ones(2, 4)}, os.path.join(tmp_path, ADAPTER_WEIGHTS_NAME))
    restored = _Recorder()
    monkeypatch.setattr(loader_mod, "restore_adapters", restored)
    return restored


def test_ep_adapter_only_resume_accepts_hub_loaded_base(tmp_path, monkeypatch):
    """An adapter-only checkpoint ships no base weights: the hub-loaded frozen base IS the run's
    base, so demanding construction from the checkpoint would make every EP/CP adapter resume DOA."""
    restored = _adapter_only_checkpoint(tmp_path, monkeypatch)
    model = _NamedSourceModel("org/base-model")
    loader = CheckpointLoader(_ctx(model, has_ep_layers=True))

    loader.load_model(str(tmp_path), model)  # must not raise

    assert len(restored.calls) == 1, "the adapters are the trained state — they must still be restored"


def test_ep_adapter_only_best_model_reload_is_allowed(tmp_path, monkeypatch):
    """``load_best_model_at_end`` on an adapter-only run is supported — the adapters reload in place,
    so the export carries the best weights. A blanket EP/CP refusal makes every EP/CP LoRA run that
    sets the flag die after training instead of exporting."""
    restored = _adapter_only_checkpoint(tmp_path, monkeypatch)
    model = _NamedSourceModel("org/base-model")
    loader = CheckpointLoader(_ctx(model, has_ep_layers=True))

    loader.load_model(str(tmp_path), model, for_best_model=True)  # must not raise

    assert len(restored.calls) == 1


def test_ep_best_model_reload_refuses_a_base_weight_checkpoint(tmp_path):
    """The other half: a full fine-tune's checkpoint DOES ship base weights, which cannot be reloaded
    in place under EP/CP — so the best-model export would silently carry the LAST weights."""
    save_file({"fc.weight": torch.ones(4, 4)}, os.path.join(tmp_path, "model.safetensors"))
    model = _NamedSourceModel("org/base-model")
    loader = CheckpointLoader(_ctx(model, has_ep_layers=True))
    with pytest.raises(ValueError, match="load_best_model_at_end"):
        loader.load_model(str(tmp_path), model, for_best_model=True)


def test_ep_base_weight_checkpoint_demands_construction_from_it(tmp_path):
    """A checkpoint that ships base weights cannot be loaded into the EP-transformed model —
    a model constructed from anywhere else would silently keep its current weights."""
    save_file({"fc.weight": torch.ones(4, 4)}, os.path.join(tmp_path, "model.safetensors"))
    model = _NamedSourceModel("org/base-model")
    loader = CheckpointLoader(_ctx(model, has_ep_layers=True))
    with pytest.raises(ValueError, match="constructed FROM the checkpoint"):
        loader.load_model(str(tmp_path), model)


# The resume skip is keyed on where the weights were READ, not on config._name_or_path.
class _ScratchInitModel(_NamedSourceModel):
    """``init_from_scratch``: built from the checkpoint's CONFIG, holding random weights.

    ``load_distributed_model`` stamps ``_loaded_weights_from = None`` for exactly this case, because
    ``AutoConfig.from_pretrained(checkpoint)`` leaves ``_name_or_path`` pointing at the checkpoint.
    """

    def __init__(self, source: str):
        super().__init__(source)
        self._loaded_weights_from = None


def _fsdp2_resume_applied(tmp_path, monkeypatch, model) -> _Recorder:
    """Run ``_load_fsdp2`` with the DTensor distribution stubbed (it needs a process group), and
    report whether the weights were applied."""
    save_file({"fc.weight": torch.ones(4, 4)}, os.path.join(tmp_path, "model.safetensors"))
    applied = _Recorder()
    monkeypatch.setattr(loader_mod, "set_model_state_dict", applied)
    CheckpointLoader(_ctx(model, fsdp_wrapped=True))._load_fsdp2(str(tmp_path), model)
    return applied


def test_fsdp2_resume_reloads_when_the_weights_were_never_read(tmp_path, monkeypatch):
    """An ``init_from_scratch`` model matches the checkpoint by ``_name_or_path`` while holding
    RANDOM weights — skipping the reload would resume at step N on a freshly initialized model."""
    applied = _fsdp2_resume_applied(tmp_path, monkeypatch, _ScratchInitModel(str(tmp_path)))

    assert len(applied.calls) == 1, "the checkpoint weights must be read"
    assert torch.equal(applied.calls[0][0][1]["fc.weight"], torch.ones(4, 4))


def test_fsdp2_resume_skips_the_reload_when_the_weights_came_from_this_checkpoint(tmp_path, monkeypatch):
    """Anti-vacuity for the test above: a model whose weights WERE read from this checkpoint already
    holds them, so re-reading a 100B+ state dict to overwrite them with themselves is pure waste."""
    model = _NamedSourceModel(str(tmp_path))
    model._loaded_weights_from = str(tmp_path)

    assert _fsdp2_resume_applied(tmp_path, monkeypatch, model).calls == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
