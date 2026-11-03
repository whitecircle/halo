"""Rank-uniform resume decisions of load_resume_weights and OptimizerShardStore.
A rank-local branch on file readability hangs peers in DTensor collectives or silently diverges optimizer state."""

import os
import re
import shutil
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from peft.utils import SAFETENSORS_WEIGHTS_NAME as ADAPTER_WEIGHTS_NAME
from safetensors.torch import load_file, save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

import halo.checkpoint.parallel.adapters as peft_mod
import halo.checkpoint.parallel.optimizer as optimizer_mod
import halo.checkpoint.parallel.resume_weights as loader_mod
from halo.checkpoint.hub_layout import ModuleLayoutView
from halo.checkpoint.parallel.adapters import restore_adapters
from halo.checkpoint.parallel.optimizer import OptimizerShardStore
from halo.checkpoint.parallel.resume_weights import load_resume_weights
from halo.checkpoint.parallel.sidecars import restore_lr_scheduler
from halo.checkpoint.weights_source import LOADED_WEIGHTS_FROM_ATTR
from halo.distributed.tensor_parallel.plain_params import register_hand_sliced
from tests.common.checkpoint_io import load_context
from tests.common.models import TINY_QWEN3_MOE_CONFIG


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
    """A per-rank head slice with no DTensor, GptOss-shaped."""

    def __init__(self):
        super().__init__()
        self.sinks = nn.Parameter(torch.zeros(2))
        register_hand_sliced(self, "sinks", 0)


def _ctx(
    model,
    optimizer=None,
    *,
    is_tp_mode=False,
    fsdp_wrapped=False,
    has_ep_layers=False,
    lr_scheduler=None,
    tp_rank=0,
):
    return load_context(
        model,
        tp_size=2 if is_tp_mode else 1,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        has_ep_layers=has_ep_layers,
        fsdp_wrapped=fsdp_wrapped,
        tp_rank=tp_rank,
        super_load_optimizer_and_scheduler=_Recorder(),
    )


def test_tp_torn_checkpoint_falls_back_uniformly(tmp_path):
    """Caught by the rank-0 probe rather than raising out of the per-rank read."""
    (tmp_path / "model.safetensors").write_bytes(b"not a safetensors file")
    model = _TinyModel()
    before = model.fc.weight.detach().clone()
    assert load_resume_weights(_ctx(model, is_tp_mode=True), str(tmp_path), model) is False  # must not raise

    assert torch.equal(model.fc.weight, before)


def test_tp_negative_rank0_verdict_forces_fallback(tmp_path, monkeypatch):
    """Without the consensus each rank decides on its own read and the world splits across branches."""
    model = _TinyModel()
    save_file({"fc.weight": torch.ones(4, 4)}, os.path.join(tmp_path, "model.safetensors"))
    monkeypatch.setattr(loader_mod, "broadcast_from_rank0", lambda local: False)
    assert load_resume_weights(_ctx(model, is_tp_mode=True), str(tmp_path), model) is False
    assert not torch.equal(model.fc.weight, torch.ones(4, 4))


def test_tp_readable_checkpoint_loads(tmp_path):
    model = _TinyModel()
    save_file({"fc.weight": torch.ones(4, 4)}, os.path.join(tmp_path, "model.safetensors"))
    assert load_resume_weights(_ctx(model, is_tp_mode=True), str(tmp_path), model) is True
    assert torch.equal(model.fc.weight, torch.ones(4, 4))


@pytest.fixture
def single_rank_gloo(tmp_path):
    """The TP+DP verdict keys on ``isinstance(p.data, DTensor)``, which no stub can satisfy."""
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        yield
    finally:
        dist.destroy_process_group()


def _dtensor_model():
    mesh = init_device_mesh("cpu", (1,))
    model = _TinyModel()
    model.fc.weight = nn.Parameter(distribute_tensor(torch.zeros(4, 4), mesh, [Shard(0)]))
    return model


def _simulate_non_rank0(monkeypatch, decisions):
    """Rank 0's broadcast answers replayed in call order: readability probe, constructed-from-checkpoint, then key
    counts."""
    answers = iter(decisions)
    monkeypatch.setattr(loader_mod, "is_global_main_process", lambda: False)
    monkeypatch.setattr(loader_mod, "broadcast_from_rank0", lambda local: next(answers))


def test_tp_skip_verdict_reads_nothing_on_non_rank0(tmp_path, single_rank_gloo, monkeypatch):
    """There is no file here at all, so a read before the skip would raise."""
    model = _dtensor_model()
    setattr(model, LOADED_WEIGHTS_FROM_ATTR, str(tmp_path))
    _simulate_non_rank0(monkeypatch, decisions=[True, True])
    ctx = _ctx(model, is_tp_mode=True)

    load_resume_weights(ctx, str(tmp_path), model)

    assert torch.equal(model.fc.weight.to_local(), torch.zeros(4, 4))


def test_tp_dp_refuses_a_model_not_built_from_the_checkpoint_without_reading(tmp_path, single_rank_gloo, monkeypatch):
    """FSDP2 over TP is a 2-D placement ``distribute_tensor`` does not invert for packed projections."""
    model = _dtensor_model()
    _simulate_non_rank0(monkeypatch, decisions=[True, False])
    ctx = _ctx(model, is_tp_mode=True, fsdp_wrapped=True)

    with pytest.raises(RuntimeError, match="constructed FROM the checkpoint"):
        load_resume_weights(ctx, str(tmp_path), model)


def test_tp_dp_best_model_reload_is_refused_naming_the_export(tmp_path, single_rank_gloo):
    """This reload can never be the constructed-from-checkpoint skip."""
    model = _dtensor_model()
    setattr(model, LOADED_WEIGHTS_FROM_ATTR, str(tmp_path))
    save_file({"fc.weight": torch.ones(4, 4)}, os.path.join(tmp_path, "model.safetensors"))
    ctx = _ctx(model, is_tp_mode=True, fsdp_wrapped=True)

    with pytest.raises(RuntimeError, match="load_best_model_at_end"):
        load_resume_weights(ctx, str(tmp_path), model, for_best_model=True)


def test_tp_distributes_the_checkpoint_into_the_live_dtensor(tmp_path, single_rank_gloo, monkeypatch):
    """The by-value proof that the resume restores trained weights."""
    model = _dtensor_model()
    save_file({"fc.weight": torch.ones(4, 4)}, os.path.join(tmp_path, "model.safetensors"))
    _simulate_non_rank0(monkeypatch, decisions=[True, False, (1, 1)])
    ctx = _ctx(model, is_tp_mode=True)

    load_resume_weights(ctx, str(tmp_path), model)

    assert torch.equal(model.fc.weight.to_local(), torch.ones(4, 4))


def test_tp_best_model_reload_reads_even_when_built_from_the_checkpoint(tmp_path, single_rank_gloo):
    """The live weights trained past the best checkpoint; skipping means the export silently carries the last
    weights."""
    model = _dtensor_model()
    setattr(model, LOADED_WEIGHTS_FROM_ATTR, str(tmp_path))
    save_file({"fc.weight": torch.ones(4, 4)}, os.path.join(tmp_path, "model.safetensors"))
    ctx = _ctx(model, is_tp_mode=True)

    load_resume_weights(ctx, str(tmp_path), model, for_best_model=True)

    assert torch.equal(model.fc.weight.to_local(), torch.ones(4, 4))


def test_tp_plain_non_rank0_still_reads_for_the_genuine_load(tmp_path, monkeypatch):
    """A replicated param is copied from the rank's own read, so a non-zero rank must still read its node's copy."""
    model = _TinyModel()
    save_file({"fc.weight": torch.ones(4, 4)}, os.path.join(tmp_path, "model.safetensors"))
    _simulate_non_rank0(monkeypatch, decisions=[True, False, (1, 1)])
    ctx = _ctx(model, is_tp_mode=True)

    load_resume_weights(ctx, str(tmp_path), model)

    assert torch.equal(model.fc.weight, torch.ones(4, 4))


def test_a_node_copy_short_of_an_expert_is_refused_before_any_write(tmp_path, monkeypatch):
    """A short copy spells the same live names as rank 0's, so only the on-disk key count tells them apart; past the
    copy, this rank alone fails while peers wait."""
    full, short = tmp_path / "full", tmp_path / "short"
    torch.manual_seed(0)
    Qwen3MoeForCausalLM(Qwen3MoeConfig(**{**TINY_QWEN3_MOE_CONFIG, "num_hidden_layers": 1})).save_pretrained(full)
    model = Qwen3MoeForCausalLM.from_pretrained(full)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    state = load_file(str(full / "model.safetensors"))
    lost = next(key for key in state if ".experts.1.up_proj." in key)
    short.mkdir()
    save_file({key: tensor for key, tensor in state.items() if key != lost}, str(short / "model.safetensors"))
    shutil.copy(full / "config.json", short)
    rank0_counts = (len(state), len(ModuleLayoutView(model, state).keys))
    _simulate_non_rank0(monkeypatch, decisions=[True, False, rank0_counts])

    refusal = (
        f"TP checkpoint resume from {short}: fewer than half of the live model's PARAMETERS match the "
        f"checkpoint's keys on at least one rank — resume would silently continue from base weights. The "
        f"checkpoint was written for a different model or wrapper layout, or a per-node copy is incomplete."
    )
    with pytest.raises(RuntimeError, match=f"^{re.escape(refusal)}$"):
        load_resume_weights(_ctx(model, is_tp_mode=True), str(short), model)

    for name, param in model.named_parameters():
        assert torch.equal(param, before[name]), f"{name}: a refused reload must not half-apply the checkpoint"


def test_tp_hand_sliced_param_takes_its_rank_slice(tmp_path):
    """The checkpoint holds every head: a whole copy would shape-mismatch, and rank 0's slice would duplicate its
    heads."""
    model = _HandSlicedModel()
    save_file(
        {"fc.weight": torch.ones(4, 4), "sinks": torch.arange(4, dtype=torch.float32)},
        os.path.join(tmp_path, "model.safetensors"),
    )
    ctx = _ctx(model, is_tp_mode=True, tp_rank=1)

    load_resume_weights(ctx, str(tmp_path), model)

    assert torch.equal(model.sinks, torch.tensor([2.0, 3.0]))
    assert torch.equal(model.fc.weight, torch.ones(4, 4))


def test_fsdp2_resume_with_no_matching_keys_raises(tmp_path):
    """strict=False reports a wholesale mismatch as `unexpected` with an empty `missing_keys`, so the call is a
    silent no-op that resumes base weights."""
    model = _TinyModel()
    before = model.fc.weight.detach().clone()
    save_file({"totally.wrong.key": torch.ones(4, 4)}, os.path.join(tmp_path, "model.safetensors"))
    refusal = (
        f"FSDP2 checkpoint resume from {tmp_path}: fewer than half of the live model's PARAMETERS match the "
        f"checkpoint's keys — resume would silently continue from base weights. The checkpoint was written "
        f"for a different model or wrapper layout (a different expert namespace, most likely)."
    )
    with pytest.raises(RuntimeError, match=f"^{re.escape(refusal)}$"):
        load_resume_weights(_ctx(model, fsdp_wrapped=True), str(tmp_path), model)

    assert torch.equal(model.fc.weight, before)


class _ThreeLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4, bias=False)
        self.fc2 = nn.Linear(4, 4, bias=False)
        self.fc3 = nn.Linear(4, 4, bias=False)


def test_fsdp2_resume_below_half_coverage_raises(tmp_path):
    """strict=False would apply the matching fraction and silently keep base weights for the rest."""
    model = _ThreeLayerModel()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    save_file({"fc1.weight": torch.ones(4, 4)}, os.path.join(tmp_path, "model.safetensors"))
    ctx = _ctx(model, fsdp_wrapped=True)

    with pytest.raises(RuntimeError, match="fewer than half of the"):
        load_resume_weights(ctx, str(tmp_path), model)

    for n, p in model.named_parameters():
        assert torch.equal(p, before[n])


def test_fsdp2_resume_majority_coverage_loads_and_keeps_absent(tmp_path):
    """Gaps like tied lm_head are real, so absent tensors keep their construction values. Needs a 1-rank gloo group
    for the real ``set_model_state_dict``."""
    model = _ThreeLayerModel()
    before_fc3 = model.fc3.weight.detach().clone()
    save_file(
        {"fc1.weight": torch.ones(4, 4), "fc2.weight": torch.full((4, 4), 2.0)},
        os.path.join(tmp_path, "model.safetensors"),
    )
    ctx = _ctx(model, fsdp_wrapped=True)

    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path}/pg_init")
    try:
        load_resume_weights(ctx, str(tmp_path), model)
    finally:
        dist.destroy_process_group()

    assert torch.equal(model.fc1.weight, torch.ones(4, 4))
    assert torch.equal(model.fc2.weight, torch.full((4, 4), 2.0))
    assert torch.equal(model.fc3.weight, before_fc3)


def test_fsdp2_resume_with_matching_keys_reaches_the_load(tmp_path, monkeypatch):
    """Anti-vacuity: the real call needs ``broadcast_from_rank0``, so it is recorded here."""
    model = _TinyModel()
    save_file({"fc.weight": torch.ones(4, 4)}, os.path.join(tmp_path, "model.safetensors"))
    applied = _Recorder()
    monkeypatch.setattr(loader_mod, "set_model_state_dict", applied)
    ctx = _ctx(model, fsdp_wrapped=True)

    load_resume_weights(ctx, str(tmp_path), model)

    assert len(applied.calls) == 1
    assert set(applied.calls[0][0][1]) == {"fc.weight"}


def test_optimizer_shards_without_meta_are_refused_as_torn(tmp_path, monkeypatch):
    """A kill between the shard and meta writes reaches this. Loading ungated maps another topology's moments onto
    this run's params."""
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    torch.save({"state": {}, "param_groups": []}, os.path.join(tmp_path, "optimizer_shard_00000.pt"))

    restore = _Recorder()
    warm_restarts = _Recorder()
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", restore)
    monkeypatch.setattr(OptimizerShardStore, "_warm_restart", lambda self, ckpt, msg: warm_restarts(ckpt, msg))

    with pytest.raises(RuntimeError, match="optimizer_meta.pt is missing beside the per-rank optimizer shards"):
        OptimizerShardStore(_ctx(model, optimizer, fsdp_wrapped=True)).load(str(tmp_path))
    assert restore.calls == [], "ungated shards were restored"
    assert warm_restarts.calls == [], "an interrupted save was warm-restarted as if complete"

    opted_in = _ctx(model, optimizer, fsdp_wrapped=True)
    opted_in.allow_optimizer_warm_restart = True
    OptimizerShardStore(opted_in).load(str(tmp_path))
    assert restore.calls == [], "ungated shards were restored under the opt-in"
    assert len(warm_restarts.calls) == 1, "the opt-in must take the warm restart"


def test_a_non_sharded_resume_refuses_per_rank_optimizer_shards(tmp_path):
    """The base Trainer's loader recognizes only optimizer.pt, so the Adam moments would reset with no warning."""
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    torch.save({"state": {}, "param_groups": []}, os.path.join(tmp_path, "optimizer_shard_00000.pt"))
    loader = OptimizerShardStore(_ctx(model, optimizer))

    with pytest.raises(RuntimeError, match="optimizer_shard"):
        loader.load(str(tmp_path))

    assert loader.ctx.super_load_optimizer_and_scheduler.calls == [], "the silent HF route must not be reached"


def test_a_meta_only_checkpoint_is_refused_too(tmp_path):
    """The load side treats meta-without-shards as a failed gate."""
    (tmp_path / "optimizer_meta.pt").write_bytes(b"x")
    loader = OptimizerShardStore(_ctx(_TinyModel()))

    with pytest.raises(RuntimeError, match="optimizer_shard"):
        loader.load(str(tmp_path))


def test_a_shard_free_checkpoint_still_takes_the_base_route(tmp_path):
    model = _TinyModel()
    torch.save({}, os.path.join(tmp_path, "optimizer.pt"))
    loader = OptimizerShardStore(_ctx(model, torch.optim.SGD(model.parameters(), lr=0.1)))

    loader.load(str(tmp_path))
    loader.load(None)

    assert len(loader.ctx.super_load_optimizer_and_scheduler.calls) == 2


def test_a_meta_write_failure_defers_to_the_collective(tmp_path, monkeypatch):
    """A raw raise (ENOSPC) on the save rank would leave its peers in the closing barrier with no diagnostic."""
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    real_save = torch.save

    def failing_meta_save(obj, f, *args, **kwargs):
        # The meta is staged beside its final name, so match the name wherever it sits.
        if "optimizer_meta.pt" in os.path.basename(str(f)):
            raise OSError("No space left on device")
        return real_save(obj, f, *args, **kwargs)

    monkeypatch.setattr(optimizer_mod.torch, "save", failing_meta_save)
    loader = OptimizerShardStore(_ctx(model, optimizer, fsdp_wrapped=True))

    # The guard's uniform re-raise, never the raw OSError a bare write would throw past the collectives.
    with pytest.raises(RuntimeError, match="No space left on device"):
        loader.save(str(tmp_path))

    assert os.path.exists(os.path.join(tmp_path, "optimizer_shard_00000.pt")), "the shard write itself succeeded"
    assert not [name for name in os.listdir(tmp_path) if "optimizer_meta" in name], (
        "a failed meta write left a file that could vouch for the shards"
    )


def _scheduler(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (step + 1))


def test_scheduler_missing_file_warm_restarts(tmp_path):
    """Legitimate under save_only_model."""
    model = _TinyModel()
    assert restore_lr_scheduler(_scheduler(model), None, str(tmp_path)) is False


def test_scheduler_torn_file_raises(tmp_path):
    """Returning False silently re-warms the LR schedule from step 0 while the dataloader skips ahead."""
    (tmp_path / "scheduler.pt").write_bytes(b"not a torch file")
    model = _TinyModel()
    with pytest.raises(RuntimeError, match="torn/corrupt"):
        restore_lr_scheduler(_scheduler(model), None, str(tmp_path))


def test_scheduler_readable_file_restores(tmp_path):
    src_model = _TinyModel()
    src_sched = _scheduler(src_model)
    for _ in range(5):
        src_sched.optimizer.step()
        src_sched.step()
    torch.save(src_sched.state_dict(), os.path.join(tmp_path, "scheduler.pt"))

    model = _TinyModel()
    sched = _scheduler(model)
    assert restore_lr_scheduler(sched, None, str(tmp_path)) is True
    assert sched.last_epoch == src_sched.last_epoch
    assert sched.get_last_lr() == src_sched.get_last_lr()


def test_scheduler_none_with_present_file_returns_false(tmp_path):
    (tmp_path / "scheduler.pt").write_bytes(b"garbage")
    assert restore_lr_scheduler(None, None, str(tmp_path)) is False


def test_adapter_attn_state_without_peft_model_raises(tmp_path):
    """Skipping the apply would continue the run from base weights while logging a successful restore."""
    save_file(
        {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(2, 4)},
        os.path.join(tmp_path, "adapter_model.safetensors"),
    )
    with pytest.raises(RuntimeError, match="no PEFT adapters"):
        restore_adapters(str(tmp_path), _TinyModel(), is_cp_mode=False)


def test_adapter_expert_only_state_without_peft_model_ok(tmp_path, monkeypatch):
    """Native EP grouped adapters have no PeftModel."""
    save_file(
        {"model.layers.0.mlp.experts.gate_up_proj.lora_A": torch.zeros(2, 4)},
        os.path.join(tmp_path, "adapter_model.safetensors"),
    )
    applied = _Recorder()
    monkeypatch.setattr(peft_mod, "load_expert_lora_adapters", applied)

    restore_adapters(str(tmp_path), _TinyModel(), is_cp_mode=False)

    assert len(applied.calls) == 1
    assert set(applied.calls[0][0][1]) == {"model.layers.0.mlp.experts.gate_up_proj.lora_A"}


class _NamedSourceModel(_TinyModel):
    def __init__(self, source: str):
        super().__init__()
        self.config = SimpleNamespace(_name_or_path=source)


def _adapter_only_checkpoint(tmp_path, monkeypatch):
    """The recorder stands in for the adapter restore, whose PEFT-shape guards are a separate concern."""
    save_file({"base_model.model.fc.lora_A.weight": torch.ones(2, 4)}, os.path.join(tmp_path, ADAPTER_WEIGHTS_NAME))
    restored = _Recorder()
    monkeypatch.setattr(loader_mod, "restore_adapters", restored)
    return restored


def test_ep_adapter_only_resume_accepts_hub_loaded_base(tmp_path, monkeypatch):
    """The hub-loaded frozen base is the run's base, so demanding construction from the checkpoint would break every
    EP/CP adapter resume."""
    restored = _adapter_only_checkpoint(tmp_path, monkeypatch)
    model = _NamedSourceModel("org/base-model")
    ctx = _ctx(model, has_ep_layers=True)

    load_resume_weights(ctx, str(tmp_path), model)

    assert len(restored.calls) == 1, "the adapters are the trained state — they must still be restored"


def test_ep_adapter_only_best_model_reload_is_allowed(tmp_path, monkeypatch):
    """The adapters reload in place; a blanket refusal would kill every EP/CP LoRA run that sets the flag after
    training."""
    restored = _adapter_only_checkpoint(tmp_path, monkeypatch)
    model = _NamedSourceModel("org/base-model")
    ctx = _ctx(model, has_ep_layers=True)

    load_resume_weights(ctx, str(tmp_path), model, for_best_model=True)

    assert len(restored.calls) == 1


def test_ep_best_model_reload_refuses_a_base_weight_checkpoint(tmp_path):
    """Base weights cannot be reloaded in place under EP/CP, so the export would silently carry the last weights."""
    save_file({"fc.weight": torch.ones(4, 4)}, os.path.join(tmp_path, "model.safetensors"))
    model = _NamedSourceModel("org/base-model")
    ctx = _ctx(model, has_ep_layers=True)
    with pytest.raises(ValueError, match="load_best_model_at_end"):
        load_resume_weights(ctx, str(tmp_path), model, for_best_model=True)


def test_ep_base_weight_checkpoint_demands_construction_from_it(tmp_path):
    """A model constructed from anywhere else would silently keep its current weights."""
    save_file({"fc.weight": torch.ones(4, 4)}, os.path.join(tmp_path, "model.safetensors"))
    model = _NamedSourceModel("org/base-model")
    ctx = _ctx(model, has_ep_layers=True)
    with pytest.raises(ValueError, match="constructed FROM the checkpoint"):
        load_resume_weights(ctx, str(tmp_path), model)


# The resume skip is keyed on where the weights were read, not on config._name_or_path.
class _ScratchInitModel(_NamedSourceModel):
    """``AutoConfig.from_pretrained(checkpoint)`` leaves ``_name_or_path`` pointing at the checkpoint while the
    weights are random."""

    def __init__(self, source: str):
        super().__init__(source)
        self._loaded_weights_from = None


def _fsdp2_resume_applied(tmp_path, monkeypatch, model) -> _Recorder:
    """The DTensor distribution is stubbed because it needs a process group."""
    save_file({"fc.weight": torch.ones(4, 4)}, os.path.join(tmp_path, "model.safetensors"))
    applied = _Recorder()
    monkeypatch.setattr(loader_mod, "set_model_state_dict", applied)
    load_resume_weights(_ctx(model, fsdp_wrapped=True), str(tmp_path), model)
    return applied


def test_fsdp2_resume_reloads_when_the_weights_were_never_read(tmp_path, monkeypatch):
    """Skipping the reload would resume at step N on a freshly initialized model."""
    applied = _fsdp2_resume_applied(tmp_path, monkeypatch, _ScratchInitModel(str(tmp_path)))

    assert len(applied.calls) == 1, "the checkpoint weights must be read"
    assert torch.equal(applied.calls[0][0][1]["fc.weight"], torch.ones(4, 4))


def test_fsdp2_resume_skips_the_reload_when_the_weights_came_from_this_checkpoint(tmp_path, monkeypatch):
    """Anti-vacuity: re-reading a 100B+ state dict to overwrite them with themselves is pure waste."""
    model = _NamedSourceModel(str(tmp_path))
    model._loaded_weights_from = str(tmp_path)

    assert _fsdp2_resume_applied(tmp_path, monkeypatch, model).calls == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
