"""CPU tests for PP checkpoint resume: the global↔local name remap and the loader's resume gates.

A PP checkpoint stores complete tensors under the UNSPLIT model's global names; resume maps them
back through each stage's ``global_parameter_name``. The gates under test:

- weight coverage: a stage tensor absent from the checkpoint raises (a silent miss = base weights);
- optimizer fingerprint mismatch under PP raises (never a quiet warm restart);
- ``pp_stage_partition`` absent or different from the live stage's layer range raises;
- per-shard FQN coverage: a trainable param with no saved moments raises, except the documented
  no-grad ``sinks`` allowlist;
- deleting every shard + meta remains the explicit warm-restart escape hatch.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

import src.distributed.checkpoint.coordination as coordination_mod
import src.distributed.checkpoint.loader as loader_mod
import src.distributed.checkpoint.optimizer as optimizer_mod
from src.checkpoint.config_export import LOADED_WEIGHTS_FROM_ATTR
from src.distributed.checkpoint.context import CheckpointLoadContext
from src.distributed.checkpoint.fingerprint import OptimizerStateFingerprint
from src.distributed.checkpoint.loader import CheckpointLoader
from src.distributed.checkpoint.optimizer import OptimizerShardStore
from src.distributed.pipeline_parallel.stage import PipelineStageModule, build_pipeline_stage
from tests.common.models import TINY_QWEN3_CONFIG

PP_SIZE = 2
N_LAYERS = 4
HIDDEN = 8


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class _StubParallelismConfig:
    def __init__(self, pp_size=PP_SIZE, pp_rank=0, ep_size=1, cp_size=1, tp_size=1):
        self.pp_size = pp_size
        self.pp_rank = pp_rank
        self.ep_size = ep_size
        self.expert_tp_size = 1
        self.cp_size = cp_size
        self.tp_size = tp_size
        self.fsdp_shard_ep1_experts = True


class _Backbone(nn.Module):
    """Fake decoder backbone: embedding on the first stage, a persistent buffer + norm on the last."""

    def __init__(self, n_layers: int, is_first: bool, is_last: bool):
        super().__init__()
        if is_first:
            self.embed_tokens = nn.Embedding(16, HIDDEN)
        self.layers = nn.ModuleList(nn.Linear(HIDDEN, HIDDEN, bias=False) for _ in range(n_layers))
        self.norm = nn.LayerNorm(HIDDEN) if is_last else nn.Identity()
        if is_last:
            self.register_buffer("pos_scale", torch.arange(HIDDEN, dtype=torch.float32))


def _fake_stage(lo: int, hi: int, n_total: int = N_LAYERS) -> PipelineStageModule:
    is_first, is_last = lo == 0, hi == n_total
    stage = PipelineStageModule(
        _Backbone(hi - lo, is_first, is_last),
        nn.Linear(HIDDEN, 16, bias=False) if is_last else None,
        is_first,
        is_last,
        backbone_prefix="model",
        head_attr="lm_head",
        layer_attr="layers",
        layer_offset=lo,
    )
    return stage


def _ctx(model, optimizer=None, *, parallelism_config=None, fsdp_wrapped=True):
    return CheckpointLoadContext(
        model=model,
        optimizer=optimizer,
        lr_scheduler=None,
        # The PP load path reads the axis sizes to reject a sharding it cannot un-shard.
        parallelism_config=parallelism_config or _StubParallelismConfig(),
        is_pp_mode=True,
        is_cp_mode=False,
        is_tp_mode=False,
        has_ep_layers=False,
        fsdp_wrapped=fsdp_wrapped,
        tp_rank=0,
        tp_size=1,
        super_load_from_checkpoint=_Recorder(),
        super_load_optimizer_and_scheduler=_Recorder(),
    )


def test_fake_two_stage_split_names_cover_unsplit_model_exactly():
    """The stages' globally-renamed state_dict keys (incl. head.* and the buffer) reassemble the
    unsplit key set with no collisions — the inverse map _load_pp_stage builds is total."""
    stages = [_fake_stage(0, 2), _fake_stage(2, 4)]
    per_stage = [{s.global_parameter_name(local): local for local in s.state_dict()} for s in stages]
    for stage, inverse in zip(stages, per_stage, strict=True):
        assert len(inverse) == len(stage.state_dict())  # injective: no two locals share a global

    union = set(per_stage[0]) | set(per_stage[1])
    assert not (set(per_stage[0]) & set(per_stage[1]))
    assert union == {
        "model.embed_tokens.weight",
        "model.layers.0.weight",
        "model.layers.1.weight",
        "model.layers.2.weight",
        "model.layers.3.weight",
        "model.norm.weight",
        "model.norm.bias",
        "model.pos_scale",
        "lm_head.weight",
    }
    # The remap re-bases layer indices: stage 1's local layer 0 is global layer 2.
    assert per_stage[1]["model.layers.2.weight"] == "model.layers.0.weight"
    assert per_stage[1]["lm_head.weight"] == "head.weight"
    assert per_stage[1]["model.pos_scale"] == "model.pos_scale"


def test_real_qwen3_split_round_trips_state_dict_keys():
    """build_pipeline_stage on a real tiny Qwen3: the stages' global names reproduce the unsplit
    model's state_dict key set exactly (nothing dropped, nothing collides, indices re-based)."""
    cfg = Qwen3Config(**TINY_QWEN3_CONFIG, pad_token_id=0, eos_token_id=1)
    reference_keys = set(Qwen3ForCausalLM(cfg).state_dict())
    seen: dict[str, int] = {}
    for pp_rank in range(PP_SIZE):
        stage = build_pipeline_stage(Qwen3ForCausalLM(cfg), pp_rank, PP_SIZE)
        for local in stage.state_dict():
            global_name = stage.global_parameter_name(local)
            seen[global_name] = seen.get(global_name, 0) + 1
    assert set(seen) == reference_keys
    assert all(count == 1 for count in seen.values())


def _write_pp_checkpoint(tmp_path, stage, *, drop: str | None = None) -> dict[str, torch.Tensor]:
    """Write the stage's tensors under GLOBAL names (optionally dropping one key)."""
    tensors = {
        stage.global_parameter_name(local): torch.randn_like(value.float()).to(value.dtype)
        for local, value in stage.state_dict().items()
    }
    if drop is not None:
        del tensors[drop]
    save_file(dict(tensors), os.path.join(tmp_path, "model.safetensors"), metadata={"format": "pt"})
    return tensors


def test_load_pp_stage_copies_checkpoint_values_into_local_names(tmp_path):
    stage = _fake_stage(2, 4)
    saved = _write_pp_checkpoint(tmp_path, stage)
    CheckpointLoader(_ctx(stage)).load_model(str(tmp_path))
    for local, value in stage.state_dict().items():
        assert torch.equal(value, saved[stage.global_parameter_name(local)]), local


def test_load_pp_stage_missing_key_raises_naming_it(tmp_path):
    stage = _fake_stage(2, 4)
    _write_pp_checkpoint(tmp_path, stage, drop="model.layers.3.weight")
    with pytest.raises(RuntimeError, match=r"model\.layers\.3\.weight"):
        CheckpointLoader(_ctx(stage)).load_model(str(tmp_path))


def test_pp_ep_guard_reads_where_weights_were_read_not_name_or_path(tmp_path):
    """The PP+EP construction-identity guard keys on ``_loaded_weights_from`` — the same signal as
    the EP/CP guard in this file's module. ``config._name_or_path`` survives a build that read no
    weights, so a model merely NAMED after the checkpoint would resume base experts silently."""
    stage = _fake_stage(2, 4)
    _write_pp_checkpoint(tmp_path, stage)
    stage.ep_moe_layers = lambda: [("model.nonexistent_moe_block", object())]
    stage.config = SimpleNamespace(_name_or_path=str(tmp_path))  # the weak signal: looks built-from
    setattr(stage, LOADED_WEIGHTS_FROM_ATTR, "/somewhere/else")  # the truth: weights read elsewhere

    with pytest.raises(ValueError, match="constructed FROM the checkpoint"):
        CheckpointLoader(_ctx(stage)).load_model(str(tmp_path))


def test_pp_ep_guard_accepts_a_model_whose_weights_were_read_from_the_checkpoint(tmp_path):
    """Anti-vacuity, and the inverse discrimination: weights genuinely read from the checkpoint pass
    even when ``_name_or_path`` points at the hub id the run was configured with."""
    stage = _fake_stage(2, 4)
    saved = _write_pp_checkpoint(tmp_path, stage)
    stage.ep_moe_layers = lambda: [("model.nonexistent_moe_block", object())]
    stage.config = SimpleNamespace(_name_or_path="org/base-model")
    setattr(stage, LOADED_WEIGHTS_FROM_ATTR, str(tmp_path))

    CheckpointLoader(_ctx(stage)).load_model(str(tmp_path))

    for local, value in stage.state_dict().items():
        assert torch.equal(value, saved[stage.global_parameter_name(local)]), local


@pytest.mark.parametrize("axis", ["tp_size", "cp_size"])
def test_load_pp_stage_refuses_axes_it_cannot_unshard(tmp_path, axis):
    """PP shards hold COMPLETE tensors and only FSDP2's dp DTensors are reconstructed. TP would leave
    plain per-rank slices and CP renames every projection, so both must fail loud here rather than
    load fragments / miss every attention key. SUPPORTED_AXIS_SETS rejects PP+TP and PP+CP already;
    this is the local invariant that keeps the shortcut honest if that allowlist widens."""
    stage = _fake_stage(0, 2)
    _write_pp_checkpoint(tmp_path, stage)
    pc = _StubParallelismConfig(**{axis: 2})

    with pytest.raises(NotImplementedError, match="does not handle"):
        CheckpointLoader(_ctx(stage, parallelism_config=pc)).load_model(str(tmp_path))


def test_pp_ep_guard_joins_the_world_even_on_a_moe_free_stage(tmp_path, monkeypatch):
    """The construction-identity verdict is joined on EVERY rank, not only inside ``if
    stage.ep_moe_layers():``. A PP split can leave a whole stage without a MoE block, and both of the
    verdict's inputs are rank-local (that module list, and a ``realpath`` on this node's filesystem),
    so a raise taken inside the gate would strand the MoE-free peers in the read consensus below."""
    stage = _fake_stage(0, 2)  # no EP layers on this stage
    _write_pp_checkpoint(tmp_path, stage)
    real_reject = loader_mod.reject_across_ranks
    joined: list[str | None] = []
    monkeypatch.setattr(
        loader_mod,
        "reject_across_ranks",
        lambda reason, what, **kwargs: joined.append(reason) or real_reject(reason, what, **kwargs),
    )

    CheckpointLoader(_ctx(stage)).load_model(str(tmp_path))

    assert joined == [None], "a MoE-free stage must still enter the identity join, reporting no reason"


def test_load_pp_stage_unreadable_checkpoint_raises_uniformly(tmp_path, monkeypatch):
    """A peer failing its shard read must raise here too (consensus), not proceed to copies."""
    stage = _fake_stage(0, 2)
    _write_pp_checkpoint(tmp_path, stage)
    # The shard read is the first of this path's two ``all_ranks_ok`` joins (the other is key
    # coverage), so failing every join still lands the raise on the read.
    monkeypatch.setattr(loader_mod, "all_ranks_ok", lambda local: False)
    with pytest.raises(RuntimeError, match="unreadable"):
        CheckpointLoader(_ctx(stage)).load_model(str(tmp_path))


def _stage_with_optimizer(lo=0, hi=2):
    stage = _fake_stage(lo, hi)
    optimizer = torch.optim.SGD(stage.parameters(), lr=0.1, momentum=0.9)
    for param in stage.parameters():
        param.grad = torch.zeros_like(param)
    optimizer.step()
    return stage, optimizer


def _write_optimizer_checkpoint(tmp_path, stage, optimizer, pc, *, meta_overrides=None, drop_fqn=None, shard=True):
    state = {name: {"momentum_buffer": torch.zeros_like(p)} for name, p in stage.named_parameters()}
    if drop_fqn is not None:
        del state[drop_fqn]
    if shard:
        torch.save({"state": state, "param_groups": []}, os.path.join(tmp_path, "optimizer_shard_00000.pt"))
    meta = {
        "num_ranks": 1,
        "fingerprint": OptimizerStateFingerprint.capture(pc, optimizer, 1).to_dict(),
        "pp_stage_partition": [[0, 2], [2, 4]],
    }
    meta.update(meta_overrides or {})
    meta = {k: v for k, v in meta.items() if v is not None}
    torch.save(meta, os.path.join(tmp_path, "optimizer_meta.pt"))


def test_pp_matching_topology_restores(tmp_path, monkeypatch):
    stage, optimizer = _stage_with_optimizer()
    pc = _StubParallelismConfig(pp_rank=0)
    _write_optimizer_checkpoint(tmp_path, stage, optimizer, pc)
    restore = _Recorder()
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", restore)

    OptimizerShardStore(_ctx(stage, optimizer, parallelism_config=pc)).load(str(tmp_path))

    assert len(restore.calls) == 1


def test_pp_fingerprint_mismatch_raises_not_warm_restart(tmp_path, monkeypatch):
    stage, optimizer = _stage_with_optimizer()
    pc = _StubParallelismConfig(pp_rank=0)
    _write_optimizer_checkpoint(tmp_path, stage, optimizer, pc)
    saved = torch.load(os.path.join(tmp_path, "optimizer_meta.pt"), weights_only=False)
    saved["fingerprint"]["pp_size"] = 4
    torch.save(saved, os.path.join(tmp_path, "optimizer_meta.pt"))
    restore = _Recorder()
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", restore)

    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        OptimizerShardStore(_ctx(stage, optimizer, parallelism_config=pc)).load(str(tmp_path))
    assert restore.calls == []


def test_pp_shifted_partition_raises(tmp_path, monkeypatch):
    stage, optimizer = _stage_with_optimizer(lo=0, hi=2)  # live range [0, 2)
    pc = _StubParallelismConfig(pp_rank=0)
    _write_optimizer_checkpoint(
        tmp_path, stage, optimizer, pc, meta_overrides={"pp_stage_partition": [[0, 3], [3, 4]]}
    )
    restore = _Recorder()
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", restore)

    with pytest.raises(RuntimeError, match="pp_stage_partition"):
        OptimizerShardStore(_ctx(stage, optimizer, parallelism_config=pc)).load(str(tmp_path))
    assert restore.calls == []


def test_pp_absent_partition_raises(tmp_path, monkeypatch):
    stage, optimizer = _stage_with_optimizer()
    pc = _StubParallelismConfig(pp_rank=0)
    _write_optimizer_checkpoint(tmp_path, stage, optimizer, pc, meta_overrides={"pp_stage_partition": None})
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", _Recorder())

    with pytest.raises(RuntimeError, match="pp_stage_partition"):
        OptimizerShardStore(_ctx(stage, optimizer, parallelism_config=pc)).load(str(tmp_path))


def test_pp_missing_fqn_in_shard_raises(tmp_path, monkeypatch):
    stage, optimizer = _stage_with_optimizer()
    pc = _StubParallelismConfig(pp_rank=0)
    _write_optimizer_checkpoint(tmp_path, stage, optimizer, pc, drop_fqn="model.layers.1.weight")
    restore = _Recorder()
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", restore)

    with pytest.raises(RuntimeError, match=r"model\.layers\.1\.weight"):
        OptimizerShardStore(_ctx(stage, optimizer, parallelism_config=pc)).load(str(tmp_path))
    assert restore.calls == []


def test_pp_sinks_allowlist_passes_fqn_gate(tmp_path, monkeypatch):
    """GptOss FA4 sinks accrue no grad, so Adam never materializes their state and the save omits
    it — the coverage gate must not mistake that for a torn shard."""
    stage, optimizer = _stage_with_optimizer()
    stage.model.sinks = nn.Parameter(torch.zeros(4))
    pc = _StubParallelismConfig(pp_rank=0)
    _write_optimizer_checkpoint(tmp_path, stage, optimizer, pc)  # state written for pre-sinks params only
    restore = _Recorder()
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", restore)

    OptimizerShardStore(_ctx(stage, optimizer, parallelism_config=pc)).load(str(tmp_path))

    assert len(restore.calls) == 1


def test_pp_no_shards_anywhere_is_the_explicit_warm_restart_escape_hatch(tmp_path, monkeypatch):
    """Deleting every optimizer_shard_*.pt and optimizer_meta.pt opts in to a warm restart."""
    stage, optimizer = _stage_with_optimizer()
    pc = _StubParallelismConfig(pp_rank=0)
    restore = _Recorder()
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", restore)

    OptimizerShardStore(_ctx(stage, optimizer, parallelism_config=pc)).load(str(tmp_path))

    assert restore.calls == []


def test_pp_torn_shard_subset_raises(tmp_path, monkeypatch):
    """This rank's shard missing while a peer holds one (consensus any=True) is a torn copy."""
    stage, optimizer = _stage_with_optimizer()
    pc = _StubParallelismConfig(pp_rank=0)
    _write_optimizer_checkpoint(tmp_path, stage, optimizer, pc, shard=False)
    for module in (optimizer_mod, coordination_mod):  # the gates ask both
        monkeypatch.setattr(module, "rank_consensus", lambda local: (local, True))

    with pytest.raises(RuntimeError, match="torn"):
        OptimizerShardStore(_ctx(stage, optimizer, parallelism_config=pc)).load(str(tmp_path))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
