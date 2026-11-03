"""PP checkpoint resume: the global↔local name remap and the loader's resume gates.
Stage tensors absent from the checkpoint raise, since a silent miss is base weights."""

import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

import halo.checkpoint.parallel.optimizer as optimizer_mod
import halo.checkpoint.parallel.pp as pp_mod
import halo.checkpoint.parallel.reading as coordination_mod
from halo.checkpoint.parallel.optimizer import OptimizerShardStore
from halo.checkpoint.parallel.optimizer_meta import OptimizerStateFingerprint
from halo.checkpoint.parallel.resume_weights import load_resume_weights
from halo.checkpoint.weights_source import LOADED_WEIGHTS_FROM_ATTR
from halo.distributed.pipeline_parallel.stage import PipelineStageModule, build_pipeline_stage
from halo.runtime.consensus import RankConsensus
from tests.common.checkpoint_io import load_context
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.pp_fakes import hand_built_stage

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
        self.is_pp_mode = pp_size > 1
        self.is_cp_mode = cp_size > 1
        self.is_tp_mode = tp_size > 1


class _Backbone(nn.Module):
    """Embedding on the first stage, a persistent buffer and norm on the last."""

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
    return hand_built_stage(
        _Backbone(hi - lo, is_first, is_last),
        nn.Linear(HIDDEN, 16, bias=False) if is_last else None,
        is_first=is_first,
        is_last=is_last,
        layer_offset=lo,
    )


def _ctx(model, optimizer=None, *, parallelism_config=None, fsdp_wrapped=True):
    # The PP load path reads the axis sizes to reject a sharding it cannot un-shard.
    return load_context(
        model,
        parallelism_config or _StubParallelismConfig(),
        optimizer=optimizer,
        fsdp_wrapped=fsdp_wrapped,
        super_load_optimizer_and_scheduler=_Recorder(),
    )


def test_fake_two_stage_split_names_cover_unsplit_model_exactly():
    """The inverse map restore_pp_stage_weights builds is total."""
    stages = [_fake_stage(0, 2), _fake_stage(2, 4)]
    per_stage = [{s.global_parameter_name(local): local for local in s.state_dict()} for s in stages]
    for stage, inverse in zip(stages, per_stage, strict=True):
        assert len(inverse) == len(stage.state_dict())  # Injective: no two locals share a global.

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
    # Stage 1's local layer 0 is global layer 2.
    assert per_stage[1]["model.layers.2.weight"] == "model.layers.0.weight"
    assert per_stage[1]["lm_head.weight"] == "head.weight"
    assert per_stage[1]["model.pos_scale"] == "model.pos_scale"


def test_real_qwen3_split_round_trips_state_dict_keys():
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
    load_resume_weights(_ctx(stage), str(tmp_path))
    for local, value in stage.state_dict().items():
        assert torch.equal(value, saved[stage.global_parameter_name(local)]), local


def test_a_stage_shard_absent_from_this_nodes_copy_raises_naming_it(tmp_path):
    """A merged index can name files a node's copy lacks; the refusal names the file holding its own tensors."""
    stage, other = _fake_stage(2, 4), _fake_stage(0, 2)
    weight_map = {
        **{other.global_parameter_name(local): "model-00001-of-00002.safetensors" for local in other.state_dict()},
        **{stage.global_parameter_name(local): "model-00002-of-00002.safetensors" for local in stage.state_dict()},
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))

    with pytest.raises(RuntimeError, match=r"PP stage 0: .* absent from this node's copy") as refused:
        load_resume_weights(_ctx(stage), str(tmp_path))

    assert "model-00002-of-00002.safetensors" in str(refused.value)
    assert "model-00001-of-00002.safetensors" not in str(refused.value)


def test_load_pp_stage_missing_key_raises_naming_it(tmp_path):
    stage = _fake_stage(2, 4)
    _write_pp_checkpoint(tmp_path, stage, drop="model.layers.3.weight")
    with pytest.raises(RuntimeError, match=r"model\.layers\.3\.weight"):
        load_resume_weights(_ctx(stage), str(tmp_path))


def test_pp_ep_guard_reads_where_weights_were_read_not_name_or_path(tmp_path, monkeypatch):
    """``config._name_or_path`` survives a build that read no weights, so a model merely named after the checkpoint
    would resume base experts."""
    stage = _fake_stage(2, 4)
    _write_pp_checkpoint(tmp_path, stage)
    monkeypatch.setattr(pp_mod, "named_ep_layers", lambda module: [("model.nonexistent_moe_block", object())])
    stage.config = SimpleNamespace(_name_or_path=str(tmp_path))  # The weak signal: looks built-from.
    setattr(stage, LOADED_WEIGHTS_FROM_ATTR, "/somewhere/else")  # The truth: weights read elsewhere.

    with pytest.raises(ValueError, match="constructed FROM the checkpoint"):
        load_resume_weights(_ctx(stage), str(tmp_path))


def test_pp_ep_guard_accepts_a_model_whose_weights_were_read_from_the_checkpoint(tmp_path, monkeypatch):
    """Anti-vacuity: the best-model load reads past the skip, so the copy proves the guard let it through."""
    stage = _fake_stage(2, 4)
    saved = _write_pp_checkpoint(tmp_path, stage)
    monkeypatch.setattr(pp_mod, "named_ep_layers", lambda module: [("model.nonexistent_moe_block", object())])
    stage.config = SimpleNamespace(_name_or_path="org/base-model")
    setattr(stage, LOADED_WEIGHTS_FROM_ATTR, str(tmp_path))

    load_resume_weights(_ctx(stage), str(tmp_path), for_best_model=True)

    for local, value in stage.state_dict().items():
        assert torch.equal(value, saved[stage.global_parameter_name(local)]), local


@pytest.mark.parametrize("axis", ["tp_size", "cp_size"])
def test_load_pp_stage_refuses_axes_it_cannot_unshard(tmp_path, axis):
    """TP would leave per-rank slices and CP renames every projection. SUPPORTED_AXIS_SETS rejects both today; this
    keeps the shortcut honest if it widens."""
    stage = _fake_stage(0, 2)
    _write_pp_checkpoint(tmp_path, stage)
    pc = _StubParallelismConfig(**{axis: 2})

    with pytest.raises(NotImplementedError, match="does not handle"):
        load_resume_weights(_ctx(stage, parallelism_config=pc), str(tmp_path))


def test_pp_ep_guard_joins_the_world_even_on_a_moe_free_stage(tmp_path, monkeypatch):
    """A split can leave a stage with no MoE block, and a raise inside the gate would strand MoE-free peers in the
    read consensus below."""
    stage = _fake_stage(0, 2)  # No EP layers on this stage.
    _write_pp_checkpoint(tmp_path, stage)
    real_reject = pp_mod.reject_across_ranks
    joined: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        pp_mod,
        "reject_across_ranks",
        lambda reason, what, **kwargs: joined.append((what, reason)) or real_reject(reason, what, **kwargs),
    )

    load_resume_weights(_ctx(stage), str(tmp_path))

    identity = [reason for what, reason in joined if what == "PP+EP resume construction identity"]
    assert identity == [None], "a MoE-free stage must still enter the identity join, reporting no reason"


def test_load_pp_stage_unreadable_checkpoint_raises_uniformly(tmp_path, monkeypatch):
    stage = _fake_stage(0, 2)
    _write_pp_checkpoint(tmp_path, stage)
    # Failing every join makes the construction verdict reload, so the raise lands on the shard read ahead of key
    # coverage.
    for module in (pp_mod, coordination_mod):
        monkeypatch.setattr(module, "rank_consensus", lambda local: RankConsensus(False, False))
    with pytest.raises(RuntimeError, match="PP checkpoint unreadable on at least one rank"):
        load_resume_weights(_ctx(stage), str(tmp_path))


def _refuse_read(*_args, **_kwargs):
    raise AssertionError("a stage constructed from the checkpoint must not re-read it")


def test_a_stage_built_from_the_checkpoint_skips_the_reload(tmp_path, monkeypatch):
    """A second full read, with its mesh collectives, would only rewrite the same values."""
    stage = _fake_stage(2, 4)
    _write_pp_checkpoint(tmp_path, stage)
    setattr(stage, LOADED_WEIGHTS_FROM_ATTR, str(tmp_path))
    constructed = {name: tensor.clone() for name, tensor in stage.state_dict().items()}
    monkeypatch.setattr(pp_mod, "joined_streaming_reader", _refuse_read)
    load_resume_weights(_ctx(stage), str(tmp_path))

    for name, tensor in stage.state_dict().items():
        assert torch.equal(tensor, constructed[name]), name


def test_the_stage_carries_its_models_construction_stamp(tmp_path, monkeypatch):
    """This stamp lets the resume recognize a stage the stage-aware loader built."""
    cfg = Qwen3Config(**TINY_QWEN3_CONFIG, pad_token_id=0, eos_token_id=1)
    model = Qwen3ForCausalLM(cfg)
    setattr(model, LOADED_WEIGHTS_FROM_ATTR, str(tmp_path))
    stage = build_pipeline_stage(model, 1, PP_SIZE)
    _write_pp_checkpoint(tmp_path, stage)
    monkeypatch.setattr(pp_mod, "joined_streaming_reader", _refuse_read)

    load_resume_weights(_ctx(stage, parallelism_config=_StubParallelismConfig(pp_rank=1)), str(tmp_path))


def test_a_stage_only_named_after_the_checkpoint_reloads(tmp_path):
    """``config._name_or_path`` also names the checkpoint for a ``from_pretrained`` that rounded the FP32 masters."""
    stage = _fake_stage(2, 4)
    saved = _write_pp_checkpoint(tmp_path, stage)
    stage.config = SimpleNamespace(_name_or_path=str(tmp_path))

    load_resume_weights(_ctx(stage), str(tmp_path))

    for local, value in stage.state_dict().items():
        assert torch.equal(value, saved[stage.global_parameter_name(local)]), local


def test_the_skip_needs_every_rank_constructed_from_the_checkpoint(tmp_path, monkeypatch):
    """``realpath`` resolves per node; peers that disagree would be left in the reload's mesh collectives."""
    stage = _fake_stage(2, 4)
    saved = _write_pp_checkpoint(tmp_path, stage)
    setattr(stage, LOADED_WEIGHTS_FROM_ATTR, str(tmp_path))
    peer_disagrees = iter([False])  # The construction verdict's join; every later join is this rank's own.
    monkeypatch.setattr(pp_mod, "rank_consensus", lambda local: RankConsensus(next(peer_disagrees, local), local))

    load_resume_weights(_ctx(stage), str(tmp_path))

    for local, value in stage.state_dict().items():
        assert torch.equal(value, saved[stage.global_parameter_name(local)]), local


def test_a_best_model_load_reads_even_a_stage_built_from_the_checkpoint(tmp_path):
    """The live weights trained past the best checkpoint."""
    stage = _fake_stage(2, 4)
    saved = _write_pp_checkpoint(tmp_path, stage)
    setattr(stage, LOADED_WEIGHTS_FROM_ATTR, str(tmp_path))

    load_resume_weights(_ctx(stage), str(tmp_path), for_best_model=True)

    for local, value in stage.state_dict().items():
        assert torch.equal(value, saved[stage.global_parameter_name(local)]), local


def test_the_reload_reshards_the_stage_before_writing(tmp_path, monkeypatch):
    """Writing to unsharded FSDP2 modules lands in transient buffers the next unshard discards."""
    stage = _fake_stage(2, 4)
    _write_pp_checkpoint(tmp_path, stage)
    events: list[str] = []
    monkeypatch.setattr(pp_mod, "reshard_fsdp2_modules", lambda module: events.append("reshard"))
    real_copy = pp_mod.copy_full_tensor
    monkeypatch.setattr(
        pp_mod, "copy_full_tensor", lambda target, value: events.append("copy") or real_copy(target, value)
    )

    load_resume_weights(_ctx(stage), str(tmp_path))

    assert events[0] == "reshard" and "copy" in events, events


def test_a_stage_shard_absent_from_this_node_names_the_placement(tmp_path):
    """A placement mismatch, not a torn save."""
    stage = _fake_stage(2, 4)
    tensors = _write_pp_checkpoint(tmp_path, stage)
    os.remove(os.path.join(tmp_path, "model.safetensors"))
    index = {"metadata": {}, "weight_map": dict.fromkeys(tensors, "model-pp00001-of-00002.safetensors")}
    with open(os.path.join(tmp_path, "model.safetensors.index.json"), "w") as handle:
        json.dump(index, handle)

    with pytest.raises(RuntimeError, match="same topology and rank placement"):
        load_resume_weights(_ctx(stage), str(tmp_path))


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
    stage, optimizer = _stage_with_optimizer(lo=0, hi=2)  # Live range [0, 2).
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


def test_pp_no_shards_anywhere_is_the_explicit_warm_restart_escape_hatch(tmp_path, monkeypatch):
    """Deleting every optimizer_shard_*.pt and optimizer_meta.pt opts in to a warm restart."""
    stage, optimizer = _stage_with_optimizer()
    pc = _StubParallelismConfig(pp_rank=0)
    restore = _Recorder()
    monkeypatch.setattr(optimizer_mod, "set_optimizer_state_dict", restore)

    OptimizerShardStore(_ctx(stage, optimizer, parallelism_config=pc)).load(str(tmp_path))

    assert restore.calls == []


def test_pp_torn_shard_subset_raises(tmp_path, monkeypatch):
    stage, optimizer = _stage_with_optimizer()
    pc = _StubParallelismConfig(pp_rank=0)
    _write_optimizer_checkpoint(tmp_path, stage, optimizer, pc, shard=False)
    for module in (optimizer_mod, coordination_mod):  # The gates ask both.
        monkeypatch.setattr(module, "rank_consensus", lambda local: RankConsensus(local, True))

    with pytest.raises(RuntimeError, match="torn"):
        OptimizerShardStore(_ctx(stage, optimizer, parallelism_config=pc)).load(str(tmp_path))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
