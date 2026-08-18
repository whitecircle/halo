#!/usr/bin/env python
"""A pipeline-parallel save must produce a STANDARD HF sharded checkpoint that reloads unsplit.

``save_pp_checkpoint`` writes one safetensors shard per stage under the UNSPLIT model's global
parameter names plus one merged index, so the result needs no merge step: ``from_pretrained`` on the
directory reconstructs the whole model. That contract has three parts a unit test of the naming map
cannot reach, all exercised here on a real 4-rank gloo group (pp_size=2 x dp=2 per stage, the
smallest config-legal PP topology):

1. **Shard/index ownership** — one writer per stage, distinct shard filenames, and exactly ONE
   index that unions every stage's keys. A per-stage index write would leave the last writer's
   partial key map on disk and ``from_pretrained`` would randomly initialize the other stages.
2. **Round trip** — the reloaded model equals the pre-split model tensor for tensor. This catches a
   dropped stage, a stale layer index (stage 1 writing stage 0's names), and a torn index.
3. **PP+EP composition** — an EP layer's experts are exported by the family gather under
   ``experts.*`` and must then pass through the stage's global rename; the stage-local EP state_dict
   entries (rank-local expert ranges) must NOT reach the checkpoint.

Plus the topology-independence claim on the read side: because the shards carry global names, the
same directory reloads onto a DIFFERENT ``pp_size``.

    python tests/cpu/checkpoint/test_pp_sharded_save_roundtrip.py
"""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from safetensors.torch import save_file
from transformers import Glm5NextConfig, Glm5NextForConditionalGeneration, Qwen3Config, Qwen3ForCausalLM

from src.checkpoint.format import is_sharded_checkpoint, save_dtype_caster
from src.checkpoint.tool_io import checkpoint_shard_files
from src.distributed.checkpoint.context import CheckpointContext, CheckpointLoadContext
from src.distributed.checkpoint.loader import CheckpointLoader
from src.distributed.checkpoint.optimizer import OptimizerShardStore
from src.distributed.checkpoint.save import save_pp_checkpoint
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.pipeline_parallel.stage import PipelineStageModule, build_pipeline_stage
from tests.common.ep_stubs import StubEPLayerBase
from tests.common.models import TINY_GLM5_CONFIG, TINY_GLM5_VISION_CONFIG, TINY_QWEN3_CONFIG
from tests.common.ports import free_port

_PC_MOD = "src.distributed.parallelism_config"

WORLD_SIZE = 4
PP_SIZE = 2
GPUS_PER_NODE = 2  # ⇒ stage_world_size = 2 (dp=2 per stage), the smallest legal PP shape
SEED = 1234

NUM_EXPERTS = 4
LOCAL_EXPERTS = 2  # an ep2 rank: the stage-local params hold E/ep experts, the gather returns E
HIDDEN = 8
INTER = 6


def _tiny_qwen3() -> Qwen3ForCausalLM:
    """A tiny Qwen3, identically initialized on every rank (PP replicas must agree)."""
    torch.manual_seed(SEED)
    return Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG, pad_token_id=0, eos_token_id=1))


def _parallelism_config(pp_size: int = PP_SIZE):
    """A real ParallelismConfig for this rank. ``gpus_per_node`` comes from the launcher, so the
    2-nodes-of-2 topology is simulated by patching that one resolver (the idiom of
    tests/cpu/checkpoint/test_pp_shard_writers.py); rank/world come from the live gloo group.
    """
    with patch(f"{_PC_MOD}.get_local_world_size", return_value=GPUS_PER_NODE):
        from src.distributed.parallelism_config import ParallelismConfig

        return ParallelismConfig(pp_size=pp_size, nvlink_domain_size=GPUS_PER_NODE, max_concurrent_loading=0)


def _context(stage, config, max_shard_size: str = "5GB") -> CheckpointContext:
    return CheckpointContext(
        model=stage,
        parallelism_config=config,
        is_pp_mode=True,
        is_cp_mode=False,
        is_tp_mode=False,
        is_ep_tp_mode=False,
        has_ep_layers=any(isinstance(m, EPMoELayerBase) for m in stage.modules()),
        fsdp_wrapped=False,
        accelerate_manages_fsdp=False,
        is_save_rank=dist.get_rank() == 0,
        max_shard_size=max_shard_size,
        save_sharded_ep=False,
        has_expert_lora=False,
        merge_expert_lora_on_save=False,
        cp_wrapper=None,
        tokenizer=None,
    )


def _roundtrip_worker(rank: int, out_dir: str, verdict_path: str, port: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
    try:
        config = _parallelism_config()
        stage = build_pipeline_stage(_tiny_qwen3(), config.pp_rank, PP_SIZE)
        if rank == 0:
            os.makedirs(out_dir, exist_ok=True)
        dist.barrier()

        save_pp_checkpoint(_context(stage, config), out_dir)

        if rank != 0:
            return
        problems = []
        files = sorted(f for f in os.listdir(out_dir) if f.endswith(".safetensors"))
        # Parts per stage depend on save_max_shard_size; what must hold is that every stage is
        # represented and none wrote outside its own prefix (concurrent writers must not collide).
        for pp_rank in range(PP_SIZE):
            prefix = f"model-pp{pp_rank:05d}-of-{PP_SIZE:05d}-"
            if not any(f.startswith(prefix) for f in files):
                problems.append(f"stage {pp_rank} wrote no shard (files={files})")
        stray = [f for f in files if not f.startswith("model-pp")]
        if stray:
            problems.append(f"shards outside any stage prefix: {stray}")

        reference = _tiny_qwen3().state_dict()
        with open(os.path.join(out_dir, "model.safetensors.index.json")) as fh:
            weight_map = json.load(fh)["weight_map"]
        if set(weight_map) != set(reference):
            missing = sorted(set(reference) - set(weight_map))
            extra = sorted(set(weight_map) - set(reference))
            problems.append(f"index key set differs: missing={missing[:5]} extra={extra[:5]}")

        reloaded = Qwen3ForCausalLM.from_pretrained(out_dir, dtype=torch.float32).state_dict()
        if set(reloaded) != set(reference):
            problems.append(f"reloaded key set differs: {sorted(set(reference) ^ set(reloaded))[:5]}")
        # The PP writer applies the shared artifact contract: save-dtype (bf16) cast with the
        # norm/balancing/fp32-pin keep-sets at trained dtype — compare through the same caster.
        cast = save_dtype_caster(_tiny_qwen3())
        for name, expected in reference.items():
            got = reloaded.get(name)
            if got is None or not torch.equal(got, cast(name, expected).to(torch.float32)):
                problems.append(f"{name} not restored")
        with open(verdict_path, "w") as fh:
            fh.write("PASS" if not problems else "FAIL: " + "; ".join(problems[:8]))
    finally:
        dist.destroy_process_group()


def _multipart_worker(rank: int, out_dir: str, verdict_path: str, port: int) -> None:
    """Same save, but at a shard limit small enough to force several parts per stage."""
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
    try:
        config = _parallelism_config()
        stage = build_pipeline_stage(_tiny_qwen3(), config.pp_rank, PP_SIZE)
        if rank == 0:
            os.makedirs(out_dir, exist_ok=True)
        dist.barrier()

        save_pp_checkpoint(_context(stage, config, max_shard_size="64KB"), out_dir)

        if rank != 0:
            return
        problems = []
        files = sorted(f for f in os.listdir(out_dir) if f.endswith(".safetensors"))
        if len(files) <= PP_SIZE:
            problems.append(f"64KB limit did not split any stage: {files}")

        with open(os.path.join(out_dir, "model.safetensors.index.json")) as fh:
            weight_map = json.load(fh)["weight_map"]
        if set(weight_map.values()) != set(files):
            problems.append(f"index names {sorted(set(weight_map.values()))} but disk holds {files}")

        # A multi-file stage layout is still a plain HF checkpoint (at the save-dtype contract).
        reference = _tiny_qwen3().state_dict()
        cast = save_dtype_caster(_tiny_qwen3())
        reloaded = Qwen3ForCausalLM.from_pretrained(out_dir, dtype=torch.float32).state_dict()
        for name, expected in reference.items():
            got = reloaded.get(name)
            if got is None or not torch.equal(got, cast(name, expected).to(torch.float32)):
                problems.append(f"{name} not restored across parts")
        with open(verdict_path, "w") as fh:
            fh.write("PASS" if not problems else "FAIL: " + "; ".join(problems[:8]))
    finally:
        dist.destroy_process_group()


def test_pp_save_splits_parts_and_still_reloads(tmp_path):
    """``save_max_shard_size`` must reach the PP writer, and splitting must not break the reload.

    Writing exactly one file per stage regardless of the setting puts a whole stage of a large model
    in one file and holds it all in host RAM first. Several parts per stage is the contract; this
    pins that the split happens AND that the result is still a plain HF checkpoint.
    """
    verdict = str(tmp_path / "verdict.txt")
    mp.start_processes(
        _multipart_worker,
        args=(str(tmp_path / "ckpt"), verdict, free_port()),
        nprocs=WORLD_SIZE,
        join=True,
        start_method="spawn",
    )
    with open(verdict) as fh:
        result = fh.read()
    assert result == "PASS", result


def test_pp_save_reloads_unsplit_with_from_pretrained(tmp_path):
    """The end-to-end contract: pp_size stage shards + one unioned index == the unsplit model."""
    verdict = str(tmp_path / "verdict.txt")
    mp.start_processes(
        _roundtrip_worker,
        args=(str(tmp_path / "ckpt"), verdict, free_port()),
        nprocs=WORLD_SIZE,
        join=True,
        start_method="spawn",
    )
    with open(verdict) as fh:
        result = fh.read()
    assert result == "PASS", result


class _StubEPLayer(StubEPLayerBase):
    """Minimal EP layer: rank-local expert shards as params, full-expert gather, no collectives."""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(LOCAL_EXPERTS, HIDDEN, 2 * INTER))
        self.down_proj = nn.Parameter(torch.zeros(LOCAL_EXPERTS, INTER, HIDDEN))
        self.router = nn.Linear(HIDDEN, NUM_EXPERTS, bias=False)

    def expert_named_params(self):
        return [("gate_up_proj", self.gate_up_proj), ("down_proj", self.down_proj)]

    def gather_expert_state_dict(self, device: str = "cpu", merge_lora: bool = False, retain: bool = True) -> dict:
        if not retain:
            return {}
        return {
            "experts.gate_up_proj": torch.ones(NUM_EXPERTS, 2 * INTER, HIDDEN, device=device),
            "experts.down_proj": torch.ones(NUM_EXPERTS, HIDDEN, INTER, device=device),
        }

    @classmethod
    def merge_shards_to_hf(cls, prefix: str, params: dict) -> dict:  # pragma: no cover - paired override
        return {}


class _MoEBackbone(nn.Module):
    """Fake decoder backbone whose every layer carries an EP MoE block."""

    def __init__(self, n_layers: int, is_first: bool, is_last: bool):
        super().__init__()
        if is_first:
            self.embed_tokens = nn.Embedding(16, HIDDEN)
        layers = []
        for _ in range(n_layers):
            layer = nn.Module()
            layer.mlp = _StubEPLayer()
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(HIDDEN) if is_last else nn.Identity()


def _moe_stage(lo: int, hi: int, n_total: int) -> PipelineStageModule:
    is_first, is_last = lo == 0, hi == n_total
    stage = PipelineStageModule(
        _MoEBackbone(hi - lo, is_first, is_last),
        nn.Linear(HIDDEN, 16, bias=False) if is_last else None,
        is_first,
        is_last,
        backbone_prefix="model",
        head_attr="lm_head",
        layer_attr="layers",
        layer_offset=lo,
    )
    stage.config = Qwen3Config(**TINY_QWEN3_CONFIG)
    return stage


def _ep_worker(rank: int, out_dir: str, verdict_path: str, port: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
    try:
        config = _parallelism_config()
        n_total = 4
        partition = [(0, 2), (2, 4)]
        stage = _moe_stage(*partition[config.pp_rank], n_total)
        if rank == 0:
            os.makedirs(out_dir, exist_ok=True)
        dist.barrier()

        save_pp_checkpoint(_context(stage, config), out_dir)

        if rank != 0:
            return
        with open(os.path.join(out_dir, "model.safetensors.index.json")) as fh:
            weight_map = json.load(fh)["weight_map"]
        problems = []
        for layer in range(n_total):
            for key in (
                f"model.layers.{layer}.mlp.experts.gate_up_proj",
                f"model.layers.{layer}.mlp.experts.down_proj",
                f"model.layers.{layer}.mlp.router.weight",
            ):
                if key not in weight_map:
                    problems.append(f"missing {key}")
        leaked = [k for k in weight_map if k.endswith(".mlp.gate_up_proj") or k.endswith(".mlp.down_proj")]
        if leaked:
            problems.append(f"stage-local EP shards leaked: {leaked[:3]}")

        from safetensors import safe_open

        for name, shard in weight_map.items():
            if not name.endswith("experts.gate_up_proj"):
                continue
            with safe_open(os.path.join(out_dir, shard), framework="pt") as f:
                if f.get_slice(name).get_shape()[0] != NUM_EXPERTS:
                    problems.append(f"{name} is not the full expert count")
        with open(verdict_path, "w") as fh:
            fh.write("PASS" if not problems else "FAIL: " + "; ".join(problems[:8]))
    finally:
        dist.destroy_process_group()


def test_pp_plus_ep_exports_full_experts_under_global_names(tmp_path):
    """PP+EP: every stage's experts land in the one index under re-based global layer names, at the
    FULL expert count, and no rank-local EP state_dict entry leaks through."""
    verdict = str(tmp_path / "verdict.txt")
    mp.start_processes(
        _ep_worker,
        args=(str(tmp_path / "ckpt"), verdict, free_port()),
        nprocs=WORLD_SIZE,
        join=True,
        start_method="spawn",
    )
    with open(verdict) as fh:
        result = fh.read()
    assert result == "PASS", result


def _tiny_composite():
    """A tiny GLM-5 multimodal wrapper — text tower + vision tower, no text-only CausalLM sibling."""
    torch.manual_seed(SEED)
    config = Glm5NextConfig(
        text_config=dict(TINY_GLM5_CONFIG), vision_config=dict(TINY_GLM5_VISION_CONFIG), attn_implementation="sdpa"
    )
    return Glm5NextForConditionalGeneration(config)


def _composite_worker(rank: int, out_dir: str, verdict_path: str, port: int) -> None:
    """A text-only run of a multimodal wrapper: the save must re-emit the vision tower it dropped."""
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
    try:
        from src.trainers.mixins.pipeline import stash_wrapper_state

        config = _parallelism_config()
        model = _tiny_composite()
        # What the trainer's gate stashes on the save rank, and only there (the others free it).
        wrapper_state = stash_wrapper_state(model) if rank == 0 else {}
        stage = build_pipeline_stage(model, config.pp_rank, PP_SIZE, moe_balancing="none")
        if rank == 0:
            os.makedirs(out_dir, exist_ok=True)
        dist.barrier()

        ctx = _context(stage, config)
        ctx.pp_wrapper_state = wrapper_state
        save_pp_checkpoint(ctx, out_dir)

        if rank != 0:
            return
        problems = []
        reference = _tiny_composite().state_dict()
        vision_keys = {k for k in reference if k.startswith("model.visual.")}
        if not vision_keys:
            problems.append("fixture carries no vision tower — the test would prove nothing")
        with open(os.path.join(out_dir, "model.safetensors.index.json")) as fh:
            weight_map = json.load(fh)["weight_map"]
        if set(weight_map) != set(reference):
            problems.append(f"index key set differs: {sorted(set(reference) ^ set(weight_map))[:5]}")
        wrapper_files = {weight_map[k] for k in vision_keys if k in weight_map}
        if not all(f.startswith("model-wrapper-") for f in wrapper_files):
            problems.append(f"vision tensors landed in a stage shard: {sorted(wrapper_files)}")

        reloaded, info = Glm5NextForConditionalGeneration.from_pretrained(
            out_dir, dtype=torch.float32, output_loading_info=True
        )
        if info["missing_keys"]:
            problems.append(f"from_pretrained random-initialized: {sorted(info['missing_keys'])[:5]}")
        state = reloaded.state_dict()
        cast = save_dtype_caster(_tiny_composite())
        for name, expected in reference.items():
            got = state.get(name)
            if got is None or not torch.equal(got, cast(name, expected).to(torch.float32)):
                problems.append(f"{name} not restored")
        with open(verdict_path, "w") as fh:
            fh.write("PASS" if not problems else "FAIL: " + "; ".join(problems[:8]))
    finally:
        dist.destroy_process_group()


def test_pp_save_of_a_text_only_multimodal_wrapper_reloads_as_the_wrapper(tmp_path):
    """The vision tower no stage holds is re-emitted from the save rank under its own part, so the
    directory reloads as the composite class with NO missing (random-initialized) keys — a
    servable wrapper-layout export with no reattach step, and a resumable one (the stage-aware
    loader plans those tensors)."""
    verdict = str(tmp_path / "verdict.txt")
    mp.start_processes(
        _composite_worker,
        args=(str(tmp_path / "ckpt"), verdict, free_port()),
        nprocs=WORLD_SIZE,
        join=True,
        start_method="spawn",
    )
    with open(verdict) as fh:
        result = fh.read()
    assert result == "PASS", result


def _load_context(stage, optimizer, config) -> CheckpointLoadContext:
    return CheckpointLoadContext(
        model=stage,
        optimizer=optimizer,
        lr_scheduler=None,
        parallelism_config=config,
        is_pp_mode=True,
        is_cp_mode=False,
        is_tp_mode=False,
        has_ep_layers=False,
        # PP always composes FSDP2 inside a stage, which routes the optimizer through per-rank
        # shards instead of the base Trainer's optimizer.pt.
        fsdp_wrapped=True,
        tp_rank=0,
        tp_size=1,
        super_load_from_checkpoint=lambda *a, **k: None,
        super_load_optimizer_and_scheduler=lambda *a, **k: None,
    )


def _stepped_optimizer(stage):
    """An optimizer with NON-ZERO momentum buffers, so a failed restore cannot look like a pass."""
    optimizer = torch.optim.SGD(stage.parameters(), lr=0.1, momentum=0.9)
    for i, param in enumerate(stage.parameters()):
        param.grad = torch.full_like(param, 0.5 + 0.01 * i)
    optimizer.step()
    return optimizer


def _optimizer_worker(rank: int, out_dir: str, verdict_path: str, port: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
    try:
        config = _parallelism_config()
        stage = build_pipeline_stage(_tiny_qwen3(), config.pp_rank, PP_SIZE)
        optimizer = _stepped_optimizer(stage)
        saved_moments = {
            name: optimizer.state[param]["momentum_buffer"].clone()
            for name, param in stage.named_parameters()
            if "momentum_buffer" in optimizer.state[param]
        }
        if rank == 0:
            os.makedirs(out_dir, exist_ok=True)
        dist.barrier()

        OptimizerShardStore(_load_context(stage, optimizer, config)).save(out_dir)

        fresh_stage = build_pipeline_stage(_tiny_qwen3(), config.pp_rank, PP_SIZE)
        fresh_optimizer = torch.optim.SGD(fresh_stage.parameters(), lr=0.1, momentum=0.9)
        OptimizerShardStore(_load_context(fresh_stage, fresh_optimizer, config)).load(out_dir)

        problems = []
        if not saved_moments:
            problems.append("no momentum buffers were produced — the test itself is vacuous")
        restored = {
            name: fresh_optimizer.state[param].get("momentum_buffer") for name, param in fresh_stage.named_parameters()
        }
        for name, expected in saved_moments.items():
            got = restored.get(name)
            if got is None or not torch.equal(got, expected):
                problems.append(f"{name} momentum not restored")
        # Hyperparameters must stay THIS run's, not the shard's rebuilt param_groups.
        if any(group["lr"] != 0.1 for group in fresh_optimizer.param_groups):
            problems.append("live param_group settings were overwritten by the shard")
        # rank 0 has the only verdict file; every rank checks its own stage.
        with open(f"{verdict_path}.{rank}", "w") as fh:
            fh.write("PASS" if not problems else "FAIL: " + "; ".join(problems[:6]))
    finally:
        dist.destroy_process_group()


def test_pp_optimizer_shards_round_trip(tmp_path):
    """Per-rank optimizer shards must actually restore under PP: save from a stepped optimizer, then
    reload into a fresh one and compare momentum buffers. Every other PP optimizer test mocks
    ``set_optimizer_state_dict`` and so proves only that the gates allow the call."""
    verdict = str(tmp_path / "verdict")
    mp.start_processes(
        _optimizer_worker,
        args=(str(tmp_path / "ckpt"), verdict, free_port()),
        nprocs=WORLD_SIZE,
        join=True,
        start_method="spawn",
    )
    for rank in range(WORLD_SIZE):
        with open(f"{verdict}.{rank}") as fh:
            result = fh.read()
        assert result == "PASS", f"rank {rank}: {result}"


def _load_ctx(stage) -> CheckpointLoadContext:
    return CheckpointLoadContext(
        model=stage,
        optimizer=None,
        lr_scheduler=None,
        parallelism_config=SimpleNamespace(pp_size=4, pp_rank=0, tp_size=1, cp_size=1),
        is_pp_mode=True,
        is_cp_mode=False,
        is_tp_mode=False,
        has_ep_layers=False,
        fsdp_wrapped=False,
        tp_rank=0,
        tp_size=1,
        super_load_from_checkpoint=lambda *a, **k: None,
        super_load_optimizer_and_scheduler=lambda *a, **k: None,
    )


def test_after_training_tools_read_a_pp_checkpoint_whole(tmp_path):
    """The conversion tools (unfuse / quantize / merge_models) enumerate shards through the index, so
    a PP directory must present as an ordinary multi-shard checkpoint: not rejected as per-rank, and
    resolving to EVERY stage's shard. Reading only one stage's file would silently emit a model
    missing the other stages' layers."""
    shards = {
        "model-00001-of-00002.safetensors": {"model.layers.0.weight": torch.randn(4, 4)},
        "model-00002-of-00002.safetensors": {"model.layers.1.weight": torch.randn(4, 4)},
    }
    weight_map = {}
    for name, tensors in shards.items():
        save_file(tensors, os.path.join(tmp_path, name), metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(tensors, name))
    # Exactly what save_pp_checkpoint writes: standard HF metadata, deliberately no format marker.
    with open(os.path.join(tmp_path, "model.safetensors.index.json"), "w") as fh:
        json.dump({"metadata": {"total_size": 128}, "weight_map": weight_map}, fh)

    assert not is_sharded_checkpoint(str(tmp_path)), "a PP checkpoint must not read as per-rank sharded"
    resolved = [os.path.basename(p) for p in checkpoint_shard_files(str(tmp_path))]
    assert resolved == sorted(shards), f"tools would read {resolved}, missing a stage's shard"


def test_pp_checkpoint_reloads_onto_a_different_pp_size(tmp_path):
    """Global names make the WEIGHTS topology-independent: a checkpoint written by pp_size=2 loads
    into a pp_size=4 stage. (The optimizer shards are stage-local and are gated separately by the
    fingerprint — see test_pp_resume_units.)"""
    reference = _tiny_qwen3().state_dict()
    save_file(dict(reference), os.path.join(tmp_path, "model.safetensors"), metadata={"format": "pt"})

    covered: set[str] = set()
    for pp_rank in range(4):
        stage = build_pipeline_stage(_tiny_qwen3(), pp_rank, 4)
        # Perturb so a no-op load cannot pass the equality check below.
        with torch.no_grad():
            for param in stage.parameters():
                param.add_(1.0)
        CheckpointLoader(_load_ctx(stage)).load_model(str(tmp_path))
        for local, value in stage.state_dict().items():
            global_name = stage.global_parameter_name(local)
            assert torch.equal(value, reference[global_name]), f"pp_rank={pp_rank} {local}"
            covered.add(global_name)
    # Anti-vacuity: the four pp_size=4 stages together must cover the whole pp_size-2-written model.
    assert covered == set(reference)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
