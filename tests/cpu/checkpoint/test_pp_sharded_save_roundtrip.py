#!/usr/bin/env python
"""A pipeline-parallel save is a standard HF checkpoint that reloads unsplit, on a real 4-rank gloo group. A
per-stage index would leave a partial key map, randomly initializing other stages."""

from __future__ import annotations

import datetime
import itertools
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, Glm5NextConfig, Glm5NextForConditionalGeneration, Qwen3Config, Qwen3ForCausalLM

from halo.checkpoint.hub_layout import export_conversions, revert_conversions
from halo.checkpoint.parallel.context import CheckpointLoadContext
from halo.checkpoint.parallel.ep import save_ep_gathered_checkpoint
from halo.checkpoint.parallel.optimizer import OptimizerShardStore
from halo.checkpoint.parallel.pp import _merge_writer_fragments, is_pp_shard_writer, save_pp_checkpoint
from halo.checkpoint.parallel.resume_weights import load_resume_weights
from halo.checkpoint.reading import checkpoint_shard_files, is_ep_sharded_checkpoint
from halo.checkpoint.weights_source import LOADED_WEIGHTS_FROM_ATTR
from halo.checkpoint.writing import save_dtype_caster
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase, named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.distributed.pipeline_parallel.stage import PipelineStageModule, build_pipeline_stage
from halo.model_loading.parallel.pp import _load_stage_lazy
from halo.models.attention.gpt_oss_sinks import neutralized_sink_value
from halo.models.introspection.param_roles import persistent_buffers
from halo.runtime import dtensor
from halo.trainers.parallelism.pipeline_trainer import stash_wrapper_state
from tests.common.checkpoint_io import load_context, safetensors_state_dict, save_context
from tests.common.ep_stubs import StubEPLayerBase
from tests.common.gloo import new_groups, run_gloo_ranks
from tests.common.models import (
    TINY_GLM5_CONFIG,
    TINY_GLM5_VISION_CONFIG,
    TINY_GPTOSS_CONFIG,
    TINY_QWEN3_CONFIG,
)
from tests.common.parallelism import single_process_ep_config
from tests.common.pp_fakes import hand_built_stage, registered_pp_backbones
from tests.common.tiny_models import (
    TINY_DENSE_FAMILIES,
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    TINY_PP_SPLIT_OVERRIDES,
    tiny_family_model,
)

_PC_MOD = "halo.distributed.parallelism_config"

WORLD_SIZE = 4
PP_SIZE = 2
GPUS_PER_NODE = 2  # ⇒ stage_world_size = 2 (dp=2 per stage), the smallest legal PP shape.
SEED = 1234

NUM_EXPERTS = 4
LOCAL_EXPERTS = 2  # An ep2 rank: the stage-local params hold E/ep experts, the gather returns E.
HIDDEN = 8
INTER = 6
ETP_SIZE = 2  # A pure-ETP stage: both ranks hold every expert at INTER / ETP_SIZE.
PG_TIMEOUT = datetime.timedelta(seconds=120)

# Each stage spans two nodes, where the per-node writer set (one rank per node) differs from the shared one (one per
# stage).
PER_NODE_WORLD = 8
RANKS_PER_NODE = 2
# A dense model, and one whose hub names form converter groups (Inkling's shared experts).
PER_NODE_FAMILIES = ("qwen3", "inkling_mm_model")

# Inkling's text tower keeps ``embed_norm`` inside the embedding, which only stage 0 may write. Laguna's renames sit in
# its EP layer, whose gather already writes them.
PP_FAMILIES = {**TINY_DENSE_FAMILIES, **TINY_MOE_FAMILIES, **TINY_MOE_VLM_FAMILIES}
PP_EXPORT_FAMILIES = (
    "deepseek_v4",
    "glm5_next",
    "step3p7",
    "laguna",
    "mistral3",
    "inkling_mm_model",
    "inkling_text",
    "qwen3_moe",
    "glm4_moe_lite",
)
PP_RENAME_FREE = frozenset({"inkling_text", "qwen3_moe", "glm4_moe_lite"})
# A run without EP layers holds them here: each decoder layer's own tensors carry its reverses.
PP_UNWRAPPED_FAMILIES = ("deepseek_v4", "glm5_next", "step3p7", "mistral3", "inkling_mm_model")
PP_FAMILY_SPLITS = {"laguna": [4, 4], "gpt_oss": [4, 4]}
UNREGISTERED_PP_BACKBONES = ("Mistral4Model",)


def _tiny_qwen3() -> Qwen3ForCausalLM:
    torch.manual_seed(SEED)
    return Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG, pad_token_id=0, eos_token_id=1))


def _parallelism_config(pp_size: int = PP_SIZE, **fields):
    """``gpus_per_node`` comes from the launcher, so the 2-nodes-of-2 topology is simulated by patching that one
    resolver."""
    with patch(f"{_PC_MOD}.get_local_world_size", return_value=GPUS_PER_NODE):
        return ParallelismConfig(pp_size=pp_size, nvlink_domain_size=GPUS_PER_NODE, max_concurrent_loading=0, **fields)


def _context(stage, config, max_shard_size: str = "5GB"):
    has_ep_layers = any(isinstance(m, EPMoELayerBase) for m in stage.modules())
    return save_context(stage, config, has_ep_layers=has_ep_layers, max_shard_size=max_shard_size)


def _roundtrip_worker(rank: int, out_dir: str, verdict_path: str) -> None:
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
    # Parts per stage depend on save_max_shard_size; concurrent writers must not collide.
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
    # The PP writer casts with the norm/balancing/fp32-pin keep-sets at trained dtype, so compare through the same
    # caster.
    cast = save_dtype_caster(_tiny_qwen3())
    for name, expected in reference.items():
        got = reloaded.get(name)
        if got is None or not torch.equal(got, cast(name, expected).to(torch.float32)):
            problems.append(f"{name} not restored")
    with open(verdict_path, "w") as fh:
        fh.write("PASS" if not problems else "FAIL: " + "; ".join(problems[:8]))


def _multipart_worker(rank: int, out_dir: str, verdict_path: str) -> None:
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

    reference = _tiny_qwen3().state_dict()
    cast = save_dtype_caster(_tiny_qwen3())
    reloaded = Qwen3ForCausalLM.from_pretrained(out_dir, dtype=torch.float32).state_dict()
    for name, expected in reference.items():
        got = reloaded.get(name)
        if got is None or not torch.equal(got, cast(name, expected).to(torch.float32)):
            problems.append(f"{name} not restored across parts")
    with open(verdict_path, "w") as fh:
        fh.write("PASS" if not problems else "FAIL: " + "; ".join(problems[:8]))


def test_pp_save_splits_parts_and_still_reloads(tmp_path):
    """One file per stage would hold a whole stage of a large model in host RAM first."""
    verdict = str(tmp_path / "verdict.txt")
    run_gloo_ranks(_multipart_worker, WORLD_SIZE, str(tmp_path / "ckpt"), verdict)
    with open(verdict) as fh:
        result = fh.read()
    assert result == "PASS", result


def test_pp_save_reloads_unsplit_with_from_pretrained(tmp_path):
    verdict = str(tmp_path / "verdict.txt")
    run_gloo_ranks(_roundtrip_worker, WORLD_SIZE, str(tmp_path / "ckpt"), verdict)
    with open(verdict) as fh:
        result = fh.read()
    assert result == "PASS", result


def _host_copy_worker(rank: int, out_dir: str, verdict_path: str) -> None:
    config = _parallelism_config()
    stage = build_pipeline_stage(_tiny_qwen3(), config.pp_rank, PP_SIZE)
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    dist.barrier()

    original = dtensor.full_cpu_tensor
    copies: list[torch.Size] = []

    def counting(tensor):
        copies.append(tensor.shape)
        return original(tensor)

    bound = [module for module in list(sys.modules.values()) if getattr(module, "full_cpu_tensor", None) is original]
    for module in bound:
        module.full_cpu_tensor = counting
    try:
        save_pp_checkpoint(_context(stage, config), out_dir)
    finally:
        for module in bound:
            module.full_cpu_tensor = original

    is_writer = is_pp_shard_writer(config, shared_fs=True)
    if is_writer and not copies:
        verdict = "FAIL: the writer resolved no tensor through the counted primitive (the probe is blind)"
    elif not is_writer and copies:
        verdict = f"FAIL: non-writer rank {rank} paid {len(copies)} host copies (first {list(copies[0])})"
    else:
        verdict = "PASS"
    with open(f"{verdict_path}.{rank}", "w") as fh:
        fh.write(verdict)


def test_pp_save_non_writers_pay_no_host_copy(tmp_path):
    """On a node that would be ``local_world_size - 1`` extra host copies of every tensor at every save."""
    verdict = str(tmp_path / "verdict")
    run_gloo_ranks(_host_copy_worker, WORLD_SIZE, str(tmp_path / "ckpt"), verdict)
    for rank in range(WORLD_SIZE):
        with open(f"{verdict}.{rank}") as fh:
            result = fh.read()
        assert result == "PASS", f"rank {rank}: {result}"


def _per_node_worker(rank: int, root: str, verdict_path: str, name: str) -> None:
    os.environ["LOCAL_RANK"] = str(rank % RANKS_PER_NODE)
    config = ParallelismConfig(pp_size=PP_SIZE, nvlink_domain_size=RANKS_PER_NODE, max_concurrent_loading=0)
    node = rank // RANKS_PER_NODE
    nodes = PER_NODE_WORLD // RANKS_PER_NODE
    node_dir = os.path.join(root, f"node{node}")
    if rank % RANKS_PER_NODE == 0:
        os.makedirs(node_dir)
    dist.barrier()

    model = _pp_family_model(name, ep_wrapped=False)
    save_pp_checkpoint(_context(_pp_family_stage(name, config.pp_rank, model), config), node_dir)

    problems = []
    shards = sorted(f for f in os.listdir(node_dir) if f.endswith(".safetensors"))
    own_prefix = f"model-pp{config.pp_rank:05d}-of-{PP_SIZE:05d}-"
    if not shards or any(not f.startswith(own_prefix) for f in shards):
        problems.append(f"node {node} (stage {config.pp_rank}) holds {shards}, not its own stage's shard alone")

    stage = _pp_family_stage(name, config.pp_rank, _pp_family_model(name, ep_wrapped=False))
    with torch.no_grad():
        for param in stage.parameters():
            param.add_(1.0)
    load_resume_weights(_load_ctx(stage), node_dir)
    unsplit = _pp_family_model(name, ep_wrapped=False)
    cast, reference = save_dtype_caster(unsplit), unsplit.state_dict()
    for local, value in stage.state_dict().items():
        global_name = stage.global_parameter_name(local)
        if not torch.equal(value, cast(global_name, reference[global_name]).to(value.dtype)):
            problems.append(f"{global_name} did not resume from node {node}'s directory")

    # Another stage's node directory has no shard for this stage's layers.
    elsewhere = os.path.join(root, f"node{(node + nodes // 2) % nodes}")
    try:
        load_resume_weights(_load_ctx(stage), elsewhere)
        problems.append(f"stage {config.pp_rank} resumed from another stage's node directory {elsewhere}")
    except RuntimeError:
        pass
    with open(f"{verdict_path}.{rank}", "w") as fh:
        fh.write("PASS" if not problems else "FAIL: " + "; ".join(problems[:6]))


@pytest.mark.parametrize("name", PER_NODE_FAMILIES)
def test_per_node_pp_checkpoint_resumes_each_node_and_refuses_a_moved_stage(tmp_path, name):
    """A stage moved onto another stage's node fails on every rank instead of resuming base weights."""
    verdict = str(tmp_path / "verdict")
    run_gloo_ranks(
        _per_node_worker,
        PER_NODE_WORLD,
        str(tmp_path),
        verdict,
        name,
        env={"DIST_OUTPUT_SHARED_FILESYSTEM": "0", "LOCAL_WORLD_SIZE": str(RANKS_PER_NODE)},
    )
    for rank in range(PER_NODE_WORLD):
        with open(f"{verdict}.{rank}") as fh:
            result = fh.read()
        assert result == "PASS", f"rank {rank}: {result}"


class _StubEPLayer(StubEPLayerBase):
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


def _full_etp_experts(layer: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integer-valued so the bf16 export holds them exactly."""
    generator = torch.Generator().manual_seed(layer)
    shapes = ((NUM_EXPERTS, HIDDEN, INTER), (NUM_EXPERTS, HIDDEN, INTER), (NUM_EXPERTS, INTER, HIDDEN))
    gate, up, down = (torch.randint(-8, 8, shape, generator=generator).float() for shape in shapes)
    return gate, up, down


class _ETPStubLayer(StubEPLayerBase):
    """Exported by the real EP gather over the stage's expert-TP group."""

    def __init__(self, layer: int, expert_tp_group, expert_tp_rank: int):
        super().__init__()
        self.expert_tp_size = ETP_SIZE
        self.expert_tp_group = expert_tp_group
        self.ep_config = SimpleNamespace(ep_size=1, dispatch_ep_group=None)
        gate, up, down = _full_etp_experts(layer)
        width = INTER // ETP_SIZE
        shard = slice(expert_tp_rank * width, (expert_tp_rank + 1) * width)
        self.gate_proj = nn.Parameter(gate[..., shard].clone())
        self.up_proj = nn.Parameter(up[..., shard].clone())
        self.down_proj = nn.Parameter(down[:, shard].clone())
        self.router = nn.Linear(HIDDEN, NUM_EXPERTS, bias=False)


class _MoEBackbone(nn.Module):
    def __init__(self, layer_range: range, is_first: bool, is_last: bool, make_mlp):
        super().__init__()
        if is_first:
            self.embed_tokens = nn.Embedding(16, HIDDEN)
        layers = []
        for index in layer_range:
            layer = nn.Module()
            layer.mlp = make_mlp(index)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(HIDDEN) if is_last else nn.Identity()


def _moe_stage(lo: int, hi: int, n_total: int, make_mlp=lambda _layer: _StubEPLayer()) -> PipelineStageModule:
    is_first, is_last = lo == 0, hi == n_total
    stage = hand_built_stage(
        _MoEBackbone(range(lo, hi), is_first, is_last, make_mlp),
        nn.Linear(HIDDEN, 16, bias=False) if is_last else None,
        is_first=is_first,
        is_last=is_last,
        layer_offset=lo,
    )
    stage.config = Qwen3Config(**TINY_QWEN3_CONFIG)
    return stage


def _ep_worker(rank: int, out_dir: str, verdict_path: str) -> None:
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

    for name, shard in weight_map.items():
        if not name.endswith("experts.gate_up_proj"):
            continue
        with safe_open(os.path.join(out_dir, shard), framework="pt") as f:
            if f.get_slice(name).get_shape()[0] != NUM_EXPERTS:
                problems.append(f"{name} is not the full expert count")
    with open(verdict_path, "w") as fh:
        fh.write("PASS" if not problems else "FAIL: " + "; ".join(problems[:8]))


def test_pp_plus_ep_exports_full_experts_under_global_names(tmp_path):
    """No rank-local EP state_dict entry may leak through."""
    verdict = str(tmp_path / "verdict.txt")
    run_gloo_ranks(_ep_worker, WORLD_SIZE, str(tmp_path / "ckpt"), verdict)
    with open(verdict) as fh:
        result = fh.read()
    assert result == "PASS", result


def _etp_worker(rank: int, out_dir: str, verdict_path: str) -> None:
    config = _parallelism_config(expert_tp_size=ETP_SIZE)
    stage_ranks = [
        list(range(base, base + config.stage_world_size)) for base in range(0, WORLD_SIZE, config.stage_world_size)
    ]
    expert_tp_group = new_groups(rank, stage_ranks, PG_TIMEOUT)
    n_total = 4
    lo, hi = [(0, 2), (2, 4)][config.pp_rank]
    stage = _moe_stage(lo, hi, n_total, lambda layer: _ETPStubLayer(layer, expert_tp_group, config.stage_local_rank))
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    dist.barrier()

    save_pp_checkpoint(_context(stage, config), out_dir)

    if rank != 0:
        return
    written = safetensors_state_dict(out_dir)
    problems = []
    for layer in range(n_total):
        gate, up, down = _full_etp_experts(layer)
        expected = {
            f"model.layers.{layer}.mlp.experts.gate_up_proj": torch.cat([gate, up], dim=-1).transpose(1, 2),
            f"model.layers.{layer}.mlp.experts.down_proj": down.transpose(1, 2),
        }
        for key, tensor in expected.items():
            got = written.get(key)
            if got is None or not torch.equal(got.float(), tensor):
                problems.append(f"{key} {'missing' if got is None else 'is not the unsplit experts'}")
        if f"model.layers.{layer}.mlp.router.weight" not in written:
            problems.append(f"layer {layer} router missing")
    leaked = [key for key in written if key.endswith((".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj"))]
    if leaked:
        problems.append(f"expert-TP shards leaked: {leaked[:3]}")
    with open(verdict_path, "w") as fh:
        fh.write("PASS" if not problems else "FAIL: " + "; ".join(problems[:8]))


def test_pp_plus_etp_exports_unsplit_experts_under_global_names(tmp_path):
    """The stage's one writer exports them whole, with both ranks in the gather."""
    verdict = str(tmp_path / "verdict.txt")
    run_gloo_ranks(_etp_worker, WORLD_SIZE, str(tmp_path / "ckpt"), verdict, pg_timeout=PG_TIMEOUT)
    with open(verdict) as fh:
        result = fh.read()
    assert result == "PASS", result


def _tiny_composite():
    torch.manual_seed(SEED)
    config = Glm5NextConfig(
        text_config=dict(TINY_GLM5_CONFIG), vision_config=dict(TINY_GLM5_VISION_CONFIG), attn_implementation="sdpa"
    )
    return Glm5NextForConditionalGeneration(config)


def _composite_worker(rank: int, out_dir: str, verdict_path: str) -> None:
    config = _parallelism_config()
    model = _tiny_composite()
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
    unsplit = _tiny_composite()
    reference = unsplit.state_dict()
    # As the EP-gathered save writes them (the tower's untouched, its text tower's hyper-connections respelled).
    conversions = export_conversions(unsplit)
    hub_keys = set(revert_conversions(unsplit, dict(reference), conversions))
    vision_keys = set(
        revert_conversions(unsplit, {k: v for k, v in reference.items() if k.startswith("model.visual.")}, conversions)
    )
    if not vision_keys:
        problems.append("fixture carries no vision tower — the test would prove nothing")
    if hub_keys == set(reference):
        problems.append("premise: the fixture's load renames nothing, so the namespace check proves nothing")
    with open(os.path.join(out_dir, "model.safetensors.index.json")) as fh:
        weight_map = json.load(fh)["weight_map"]
    if set(weight_map) != hub_keys:
        problems.append(f"index key set differs from the hub's: {sorted(hub_keys ^ set(weight_map))[:5]}")
    wrapper_files = {weight_map.get(k) for k in vision_keys}
    if not all(f is not None and f.startswith("model-wrapper-") for f in wrapper_files):
        problems.append(f"vision tensors missing or in a stage shard: {sorted(map(str, wrapper_files))}")

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


def test_pp_save_of_a_text_only_multimodal_wrapper_reloads_as_the_wrapper(tmp_path):
    """Reloads as the composite class with no random-initialized keys, so no reattach step is needed."""
    verdict = str(tmp_path / "verdict.txt")
    run_gloo_ranks(_composite_worker, WORLD_SIZE, str(tmp_path / "ckpt"), verdict)
    with open(verdict) as fh:
        result = fh.read()
    assert result == "PASS", result


def test_a_writers_stage_and_wrapper_bytes_count_apart():
    """Each node's save rank writes the same-named wrapper part, so a writer's one fragment would be skipped whole."""
    wrapper = ({"model.visual.weight": "model-wrapper-00001.safetensors"}, 7)
    stages = [
        ({f"model.layers.{stage}.weight": f"model-pp{stage:05d}-of-00002-00001.safetensors"}, 100)
        for stage in range(2)
    ]

    weight_map, total = _merge_writer_fragments([[stages[0], wrapper], None, [stages[1], wrapper], None])

    assert total == 207
    assert set(weight_map) == {"model.layers.0.weight", "model.layers.1.weight", "model.visual.weight"}


def _per_node_wrapper_worker(rank: int, root: str, verdict_path: str) -> None:
    os.environ["LOCAL_RANK"] = str(rank % GPUS_PER_NODE)
    config = _parallelism_config()
    model = _tiny_composite()
    is_node_writer = rank % GPUS_PER_NODE == 0
    wrapper_state = stash_wrapper_state(model) if is_node_writer else {}
    stage = build_pipeline_stage(model, config.pp_rank, PP_SIZE, moe_balancing="none")
    node_dirs = [os.path.join(root, f"node{node}") for node in range(WORLD_SIZE // GPUS_PER_NODE)]
    if is_node_writer:
        os.makedirs(node_dirs[rank // GPUS_PER_NODE])
    dist.barrier()

    ctx = _context(stage, config)
    ctx.pp_wrapper_state = wrapper_state
    save_pp_checkpoint(ctx, node_dirs[rank // GPUS_PER_NODE])
    dist.barrier()

    if rank != 0:
        return
    payload_bytes: dict[str, int] = {}
    for node_dir in node_dirs:
        for name in os.listdir(node_dir):
            if name.endswith(".safetensors"):
                with safe_open(os.path.join(node_dir, name), framework="pt") as reader:
                    payload_bytes[name] = sum(reader.get_tensor(key).nbytes for key in reader.keys())  # noqa: SIM118
    problems = []
    if sum(name.startswith("model-wrapper-") for name in payload_bytes) != 1:
        problems.append(f"premise: both nodes write one same-named wrapper part: {sorted(payload_bytes)}")
    with open(os.path.join(node_dirs[0], "model.safetensors.index.json")) as fh:
        total_size = json.load(fh)["metadata"]["total_size"]
    if total_size != sum(payload_bytes.values()):
        problems.append(f"total_size {total_size} != {sum(payload_bytes.values())} bytes across {payload_bytes}")
    with open(verdict_path, "w") as fh:
        fh.write("PASS" if not problems else "FAIL: " + "; ".join(problems))


def test_a_per_node_pp_save_of_a_multimodal_wrapper_indexes_every_parts_bytes(tmp_path):
    verdict = str(tmp_path / "verdict.txt")
    run_gloo_ranks(
        _per_node_wrapper_worker,
        WORLD_SIZE,
        str(tmp_path),
        verdict,
        env={"DIST_OUTPUT_SHARED_FILESYSTEM": "0", "LOCAL_WORLD_SIZE": str(GPUS_PER_NODE)},
    )
    with open(verdict) as fh:
        result = fh.read()
    assert result == "PASS", result


def _load_context(stage, optimizer, config) -> CheckpointLoadContext:
    # PP always composes FSDP2 inside a stage, which routes the optimizer through per-rank shards.
    return load_context(stage, config, optimizer=optimizer, fsdp_wrapped=True)


def _stepped_optimizer(stage):
    """Non-zero momentum buffers, so a failed restore cannot look like a pass."""
    optimizer = torch.optim.SGD(stage.parameters(), lr=0.1, momentum=0.9)
    for i, param in enumerate(stage.parameters()):
        param.grad = torch.full_like(param, 0.5 + 0.01 * i)
    optimizer.step()
    return optimizer


def _optimizer_worker(rank: int, out_dir: str, verdict_path: str) -> None:
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
    # Hyperparameters stay this run's, not the shard's rebuilt param_groups.
    if any(group["lr"] != 0.1 for group in fresh_optimizer.param_groups):
        problems.append("live param_group settings were overwritten by the shard")
    with open(f"{verdict_path}.{rank}", "w") as fh:
        fh.write("PASS" if not problems else "FAIL: " + "; ".join(problems[:6]))


def test_pp_optimizer_shards_round_trip(tmp_path):
    """Every other PP optimizer test mocks ``set_optimizer_state_dict`` and so proves only that the gates allow the
    call."""
    verdict = str(tmp_path / "verdict")
    run_gloo_ranks(_optimizer_worker, WORLD_SIZE, str(tmp_path / "ckpt"), verdict)
    for rank in range(WORLD_SIZE):
        with open(f"{verdict}.{rank}") as fh:
            result = fh.read()
        assert result == "PASS", f"rank {rank}: {result}"


def _load_ctx(stage) -> CheckpointLoadContext:
    config = SimpleNamespace(
        pp_size=4, pp_rank=0, tp_size=1, cp_size=1, is_pp_mode=True, is_cp_mode=False, is_tp_mode=False
    )
    return load_context(stage, config)


def test_after_training_tools_read_a_pp_checkpoint_whole(tmp_path):
    """Reading only one stage's file would silently emit a model missing the other stages' layers."""
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

    assert not is_ep_sharded_checkpoint(str(tmp_path)), "a PP checkpoint must not read as per-rank sharded"
    resolved = [os.path.basename(p) for p in checkpoint_shard_files(str(tmp_path))]
    assert resolved == sorted(shards), f"tools would read {resolved}, missing a stage's shard"


def test_pp_checkpoint_reloads_onto_a_different_pp_size(tmp_path):
    """The optimizer shards are stage-local and gated separately by the fingerprint (test_pp_resume_units)."""
    reference = _tiny_qwen3().state_dict()
    save_file(dict(reference), os.path.join(tmp_path, "model.safetensors"), metadata={"format": "pt"})

    covered: set[str] = set()
    for pp_rank in range(4):
        stage = build_pipeline_stage(_tiny_qwen3(), pp_rank, 4)
        # Perturb so a no-op load cannot pass the equality check below.
        with torch.no_grad():
            for param in stage.parameters():
                param.add_(1.0)
        load_resume_weights(_load_ctx(stage), str(tmp_path))
        for local, value in stage.state_dict().items():
            global_name = stage.global_parameter_name(local)
            assert torch.equal(value, reference[global_name]), f"pp_rank={pp_rank} {local}"
            covered.add(global_name)
    # Anti-vacuity: the four pp_size=4 stages together cover the whole model.
    assert covered == set(reference)


def _pp_family_model(name: str, *, ep_wrapped: bool = True):
    """``ep_wrapped`` EP-patches at ``ep_size=1``, as grouped GEMM does every MoE stage; without it the family's own
    blocks stay."""
    torch.manual_seed(SEED)
    model = tiny_family_model(PP_FAMILIES[name], overrides=TINY_PP_SPLIT_OVERRIDES.get(name))
    if ep_wrapped:
        apply_ep_layers(model, single_process_ep_config())
    return model


@registered_pp_backbones(*UNREGISTERED_PP_BACKBONES)
def _pp_family_stage(name: str, pp_rank: int, model) -> PipelineStageModule:
    return build_pipeline_stage(model, pp_rank, PP_SIZE, moe_balancing="none", pp_split=PP_FAMILY_SPLITS.get(name))


def _family_save_worker(rank: int, name: str, out_dir: str, ep_wrapped: bool) -> None:
    config = _parallelism_config()
    model = _pp_family_model(name, ep_wrapped=ep_wrapped)
    wrapper_state = stash_wrapper_state(model) if rank == 0 else {}
    stage = _pp_family_stage(name, config.pp_rank, model)
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    dist.barrier()
    ctx = _context(stage, config)
    ctx.pp_wrapper_state = wrapper_state
    save_pp_checkpoint(ctx, out_dir)


def _family_reload_worker(rank: int, name: str, checkpoint: str, ep_wrapped: bool) -> None:
    config = _parallelism_config()
    expected, stage = (
        _pp_family_stage(name, config.pp_rank, _pp_family_model(name, ep_wrapped=ep_wrapped)) for _ in range(2)
    )
    with torch.no_grad():
        for _, tensor in itertools.chain(stage.named_parameters(), persistent_buffers(stage)):
            if tensor.is_floating_point():
                tensor.add_(1.0)
    # EP experts load only at construction; the stamp passes that gate, and a best-model load still reads.
    setattr(stage, LOADED_WEIGHTS_FROM_ATTR, checkpoint)
    load_resume_weights(_load_context(stage, None, config), checkpoint, for_best_model=True)

    cast = save_dtype_caster(expected)
    want, got = expected.state_dict(), stage.state_dict()
    locals_ = list(stage.checkpoint_name_map().values())
    assert locals_, f"{name}: stage {config.pp_rank} reloads nothing"
    for local in locals_:
        assert torch.equal(got[local], cast(local, want[local]).to(got[local].dtype)), f"{name}: {local}"


def _stage_tensors(stage: PipelineStageModule) -> dict[str, torch.Tensor]:
    state = stage.state_dict()
    tensors = {name: state[local] for name, local in stage.checkpoint_name_map().items()}
    for layer_name, module in named_ep_layers(stage):
        for key, tensor in module.gather_layer_state_dict(layer_name, merge_lora=False, retain=True).items():
            tensors[stage.global_parameter_name(key)] = tensor
    return tensors


@pytest.mark.parametrize("name", PP_EXPORT_FAMILIES)
def test_pp_save_writes_the_names_and_values_of_the_ep_gathered_save(tmp_path, name):
    """A stage holds part of the model, so it reverts through what it took from the whole at the split; otherwise it
    keeps live names a serving engine does not read."""
    renames = bool(export_conversions(_pp_family_model(name)))
    assert renames == (name not in PP_RENAME_FREE), f"premise: {name}'s load renames iff it is not a control"
    ep_dir, pp_dir = str(tmp_path / "ep"), str(tmp_path / "pp")
    save_ep_gathered_checkpoint(save_context(_pp_family_model(name)), ep_dir)
    run_gloo_ranks(_family_save_worker, WORLD_SIZE, name, pp_dir, True)

    expected, written = safetensors_state_dict(ep_dir), safetensors_state_dict(pp_dir)
    assert set(written) == set(expected), sorted(set(written) ^ set(expected))[:6]
    unequal = [key for key, tensor in expected.items() if not torch.equal(written[key], tensor)]
    assert not unequal, f"{name}: values differ from the EP-gathered save: {unequal[:6]}"


@pytest.mark.parametrize("name", PP_EXPORT_FAMILIES)
def test_a_pp_save_resumes_through_both_reads(tmp_path, name):
    """The stage-aware loader and a best-model load both map the hub's names back onto the stage's own."""
    pp_dir = str(tmp_path / "pp")
    run_gloo_ranks(_family_save_worker, WORLD_SIZE, name, pp_dir, True)

    family = PP_FAMILIES[name]
    for pp_rank in range(PP_SIZE):
        with registered_pp_backbones(*UNREGISTERED_PP_BACKBONES):
            loaded = _load_stage_lazy(
                pp_dir,
                pp_rank,
                PP_SIZE,
                config=AutoConfig.from_pretrained(pp_dir),
                pp_split=PP_FAMILY_SPLITS.get(name),
                ep_config=single_process_ep_config(),
                model_class=family.load_class,
                trust_remote_code=False,
            )
        built = _stage_tensors(_pp_family_stage(name, pp_rank, loaded))
        reference = _pp_family_stage(name, pp_rank, _pp_family_model(name))
        cast = save_dtype_caster(reference)
        want = _stage_tensors(reference)
        assert set(built) == set(want), f"{name} stage {pp_rank}: {sorted(set(built) ^ set(want))[:6]}"
        for key, tensor in want.items():
            local = reference.local_parameter_name(key)
            # The loader reads at the run dtype, so compare at the dtype it landed in.
            assert torch.equal(built[key], cast(local, tensor).to(built[key].dtype)), f"{name} stage {pp_rank}: {key}"

    run_gloo_ranks(_family_reload_worker, WORLD_SIZE, name, pp_dir, True)


@pytest.mark.parametrize("name", PP_UNWRAPPED_FAMILIES)
def test_pp_save_without_ep_layers_writes_the_whole_model_revert(tmp_path, name):
    """Inkling's shared experts' gate and up form one ``shared_w13_weight``, so the save converts each layer whole."""
    pp_dir = str(tmp_path / "pp")
    run_gloo_ranks(_family_save_worker, WORLD_SIZE, name, pp_dir, False)

    model = _pp_family_model(name, ep_wrapped=False)
    cast = save_dtype_caster(model)
    state = {key: cast(key, tensor) for key, tensor in model.state_dict().items()}
    expected = revert_conversions(model, dict(state), export_conversions(model))
    assert set(expected) != set(state), f"premise: {name}'s load renames nothing"
    written = safetensors_state_dict(pp_dir)
    assert set(written) == set(expected), sorted(set(written) ^ set(expected))[:6]
    unequal = [key for key, tensor in expected.items() if not torch.equal(written[key], tensor)]
    assert not unequal, f"{name}: values differ from the unsplit model's revert: {unequal[:6]}"

    run_gloo_ranks(_family_reload_worker, WORLD_SIZE, name, pp_dir, False)


def _gptoss_dropped_sinks_worker(rank: int, out_dir: str) -> None:
    config = _parallelism_config()
    model = _pp_family_model("gpt_oss", ep_wrapped=False).to(torch.bfloat16)
    for layer in model.model.layers:
        layer.self_attn.sinks = None
    stage = _pp_family_stage("gpt_oss", config.pp_rank, model)
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    dist.barrier()
    save_pp_checkpoint(_context(stage, config), out_dir)


def test_pp_save_writes_the_sinks_fa2_dropped_neutralized_under_global_names(tmp_path):
    """Under FA2 the sinks leave the stage's tensors; the save writes each back neutralized, in its own stage's
    shard under the unsplit index."""
    out_dir = str(tmp_path / "pp")
    run_gloo_ranks(_gptoss_dropped_sinks_worker, WORLD_SIZE, out_dir)

    with open(os.path.join(out_dir, "model.safetensors.index.json")) as fh:
        weight_map = json.load(fh)["weight_map"]
    sinks = {key: shard for key, shard in weight_map.items() if key.endswith(".self_attn.sinks")}
    layers = range(TINY_GPTOSS_CONFIG["num_hidden_layers"])
    assert set(sinks) == {f"model.layers.{i}.self_attn.sinks" for i in layers}
    first_stage_layers = PP_FAMILY_SPLITS["gpt_oss"][0]
    for i in layers:
        stage = 0 if i < first_stage_layers else 1
        shard = sinks[f"model.layers.{i}.self_attn.sinks"]
        assert shard.startswith(f"model-pp{stage:05d}-of-{PP_SIZE:05d}-"), f"layer {i}'s sink is in {shard}"
    written = safetensors_state_dict(out_dir)
    for key in sinks:
        sink = written[key]
        assert sink.dtype == torch.bfloat16, f"{key}: {sink.dtype}"
        assert torch.all(sink == neutralized_sink_value(torch.bfloat16)), f"{key} is not neutralized: {sink}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
