#!/usr/bin/env python
"""A pipeline-parallel save must produce a STANDARD HF sharded checkpoint that reloads unsplit.

``save_pp_checkpoint`` writes one safetensors shard per stage, holding the UNSPLIT model's tensors
under the hub's names, plus one merged index, so the result needs no merge step: ``from_pretrained``
on the directory reconstructs the whole model. That contract has four parts a unit test of the
naming map cannot reach, all exercised here on a real 4-rank gloo group (pp_size=2 x dp=2 per
stage, the smallest config-legal PP topology):

1. **Shard/index ownership** — one writer per stage, distinct shard filenames, and exactly ONE
   index that unions every stage's keys. A per-stage index write would leave the last writer's
   partial key map on disk and ``from_pretrained`` would randomly initialize the other stages.
2. **Round trip** — the reloaded model equals the pre-split model tensor for tensor. This catches a
   dropped stage, a stale layer index (stage 1 writing stage 0's names), and a torn index.
3. **PP+EP composition** — an EP layer's experts are exported by the family gather under
   ``experts.*`` and must then pass through the stage's global rename; the stage-local EP state_dict
   entries (rank-local expert ranges) must NOT reach the checkpoint.
4. **Hub namespace** — names and values equal the EP-gathered save (without EP wrappers, the
   save-side revert of the whole model), and both PP reads map them back onto the stage.

Plus the topology-independence claim on the read side: because the shards carry the unsplit model's
names, the same directory reloads onto a DIFFERENT ``pp_size``.

    python tests/cpu/checkpoint/test_pp_sharded_save_roundtrip.py
"""

from __future__ import annotations

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
from accelerate import PartialState
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, Glm5NextConfig, Glm5NextForConditionalGeneration, Qwen3Config, Qwen3ForCausalLM

from src.checkpoint.hub_layout import revert_conversions
from src.checkpoint.reading import checkpoint_shard_files, is_ep_sharded_checkpoint
from src.checkpoint.weights_source import LOADED_WEIGHTS_FROM_ATTR
from src.checkpoint.writing import save_dtype_caster
from src.distributed.checkpoint.context import CheckpointContext, CheckpointLoadContext
from src.distributed.checkpoint.ep_save import save_ep_model
from src.distributed.checkpoint.loader import CheckpointLoader
from src.distributed.checkpoint.optimizer import OptimizerShardStore
from src.distributed.checkpoint.save import is_pp_shard_writer, save_pp_checkpoint
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.expert_weights import gather_ep_layer_weights
from src.distributed.expert_parallel.hub_conversion import gathered_export_conversions
from src.distributed.expert_parallel.patching import patch_moe_model_for_ep
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.pipeline_parallel.lazy_loader import load_pp_stage_model
from src.distributed.pipeline_parallel.stage import PipelineStageModule, build_pipeline_stage
from src.models.param_roles import persistent_buffers
from src.models.patches.gpt_oss_sinks import neutralized_sink_value
from src.runtime import dtensor
from src.runtime.filesystem import is_fs_save_rank
from src.trainers.mixins.pipeline import stash_wrapper_state
from tests.common.ep_stubs import StubEPLayerBase
from tests.common.gloo import run_gloo_ranks
from tests.common.models import (
    TINY_GLM5_CONFIG,
    TINY_GLM5_VISION_CONFIG,
    TINY_GPTOSS_CONFIG,
    TINY_INKLING_PP_OVERRIDES,
    TINY_LAGUNA_CONFIG,
    TINY_QWEN3_CONFIG,
)
from tests.common.tiny_models import TINY_DENSE_FAMILIES, TINY_MOE_FAMILIES, TINY_MOE_VLM_FAMILIES, tiny_family_model
from tests.common.utils import safetensors_state_dict

PartialState()  # save_ep_model logs through accelerate's logger

_PC_MOD = "src.distributed.parallelism_config"

WORLD_SIZE = 4
PP_SIZE = 2
GPUS_PER_NODE = 2  # ⇒ stage_world_size = 2 (dp=2 per stage), the smallest legal PP shape
SEED = 1234

NUM_EXPERTS = 4
LOCAL_EXPERTS = 2  # an ep2 rank: the stage-local params hold E/ep experts, the gather returns E
HIDDEN = 8
INTER = 6

# Per-node output storage: 8 ranks as 4 simulated nodes of 2, so each stage spans two nodes — the
# shape where the per-node writer set (one rank per node) differs from the shared one (one per stage).
PER_NODE_WORLD = 8
RANKS_PER_NODE = 2
# A dense model, and one whose hub names form converter groups (Inkling's shared experts): a stage reads
# only its own layers' groups.
PER_NODE_FAMILIES = ("qwen3", "inkling_mm_model")

# The PP families whose load renames tensors, and controls that rename none (Inkling's text tower keeps
# ``embed_norm`` beside the embedding, which only stage 0 may write). Laguna's renames all sit in its EP
# layer, whose gather already writes them; its attention pattern repeats every four layers, so it is
# doubled to split in two on a period boundary.
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
)
PP_RENAME_FREE = frozenset({"inkling_text", "qwen3_moe"})
# The renaming families as a run without EP wrappers (``ep_size=1``, no grouped GEMM) holds them: each
# decoder layer's own tensors carry its reverses, several at once for Inkling's shared experts.
PP_UNWRAPPED_FAMILIES = ("deepseek_v4", "glm5_next", "step3p7", "mistral3", "inkling_mm_model")
PP_FAMILY_OVERRIDES = {
    "laguna": {
        "num_hidden_layers": 8,
        "num_attention_heads_per_layer": TINY_LAGUNA_CONFIG["num_attention_heads_per_layer"] * 2,
        "layer_types": TINY_LAGUNA_CONFIG["layer_types"] * 2,
        "mlp_layer_types": TINY_LAGUNA_CONFIG["mlp_layer_types"] * 2,
    },
    "inkling_text": TINY_INKLING_PP_OVERRIDES,
    "inkling_mm_model": TINY_INKLING_PP_OVERRIDES,
}
PP_FAMILY_SPLITS = {"laguna": [4, 4], "gpt_oss": [4, 4]}
SINGLE_RANK_EP_CONFIG = {
    "ep_size": 1,
    "stage_world_size": 1,
    "nvlink_domain_size": 1,
    "use_grouped_gemm": False,
    "experts_fsdp_managed": True,
}


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
        is_save_rank=is_fs_save_rank(),
        max_shard_size=max_shard_size,
        save_sharded_ep=False,
        has_expert_lora=False,
        merge_expert_lora_on_save=False,
        cp_wrapper=None,
        tokenizer=None,
    )


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


def _multipart_worker(rank: int, out_dir: str, verdict_path: str) -> None:
    """Same save, but at a shard limit small enough to force several parts per stage."""
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


def test_pp_save_splits_parts_and_still_reloads(tmp_path):
    """``save_max_shard_size`` must reach the PP writer, and splitting must not break the reload.

    Writing exactly one file per stage regardless of the setting puts a whole stage of a large model
    in one file and holds it all in host RAM first. Several parts per stage is the contract; this
    pins that the split happens AND that the result is still a plain HF checkpoint.
    """
    verdict = str(tmp_path / "verdict.txt")
    run_gloo_ranks(_multipart_worker, WORLD_SIZE, str(tmp_path / "ckpt"), verdict)
    with open(verdict) as fh:
        result = fh.read()
    assert result == "PASS", result


def test_pp_save_reloads_unsplit_with_from_pretrained(tmp_path):
    """The end-to-end contract: pp_size stage shards + one unioned index == the unsplit model."""
    verdict = str(tmp_path / "verdict.txt")
    run_gloo_ranks(_roundtrip_worker, WORLD_SIZE, str(tmp_path / "ckpt"), verdict)
    with open(verdict) as fh:
        result = fh.read()
    assert result == "PASS", result


def _host_copy_worker(rank: int, out_dir: str, verdict_path: str) -> None:
    """Count this rank's host copies through the runtime primitive, wherever a module bound it."""
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
    """Every stage rank enters each tensor's gather, but only the stage's writer keeps the result.

    A non-writer that resolves the full tensor to host memory pays the device-to-host copy and holds
    the tensor for nothing; on a node that is ``local_world_size - 1`` such copies of every tensor at
    every save, on top of the writer's own.
    """
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
    CheckpointLoader(_load_ctx(stage)).load_model(node_dir)
    unsplit = _pp_family_model(name, ep_wrapped=False)
    cast, reference = save_dtype_caster(unsplit), unsplit.state_dict()
    for local, value in stage.state_dict().items():
        global_name = stage.global_parameter_name(local)
        if not torch.equal(value, cast(global_name, reference[global_name]).to(value.dtype)):
            problems.append(f"{global_name} did not resume from node {node}'s directory")

    # A relaunch that hands this stage another stage's node directory has no shard for its layers.
    elsewhere = os.path.join(root, f"node{(node + nodes // 2) % nodes}")
    try:
        CheckpointLoader(_load_ctx(stage)).load_model(elsewhere)
        problems.append(f"stage {config.pp_rank} resumed from another stage's node directory {elsewhere}")
    except RuntimeError:
        pass
    with open(f"{verdict_path}.{rank}", "w") as fh:
        fh.write("PASS" if not problems else "FAIL: " + "; ".join(problems[:6]))


@pytest.mark.parametrize("name", PER_NODE_FAMILIES)
def test_per_node_pp_checkpoint_resumes_each_node_and_refuses_a_moved_stage(tmp_path, name):
    """Per-node output storage: every node's directory carries its own stage's shard and the full
    index, so each node's ranks resume their stage from local disk alone, reading only the checkpoint
    keys their own tensors convert from; a stage moved onto another stage's node fails on every rank
    instead of resuming base weights."""
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
    """PP+EP: every stage's experts land in the one index under re-based global layer names, at the
    FULL expert count, and no rank-local EP state_dict entry leaks through."""
    verdict = str(tmp_path / "verdict.txt")
    run_gloo_ranks(_ep_worker, WORLD_SIZE, str(tmp_path / "ckpt"), verdict)
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


def _composite_worker(rank: int, out_dir: str, verdict_path: str) -> None:
    """A text-only run of a multimodal wrapper: the save must re-emit the vision tower it dropped."""
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
    unsplit = _tiny_composite()
    reference = unsplit.state_dict()
    # Under the hub's names, as the EP-gathered save writes them (the tower's untouched, its text
    # tower's hyper-connections respelled).
    conversions = gathered_export_conversions(unsplit)
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
    """The vision tower no stage holds is re-emitted from the save rank under its own part, so the
    directory reloads as the composite class with NO missing (random-initialized) keys — a
    servable wrapper-layout export with no reattach step, and a resumable one (the stage-aware
    loader plans those tensors)."""
    verdict = str(tmp_path / "verdict.txt")
    run_gloo_ranks(_composite_worker, WORLD_SIZE, str(tmp_path / "ckpt"), verdict)
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
    # Hyperparameters must stay THIS run's, not the shard's rebuilt param_groups.
    if any(group["lr"] != 0.1 for group in fresh_optimizer.param_groups):
        problems.append("live param_group settings were overwritten by the shard")
    # Every rank checks its own stage and writes its own verdict file.
    with open(f"{verdict_path}.{rank}", "w") as fh:
        fh.write("PASS" if not problems else "FAIL: " + "; ".join(problems[:6]))


def test_pp_optimizer_shards_round_trip(tmp_path):
    """Per-rank optimizer shards must actually restore under PP: save from a stepped optimizer, then
    reload into a fresh one and compare momentum buffers. Every other PP optimizer test mocks
    ``set_optimizer_state_dict`` and so proves only that the gates allow the call."""
    verdict = str(tmp_path / "verdict")
    run_gloo_ranks(_optimizer_worker, WORLD_SIZE, str(tmp_path / "ckpt"), verdict)
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

    assert not is_ep_sharded_checkpoint(str(tmp_path)), "a PP checkpoint must not read as per-rank sharded"
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


def _pp_family_model(name: str, *, ep_wrapped: bool = True):
    """``name``'s tiny model, seeded. ``ep_wrapped`` EP-patches it at ``ep_size=1``, as grouped GEMM does
    every MoE stage; without it the family's own MoE blocks stay, as under ``use_grouped_gemm: false``."""
    torch.manual_seed(SEED)
    model = tiny_family_model(PP_FAMILIES[name], overrides=PP_FAMILY_OVERRIDES.get(name))
    if ep_wrapped:
        patch_moe_model_for_ep(model, EPConfig(**SINGLE_RANK_EP_CONFIG))
    return model


def _pp_family_stage(name: str, pp_rank: int, model) -> PipelineStageModule:
    return build_pipeline_stage(model, pp_rank, PP_SIZE, moe_balancing="none", pp_split=PP_FAMILY_SPLITS.get(name))


def _family_save_worker(rank: int, name: str, out_dir: str, ep_wrapped: bool) -> None:
    config = _parallelism_config()
    model = _pp_family_model(name, ep_wrapped=ep_wrapped)
    # What the trainer's multimodal gate stashes on the save rank: the tensors no stage holds.
    wrapper_state = stash_wrapper_state(model) if rank == 0 else {}
    stage = _pp_family_stage(name, config.pp_rank, model)
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    dist.barrier()
    ctx = _context(stage, config)
    ctx.pp_wrapper_state = wrapper_state
    save_pp_checkpoint(ctx, out_dir)


def _family_reload_worker(rank: int, name: str, checkpoint: str, ep_wrapped: bool) -> None:
    """A best-model load of a perturbed stage from the PP save restores every tensor outside an EP layer."""
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
    CheckpointLoader(_load_context(stage, None, config)).load_model(checkpoint, for_best_model=True)

    cast = save_dtype_caster(expected)
    want, got = expected.state_dict(), stage.state_dict()
    locals_ = list(stage.checkpoint_name_map().values())
    assert locals_, f"{name}: stage {config.pp_rank} reloads nothing"
    for local in locals_:
        assert torch.equal(got[local], cast(local, want[local]).to(got[local].dtype)), f"{name}: {local}"


def _stage_tensors(stage: PipelineStageModule) -> dict[str, torch.Tensor]:
    """A stage's tensors under the unsplit model's names: its own entries, and each EP layer's gather."""
    state = stage.state_dict()
    tensors = {name: state[local] for name, local in stage.checkpoint_name_map().items()}
    for layer_name, module in stage.ep_moe_layers():
        for key, tensor in gather_ep_layer_weights(layer_name, module, merge_lora=False, retain=True).items():
            tensors[stage.global_parameter_name(key)] = tensor
    return tensors


@pytest.mark.parametrize("name", PP_EXPORT_FAMILIES)
def test_pp_save_writes_the_names_and_values_of_the_ep_gathered_save(tmp_path, name):
    """The PP save writes the hub's namespace, the one the EP-gathered save writes: transformers'
    save-side revert of the unsplit model's load conversions, the per-expert merges excepted. A stage
    holds only part of the model, so it reverts through what it took from the whole one at the split;
    without that a renamed family's PP save keeps the live names a serving engine does not read
    (DeepSeek-V4's ``attn_hc.*``, GLM-5 Next's hyper-connections, Step-3.7's ``model.language_model.*``)."""
    renames = bool(gathered_export_conversions(_pp_family_model(name)))
    assert renames == (name not in PP_RENAME_FREE), f"premise: {name}'s load renames iff it is not a control"
    ep_dir, pp_dir = str(tmp_path / "ep"), str(tmp_path / "pp")
    save_ep_model(_pp_family_model(name), ep_dir)
    run_gloo_ranks(_family_save_worker, WORLD_SIZE, name, pp_dir, True)

    expected, written = safetensors_state_dict(ep_dir), safetensors_state_dict(pp_dir)
    assert set(written) == set(expected), sorted(set(written) ^ set(expected))[:6]
    unequal = [key for key, tensor in expected.items() if not torch.equal(written[key], tensor)]
    assert not unequal, f"{name}: values differ from the EP-gathered save: {unequal[:6]}"


@pytest.mark.parametrize("name", PP_EXPORT_FAMILIES)
def test_a_pp_save_resumes_through_both_reads(tmp_path, name):
    """A PP resume reads its checkpoint two ways: the stage-aware loader builds each stage from it, and
    a best-model load (or a stage not built from it) reloads the stage. Both map the hub's names back
    onto the stage's own."""
    pp_dir = str(tmp_path / "pp")
    run_gloo_ranks(_family_save_worker, WORLD_SIZE, name, pp_dir, True)

    family = PP_FAMILIES[name]
    for pp_rank in range(PP_SIZE):
        loaded = load_pp_stage_model(
            pp_dir,
            pp_rank,
            PP_SIZE,
            config=AutoConfig.from_pretrained(pp_dir),
            pp_split=PP_FAMILY_SPLITS.get(name),
            ep_config=EPConfig(**SINGLE_RANK_EP_CONFIG),
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
            # The loader reads at the run dtype, so the saved value is compared at the dtype it landed in.
            assert torch.equal(built[key], cast(local, tensor).to(built[key].dtype)), f"{name} stage {pp_rank}: {key}"

    run_gloo_ranks(_family_reload_worker, WORLD_SIZE, name, pp_dir, True)


@pytest.mark.parametrize("name", PP_UNWRAPPED_FAMILIES)
def test_pp_save_without_ep_wrappers_writes_the_whole_model_revert(tmp_path, name):
    """Without EP wrappers the reverses read a decoder layer's own tensors, several at once for Inkling
    (its shared experts' gate and up projections form one ``shared_w13_weight``): the save converts
    each layer whole, writing what transformers' save-side revert of the unsplit model writes, and a
    best-model load reads it back."""
    pp_dir = str(tmp_path / "pp")
    run_gloo_ranks(_family_save_worker, WORLD_SIZE, name, pp_dir, False)

    model = _pp_family_model(name, ep_wrapped=False)
    cast = save_dtype_caster(model)
    state = {key: cast(key, tensor) for key, tensor in model.state_dict().items()}
    expected = revert_conversions(model, dict(state), gathered_export_conversions(model))
    assert set(expected) != set(state), f"premise: {name}'s load renames nothing"
    written = safetensors_state_dict(pp_dir)
    assert set(written) == set(expected), sorted(set(written) ^ set(expected))[:6]
    unequal = [key for key, tensor in expected.items() if not torch.equal(written[key], tensor)]
    assert not unequal, f"{name}: values differ from the unsplit model's revert: {unequal[:6]}"

    run_gloo_ranks(_family_reload_worker, WORLD_SIZE, name, pp_dir, False)


def _gptoss_dropped_sinks_worker(rank: int, out_dir: str) -> None:
    """A bf16 GptOss with every sink dropped as FA2 drops it, saved at pp2."""
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
    """Under FA2 GptOss's sinks leave the stage's tensors. The save writes every layer's back at the
    neutralized value, each in its own stage's shard under the unsplit model's layer index, rather
    than none, or stage 1's under stage 0's names."""
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
