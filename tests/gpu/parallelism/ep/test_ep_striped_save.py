#!/usr/bin/env python
"""An EP gathered save striped over both ranks of a node (``HALO_CHECKPOINT_WRITERS_PER_NODE=2``) holds the single-writer
save's tensors, keys, dtypes and index size on every EP family's layout, and loads through the coverage gate."""

from __future__ import annotations

import json
import os
import shutil

import torch
import torch.distributed as dist

from halo.checkpoint.format import SAFETENSORS_INDEX_FILE
from halo.checkpoint.parallel.ep import save_ep_gathered_checkpoint
from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.model_loading.model_preparation import auto_load_model
from tests.common.checkpoint_io import safetensors_state_dict, save_context
from tests.common.distributed import shared_scratch_dir
from tests.common.harness import gpu_test_main
from tests.common.rank_log import log
from tests.common.tiny_models import TINY_MOE_FAMILIES, TINY_MOE_VLM_FAMILIES, tiny_family_model

EP_SIZE = 2
WRITERS_ENV = "HALO_CHECKPOINT_WRITERS_PER_NODE"
# Small enough that each writer's layers span several parts.
SHARD_SIZE = "64KB"
_ROSTER = {**TINY_MOE_FAMILIES, **TINY_MOE_VLM_FAMILIES}
_FAMILIES = (
    "gpt_oss",
    "qwen3_moe",
    "qwen3_5_moe_text",
    "laguna",
    "cohere2_moe",
    "deepseek_v4",
    "glm5_next",
    "qwen3_5_moe",
)


def _ep_patched(family: str, device: torch.device) -> torch.nn.Module:
    torch.manual_seed(0)
    model = tiny_family_model(_ROSTER[family]).to(device=device, dtype=torch.bfloat16)
    config = EPConfig(
        ep_size=EP_SIZE, stage_world_size=EP_SIZE, nvlink_domain_size=EP_SIZE, experts_fsdp_managed=False
    )
    return apply_ep_layers(model, config)


def _save(model: torch.nn.Module, directory: str, writers_per_node: int) -> None:
    os.environ[WRITERS_ENV] = str(writers_per_node)
    save_ep_gathered_checkpoint(save_context(model, has_ep_layers=True, max_shard_size=SHARD_SIZE), directory)
    dist.barrier()


def _compare(single_dir: str, striped_dir: str) -> list[str]:
    single, striped = safetensors_state_dict(single_dir), safetensors_state_dict(striped_dir)
    problems = []
    if set(single) != set(striped):
        problems.append(f"key sets differ: {sorted(set(single) ^ set(striped))[:6]}")
    for key in sorted(set(single) & set(striped)):
        if single[key].dtype != striped[key].dtype or not torch.equal(single[key], striped[key]):
            problems.append(f"{key}: striped differs from single-writer")
    sizes = []
    for directory in (single_dir, striped_dir):
        with open(os.path.join(directory, SAFETENSORS_INDEX_FILE)) as fh:
            sizes.append(json.load(fh)["metadata"])
    if sizes[0] != sizes[1]:
        problems.append(f"index metadata differs: {sizes}")
    return problems


def _roundtrip(family: str, root: str, device: torch.device, rank: int) -> list[str]:
    single_dir, striped_dir = (os.path.join(root, family, kind) for kind in ("single", "striped"))
    model = _ep_patched(family, device)
    _save(model, single_dir, 1)
    _save(model, striped_dir, EP_SIZE)
    if rank != 0:
        return []
    problems = _compare(single_dir, striped_dir)
    try:
        auto_load_model(striped_dir, dtype=torch.bfloat16)
    except Exception as exc:
        problems.append(f"striped checkpoint does not load: {type(exc).__name__}: {exc}")
    return problems


def run(ctx) -> dict:
    root = shared_scratch_dir("halo_ep_striped_save")
    if ctx.rank == 0:
        shutil.rmtree(root, ignore_errors=True)
        ctx.on_teardown(lambda: shutil.rmtree(root, ignore_errors=True))
    ctx.barrier()
    checks = {}
    for family in _FAMILIES:
        # Rank 0 alone reads back, so its verdict travels as data: a rank-0-only raise would strand the peers.
        verdict = [_roundtrip(family, root, ctx.device, ctx.rank)]
        dist.broadcast_object_list(verdict, src=0)
        checks[f"{family}_striped_equals_single_writer"] = not verdict[0]
        log(f"{family}: {'OK' if not verdict[0] else chr(10).join(verdict[0])}")
    return {"checks": checks}


main = gpu_test_main(exact_world_size=EP_SIZE, prefix="ep_striped_save")(run)

if __name__ == "__main__":
    main()
