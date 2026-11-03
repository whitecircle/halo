#!/usr/bin/env python
"""``merge_ep_shards.py`` on a real per-rank EP save reproduces the gathered save bit for bit, on tiny models without
DeepEP dispatch.

A model whose names the gather reverts to a namespace the key-by-key merge can't respell has its sharded save refused
on every rank.
"""

from __future__ import annotations

import os
import shutil

import torch
import torch.distributed as dist

from halo.checkpoint.parallel.ep import save_ep_gathered_checkpoint, save_ep_sharded_checkpoint
from halo.checkpoint.tools.ep_shard_merge import merge_ep_shards
from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.model_loading.model_preparation import auto_load_model
from tests.common.checkpoint_io import safetensors_state_dict, save_context
from tests.common.distributed import shared_scratch_dir
from tests.common.harness import gpu_test_main
from tests.common.rank_log import log
from tests.common.tiny_models import TINY_MOE_FAMILIES, TINY_MOE_VLM_FAMILIES, tiny_family_model

EP_SIZE = 2

_ROSTER = {**TINY_MOE_FAMILIES, **TINY_MOE_VLM_FAMILIES}
# One family per expert layout the merge must invert: interleaved fused, per-expert, fused, per-expert under key
# renames, fused with tied embeddings, and a fused text tower in a composite VLM.
_FAMILIES = ("gpt_oss", "qwen3_moe", "qwen3_5_moe_text", "laguna", "cohere2_moe", "qwen3_5_moe")
# A vendor namespace (DeepSeek-V4, GLM-5 Next) and a SigLIP tower's ``vision_model`` level (Command A+).
_REFUSED = ("deepseek_v4", "glm5_next", "cohere2_vision")


def _ep_patched(family: str, device: torch.device) -> torch.nn.Module:
    torch.manual_seed(0)
    model = tiny_family_model(_ROSTER[family]).to(device=device, dtype=torch.bfloat16)
    config = EPConfig(
        ep_size=EP_SIZE, stage_world_size=EP_SIZE, nvlink_domain_size=EP_SIZE, experts_fsdp_managed=False
    )
    return apply_ep_layers(model, config)


def _compare(merged_dir: str, gathered_dir: str) -> list[str]:
    merged, gathered = safetensors_state_dict(merged_dir), safetensors_state_dict(gathered_dir)
    problems = []
    if set(merged) != set(gathered):
        problems.append(
            f"key sets differ: merged-only {sorted(set(merged) - set(gathered))[:6]}, "
            f"gathered-only {sorted(set(gathered) - set(merged))[:6]}"
        )
    for key in sorted(set(merged) & set(gathered)):
        want, got = gathered[key], merged[key]
        if got.dtype != want.dtype or got.shape != want.shape:
            problems.append(
                f"{key}: merged {got.dtype} {tuple(got.shape)} vs gathered {want.dtype} {tuple(want.shape)}"
            )
        elif not torch.equal(got, want):
            problems.append(f"{key}: values differ (max |diff| {(got.float() - want.float()).abs().max().item():.3e})")
    return problems


def _roundtrip(family: str, root: str, device: torch.device, rank: int) -> list[str]:
    gathered_dir, sharded_dir, merged_dir = (
        os.path.join(root, family, kind) for kind in ("gathered", "sharded", "merged")
    )
    model = _ep_patched(family, device)

    save_ep_gathered_checkpoint(save_context(model), gathered_dir)
    save_ep_sharded_checkpoint(save_context(model), sharded_dir)
    dist.barrier()
    if rank != 0:
        return []

    merge_ep_shards(sharded_dir, merged_dir, verbose=False)
    problems = _compare(merged_dir, gathered_dir)
    try:
        # The merged artifact must also pass the coverage gate, which raises on anything missing.
        auto_load_model(merged_dir, dtype=torch.bfloat16)
    except Exception as exc:
        problems.append(f"merged checkpoint does not load: {type(exc).__name__}: {exc}")
    return problems


def run(ctx) -> dict:
    root = shared_scratch_dir("halo_ep_sharded_merge_roundtrip")
    if ctx.rank == 0:
        shutil.rmtree(root, ignore_errors=True)
        ctx.on_teardown(lambda: shutil.rmtree(root, ignore_errors=True))
    ctx.barrier()

    checks = {}
    for family in _FAMILIES:
        # Rank 0 alone reads back, so its verdict travels as data: a rank-0-only raise would leave peers in the next
        # family's collectives.
        verdict = [_roundtrip(family, root, ctx.device, ctx.rank)]
        dist.broadcast_object_list(verdict, src=0)
        problems = verdict[0]
        checks[f"{family}_merged_equals_gathered"] = not problems
        log(f"{family}: {'OK' if not problems else chr(10).join(problems)}")
    for family in _REFUSED:
        # Name-only and raised before any collective, so every rank refuses alike.
        try:
            save_ep_sharded_checkpoint(
                save_context(_ep_patched(family, ctx.device)), os.path.join(root, family, "sharded")
            )
        except ValueError as refusal:
            checks[f"{family}_sharded_save_refused"] = "key-by-key stream cannot" in str(refusal)
        else:
            checks[f"{family}_sharded_save_refused"] = False
    return {"checks": checks}


main = gpu_test_main(exact_world_size=EP_SIZE, prefix="ep_sharded_merge_roundtrip")(run)

if __name__ == "__main__":
    main()
