#!/usr/bin/env python
"""``merge_ep_shards.py`` on a REAL per-rank EP save must reproduce the gathered save bit for bit.

The sharded EP layout (``save_sharded_ep: true``) is loadable only through the merge, and the CPU
suite pins that merge against synthetic shard dicts and a single-process gather. This
is the end-to-end pin: on 2 GPUs, each family's tiny model is EP-patched at ``ep_size=2``, saved
BOTH ways (``save_ep_gathered_checkpoint`` / ``save_ep_sharded_checkpoint``), the per-rank shards
are merged by the real script, and the merged directory must be key-and-tensor identical to the
gathered one — and load back through the toolkit's verified ``from_pretrained``. Any drift between the
merge and the gather around the family's one ``ExpertLayout`` (a transpose, a lost re-interleave, a
hub rename applied on one side only, a balancing tensor cast on one side only) shows up as a tensor
diff here. A model whose names the gathered save reverts to a namespace the
key-by-key merge cannot respell must have its sharded save refused instead, on every rank.

Hermetic: tiny random-init models, no download, and no DeepEP — the transport buffer is built at
the first dispatch, which a save never issues.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 tests/gpu/parallelism/ep/test_ep_sharded_merge_roundtrip.py
"""

from __future__ import annotations

import os
import shutil

import torch
import torch.distributed as dist

from halo.checkpoint.parallel.ep import save_ep_gathered_checkpoint, save_ep_sharded_checkpoint
from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.model_loading.model_preparation import auto_load_model
from scripts.after_training.merge_ep_shards import merge_ep_shards
from tests.common.checkpoint_io import save_context
from tests.common.distributed import shared_scratch_dir
from tests.common.harness import gpu_test_main
from tests.common.tiny_models import TINY_MOE_FAMILIES, TINY_MOE_VLM_FAMILIES, tiny_family_model
from tests.common.utils import log, safetensors_state_dict

EP_SIZE = 2

_ROSTER = {**TINY_MOE_FAMILIES, **TINY_MOE_VLM_FAMILIES}
# One family per expert layout the merge has to invert: interleaved fused (GptOss, stored
# de-interleaved under grouped GEMM), per-expert (Qwen3), fused (Qwen3.5), per-expert under the
# family's own key renames (Laguna), fused with tied embeddings (Cohere2), and a fused text tower
# inside a composite VLM wrapper (Qwen3.5). Every family here is one the sharded save admits.
_FAMILIES = ("gpt_oss", "qwen3_moe", "qwen3_5_moe_text", "laguna", "cohere2_moe", "qwen3_5_moe")
# Whose gathered save writes a namespace the merge cannot respell: a vendor one (DeepSeek-V4, GLM-5
# Next) and a SigLIP tower's ``vision_model`` level (Command A+).
_REFUSED = ("deepseek_v4", "glm5_next", "cohere2_vision")


def _ep_patched(family: str, device: torch.device) -> torch.nn.Module:
    """The family's tiny model, identically initialized on every rank, EP-patched at ``EP_SIZE``."""
    torch.manual_seed(0)
    model = tiny_family_model(_ROSTER[family]).to(device=device, dtype=torch.bfloat16)
    config = EPConfig(
        ep_size=EP_SIZE, stage_world_size=EP_SIZE, nvlink_domain_size=EP_SIZE, experts_fsdp_managed=False
    )
    return apply_ep_layers(model, config)


def _compare(merged_dir: str, gathered_dir: str) -> list[str]:
    """Every key of the gathered save, at the same dtype and bytes, in the merged one — and nothing else."""
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
        # The merged artifact must also be what the loaders consume: every key present, at the
        # family's hub spelling, through the coverage gate that raises on anything missing.
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
        # Rank 0 alone reads back, so its verdict travels to the peers as data: a rank-0-only raise
        # would leave them in the next family's collectives.
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
