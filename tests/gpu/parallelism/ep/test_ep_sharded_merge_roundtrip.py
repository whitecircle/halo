#!/usr/bin/env python
"""``merge_ep_shards.py`` on a REAL per-rank EP save must reproduce the gathered save bit for bit.

The sharded EP layout (``save_sharded_ep: true``) is loadable only through the merge, and the CPU
suite pins that merge against synthetic shard dicts and a stubbed class-level gather oracle. This
is the end-to-end pin: on 2 GPUs, each family's tiny model is EP-patched at ``ep_size=2``, saved
BOTH ways by ``save_ep_model``, the per-rank shards are merged by the real script, and the merged
directory must be key-and-tensor identical to the gathered one — and load back through the
toolkit's verified ``from_pretrained``. Any drift between a family's ``merge_shards_to_hf`` and its
``gather_expert_state_dict`` (a transpose, a lost re-interleave, a hub rename applied on one side
only, a balancing tensor cast on one side only) shows up as a tensor diff here.

Hermetic: tiny random-init models, no download, and no DeepEP — the transport buffer is built at
the first dispatch, which a save never issues.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 tests/gpu/parallelism/ep/test_ep_sharded_merge_roundtrip.py
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable

import torch
import torch.distributed as dist
from transformers import (
    Cohere2MoeConfig,
    Cohere2MoeForCausalLM,
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    Glm5NextConfig,
    Glm5NextForConditionalGeneration,
    GptOssConfig,
    GptOssForCausalLM,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from scripts.after_training.merge_ep_shards import merge_ep_shards
from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.patching import patch_moe_model_for_ep
from src.distributed.expert_parallel.saving import save_ep_model
from src.models.loading.model_preparation import auto_load_model
from tests.common.distributed import shared_scratch_dir
from tests.common.harness import gpu_test_main
from tests.common.models import (
    TINY_COHERE2_MOE_CONFIG,
    TINY_DSV4_CONFIG,
    TINY_GLM5_CONFIG,
    TINY_GLM5_VISION_CONFIG,
    TINY_GPTOSS_CONFIG,
    TINY_QWEN3_MOE_CONFIG,
)
from tests.common.utils import log, safetensors_state_dict
from tests.cpu.models.test_deepseek_v4_support import randomize_tid2eid

EP_SIZE = 2


def _tiny_deepseek_v4() -> DeepseekV4ForCausalLM:
    """Random init leaves the hash router's tid2eid all-zero, which the EP wrapper refuses."""
    model = DeepseekV4ForCausalLM(DeepseekV4Config(**TINY_DSV4_CONFIG))
    randomize_tid2eid(model)
    return model


# A MoE text tower in the Qwen3.5 family: fused hub layout, linear-attention interleave.
_TINY_QWEN35_MOE_TEXT = {
    "hidden_size": 256,
    "intermediate_size": 256,
    "num_hidden_layers": 4,
    "full_attention_interval": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 64,
    "linear_value_head_dim": 64,
    "linear_conv_kernel_dim": 2,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 128,
    "shared_expert_intermediate_size": 128,
    "vocab_size": 1024,
    "max_position_embeddings": 512,
    "tie_word_embeddings": False,
}

# One builder per expert layout the merge has to invert: interleaved fused (GptOss, stored
# de-interleaved under grouped GEMM), per-expert (Qwen3), fused (Qwen3.5), fused behind a read-side
# hub-conversion bridge (DeepSeek-V4), fused with tied embeddings (Cohere2), and a fused text tower
# inside a composite VLM wrapper (GLM-5 Next). Every family here is one the sharded save admits.
_FAMILIES: dict[str, Callable[[], torch.nn.Module]] = {
    "gpt_oss": lambda: GptOssForCausalLM(GptOssConfig(**TINY_GPTOSS_CONFIG, pad_token_id=0, eos_token_id=1)),
    "qwen3_moe": lambda: Qwen3MoeForCausalLM(Qwen3MoeConfig(**TINY_QWEN3_MOE_CONFIG)),
    "qwen3_5_moe": lambda: Qwen3_5MoeForCausalLM(Qwen3_5MoeTextConfig(**_TINY_QWEN35_MOE_TEXT)),
    "deepseek_v4": _tiny_deepseek_v4,
    "cohere2_moe": lambda: Cohere2MoeForCausalLM(Cohere2MoeConfig(**TINY_COHERE2_MOE_CONFIG)),
    "glm5_next": lambda: Glm5NextForConditionalGeneration(
        Glm5NextConfig(text_config=dict(TINY_GLM5_CONFIG), vision_config=dict(TINY_GLM5_VISION_CONFIG))
    ),
}


def _ep_patched(family: str, device: torch.device) -> torch.nn.Module:
    """The family's tiny model, identically initialized on every rank, EP-patched at ``EP_SIZE``."""
    torch.manual_seed(0)
    model = _FAMILIES[family]().to(device=device, dtype=torch.bfloat16)
    config = EPConfig(ep_size=EP_SIZE, world_size=EP_SIZE, gpus_per_node=EP_SIZE)
    return patch_moe_model_for_ep(model, config)


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

    save_ep_model(model, gathered_dir, sharded=False)
    save_ep_model(model, sharded_dir, sharded=True)
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
    return {"checks": checks}


main = gpu_test_main(exact_world_size=EP_SIZE, prefix="ep_sharded_merge_roundtrip")(run)

if __name__ == "__main__":
    main()
