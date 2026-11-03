#!/usr/bin/env python
"""The sharded EP save under TP writes complete GptOss attention sinks, hand-sliced per rank since they can't be
DTensors.

Falling through to the plain non-expert branch writes rank 0's half under the full key, and ``merge_ep_shards.py``
passes it through truncated. No EP layers needed: sinks travel the non-expert path.
"""

import json
import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.distributed.device_mesh import init_device_mesh
from transformers import GptOssConfig, GptOssForCausalLM

from halo.checkpoint.parallel.ep import save_ep_sharded_checkpoint
from halo.distributed.tensor_parallel.parallelize_attention import shard_sinks_param
from tests.common.checkpoint_io import save_context
from tests.common.distributed import shared_scratch_dir
from tests.common.harness import gpu_test_main
from tests.common.rank_log import log

NUM_LAYERS = 2
NUM_HEADS = 4
TP_SIZE = 2


def _tiny_gpt_oss(device: torch.device) -> GptOssForCausalLM:
    config = GptOssConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=2,
        head_dim=16,
        num_local_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    return GptOssForCausalLM(config).to(device=device, dtype=torch.bfloat16)


def _full_sinks(layer_idx: int, device: torch.device) -> torch.Tensor:
    """Distinct per head and per layer, so a truncated or wrong-layer write is visible."""
    return (torch.arange(NUM_HEADS, device=device, dtype=torch.bfloat16) + 1) * (layer_idx + 1)


def _check_saved_sinks(save_dir: str, world_size: int, expected: dict) -> list[str]:
    shard = load_file(os.path.join(save_dir, f"model-{0:05d}-of-{world_size:05d}.safetensors"))
    weight_map = json.loads(Path(save_dir, "model.safetensors.index.json").read_text())["weight_map"]
    problems = []
    for key, want in expected.items():
        if key not in shard:
            problems.append(f"{key}: missing from the sharded EP checkpoint ({len(shard)} keys written)")
            continue
        if key not in weight_map:
            problems.append(f"{key}: missing from the shard index")
        got = shard[key]
        if tuple(got.shape) != (NUM_HEADS,):
            problems.append(
                f"{key}: TRUNCATED sink {tuple(got.shape)}, expected all {NUM_HEADS} heads "
                f"(rank 0's TP slice written under the full-tensor key)"
            )
        elif not torch.equal(got, want):
            problems.append(f"{key}: saved {got.tolist()} != pre-shard {want.tolist()}")
    return problems


def run(ctx) -> dict:
    save_dir = shared_scratch_dir("halo_ep_sharded_tp_sinks")
    if ctx.rank == 0:
        ctx.on_teardown(lambda: shutil.rmtree(save_dir, ignore_errors=True))

    model = _tiny_gpt_oss(ctx.device)
    mesh = init_device_mesh("cuda", (TP_SIZE,), mesh_dim_names=("tp",))
    model._device_mesh = mesh

    expected = {}
    for i, layer in enumerate(model.model.layers):
        full = _full_sinks(i, ctx.device)
        with torch.no_grad():
            layer.self_attn.sinks.copy_(full)
        expected[f"model.layers.{i}.self_attn.sinks"] = full.cpu()
        shard_sinks_param(model, layer.self_attn, "self_attn", mesh)
        assert layer.self_attn.sinks.shape[0] == NUM_HEADS // TP_SIZE, "TP sink slicing did not apply"

    if ctx.rank == 0:
        shutil.rmtree(save_dir, ignore_errors=True)
    ctx.barrier()

    save_ep_sharded_checkpoint(save_context(model), save_dir)
    ctx.barrier()

    # Only rank 0 writes the non-expert params. The verdict travels as data: an early rank-0 exit strands peers in the
    # next collective.
    problems = _check_saved_sinks(save_dir, ctx.world_size, expected) if ctx.rank == 0 else None
    verdict = [problems]
    dist.broadcast_object_list(verdict, src=0)
    if verdict[0]:
        log("\n".join(verdict[0]))
    else:
        log(f"sharded EP save wrote complete {NUM_HEADS}-head sinks for all {NUM_LAYERS} layers")

    return {"checks": {"sinks_complete": not verdict[0]}}


main = gpu_test_main(exact_world_size=TP_SIZE, prefix="ep_sharded_save_tp_sinks")(run)

if __name__ == "__main__":
    main()
