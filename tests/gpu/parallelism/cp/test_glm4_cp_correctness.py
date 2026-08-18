#!/usr/bin/env python
"""CP correctness for GLM4 MoE Lite (DeepSeek-V3-style MLA) attention.

GLM4 MoE Lite attention is MLA: ``kv_a_proj_with_mqa`` emits a compressed KV that
splits into a layer-normed/expanded ``k_pass`` and a single shared rotary
``k_rot`` (broadcast to every KV head), with ``qk_head_dim`` (a nope+rope split)
distinct from ``v_head_dim``. The Ulysses wrapper must reproduce that projection
exactly and pad V to ``qk_head_dim`` for flash — a Qwen3-style wrapper (feeding
the full compressed KV straight through ``kv_a_layernorm``/``kv_b_proj``) crashes
on the first forward with a shape mismatch.

This test builds a tiny all-MLA GLM4-lite checkpoint and compares the CP=2 loss
against the single-GPU reference, for both ``rope_interleave`` settings (the
interleaved RoPE path and the rotate-half path).

Run (2 GPUs):

    torchrun --nproc_per_node=2 \\
        tests/gpu/parallelism/cp/test_glm4_cp_correctness.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist
from transformers.models.glm4_moe_lite import (
    Glm4MoeLiteConfig,
    Glm4MoeLiteForCausalLM,
)

from src.distributed.context_parallel.layers.glm4 import Glm4MoeLiteUlyssesAttention
from src.distributed.context_parallel.wrapper import UlyssesCPModelWrapper
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.distributed import shared_scratch_dir
from tests.common.harness import gpu_test_main
from tests.common.utils import cleanup_memory, log, log_all

# Tiny MLA config: num_attention_heads=8 → 4 per CP rank, num_key_value_heads=8
# → 4 per rank. q_lora_rank set (exercises the q_a/q_b LoRA-Q path); MLA dims are
# distinct (qk_head_dim = 16+16 = 32, v_head_dim = 32). Layer 0 is dense, the
# rest MoE (config.mlp_layer_types). n_group=1 so group-limited routing is off.
TINY_CONFIG = {
    "vocab_size": 512,
    "hidden_size": 256,
    "intermediate_size": 512,
    "moe_intermediate_size": 64,
    "num_hidden_layers": 3,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "n_shared_experts": 1,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "kv_lora_rank": 64,
    "q_lora_rank": 48,
    "qk_rope_head_dim": 16,
    "v_head_dim": 32,
    "qk_nope_head_dim": 16,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "routed_scaling_factor": 1.0,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "max_position_embeddings": 512,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
}

# MLA adds an all-to-all + V-pad/crop per layer on top of bf16 / flash-attn
# ordering noise and the CP per-rank-sum vs reference-mean normalization
# difference. The crash/corruption modes this guards against (wrong projection or
# wrong interleave) produce order-of-magnitude errors, so a 1e-1 tolerance both
# admits normal noise and catches the real bug.
LOSS_TOLERANCE = 1e-1


def build_tiny_ckpt(out_dir: Path, rope_interleave: bool, seed: int = 0) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    config = Glm4MoeLiteConfig(**TINY_CONFIG, rope_interleave=rope_interleave)
    model = Glm4MoeLiteForCausalLM(config).to(torch.bfloat16)
    model.save_pretrained(out_dir, safe_serialization=True)
    return out_dir


def make_inputs(vocab_size: int, batch: int, seq: int, device: str):
    torch.manual_seed(123)
    ids = torch.randint(0, vocab_size, (batch, seq), device=device)
    labels = ids.clone()
    labels[:, ::4] = -100  # leave every rank some active labels
    return ids, labels


def reference_forward(checkpoint_dir: str, ids: torch.Tensor, labels: torch.Tensor, device: str) -> float:
    model = Glm4MoeLiteForCausalLM.from_pretrained(
        checkpoint_dir,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    model.eval()
    with torch.no_grad():
        out = model(input_ids=ids, labels=labels, use_cache=False)
    loss = out.loss.item()
    log(f"  Reference loss: {loss:.6f}")
    del model
    cleanup_memory()
    return loss


def cp_forward(checkpoint_dir: str, ids: torch.Tensor, labels: torch.Tensor, rank: int, local_rank: int) -> float:
    pc = ParallelismConfig(cp_size=2)
    device = f"cuda:{local_rank}"
    base_model = Glm4MoeLiteForCausalLM.from_pretrained(
        checkpoint_dir,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)

    model = UlyssesCPModelWrapper(base_model, pc.create_cp_config())

    wrappers = [m for m in model.modules() if isinstance(m, Glm4MoeLiteUlyssesAttention)]
    log(f"  Glm4MoeLiteUlyssesAttention layers: {len(wrappers)}")
    assert len(wrappers) == TINY_CONFIG["num_hidden_layers"], (
        f"Expected {TINY_CONFIG['num_hidden_layers']} CP-wrapped layers, got {len(wrappers)}"
    )
    # The loss tolerance can't distinguish RoPE pairings on a random tiny model — pin the wrapper's
    # resolved flag to the config field (catches a wrong/renamed attribute lookup outright).
    for w in wrappers:
        assert w.rope_interleave == base_model.config.rope_interleave, (
            f"wrapper rope_interleave={w.rope_interleave} != config.rope_interleave="
            f"{base_model.config.rope_interleave}"
        )
    for layer in base_model.model.layers:
        assert isinstance(layer.self_attn, Glm4MoeLiteUlyssesAttention), (
            f"self_attn is {type(layer.self_attn).__name__}, expected Glm4MoeLiteUlyssesAttention"
        )

    model.eval()
    with torch.no_grad():
        out = model(input_ids=ids, labels=labels, use_cache=False)
    loss = out.loss.item()
    log_all(f"  CP loss (rank {rank}): {loss:.6f}")
    del model, base_model
    cleanup_memory()
    return loss


def run_correctness(rope_interleave: bool, ctx) -> tuple[bool, float]:
    label = "interleaved" if rope_interleave else "rotate-half"
    device = str(ctx.device)
    log(f"\n--- GLM4 MLA CP correctness [{label} RoPE] ---")
    ckpt_dir = shared_scratch_dir(f"glm4_tiny_mla_{label}")

    if ctx.rank == 0:
        build_tiny_ckpt(Path(ckpt_dir), rope_interleave=rope_interleave)
    ctx.barrier()

    ids, labels = make_inputs(TINY_CONFIG["vocab_size"], batch=2, seq=64, device=device)
    dist.broadcast(ids, src=0)
    dist.broadcast(labels, src=0)

    ref_loss_t = torch.zeros(1, dtype=torch.float64, device=device)
    if ctx.rank == 0:
        ref_loss_t[0] = reference_forward(ckpt_dir, ids, labels, device)
    ctx.barrier()
    dist.broadcast(ref_loss_t, src=0)
    ref_loss = ref_loss_t.item()

    cp_loss = cp_forward(ckpt_dir, ids, labels, ctx.rank, ctx.local_rank)
    all_losses = [torch.zeros(1, device=device, dtype=torch.float64) for _ in range(ctx.world_size)]
    dist.all_gather(all_losses, torch.tensor([cp_loss], device=device, dtype=torch.float64))
    per_rank = [t.item() for t in all_losses]
    cp_mean = sum(per_rank) / len(per_rank)
    delta = abs(cp_mean - ref_loss)

    ok = delta <= LOSS_TOLERANCE
    if ctx.rank == 0:
        log(f"  reference={ref_loss:.6f}  cp_mean={cp_mean:.6f}  |Δ|={delta:.3e}  tol={LOSS_TOLERANCE:.1e}")
        log(f"  {'PASS' if ok else 'FAIL'}: GLM4 MLA CP [{label}]")
    return ok, delta


def run(ctx):
    log("=" * 70)
    log("  GLM4 MoE Lite CP CORRECTNESS — DeepSeek-V3 MLA projection + RoPE")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log("=" * 70)

    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    for interleave in (True, False):
        key = "interleaved" if interleave else "rotate_half"
        ok, delta = run_correctness(interleave, ctx)
        checks[f"cp_loss_matches_reference_{key}"] = ok
        metrics[f"loss_delta_{key}"] = delta

    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(min_world_size=2, exact_world_size=2, prefix="glm4_cp_correctness")(run)

if __name__ == "__main__":
    main()
