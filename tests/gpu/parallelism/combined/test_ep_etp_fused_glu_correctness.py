#!/usr/bin/env python
"""ETP correctness for fused-GLU contiguous-halves MoE layers.

This test catches a class of bugs in the shared ``EPMoELayerBase`` fused-GLU
helper: naively slicing ``gate_up_proj [E, H, 2M]`` along dim 2 splits the
contiguous halves layout (``[gate(M) | up(M)]``) so each rank ends up with
only gate or only up, and ``chunk(2, dim=-1)`` mis-pairs them. The helper
therefore stores ``gate_proj`` and ``up_proj`` as separate shards under ETP.

It loads a small synthetic Mistral4 checkpoint (text backbone of mistral3
VLMs, which uses the same fused-GLU contiguous-halves layout as GLM-4 MoE
Lite, LFM-2, Qwen3.5/3.6, Gemma-4 and Zaya), runs two configurations and
compares forward losses:

1. **Reference** — rank 0 loads the model with plain ``from_pretrained``
   (no Halo loader, no EP wrapper), so the math runs through the standard
   ``Mistral4MoE`` block on full weights.
2. **ETP** — both ranks load the model with ``ep_size=1,
   expert_tp_size=2``, so the fused-GLU experts go through the shared
   ETP shard-and-all-reduce path.

Correct ETP sharding puts both losses within bf16 noise; slicing the fused
tensor instead moves the ETP loss orders of magnitude off.

Run (2 GPUs):

    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/combined/test_ep_etp_fused_glu_correctness.py
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.distributed as dist
from transformers.models.mistral4 import Mistral4ForCausalLM

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.distributed import shared_scratch_dir, world_spread
from tests.common.ep_reference import ep_layers, random_token_batch
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_MISTRAL4_CONFIG
from tests.common.tiny_models import build_tiny_mistral4_checkpoint
from tests.common.utils import cleanup_memory, gpu_mem_gb, log, log_all

# bf16 ulp at ~1.0 is ~7.8e-3; ETP adds one all-reduce per layer (4 layers in the tiny config) and
# bf16 grouped-mm reorders accumulation. 1e-2 sits above that noise and far below the shift of the
# gate/up pairs the contiguous-halves split bug mismatches.
LOSS_TOLERANCE = 1e-2
# Both ETP partners end every layer on the same all-reduced sum of the same batch, so their losses
# agree to reduction order.
ETP_RANK_SPREAD_ABS = 1e-4


def reference_forward(checkpoint_dir: str, ids: torch.Tensor, labels: torch.Tensor, device: str) -> float:
    """Load the model with no distribution and compute the reference forward loss.

    Only invoked on rank 0; other ranks read the broadcast value.
    """
    log("  Loading reference (no-EP, no-ETP) model on rank 0...")
    model = Mistral4ForCausalLM.from_pretrained(
        checkpoint_dir,
        dtype=torch.bfloat16,
        trust_remote_code=True,
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


def etp_forward(checkpoint_dir: str, ids: torch.Tensor, labels: torch.Tensor, rank: int) -> float:
    """Load the model under ep_size=1, expert_tp_size=2 and compute forward loss."""
    pc = ParallelismConfig(
        ep_size=1,
        expert_tp_size=2,
        max_concurrent_loading=0,  # tiny model, parallel load is fine
    )
    log(f"  Loading ETP model (ep_size=1, expert_tp_size=2) — {pc.mode_string}")
    log(f"  GPU memory before load: {gpu_mem_gb():.2f}GB")

    model, _ = load_distributed_model(
        model_name_or_path=checkpoint_dir,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    log(f"  GPU memory after load: {gpu_mem_gb():.2f}GB")

    layers = ep_layers(model)
    etp_layers = [m for m in layers if getattr(m, "expert_tp_size", 1) > 1]
    log(f"  EP layers: {len(layers)}, ETP layers: {len(etp_layers)}")
    # Without ETP-wrapped layers the loss comparison below measures the plain dense MoE block.
    assert etp_layers and len(etp_layers) == len(layers), (
        f"expected every MoE layer ETP-wrapped, got {len(etp_layers)} of {len(layers)}"
    )
    first = etp_layers[0]
    assert hasattr(first, "gate_proj") and hasattr(first, "up_proj"), (
        "ETP fused-GLU path must store gate_proj/up_proj separately, "
        f"got attrs: {[n for n, _ in first.named_parameters()]}"
    )
    assert not hasattr(first, "gate_up_proj"), (
        "ETP path should not retain the fused gate_up_proj parameter "
        "(slicing the fused tensor splits gate/up across ranks)"
    )
    log(
        f"  ✓ ETP layer has split gate_proj/up_proj shards: "
        f"gate_proj.shape={tuple(first.gate_proj.shape)}, "
        f"up_proj.shape={tuple(first.up_proj.shape)}, "
        f"down_proj.shape={tuple(first.down_proj.shape)}"
    )

    model.eval()
    with torch.no_grad():
        out = model(input_ids=ids, labels=labels, use_cache=False)
    loss = out.loss.item()
    log_all(f"  ETP loss (rank {rank}): {loss:.6f}")
    del model
    cleanup_memory()
    return loss


def run(ctx):
    device = f"cuda:{ctx.local_rank}"

    log("=" * 70)
    log("  ETP CORRECTNESS — fused-GLU contiguous halves (Mistral4 tiny)")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log("=" * 70)

    ckpt_dir = shared_scratch_dir("mistral4_tiny")

    if ctx.rank == 0:
        log(f"\nBuilding synthetic Mistral4 checkpoint at {ckpt_dir}")
        build_tiny_mistral4_checkpoint(Path(ckpt_dir))
    ctx.barrier()

    # Same input on every rank.
    ids, labels = random_token_batch(
        TINY_MISTRAL4_CONFIG["vocab_size"],
        batch=2,
        seq=64,
        device=device,
    )
    dist.broadcast(ids, src=0)
    dist.broadcast(labels, src=0)

    # --- Phase 1: reference forward (rank 0 only, no distributed ops) ---
    ref_loss_t = torch.zeros(1, dtype=torch.float64, device=device)
    if ctx.rank == 0:
        log("\n[1/2] Reference forward (no parallelism)")
        ref_loss_t[0] = reference_forward(ckpt_dir, ids, labels, device)
    ctx.barrier()
    dist.broadcast(ref_loss_t, src=0)
    ref_loss = ref_loss_t.item()

    # --- Phase 2: ETP forward (all ranks) ---
    log("\n[2/2] ETP forward (ep_size=1, expert_tp_size=2)")
    etp_loss = etp_forward(ckpt_dir, ids, labels, ctx.rank)

    # All ranks should agree on the ETP loss (same input, ETP reduces inside).
    cross_rank_diff = world_spread(etp_loss)
    ref_vs_etp = abs(etp_loss - ref_loss)

    log("\n" + "=" * 70)
    log(f"  Reference loss:         {ref_loss:.6f}")
    log(f"  Cross-rank ETP spread:  {cross_rank_diff:.3e}")
    log(f"  |ETP - reference|:      {ref_vs_etp:.3e}")
    log(f"  Tolerance:              {LOSS_TOLERANCE:.3e}")
    log("=" * 70)

    # Every rank holds the broadcast reference, so each judges its own ETP loss against it.
    checks = {
        "losses_finite": math.isfinite(ref_loss) and math.isfinite(etp_loss),
        "etp_ranks_agree": cross_rank_diff < ETP_RANK_SPREAD_ABS,
        "etp_matches_reference": ref_vs_etp < LOSS_TOLERANCE,
    }
    if not checks["etp_ranks_agree"]:
        log(f"  FAIL: ETP losses inconsistent across ranks (spread {cross_rank_diff:.3e})")
    if not checks["etp_matches_reference"]:
        log_all(
            f"  FAIL: ETP loss disagrees with reference "
            f"(|{etp_loss:.6f} - {ref_loss:.6f}| >= {LOSS_TOLERANCE:.3e}). "
            f"This is the contiguous-halves slicing bug "
            f"(gate_proj/up_proj must be sharded separately, not gate_up_proj)."
        )

    return {"checks": checks}


main = gpu_test_main(exact_world_size=2, prefix="ep_etp_fused_glu_correctness")(run)

if __name__ == "__main__":
    main()
