#!/usr/bin/env python
"""Numerical correctness of combined/complex parallelism shapes vs a single-GPU reference.

``tests/gpu/parallelism/test_mistral4_all_parallelism.py`` proves each shape *runs* (finite loss,
rank-consistent reduced loss, EP wrappers land, checkpoint roundtrips). It does **not**
prove the shape computes the *same math* as the undistributed model. This test closes
that gap for the non-CP shapes on Mistral4 — EP, TP, pure ETP, **EP+TP** and **EP+ETP** —
by comparing each shape's forward loss and first router gradient to a plain single-GPU
reference on byte-identical weights and inputs.

Reference: rank 0 loads the *same* synthetic checkpoint with a plain
``Mistral4ForCausalLM.from_pretrained`` (no distribution) and runs one forward+backward. The
loss and router gradient are broadcast; every rank asserts its reduced loss matches within
bf16 + grouped-mm + all-reduce reordering tolerance. Because these modes feed the FULL
sequence to every rank (EP/TP/ETP do not shard the sequence — only CP does), the reduced
per-rank loss must equal the reference; a sharding/gather/reduce bug moves it far outside
tolerance. CP is covered separately (``cp/test_cp_train_correctness.py`` on Qwen3,
``cp/test_glm4_cp_correctness.py``) because its per-rank chunk losses require token-weighted
aggregation to recover the reference.

One shape per ``--mode`` invocation (a runner chains them):

    torchrun --nproc_per_node=4 \
        tests/gpu/parallelism/combined/test_combined_ref_correctness.py --mode ep_tp --ep 2 --tp 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from transformers.models.mistral4 import Mistral4ForCausalLM

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.distributed import shared_scratch_dir, world_spread
from tests.common.ep_reference import (
    broadcast_reference,
    ep_layers,
    find_router_weight,
    full_grad,
    random_token_batch,
)
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_MISTRAL4_CONFIG
from tests.common.tiny_models import build_tiny_mistral4_checkpoint
from tests.common.tolerances import TOL
from tests.common.utils import cleanup_memory, fro_rel_err, log, log_all

# bf16 ulp near a loss of ~O(1-10) is ~1e-2. EP/TP/ETP each add extra all-reduce / grouped-mm
# accumulation in a different float order than the dense reference. 3e-2 is tight enough to catch
# a real sharding/gather/reduce bug (those move the loss by >100%) but loose enough not to trip on
# legitimate reordering noise on the 4-layer tiny model.
LOSS_TOL = 3e-2

# Relative L2 tolerance for the router gradient vs the single-GPU reference. Measured noise is
# direction-only (norm ratio stays within 0.4%) and peaks at 0.14 on tp2, where the router gradient is
# small (norm ~1e-2) and bf16 reduce order differs most. A missing cross-rank reduction scales the
# gradient by 1/axis_size, i.e. relative error 0.5 (size 2) to 0.875 (size 8) — so 0.35 sits with 2.5x
# headroom over noise and 1.4x under the weakest bug signal.
ROUTER_GRAD_TOL = 0.35

_MODES = ("ep", "tp", "etp", "ep_tp", "ep_etp")


def _reference_loss_and_router_grad(
    ckpt_dir: str, ids: torch.Tensor, labels: torch.Tensor, device: str
) -> tuple[float, torch.Tensor]:
    """Plain single-GPU forward+backward on the same weights (rank 0 only).

    Returns the loss and the router gradient. The gradient half is what catches a missing cross-rank
    reduction: a dropped one leaves the loss exact and silently scales the router gradient. Loads
    ``Mistral4ForCausalLM`` directly because ``AutoModelForCausalLM``, which ``dense_reference`` goes
    through, has no mistral4 mapping.
    """
    model = Mistral4ForCausalLM.from_pretrained(
        ckpt_dir,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(device)
    model.train()
    out = model(input_ids=ids, labels=labels, use_cache=False)
    out.loss.backward()
    loss = out.loss.item()
    grad = full_grad(find_router_weight(model)[1])
    del model, out
    cleanup_memory()
    return loss, grad


def run(ctx):
    args = ctx.cli
    device = f"cuda:{ctx.local_rank}"
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}

    # Build the tiny synthetic Mistral4 checkpoint on rank 0, then all ranks read it.
    ckpt_dir = shared_scratch_dir("combined_ref")
    if ctx.rank == 0:
        build_tiny_mistral4_checkpoint(Path(ckpt_dir))
    ctx.barrier()

    pc = ParallelismConfig(
        ep_size=args.ep,
        cp_size=1,
        tp_size=args.tp,
        expert_tp_size=args.etp,
        ep_fp32_router=False,
        ep_fp32_experts=False,
        max_concurrent_loading=0,
    )
    log(f"\n=== mode={args.mode}  ep={args.ep} tp={args.tp} etp={args.etp}  {pc.mode_string} ===")
    log(f"  data_parallel_size={pc.data_parallel_size}")

    # Identical inputs on every rank.
    ids, labels = random_token_batch(TINY_MISTRAL4_CONFIG["vocab_size"], batch=2, seq=64, device=device)
    dist.broadcast(ids, src=0)
    dist.broadcast(labels, src=0)

    # ── Reference (rank 0, no distribution) ──────────────────────────────────
    ref_loss_local, ref_grad = 0.0, None
    if ctx.rank == 0:
        ref_loss_local, ref_grad = _reference_loss_and_router_grad(ckpt_dir, ids, labels, device)
        log(f"  Reference loss (single-GPU): {ref_loss_local:.6f}  router grad norm: {ref_grad.norm():.6e}")
    # Every rank needs the reference gradient to compare against its own.
    ref_loss, ref_grad = broadcast_reference(ref_loss_local, ref_grad, device, ctx.rank)
    metrics["ref_loss"] = ref_loss

    # ── Parallel shape ───────────────────────────────────────────────────────
    model, _ = load_distributed_model(
        model_name_or_path=ckpt_dir,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    layers = ep_layers(model)
    log(f"  EP/ETP wrapper layers: {len(layers)}")
    # Every mode wraps each MoE block (at ep_size=1 for the grouped-GEMM expert compute), and
    # first_k_dense_replace=0 makes every layer one; unwrapped, this compares the dense block to itself.
    checks["moe_layers_wrapped"] = len(layers) == TINY_MISTRAL4_CONFIG["num_hidden_layers"]

    model.train()
    out = model(input_ids=ids, labels=labels, use_cache=False)
    loss = out.loss
    log_all(f"  [{args.mode}] parallel loss: {loss.item():.6f}  |Δref|={abs(loss.item() - ref_loss):.3e}")
    metrics["parallel_loss"] = loss.item()
    checks["loss_finite"] = bool(torch.isfinite(loss))

    # Reduced per-rank loss must match the dense reference (full sequence, EP⊥DP / TP-reduced).
    checks["loss_matches_reference"] = abs(loss.item() - ref_loss) < LOSS_TOL

    # Every rank must agree on the reduced loss (a broken gather desyncs ranks).
    spread = world_spread(loss.item())
    metrics["rank_loss_spread"] = spread
    checks["losses_agree_across_ranks"] = TOL.identical_batch_ranks_agree(spread)

    loss.backward()
    router_name, router_weight = find_router_weight(model)
    got = full_grad(router_weight)
    checks["router_grad_finite"] = bool(torch.isfinite(got).all())

    # Every rank sees identical inputs, so after the router hook's cross-rank average each rank's
    # router gradient must equal the single-GPU reference. A missing reduction on any axis shows up
    # here as a clean 1/axis_size scale — and nowhere else, since the forward stays exact.
    rel = fro_rel_err(got, ref_grad)
    metrics["router_grad_rel_err"] = rel
    metrics["router_grad_norm_ratio"] = got.norm().item() / ref_grad.norm().item()
    log_all(
        f"  [{args.mode}] router grad ({router_name}) vs reference: rel_err={rel:.4e} "
        f"norm_ratio={metrics['router_grad_norm_ratio']:.4f}"
    )
    checks["router_grad_matches_reference"] = rel < ROUTER_GRAD_TOL

    return {"checks": checks, "metrics": metrics}


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=_MODES)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--etp", type=int, default=1)
    return p.parse_args()


# gpu_test_main owns the lifecycle; stash parsed CLI on the ctx via a thin wrapper.
_ARGS = _parse() if __name__ == "__main__" else None


@gpu_test_main(min_world_size=2, prefix="combined_ref")
def _run(ctx):
    ctx.cli = _ARGS
    return run(ctx)


if __name__ == "__main__":
    _run()
