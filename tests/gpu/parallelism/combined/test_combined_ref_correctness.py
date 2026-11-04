#!/usr/bin/env python
"""Combined parallel shapes on Mistral4 compute the same math as a single-GPU reference, on byte-identical weights and
inputs.

EP, TP and ETP feed the full sequence to every rank, so the reduced loss must equal the reference; CP is covered
separately.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from transformers.models.mistral4 import Mistral4ForCausalLM

from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from tests.common.distributed import shared_scratch_dir, world_spread
from tests.common.ep_reference import (
    broadcast_reference,
    ep_layers,
    find_router_weight,
    full_grad,
    random_token_batch,
)
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.models import TINY_MISTRAL4_CONFIG
from tests.common.numerics import fro_rel_err
from tests.common.rank_log import log, log_all
from tests.common.tiny_models import build_tiny_mistral4_checkpoint
from tests.common.tolerances import TOL

# bf16 ulp near a loss of ~O(1-10) is ~1e-2, and EP/TP/ETP add accumulation in another float order. A real sharding or
# reduce bug moves the loss by >100%.
LOSS_TOL = 3e-2

# Measured noise is direction-only (norm ratio within 0.4%) and peaks at 0.14 on tp2. A missing reduction scales the
# gradient by 1/axis_size, relative error 0.5 (size 2) to 0.875 (size 8), so 0.35 has 2.5x headroom and sits 1.4x under
# the weakest bug.
ROUTER_GRAD_TOL = 0.35

_MODES = ("ep", "tp", "etp", "ep_tp", "ep_etp")


def _reference_loss_and_router_grad(
    ckpt_dir: str, ids: torch.Tensor, labels: torch.Tensor, device: str
) -> tuple[float, torch.Tensor]:
    """A dropped cross-rank reduction leaves the loss exact and silently scales the router gradient.
    ``Mistral4ForCausalLM`` directly, since ``AutoModelForCausalLM`` has no mistral4 mapping."""
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

    ids, labels = random_token_batch(TINY_MISTRAL4_CONFIG["vocab_size"], batch=2, seq=64, device=device)
    dist.broadcast(ids, src=0)
    dist.broadcast(labels, src=0)

    ref_loss_local, ref_grad = 0.0, None
    if ctx.rank == 0:
        ref_loss_local, ref_grad = _reference_loss_and_router_grad(ckpt_dir, ids, labels, device)
        log(f"  Reference loss (single-GPU): {ref_loss_local:.6f}  router grad norm: {ref_grad.norm():.6e}")
    # Every rank compares against the reference gradient.
    ref_loss, ref_grad = broadcast_reference(ref_loss_local, ref_grad, device, ctx.rank)
    metrics["ref_loss"] = ref_loss

    model, _ = load_distributed_model(
        model_name_or_path=ckpt_dir,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    layers = ep_layers(model)
    log(f"  EP/ETP wrapper layers: {len(layers)}")
    # first_k_dense_replace=0 makes every layer MoE; unwrapped, this compares the dense block to itself.
    checks["moe_layers_wrapped"] = len(layers) == TINY_MISTRAL4_CONFIG["num_hidden_layers"]

    model.train()
    out = model(input_ids=ids, labels=labels, use_cache=False)
    loss = out.loss
    log_all(f"  [{args.mode}] parallel loss: {loss.item():.6f}  |Δref|={abs(loss.item() - ref_loss):.3e}")
    metrics["parallel_loss"] = loss.item()
    checks["loss_finite"] = bool(torch.isfinite(loss))

    # Full sequence, EP⊥DP / TP-reduced.
    checks["loss_matches_reference"] = abs(loss.item() - ref_loss) < LOSS_TOL

    # A broken gather desyncs ranks.
    spread = world_spread(loss.item())
    metrics["rank_loss_spread"] = spread
    checks["losses_agree_across_ranks"] = TOL.identical_batch_ranks_agree(spread)

    loss.backward()
    router_name, router_weight = find_router_weight(model)
    got = full_grad(router_weight)
    checks["router_grad_finite"] = bool(torch.isfinite(got).all())

    # A missing reduction on any axis shows here as a clean 1/axis_size scale, and nowhere else: the forward stays
    # exact.
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


_ARGS = _parse() if __name__ == "__main__" else None


@gpu_test_main(min_world_size=2, prefix="combined_ref")
def _run(ctx):
    ctx.cli = _ARGS
    return run(ctx)


if __name__ == "__main__":
    _run()
