#!/usr/bin/env python
"""End-to-end EP / CP / TP / EP+CP / EP+TP test for Mistral4.

Validates EP (Mistral4MoE → DeepEP), CP (Mistral4Attention → Ulysses), and TP
(selective DTensor attention) integrations for the text backbone of mistral3
VLMs (``mistralai/Mistral-Small-4-119B-2603`` and similar). Each invocation of
this script runs **one** parallelism mode determined by ``--mode``; the
manifest's args entries chain them on 8 GPUs.

Build path: instead of downloading the 119 B fp8 checkpoint, the test
materializes ``tests.common.models.TINY_MISTRAL4_CONFIG``, which exercises the
same code paths (MLA + MoE + YARN + llama-4 scaling + shared expert + group
routing). A synthetic checkpoint is written to ``--checkpoint-dir`` (rank-0) so
the EP lazy loader and ``load_distributed_model`` paths are exercised end-to-end.

Run (8 GPUs):

    torchrun --nproc_per_node=8 \
        tests/gpu/parallelism/test_mistral4_all_parallelism.py \
        --mode ep --ep 8
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from liger_kernel.transformers import LigerRMSNorm
from torch.distributed.tensor import DTensor
from transformers.models.mistral4 import modeling_mistral4

from src.distributed.expert_parallel.saving import save_ep_model
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.distributed import cleanup_dirs, shared_scratch_dir, world_spread
from tests.common.ep_reference import ep_layers, random_token_batch
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_MISTRAL4_CONFIG
from tests.common.tiny_models import build_tiny_mistral4_checkpoint
from tests.common.tolerances import TOL
from tests.common.utils import cleanup_memory, gpu_mem_gb, log


def find_attention_classes(model: torch.nn.Module) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, mod in model.named_modules():
        cls = type(mod).__name__
        if "Attention" in cls or "Ulysses" in cls:
            counts[cls] = counts.get(cls, 0) + 1
    return counts


def run_mode(
    *,
    mode: str,
    ep: int,
    cp: int,
    tp: int,
    expert_tp: int,
    checkpoint_dir: str,
    reload_dir: str,
    use_liger: bool,
    rank: int,
    world_size: int,
    local_rank: int,
) -> tuple[dict[str, bool], dict]:
    """Load Mistral4 with the requested parallelism and run forward + backward.

    Returns ``(checks, metrics)``.
    """
    pc = ParallelismConfig(
        ep_size=ep,
        cp_size=cp,
        tp_size=tp,
        expert_tp_size=expert_tp,
        ep_fp32_router=False,
        ep_fp32_experts=False,
        max_concurrent_loading=0,  # All ranks in parallel — tiny model fits easily.
    )

    log(f"\n=== mode={mode}  ep={ep}  cp={cp}  tp={tp}  etp={expert_tp}  liger={use_liger} ===")
    log(f"  GPU memory before load: {gpu_mem_gb():.2f}GB")
    log(f"  world_size={world_size}  data_parallel_size={pc.data_parallel_size}")

    model, tokenizer = load_distributed_model(
        model_name_or_path=checkpoint_dir,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        use_liger_kernel=use_liger,
    )

    log(f"  GPU memory after load: {gpu_mem_gb():.2f}GB")

    checks: dict[str, bool] = {}
    moe_layers = ep_layers(model)
    attn_classes = find_attention_classes(model)

    # ── Sanity ─────────────────────────────────────────────────────────────
    # Mistral4 is MoE and ``use_grouped_gemm`` defaults on, so ``needs_ep_wrappers`` holds in every
    # mode: EP / EP+TP / EP+ETP / TP+MoE take the EP wrappers, and pure CP takes the
    # ``ep_size == 1`` grouped-GEMM wrappers (``_load_cp_model`` hands ``load_model_for_cp`` an
    # ep_config whenever ``needs_ep_wrappers`` holds, so its experts do not fall back to the stock
    # per-expert loop while still paying the Liger swiglu/geglu force-off).
    expected_layers = TINY_MISTRAL4_CONFIG["num_hidden_layers"]
    checks["ep_layers_present"] = len(moe_layers) == expected_layers
    log(f"  EP layers found: {len(moe_layers)} (expected {expected_layers})")
    if moe_layers:
        first = moe_layers[0]
        log(f"  experts_per_rank={first.experts_per_rank}, range=[{first.expert_start}, {first.expert_end})")
        log(f"  use_grouped_mm={first._use_grouped_mm}")
        checks["correct_ep_layer_class"] = type(first).__name__ == "EPMistral4MoELayer"
        # The wrappers exist in EVERY mode (grouped GEMM installs them at ep_size == 1), so their
        # presence proves nothing about the mode under test. The shard widths do: a config whose
        # ep/etp silently collapsed to 1 gets the full expert count and the full FFN width.
        checks["ep_shard_width"] = first.experts_per_rank == TINY_MISTRAL4_CONFIG["n_routed_experts"] // ep
        checks["etp_shard_width"] = first.expert_tp_size == expert_tp
        log(f"  expert_tp_size={first.expert_tp_size} (expected {expert_tp})")

    if pc.is_cp_mode:
        # Validate that attention was patched to the Ulysses variant.
        checks["cp_attention_patched"] = "Mistral4UlyssesAttention" in attn_classes
        log(f"  Attention classes after CP: {attn_classes}")
    if pc.is_tp_mode:
        # Selective TP keeps the outer attention class (Mistral4FlashAttention2) and turns
        # q_b_proj / kv_b_proj into DTensors — so the DTensors, not the class name, are the proof
        # the mode engaged. Without this the --mode tp and --mode ep_tp nodes assert no TP at all.
        checks["tp_engaged"] = any(isinstance(p.data, DTensor) for p in model.parameters())
        log(f"  Attention classes after TP: {attn_classes}; DTensor params present: {checks['tp_engaged']}")

    # Liger validation (only for paths that don't carry MoE replacement
    # incompatibilities — Liger swiglu disabled when EP wrappers active).
    if use_liger:
        # The toolkit patches by SUBCLASSING the family's own norm and binding Liger's forward, so
        # the class keeps its name and its `isinstance` relation; identity against LigerRMSNorm
        # would answer False for a correctly patched model.
        patched_norm = modeling_mistral4.Mistral4RMSNorm
        checks["liger_rmsnorm_patched"] = (
            getattr(patched_norm, "_halo_liger_patched_role", None) == "rms_norm"
            and patched_norm.forward is LigerRMSNorm.forward
        )
        log(f"  Liger Mistral4RMSNorm patched: {checks['liger_rmsnorm_patched']}")

    # ── Forward + backward ────────────────────────────────────────────────
    # Same input on every rank (broadcast for safety with random init).
    ids, labels = random_token_batch(TINY_MISTRAL4_CONFIG["vocab_size"], batch=2, seq=64, device=f"cuda:{local_rank}")
    dist.broadcast(ids, src=0)
    dist.broadcast(labels, src=0)

    # ``UlyssesCPModelWrapper.forward`` shards the inputs along the seq axis,
    # so pass the FULL sequence. For all other modes the model gets the full
    # sequence too — the only difference is that EP/TP do not require slicing.
    model.train()
    out = model(input_ids=ids, labels=labels, use_cache=False)

    loss = out.loss
    log(f"  Forward loss: {loss.item():.6f}")
    checks["loss_finite"] = torch.isfinite(loss).item()

    # All ranks should observe the same loss within a tight tolerance
    # (modulo CP sharding, which divides the per-token loss across ranks).
    spread = world_spread(loss.item())
    if pc.is_cp_mode:
        # Each CP rank computes loss on its sequence chunk → values differ;
        # only require finiteness (the spread is inf when any rank is non-finite).
        checks["losses_finite_across_ranks"] = math.isfinite(spread)
    else:
        checks["losses_consistent_across_ranks"] = spread < TOL.ep_identical_batch_rank_spread_abs
        log(f"  Cross-rank max loss diff: {spread:.6e}")

    # Backward (a raising backward fails the run through the harness, so no check entry here).
    loss.backward()
    cleanup_memory()

    # ── Gradient check on a known parameter ───────────────────────────────
    # EP: gate weight must have a gradient; expert weights live in the wrapper.
    if moe_layers:
        gate_w = moe_layers[0].gate.weight
        gate_grad = gate_w.grad
        gate_grad_norm = float(gate_grad.norm()) if gate_grad is not None else float("nan")
        # A norm of exactly zero means no router signal reached the gate — present but useless,
        # which a `is not None and isfinite` check accepts.
        checks["ep_router_grad_present"] = (
            gate_grad is not None and torch.isfinite(gate_grad).all().item() and gate_grad_norm > 0.0
        )
        log(f"  Router (gate) grad norm: {gate_grad_norm:.6e}")

    # ── EP checkpoint roundtrip ───────────────────────────────────────────
    # Save the model and reload it from the saved directory using the same
    # parallelism config. Validates that gathered EP weights are bit-exact
    # enough to reproduce the forward loss.
    saved_loss = float("nan")
    if pc.is_ep_mode and not (pc.is_cp_mode or pc.is_tp_mode):
        save_dir = Path(reload_dir)
        # Same set of save ranks each iteration — clear stale state first.
        dist.barrier()
        if rank == 0:
            shutil.rmtree(save_dir, ignore_errors=True)
        dist.barrier()

        log(f"  Saving EP model to {save_dir}")
        save_ep_model(model, str(save_dir), tokenizer=tokenizer, sharded=False)
        dist.barrier()

        if rank == 0:
            files = sorted(p.name for p in save_dir.iterdir())
            log(f"  Saved files: {files}")
            checks["ckpt_files_present"] = "model.safetensors" in files and "config.json" in files

        # Pre-save eval-mode LOGITS on the same batch — the reload comparison target (train-mode
        # loss is not comparable). Logits, not loss: this model is randomly initialized and never
        # optimizer-stepped, so its cross-entropy sits at ln(vocab) within ~0.03 whatever the
        # experts contain — a loss comparison cannot distinguish a correct gather from a zeroed
        # one. Runs on every rank: the EP forward is collective.
        model.eval()
        with torch.no_grad():
            pre_save_out = model(input_ids=ids, labels=labels, use_cache=False)
            pre_save_loss = pre_save_out.loss.item()
            pre_save_logits = pre_save_out.logits.float().cpu()

        # Free the trained model before reloading.
        del model
        cleanup_memory()

        reloaded, _ = load_distributed_model(
            model_name_or_path=str(save_dir),
            parallelism_config=pc,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            use_liger_kernel=False,
        )
        reloaded.eval()
        with torch.no_grad():
            out2 = reloaded(input_ids=ids, labels=labels, use_cache=False)
        saved_loss = out2.loss.item()
        log(f"  Reloaded forward loss: {saved_loss:.6f} (pre-save eval loss: {pre_save_loss:.6f})")
        checks["reload_loss_finite"] = torch.isfinite(out2.loss).item()
        # The gathered save must reproduce the forward, not merely stay finite — a gather that
        # drops or mangles expert tensors keeps the loss finite while the model is wrong.
        logit_delta = (out2.logits.float().cpu() - pre_save_logits).abs().max().item()
        log(f"  Max |Δlogit| across the save/reload roundtrip: {logit_delta:.3e}")
        checks["reload_logits_match"] = logit_delta < 1e-2
        del reloaded
        cleanup_memory()

    metrics = {
        "loss": loss.item() if torch.is_tensor(loss) else loss,
        "reload_loss": saved_loss,
        "ep_layer_count": len(moe_layers),
        "attention_classes": attn_classes,
    }
    return checks, metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        required=True,
        choices=["ep", "cp", "tp", "ep_cp", "ep_tp", "ep_etp"],
        help="Which parallelism configuration to test.",
    )
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--cp", type=int, default=1)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--etp", type=int, default=1, help="Expert TP size (only used when mode includes ETP).")
    p.add_argument("--checkpoint-dir", default=None, help="Directory holding the synthetic Mistral4 checkpoint.")
    p.add_argument("--liger", action="store_true", help="Apply Liger Kernel patches before loading.")
    return p.parse_args()


@gpu_test_main(min_world_size=2, prefix="mistral4_all_parallelism")
def run(ctx):
    args = parse_args()
    # Build the synthetic checkpoint on rank 0, barrier, then all ranks read it.
    ckpt_dir = args.checkpoint_dir or shared_scratch_dir("mistral4_tiny")
    reload_dir = shared_scratch_dir(f"mistral4_reload_{args.mode}")
    if ctx.rank == 0:
        ctx.on_teardown(lambda: cleanup_dirs(reload_dir))
        if args.checkpoint_dir is None:
            ctx.on_teardown(lambda: cleanup_dirs(ckpt_dir))
        log(f"Building synthetic Mistral4 checkpoint at {ckpt_dir}")
        build_tiny_mistral4_checkpoint(Path(ckpt_dir))
    ctx.barrier()

    checks, metrics = run_mode(
        mode=args.mode,
        ep=args.ep,
        cp=args.cp,
        tp=args.tp,
        expert_tp=args.etp,
        checkpoint_dir=ckpt_dir,
        reload_dir=reload_dir,
        use_liger=args.liger,
        rank=ctx.rank,
        world_size=ctx.world_size,
        local_rank=ctx.local_rank,
    )
    log(f"Metrics ({args.mode}): {json.dumps(metrics, indent=2, default=str)}")
    # A peer can still be reading reload_dir, which rank 0's teardown removes.
    ctx.barrier()
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
