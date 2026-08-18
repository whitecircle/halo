#!/usr/bin/env python
"""
EP vs Non-EP Correctness Test: Gold Standard Comparison.

Loads the SAME MoE model twice — once without EP (standard FSDP, all experts
on every rank) and once with EP=2 (experts distributed across ranks via DeepEP) —
and compares forward pass loss and per-token logits on identical input, so that EP
dispatch → expert compute → combine is measured directly against having all experts
locally available.

The baseline is loaded through the same loader with no parallelism axes — grouped GEMM
(the SM90+ default) still wraps its experts, so the baseline is itself an approximation of the
undistributed model, not the undistributed model. Both sides are therefore scored against a plain
``AutoModelForCausalLM`` reference rather than against each other: each carries its own independent
router-pick-flip deviation from that reference, and the DIFFERENCE of two such deviations is not
bounded by a tolerance derived for one of them (see LOSS_ABS_TOL).

Both sides are loaded over the checkpoint's PRETRAINED attention sinks, on the one backend that
carries them — see ATTN_IMPLEMENTATION. Under the loader's fine-tuning sink reset the model runs
off-distribution, both sides reroute against each other, and the file reports that as an EP defect.
This is the nightly counterpart of the core-tier
tests/gpu/parallelism/ep/test_ep_correctness.py: it scores the ep1 wrapper alongside the EP model
and adds the per-token logit and top-1 comparisons that one leaves out.

Test Matrix:
  1. Forward pass: both the non-EP baseline and the EP loss match the undistributed reference
  2. Logit comparison: EP logits ≈ non-EP logits (cosine similarity)
  3. Router gradient comparison: EP router grads ≈ non-EP router grads
  4. Per-expert output: verifies routed tokens produce same expert output

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/ep/test_ep_vs_no_ep.py

Requirements:
    - 2x GPUs with >=100GB memory each (rank 0 briefly holds the dense reference and its gradients)
    - DeepEP installed
    - Model: unsloth/gpt-oss-20b-BF16 (auto-downloaded)
"""

import sys
import traceback

import torch
import torch.distributed as dist
from accelerate import PartialState
from transformers import AutoTokenizer

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from tests.common.distributed import ensure_model_downloaded, init_distributed, teardown_distributed
from tests.common.ep_reference import broadcast_reference, dense_reference, fixed_chat_batch
from tests.common.models import GPT_OSS_20B
from tests.common.tolerances import TOL
from tests.common.utils import cleanup_memory, gpu_mem_gb, log, log_all

MODEL_NAME = GPT_OSS_20B
EP_SIZE = 2
SEQ_LEN = 128
SEED = 42

# Live pretrained sinks (`reset_sinks=False`) are what make this an equivalence statement about
# parallelism: neutralising them puts gpt-oss off-distribution onto near-tied top-4-of-32 router
# margins, where bf16 reassociation between ep1 and ep2 reroutes ~30% of top-1 and the router-grad
# ratio wanders to 0.48-2.53 regardless of matched backends. Eager is the only backend carrying live
# sinks (FA2/SDPA are refused). Same settings/reasons as the core-tier test_ep_correctness.py.
ATTN_IMPLEMENTATION = "eager"

# bf16 reorders the expert sum feeding a near-tied top-4-of-32 router, so loss rides the
# router-pick-flip bound rather than a bitwise one. It bounds ONE side against the undistributed
# reference, which is why both sides are scored against that reference and never against each other:
# swept over nine renderings of this fixture each side stays inside 0.086 of the reference while the
# gap between them reaches 0.128. Nor can that gap simply be gated wider — rotating an expert bank
# moves it by as little as 0.17, so a bound loose enough for the noise would pass a wrongly-routed model.
LOSS_ABS_TOL = TOL.router_pick_flip_loss_abs
LOGIT_COSINE_MIN = 0.95
ROUTER_GRAD_COSINE_MIN = TOL.grad_direction_cosine_min


def cosine_sim(a, b):
    """Compute cosine similarity between two flat tensors."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def extract_router_grads(model):
    """Extract router/gate weight gradients from a model."""
    grads = {}
    for name, param in model.named_parameters():
        if ("router" in name or "gate" in name) and "weight" in name and param.grad is not None:
            grads[name] = param.grad.data.clone().float()
    return grads


def run_baseline_forward(batch):
    """Load model WITHOUT EP and run forward pass.

    Returns loss, logits (last token), and router gradients on rank 0.
    """
    input_ids, attention_mask, labels = batch

    log(f"\n{'=' * 70}")
    log("PHASE 1: Non-EP Baseline (Standard FSDP, All Experts Local)")
    log(f"{'=' * 70}")

    log("  Loading model WITHOUT EP (via load_distributed_model, no parallelism)...")
    log(f"  GPU memory before: {gpu_mem_gb():.2f} GB")

    # Same loader as the EP side so attn_implementation validation matches (GptOss rejects FA2 in tf5).
    no_ep_config = ParallelismConfig()
    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=no_ep_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=ATTN_IMPLEMENTATION,
        reset_sinks=False,
    )
    model.eval()

    log(f"  GPU memory after load: {gpu_mem_gb():.2f} GB")
    total_params = sum(p.numel() for p in model.parameters())
    log(f"  Total params: {total_params / 1e9:.2f}B")

    log(f"  Input shape: {input_ids.shape}")

    model.train()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    loss = outputs.loss
    logits = outputs.logits.detach().clone()
    loss_val = loss.item()

    log_all(f"  Baseline loss: {loss_val:.6f}")

    loss.backward()
    router_grads = extract_router_grads(model)
    log(f"  Router grad params extracted: {len(router_grads)}")

    del model, outputs
    cleanup_memory()

    log(f"  GPU memory after cleanup: {gpu_mem_gb():.2f} GB")

    return loss_val, logits, router_grads


def run_ep_forward(batch):
    """Load model WITH EP=2 and run forward pass.

    Returns loss, logits (last token), and router gradients on rank 0.
    """
    rank = dist.get_rank()
    input_ids, attention_mask, labels = batch

    log(f"\n{'=' * 70}")
    log("PHASE 2: EP=2 (Experts Distributed via DeepEP)")
    log(f"{'=' * 70}")

    log("  Loading model WITH EP=2...")
    log(f"  GPU memory before: {gpu_mem_gb():.2f} GB")

    parallelism_config = ParallelismConfig(
        ep_size=EP_SIZE,
        cp_size=1,
        tp_size=1,
        ep_fp32_router=False,
        ep_fp32_experts=False,
    )
    log(f"  Config: {parallelism_config.summary()}")

    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=ATTN_IMPLEMENTATION,
        reset_sinks=False,
    )

    log(f"  GPU memory after load: {gpu_mem_gb():.2f} GB")

    ep_layers = [m for _, m in model.named_modules() if hasattr(m, "ep_config")]
    log(f"  EP layers found: {len(ep_layers)}")
    # Without EP wrappers this compares a dense model to itself and every check below scores perfectly.
    if not ep_layers:
        raise RuntimeError("EP=2 load produced no EP-wrapped layers — the comparison would be vacuous")

    if ep_layers:
        layer = ep_layers[0]
        log(f"  Expert range (rank {rank}): [{layer.expert_start}, {layer.expert_end})")
        log(f"  Experts per rank: {layer.experts_per_rank}")

    log(f"  Input shape: {input_ids.shape}")

    model.train()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    loss = outputs.loss
    logits = outputs.logits.detach().clone()
    loss_val = loss.item()

    log_all(f"  EP loss: {loss_val:.6f}")

    loss.backward()
    router_grads = extract_router_grads(model)
    log(f"  Router grad params extracted: {len(router_grads)}")

    del model, outputs
    cleanup_memory()

    log(f"  GPU memory after cleanup: {gpu_mem_gb():.2f} GB")

    return loss_val, logits, router_grads


def compare_results(
    reference_loss,
    baseline_loss,
    baseline_logits,
    baseline_router_grads,
    ep_loss,
    ep_logits,
    ep_router_grads,
    local_rank,
):
    """Score both sides against the undistributed reference. Returns (passed, results dict)."""
    rank = dist.get_rank()
    device = f"cuda:{local_rank}"

    log(f"\n{'=' * 70}")
    log("PHASE 3: Comparison (EP vs Non-EP)")
    log(f"{'=' * 70}")

    passed = True
    results = {}

    baseline_losses = [None] * dist.get_world_size()
    ep_losses = [None] * dist.get_world_size()
    dist.all_gather_object(baseline_losses, baseline_loss)
    dist.all_gather_object(ep_losses, ep_loss)

    if rank == 0:
        log("\n  --- Loss Comparison ---")
        log(f"  Baseline losses (per rank): {[f'{l:.6f}' for l in baseline_losses]}")
        log(f"  EP losses (per rank):       {[f'{l:.6f}' for l in ep_losses]}")

        all_finite = all(torch.isfinite(torch.tensor(l)) for l in baseline_losses + ep_losses)
        results["all_finite"] = all_finite
        log(f"  All losses finite: {'PASS' if all_finite else 'FAIL'}")
        if not all_finite:
            passed = False

        log(f"  Reference loss (undistributed): {reference_loss:.6f}")
        for side, value in (("baseline", baseline_losses[0]), ("ep", ep_losses[0])):
            delta = abs(value - reference_loss)
            ok = delta < LOSS_ABS_TOL
            results[f"{side}_matches_reference"] = ok
            results[f"{side}_reference_loss_diff"] = delta
            log(f"  |{side} - reference|: {delta:.6f} (tol={LOSS_ABS_TOL}): {'PASS' if ok else 'FAIL'}")
            if not ok:
                passed = False

        # Reported, never gated: it is the DIFFERENCE of the two deviations above, so it carries
        # both and no bound on it can separate the noise from a wrong-expert routing defect.
        loss_diff = abs(baseline_losses[0] - ep_losses[0])
        results["loss_diff"] = loss_diff
        log(f"  Baseline-vs-EP loss gap (reported, not gated): {loss_diff:.6f}")

        avg_baseline = sum(baseline_losses) / len(baseline_losses)
        avg_ep = sum(ep_losses) / len(ep_losses)
        avg_diff = abs(avg_baseline - avg_ep)
        results["avg_loss_diff"] = avg_diff
        log(f"  Avg loss diff: {avg_diff:.6f}")

        # Every rank ran the identical broadcast batch, so any spread is a desync, not data noise —
        # hence the tight identical-batch bound rather than the DP one.
        baseline_spread = max(baseline_losses) - min(baseline_losses)
        results["baseline_spread"] = baseline_spread
        results["baseline_ranks_consistent"] = baseline_spread < TOL.ep_identical_batch_rank_spread_abs
        if not results["baseline_ranks_consistent"]:
            passed = False
        log(f"  Baseline cross-rank spread: {baseline_spread:.8f} (tol={TOL.ep_identical_batch_rank_spread_abs})")

        ep_spread = max(ep_losses) - min(ep_losses)
        results["ep_spread"] = ep_spread
        results["ep_ranks_consistent"] = ep_spread < TOL.ep_identical_batch_rank_spread_abs
        if not results["ep_ranks_consistent"]:
            passed = False
        log(f"  EP cross-rank spread: {ep_spread:.8f} (tol={TOL.ep_identical_batch_rank_spread_abs})")

    if rank == 0:
        log("\n  --- Logit Comparison ---")
        log(f"  Baseline logits shape: {baseline_logits.shape}")
        log(f"  EP logits shape: {ep_logits.shape}")

        logit_cos = cosine_sim(baseline_logits, ep_logits)
        results["logit_cosine"] = logit_cos
        logit_ok = logit_cos >= LOGIT_COSINE_MIN
        log(f"  Logit cosine similarity: {logit_cos:.6f} (min={LOGIT_COSINE_MIN}): {'PASS' if logit_ok else 'FAIL'}")
        if not logit_ok:
            passed = False

        B, S, V = baseline_logits.shape
        position_cosines = []
        for s in range(S):
            pos_cos = cosine_sim(baseline_logits[0, s], ep_logits[0, s])
            position_cosines.append(pos_cos)

        avg_pos_cos = sum(position_cosines) / len(position_cosines)
        min_pos_cos = min(position_cosines)
        worst_pos = position_cosines.index(min_pos_cos)
        results["avg_position_cosine"] = avg_pos_cos
        results["min_position_cosine"] = min_pos_cos
        log(f"  Avg per-position cosine: {avg_pos_cos:.6f}")
        log(f"  Min per-position cosine: {min_pos_cos:.6f} (position {worst_pos})")

        abs_diff = (baseline_logits.float() - ep_logits.float()).abs()
        log(f"  Logit abs diff — mean: {abs_diff.mean():.6f}, max: {abs_diff.max():.6f}, std: {abs_diff.std():.6f}")

        baseline_top1 = baseline_logits.argmax(dim=-1)
        ep_top1 = ep_logits.argmax(dim=-1)
        agreement = (baseline_top1 == ep_top1).float().mean().item()
        results["top1_agreement"] = agreement
        log(f"  Top-1 token agreement: {agreement:.4f} ({agreement * 100:.1f}%)")

    # The router weight is REPLICATED and world-synced (EP distributes only experts), so its grad is
    # reduced exactly once. Two gates: SCALE (a dropped/doubled reduction scales the norm ratio by the
    # axis size) and DIRECTION (a routing/permutation corruption reorients it without moving the norm).
    if rank == 0:
        log("\n  --- Router Gradient Comparison (pass/fail: norm ratio + cosine) ---")

        if not baseline_router_grads or not ep_router_grads:
            # A severed router is the regression this exists to catch — must fail, not skip.
            log("  FAIL: no router grads captured on one or both sides")
            results["router_grad_cosine"] = None
            passed = False
        else:
            baseline_keys = sorted(baseline_router_grads.keys())
            ep_keys = sorted(ep_router_grads.keys())
            log(f"  Baseline router grad keys ({len(baseline_keys)} layers): {baseline_keys[:3]}...")
            log(f"  EP router grad keys ({len(ep_keys)} layers): {ep_keys[:3]}...")

            if len(baseline_keys) == len(ep_keys):
                all_baseline_grads = torch.cat([baseline_router_grads[k].flatten() for k in baseline_keys])
                all_ep_grads = torch.cat([ep_router_grads[k].flatten() for k in ep_keys])

                grad_cos = cosine_sim(all_baseline_grads, all_ep_grads)
                results["router_grad_cosine"] = grad_cos
                # DIRECTION gate: bf16 reassociation alone leaves this at 0.978 over pretrained sinks.
                grad_cos_ok = grad_cos > ROUTER_GRAD_COSINE_MIN
                results["router_grad_cosine_ok"] = grad_cos_ok
                log(
                    f"  Router grad cosine similarity: {grad_cos:.6f} "
                    f"(min={ROUTER_GRAD_COSINE_MIN}): {'PASS' if grad_cos_ok else 'FAIL'}"
                )
                if not grad_cos_ok:
                    passed = False

                # SCALE gate: the band admits 1.25x, so a 2x reduction error cannot hide (measures 0.9965).
                baseline_norm = all_baseline_grads.norm().item()
                ep_norm = all_ep_grads.norm().item()
                norm_ratio = ep_norm / baseline_norm if baseline_norm > 0 else float("inf")
                results["router_grad_norm_ratio"] = norm_ratio
                norm_ratio_ok = 1 / TOL.grad_norm_ratio_max < norm_ratio < TOL.grad_norm_ratio_max
                results["router_grad_norm_ratio_ok"] = norm_ratio_ok
                log(f"  Baseline grad norm: {baseline_norm:.6f}")
                log(f"  EP grad norm: {ep_norm:.6f}")
                log(
                    f"  Norm ratio (EP/baseline): {norm_ratio:.4f} "
                    f"(band {1 / TOL.grad_norm_ratio_max:.2f}–{TOL.grad_norm_ratio_max}): "
                    f"{'PASS' if norm_ratio_ok else 'FAIL'}"
                )
                if not norm_ratio_ok:
                    passed = False

                layer_cosines = []
                for bk, ek in zip(baseline_keys, ep_keys, strict=False):
                    lcos = cosine_sim(baseline_router_grads[bk], ep_router_grads[ek])
                    layer_cosines.append(lcos)
                log(
                    f"  Per-layer grad cosine: min={min(layer_cosines):.4f}, "
                    f"max={max(layer_cosines):.4f}, avg={sum(layer_cosines) / len(layer_cosines):.4f}"
                )

                worst_layer = layer_cosines.index(min(layer_cosines))
                best_layer = layer_cosines.index(max(layer_cosines))
                log(f"  Best layer: {baseline_keys[best_layer]} (cos={layer_cosines[best_layer]:.4f})")
                log(f"  Worst layer: {baseline_keys[worst_layer]} (cos={layer_cosines[worst_layer]:.4f})")

                baseline_nonzero = (all_baseline_grads.abs() > 1e-8).float().mean().item()
                ep_nonzero = (all_ep_grads.abs() > 1e-8).float().mean().item()
                grads_alive = baseline_nonzero > 0.5 and ep_nonzero > 0.5
                results["grads_alive"] = grads_alive
                log(f"  Baseline grad non-zero fraction: {baseline_nonzero:.4f}")
                log(f"  EP grad non-zero fraction: {ep_nonzero:.4f}")
                log(f"  Gradients alive (both >50% non-zero): {'PASS' if grads_alive else 'FAIL'}")
                if not grads_alive:
                    passed = False
            else:
                log(f"  FAIL: key count mismatch: baseline={len(baseline_keys)}, ep={len(ep_keys)}")
                results["router_grad_cosine"] = None
                passed = False

    pass_tensor = torch.tensor([1 if passed else 0], device=device, dtype=torch.int32)
    dist.broadcast(pass_tensor, src=0)
    passed = pass_tensor.item() == 1

    return passed, results


def main():
    rank, world_size, local_rank = init_distributed()
    PartialState()

    log(f"\n{'#' * 70}")
    log("  EP vs Non-EP Correctness Test (Gold Standard)")
    log(f"  World size: {world_size}, EP size: {EP_SIZE}")
    log(f"  Model: {MODEL_NAME}")
    log(f"  Seq len: {SEQ_LEN}, Seed: {SEED}")
    log(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
    log(
        f"  Tolerances: loss_abs={LOSS_ABS_TOL}, logit_cos={LOGIT_COSINE_MIN}, "
        f"router_grad_cos={ROUTER_GRAD_COSINE_MIN}"
    )
    log(f"{'#' * 70}")

    if world_size != EP_SIZE:
        log(f"\nERROR: This test requires exactly {EP_SIZE} GPUs, got {world_size}")
        teardown_distributed()
        return 1

    try:
        log("\nEnsuring model is downloaded...")
        ensure_model_downloaded(MODEL_NAME, rank)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # One batch for all three models, broadcast once: a per-phase rebuild would leave the sides
        # comparable only for as long as the fixture stayed byte-identical between calls.
        device = f"cuda:{local_rank}"
        batch = fixed_chat_batch(tokenizer, SEQ_LEN, device, seed=SEED)
        for tensor in batch:
            dist.broadcast(tensor, src=0)

        # Undistributed reference: neither side of the comparison below is one, so both are scored
        # against this. Rank 0 builds and frees it before the parallel loads (see LOSS_ABS_TOL).
        reference_loss_local = 0.0
        if rank == 0:
            reference_loss_local, _ = dense_reference(
                MODEL_NAME, *batch, device, attn_implementation=ATTN_IMPLEMENTATION
            )
            log(f"  Reference (single-GPU dense): loss={reference_loss_local:.6f}")
        reference_loss, _ = broadcast_reference(reference_loss_local, None, device, rank, with_grad=False)

        barrier()
        cleanup_memory()

        baseline_loss, baseline_logits, baseline_router_grads = run_baseline_forward(batch)

        barrier()
        cleanup_memory()

        ep_loss, ep_logits, ep_router_grads = run_ep_forward(batch)

        barrier()
        cleanup_memory()

        passed, results = compare_results(
            reference_loss,
            baseline_loss,
            baseline_logits,
            baseline_router_grads,
            ep_loss,
            ep_logits,
            ep_router_grads,
            local_rank,
        )

    except Exception as e:
        log(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        passed = False
        results = {}

    barrier()

    log(f"\n{'#' * 70}")
    if passed:
        log("  EP vs NON-EP CORRECTNESS TEST: PASSED")
        log("  EP produces numerically equivalent results to non-EP baseline")
    else:
        log("  EP vs NON-EP CORRECTNESS TEST: FAILED")
        if results:
            for k, v in results.items():
                log(f"    {k}: {v}")
    log(f"{'#' * 70}\n")

    barrier()
    teardown_distributed()

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
