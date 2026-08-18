#!/usr/bin/env python
"""Qwen3-MoE under EP=2 must compute what the same checkpoint computes with no distribution at all.

Every way EP can be wrong on this family — a token dispatched to the wrong expert, an expert bank
sliced at the wrong offset, a combine that drops one rank's contribution, a router gradient reduced
by the wrong divisor — yields a finite loss, finite gradients, correctly-shaped EP wrappers and
per-rank agreement. Structural inspection cannot see any of it. The only thing that can is the same
math on the same weights with nothing sharded, so that is what this test compares against:

* rank 0 runs a plain ``AutoModelForCausalLM`` forward+backward on the same checkpoint and the same
  batch, then frees it and broadcasts its loss and first router gradient (an undistributed reference
  is what FSDP2 data-parallel reproduces exactly — FSDP shards storage, not arithmetic — so it is
  the reference for the dense side, at one model's worth of memory instead of two);
* every rank's EP loss must match it (EP feeds the FULL sequence to every rank — only CP shards it —
  so the per-rank loss equals the undistributed one with no aggregation);
* every rank's router gradient must match it in scale AND direction after the EP router hook's
  cross-rank average — the two independent bug classes, a mis-scaled reduction and a routing
  corruption, are visible in exactly one of those two numbers each;
* an expert-identity NEGATIVE CONTROL rotates each rank's local expert bank by one expert and
  re-runs forward and backward: if the comparisons above were insensitive to *which* expert receives
  a token, the rotated model would match too. It must miss by a wide margin instead.

Both sides load with ``flash_attention_2``: the reference exists to isolate the parallelism, so any
other difference between the two models (attention backend included) would be measured as one.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/ep/test_ep_vs_reference_qwen3_moe.py

Requirements:
    - 2x GPUs, >=140GB each (rank 0 briefly holds the full dense model plus its gradients;
      measured peak 122GB)
    - DeepEP installed
    - Model: Qwen/Qwen3-30B-A3B-Instruct-2507 (auto-downloaded)
"""

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.distributed import ensure_model_downloaded
from tests.common.ep_reference import (
    broadcast_reference,
    compare_grad,
    dense_reference,
    ep_layers,
    find_router_weight,
    fixed_chat_batch,
    full_grad,
    roll_expert_banks,
)
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_30B_A3B
from tests.common.tolerances import TOL
from tests.common.utils import gpu_peak_mem_gb, log, log_all

MODEL_NAME = QWEN3_30B_A3B
EP_SIZE = 2
MAX_SEQ_LENGTH = 256
SEED = 42
ATTN_IMPLEMENTATION = "flash_attention_2"
EXPECTED_LAYER_TYPE = "EPQwen3MoELayer"

# Both sides are the same 48-layer bf16 checkpoint on the same batch. EP reorders the bf16 expert
# sum, and from layer 1 on that reordered output is what the next router scores — which flips a
# handful of near-tied picks out of top-8-of-128 with no sync bug present. That is exactly the
# mechanism ``router_pick_flip_loss_abs`` is derived for. Measured here: reference 3.037 vs EP 3.005
# (|Δ| = 0.032), while the rotated-expert control lands at 6.271 (|Δ| = 3.23) — so this bound still
# separates health from breakage by >30x.
LOSS_TOL = TOL.router_pick_flip_loss_abs


@gpu_test_main(exact_world_size=EP_SIZE, prefix="ep_vs_dense_qwen3_moe")
def run(ctx):
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    device = f"cuda:{ctx.local_rank}"

    log(f"EP vs dense reference: ep={EP_SIZE} world={ctx.world_size} model={MODEL_NAME}")
    ensure_model_downloaded(MODEL_NAME, ctx.rank)
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Identical batch on every rank — what makes the per-rank loss directly comparable to the
    # undistributed reference, and a rank-loss spread a real desync signal rather than a data one.
    input_ids, attention_mask, labels = fixed_chat_batch(tokenizer, MAX_SEQ_LENGTH, device, seed=SEED)
    dist.broadcast(input_ids, src=0)
    dist.broadcast(attention_mask, src=0)
    dist.broadcast(labels, src=0)

    # ── Reference: no EP, no sharding of any kind (rank 0 builds and frees it, then broadcasts) ──
    ref_loss_local, ref_grad = 0.0, None
    if ctx.rank == 0:
        ref_loss_local, ref_grad = dense_reference(
            MODEL_NAME, input_ids, attention_mask, labels, device, attn_implementation=ATTN_IMPLEMENTATION
        )
        log(f"  reference (single-GPU dense): loss={ref_loss_local:.6f} router|g|={ref_grad.norm():.6e}")
    ref_loss, ref_grad = broadcast_reference(ref_loss_local, ref_grad, device, ctx.rank)
    ref_grad_norm = ref_grad.norm().item()
    metrics["reference_loss"] = ref_loss
    metrics["reference_router_grad_norm"] = ref_grad_norm

    # ANTI-VACUITY: the compared quantities must be non-trivial. A zero reference loss or a zero
    # router gradient would let every comparison below pass on a model that computes nothing.
    checks["reference_loss_nontrivial"] = 0.5 < ref_loss < 20.0
    checks["reference_router_grad_nonzero"] = ref_grad_norm > 1e-8

    # ── EP model ─────────────────────────────────────────────────────────────────────────────
    parallelism_config = ParallelismConfig(ep_size=EP_SIZE, cp_size=1, tp_size=1)
    log(f"  {parallelism_config.summary()}")
    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=ATTN_IMPLEMENTATION,
        use_liger_kernel=False,
    )

    layers = ep_layers(model)
    checks["ep_layers_wrapped"] = len(layers) > 0
    checks["ep_layers_are_the_qwen3_family_class"] = bool(layers) and all(
        type(layer).__name__ == EXPECTED_LAYER_TYPE for layer in layers
    )
    metrics["ep_layers"] = float(len(layers))
    # The bank really is split: sharing the full expert set on every rank would reproduce the
    # reference perfectly and turn every numeric check below into a statement about nothing.
    first = layers[0]
    checks["expert_bank_split_ep_way"] = (
        first.experts_per_rank == first.num_experts // EP_SIZE
        and first.expert_end - first.expert_start == first.experts_per_rank
        and first.experts_per_rank < first.num_experts
    )
    metrics["experts_per_rank"] = float(first.experts_per_rank)
    log_all(f"  rank owns experts [{first.expert_start}, {first.expert_end}) of {first.num_experts}")

    model.train()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
    loss = outputs.loss
    delta = abs(loss.item() - ref_loss)
    log_all(f"  EP loss={loss.item():.6f}  |Δreference|={delta:.3e}")
    metrics["ep_loss"] = loss.item()
    metrics["loss_abs_err"] = delta
    checks["loss_matches_reference"] = delta < LOSS_TOL

    gathered = [torch.zeros_like(loss.detach()) for _ in range(ctx.world_size)]
    dist.all_gather(gathered, loss.detach())
    spread = max(abs(g.item() - gathered[0].item()) for g in gathered)
    metrics["rank_loss_spread"] = spread
    checks["losses_agree_across_ranks"] = spread < TOL.ep_identical_batch_rank_spread_abs

    # ── Router gradient vs the reference ─────────────────────────────────────────────────────
    loss.backward()
    router_name, router_weight = find_router_weight(model)
    ratio, cosine = compare_grad(full_grad(router_weight), ref_grad)
    metrics["router_grad_norm_ratio"] = ratio
    metrics["router_grad_cosine"] = cosine
    log_all(f"  router grad ({router_name}) vs reference: norm_ratio={ratio:.4f} cosine={cosine:.4f}")
    checks["router_grad_scale_matches_reference"] = 1 / TOL.grad_norm_ratio_max < ratio < TOL.grad_norm_ratio_max
    checks["router_grad_direction_matches_reference"] = cosine > TOL.grad_direction_cosine_min

    expert_grads = [p.grad for layer in layers for _, p in layer.expert_named_params()]
    checks["expert_grads_present_and_finite"] = bool(expert_grads) and all(
        g is not None and torch.isfinite(g).all() for g in expert_grads
    )

    # ── NEGATIVE CONTROL: rotate each rank's expert bank, re-run forward AND backward ────────
    rotated = roll_expert_banks(model)
    metrics["control_tensors_rotated"] = float(rotated)
    checks["control_engaged"] = rotated > 0
    model.zero_grad(set_to_none=True)
    control_out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
    control_out.loss.backward()
    control_loss = control_out.loss.item()
    control_shift = abs(control_loss - ref_loss)
    _, control_cosine = compare_grad(full_grad(router_weight), ref_grad)
    log_all(
        f"  control (expert bank rotated by 1): loss={control_loss:.6f} |Δreference|={control_shift:.3e} "
        f"router grad cosine={control_cosine:.4f}"
    )
    metrics["control_loss"] = control_loss
    metrics["control_loss_shift"] = control_shift
    metrics["control_router_grad_cosine"] = control_cosine
    checks["control_wrong_expert_breaks_loss_match"] = control_shift > TOL.control_min_loss_shift(LOSS_TOL)
    checks["control_wrong_expert_breaks_grad_direction"] = control_cosine < TOL.grad_direction_cosine_min

    metrics["peak_gb"] = gpu_peak_mem_gb()
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
