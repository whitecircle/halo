#!/usr/bin/env python
"""ETP numerical correctness on gpt-oss-20b vs an undistributed single-GPU reference.

Expert Tensor Parallelism splits each expert's FFN *within* the expert rather than splitting the
expert bank: at ``ep_size=1, expert_tp_size=2`` both ranks hold all 32 experts, each expert's
interleaved ``gate_up_proj`` de-interleaved and column-sharded and its ``down_proj`` row-sharded,
with the partial sums reduced in TOKEN space outside the DeepEP dispatch→combine span. A shard cut
on the wrong axis, a de-interleave that pairs gate with the wrong up column, or a missing partial-sum
reduce all yield a finite loss and finite gradients, so this test pins the math instead:

* rank 0 runs a plain ``AutoModelForCausalLM`` forward+backward on the *same* checkpoint and batch
  with no sharding at all, and broadcasts its loss and first router gradient;
* every rank's ETP loss must match that reference (ETP feeds the FULL sequence to every rank, so
  the per-rank loss equals the undistributed one);
* every rank's router gradient must match the reference after the cross-rank average, which is
  where a mis-scaled reduction shows up and nowhere else — the forward stays exact;
* an expert-identity NEGATIVE CONTROL rotates each rank's expert bank by one expert and re-runs the
  forward: if the loss match were insensitive to which expert a token reaches, the rotated loss
  would still match. It has to move well outside the tolerance instead.

Two settings make this an equivalence statement about *sharding* rather than about preprocessing:
``eager`` attention on both sides (gpt-oss carries live attention sinks and a sink-dropping backend
moves the loss by nats) and ``reset_sinks=False`` on the sharded load (the loader's default resets
them to dtype-min for fine-tuning, which on its own shifts this batch's loss by 2.65 — measured).

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/combined/test_ep_etp_correctness.py
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
from tests.common.models import GPT_OSS_20B
from tests.common.tolerances import TOL
from tests.common.utils import gpu_peak_mem_gb, log, log_all

MODEL_NAME = GPT_OSS_20B
EP_SIZE = 1
EXPERT_TP_SIZE = 2
SEQ_LEN = 128
SEED = 42

# The token-space partial-sum reduce reorders what the router sees, and on gpt-oss (top-4-of-32)
# that flips a few near-tied expert picks — the same mechanism EP+TP hits, at the same size
# (0.064 here, 0.060 there), so it takes the same bound. This is a property of THIS checkpoint's
# routing, not of ETP: the Mistral4 ETP test matches its reference to 1e-2. What still holds ETP
# to the sharding math is the gradient pair below, which a reduce bug moves and a pick flip does
# not, plus the rotated-expert control at 0.457.
LOSS_TOL = TOL.router_pick_flip_loss_abs


@gpu_test_main(exact_world_size=2, prefix="ep_etp_correctness")
def run(ctx):
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    device = f"cuda:{ctx.local_rank}"

    log(f"ETP correctness: ep={EP_SIZE} expert_tp={EXPERT_TP_SIZE} world={ctx.world_size} model={MODEL_NAME}")
    ensure_model_downloaded(MODEL_NAME, ctx.rank)
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids, attention_mask, labels = fixed_chat_batch(tokenizer, SEQ_LEN, device, seed=SEED)
    dist.broadcast(input_ids, src=0)
    dist.broadcast(attention_mask, src=0)
    dist.broadcast(labels, src=0)

    # ── Reference: the same checkpoint with no sharding of any kind (rank 0, then broadcast) ───
    ref_loss_local, ref_grad = 0.0, None
    if ctx.rank == 0:
        ref_loss_local, ref_grad = dense_reference(
            MODEL_NAME, input_ids, attention_mask, labels, device, attn_implementation="eager"
        )
        log(f"  reference (single-GPU dense): loss={ref_loss_local:.6f} router|g|={ref_grad.norm():.6e}")
    ref_loss, ref_grad = broadcast_reference(ref_loss_local, ref_grad, device, ctx.rank)
    ref_grad_norm = ref_grad.norm().item()
    metrics["reference_loss"] = ref_loss
    metrics["reference_router_grad_norm"] = ref_grad_norm

    # ANTI-VACUITY: a zero reference loss or a zero reference gradient would let every comparison
    # below pass against a model that computes nothing.
    checks["reference_loss_nontrivial"] = 0.5 < ref_loss < 20.0
    checks["reference_router_grad_nonzero"] = ref_grad_norm > 1e-8

    # ── ETP model ─────────────────────────────────────────────────────────────────────────────
    parallelism_config = ParallelismConfig(ep_size=EP_SIZE, tp_size=1, cp_size=1, expert_tp_size=EXPERT_TP_SIZE)
    log(f"  {parallelism_config.summary()}  ep_group_size={parallelism_config.ep_group_size}")
    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
        # Keep the checkpoint's pretrained sinks: the reference is a plain from_pretrained, and the
        # loader's fine-tuning reset would make the two models different models.
        reset_sinks=False,
    )
    layers = ep_layers(model)
    checks["ep_layers_wrapped"] = len(layers) > 0
    # The expert FFN must actually be sharded — otherwise this is an EP test wearing an ETP label.
    etp_layers = [m for m in layers if getattr(m, "expert_tp_size", 1) == EXPERT_TP_SIZE]
    checks["expert_ffn_sharded_etp_way"] = len(etp_layers) == len(layers) and bool(layers)
    metrics["ep_layers"] = float(len(layers))
    metrics["etp_layers"] = float(len(etp_layers))

    model.train()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
    loss = outputs.loss
    delta = abs(loss.item() - ref_loss)
    log_all(f"  ETP loss={loss.item():.6f}  |Δreference|={delta:.3e}")
    metrics["etp_loss"] = loss.item()
    metrics["loss_abs_err"] = delta
    checks["loss_matches_reference"] = delta < LOSS_TOL

    gathered = [torch.zeros_like(loss.detach()) for _ in range(ctx.world_size)]
    dist.all_gather(gathered, loss.detach())
    spread = max(abs(g.item() - gathered[0].item()) for g in gathered)
    metrics["rank_loss_spread"] = spread
    checks["losses_agree_across_ranks"] = spread < TOL.rank_loss_consistency_abs

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
    # Proves both comparisons above are statements about expert identity rather than two numbers
    # that would look close under any expert assignment.
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
