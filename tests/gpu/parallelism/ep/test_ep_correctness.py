#!/usr/bin/env python
"""EP=2 on gpt-oss-20b must reproduce the undistributed forward and its router gradient.

An EP MoE forward that dispatches every token to the WRONG expert still produces a finite loss, a
plausible loss value, and perfect agreement between ranks — every rank is wrong in the same way. So
finiteness, range and rank-consistency checks cannot see the defect class this file is named for.
What can is the same math on the same weights with nothing distributed:

* rank 0 runs a plain ``AutoModelForCausalLM`` forward+backward on the same checkpoint and batch,
  frees it, and broadcasts its loss and first router gradient;
* the EP loss must match it, and the EP router gradient must match it in scale AND direction (a
  mis-scaled cross-rank reduction moves only the scale; a routing corruption moves only the direction);
* repeating the forward on unchanged weights must reproduce the loss EXACTLY — DeepEP's dispatch
  reuses arena buffers between calls, and a stale or partially-overwritten buffer shows up here and
  nowhere else;
* an expert-identity NEGATIVE CONTROL rotates each rank's local expert bank by one expert: if the
  comparisons above tolerated any expert assignment, the rotated model would pass them too.

Two settings make this an equivalence statement about parallelism rather than about preprocessing,
and both are load-bearing on this model: ``eager`` attention on both sides (gpt-oss carries live
attention sinks and a sink-dropping backend moves the loss by nats), and ``reset_sinks=False`` on the
parallel load (the loader's fine-tuning reset would make the two models different models).

Division of labour: this is the per-PR (``core``) gate — loss, router gradient, determinism, control,
one parallel model plus a briefly-held reference. ``test_ep_vs_no_ep.py`` is the nightly (``full``)
counterpart that holds both models at once and additionally compares per-token logits and top-1
agreement.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/ep/test_ep_correctness.py

Requirements:
    - 2x GPUs with >=100GB memory each (rank 0 briefly holds the dense model and its gradients)
    - DeepEP installed
    - Model: unsloth/gpt-oss-20b-BF16 (auto-downloaded)
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
EP_SIZE = 2
SEQ_LEN = 128
SEED = 42
# gpt-oss sinks: the only backend that carries them is eager, and both sides must agree.
ATTN_IMPLEMENTATION = "eager"

# Same bf16 checkpoint, same batch, both undistributed-equivalent. EP reorders the bf16 expert sum,
# and from layer 1 on that reordered output is what the next router scores, which flips a few
# near-tied picks out of top-4-of-32 with no bug present — the mechanism this tolerance is derived
# for (on this very model). Measured here: reference 5.607 vs EP 5.589 (|Δ| = 0.018), rotated-expert
# control 5.224 (|Δ| = 0.383), i.e. the same separation the tolerance's derivation records.
LOSS_TOL = TOL.router_pick_flip_loss_abs


@gpu_test_main(exact_world_size=EP_SIZE, prefix="ep_correctness")
def run(ctx):
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    device = f"cuda:{ctx.local_rank}"

    log(f"EP correctness: ep={EP_SIZE} world={ctx.world_size} model={MODEL_NAME} attn={ATTN_IMPLEMENTATION}")
    ensure_model_downloaded(MODEL_NAME, ctx.rank)
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Every rank sees the SAME batch — that is what makes the per-rank loss directly comparable to
    # the undistributed reference, and a rank-loss spread a desync signal rather than a data one.
    input_ids, attention_mask, labels = fixed_chat_batch(tokenizer, SEQ_LEN, device, seed=SEED)
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

    # ANTI-VACUITY: a zero reference loss or a zero router gradient would let every comparison
    # below pass on a model that computes nothing.
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
        # Keep the checkpoint's pretrained sinks: the reference is a plain from_pretrained, so the
        # loader's fine-tuning reset would compare two different models.
        reset_sinks=False,
    )

    layers = ep_layers(model)
    # Gating, not a warning: without EP wrappers this file measures a plain dense model against a
    # plain dense reference and reports every check green.
    checks["ep_layers_wrapped"] = len(layers) > 0
    metrics["ep_layers"] = float(len(layers))
    first = layers[0]
    checks["expert_bank_split_ep_way"] = (
        first.experts_per_rank == first.num_experts // EP_SIZE
        and first.expert_end - first.expert_start == first.experts_per_rank
        and first.experts_per_rank < first.num_experts
    )
    metrics["experts_per_rank"] = float(first.experts_per_rank)
    log_all(f"  rank owns experts [{first.expert_start}, {first.expert_end}) of {first.num_experts}")

    # ── Repeated forward on unchanged weights ────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        first_loss = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False
        ).loss.item()
        second_loss = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False
        ).loss.item()
    repeat_diff = abs(first_loss - second_loss)
    metrics["repeat_forward_diff"] = repeat_diff
    log_all(f"  repeated forward: {first_loss:.6f} vs {second_loss:.6f} (diff={repeat_diff:.3e})")
    # Exact, not "small": any difference means state survived between calls (a reused DeepEP arena
    # buffer, a mutated routing cache), and a bound wide enough to absorb "bf16 noise" would absorb
    # that defect too. This does NOT contradict the run-to-run reassociation the arena test
    # describes: DeepEP's receive order varies across PROCESSES, while two back-to-back forwards on
    # unchanged weights in one process dispatch identically. Measured 0.0 on this configuration.
    checks["repeated_forward_is_deterministic"] = repeat_diff == 0.0

    # ── Loss and router gradient vs the reference ────────────────────────────────────────────
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
