#!/usr/bin/env python
"""EP+TP numerical correctness on gpt-oss-20b vs an undistributed single-GPU reference.

EP+TP splits a MoE model along two axes at once: attention is DTensor-sharded and all-reduced
(TP), the expert bank is split across ranks and reached over DeepEP's dispatch/combine (EP).
Every failure mode of that combination — a token dispatched to the wrong expert, an expert bank
sliced at the wrong offset, a TP output left un-reduced, a router gradient reduced by the wrong
divisor — produces a perfectly finite loss and perfectly finite gradients. So this test pins the
math, not the finiteness:

* rank 0 runs a plain ``AutoModelForCausalLM`` forward+backward on the *same* checkpoint and the
  same batch with no distribution at all, and broadcasts its loss and first router gradient;
* every rank's EP+TP loss must match that reference (EP/TP feed the FULL sequence to every rank —
  only CP shards it — so the per-rank loss equals the undistributed one, no aggregation needed);
* every rank's router gradient must match the reference after the EP router hook's cross-rank
  average, which is where a mis-scaled reduction shows up and nowhere else;
* an expert-identity NEGATIVE CONTROL rotates each rank's local expert bank by one expert and
  re-runs the forward. If the loss match above were insensitive to which expert receives which
  token — the exact defect a finiteness-only check cannot see — the rotated loss would still match.
  It has to move well outside the tolerance instead.

Two settings make the comparison an equivalence statement about *parallelism* rather than about
preprocessing, and both are load-bearing: ``eager`` attention on both sides (gpt-oss carries live
attention sinks, and a sink-dropping backend moves the loss by nats), and ``reset_sinks=False`` on
the parallel load (the loader's default resets the sinks to dtype-min for fine-tuning, which on
its own shifts this batch's loss by 2.65 — measured — and would swamp everything below). FA2/FA4
under EP+TP is covered by ``test_ep_tp_replicated_grad_sync.py``.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/combined/test_ep_tp_correctness.py
"""

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.tensor_parallel.state_dict import get_tp_mesh
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
TP_SIZE = 2
SEQ_LEN = 128
SEED = 42

# Both models are the same 24-layer bf16 checkpoint on the same batch, so the reference loss is
# O(1) and this bound is far above bf16 + grouped-mm + TP-all-reduce reordering noise — including
# the near-tied top-k router picks TP's reduction order flips (see the tolerance's derivation) —
# while far below any structural bug (a wrong expert, a dropped TP reduce, a wrong expert-slice
# offset all move a language-modelling loss by O(1) or more).
LOSS_TOL = TOL.router_pick_flip_loss_abs


@gpu_test_main(exact_world_size=2, prefix="ep_tp_correctness")
def run(ctx):
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    device = f"cuda:{ctx.local_rank}"

    log(f"EP+TP correctness: ep={EP_SIZE} tp={TP_SIZE} world={ctx.world_size} model={MODEL_NAME}")
    ensure_model_downloaded(MODEL_NAME, ctx.rank)
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Identical batch on every rank — that is what makes the per-rank loss directly comparable to
    # the undistributed reference (and what makes a rank-loss spread a real desync signal).
    input_ids, attention_mask, labels = fixed_chat_batch(tokenizer, SEQ_LEN, device, seed=SEED)
    dist.broadcast(input_ids, src=0)
    dist.broadcast(attention_mask, src=0)
    dist.broadcast(labels, src=0)

    # ── Reference: no EP, no TP, no sharding of any kind (rank 0, then broadcast) ─────────────
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

    # ANTI-VACUITY: the compared quantities must be non-trivial. A reference loss of 0 (or a
    # zero router gradient) would let every comparison below pass on a model that computes nothing.
    checks["reference_loss_nontrivial"] = 0.5 < ref_loss < 20.0
    checks["reference_router_grad_nonzero"] = ref_grad_norm > 1e-8

    # ── EP+TP model ──────────────────────────────────────────────────────────────────────────
    parallelism_config = ParallelismConfig(ep_size=EP_SIZE, tp_size=TP_SIZE, cp_size=1, expert_tp_size=1)
    log(f"  {parallelism_config.summary()}")
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
    metrics["ep_layers"] = float(len(layers))
    # Attention must actually be TP-sharded — otherwise this is an EP test wearing an EP+TP label
    # and every check below would still pass. A sharded projection is a DTensor whose local shape
    # is a slice of its global one; a silently un-sharded TP leaves both equal.
    tp_mesh = get_tp_mesh(model)
    sharded_attn = [
        name
        for name, param in model.named_parameters()
        if ".self_attn." in name and hasattr(param, "to_local") and param.to_local().shape != param.shape
    ]
    checks["tp_mesh_sized_tp_way"] = tp_mesh is not None and tp_mesh.size() == TP_SIZE
    checks["attention_sharded_tp_way"] = len(sharded_attn) > 0
    metrics["tp_sharded_attn_params"] = float(len(sharded_attn))

    model.train()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
    loss = outputs.loss
    delta = abs(loss.item() - ref_loss)
    log_all(f"  EP+TP loss={loss.item():.6f}  |Δreference|={delta:.3e}")
    metrics["ep_tp_loss"] = loss.item()
    metrics["loss_abs_err"] = delta
    # Decided below, once the corrupted control's shift is known: the absolute bound alone sits on
    # the flip-noise boundary (measured 0.09-0.12 on one host across loaded vs quiet runs — NCCL
    # reduction order shifts with topology state, and near-tied top-k picks flip with it).

    # Every rank saw the same batch, so a spread means the gather/reduce desynced them.
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

    # Two-sided: inside the flip band, or marginally over it while WELL separated from the
    # corrupted control — a real expert-identity/reduction bug lives in the control's regime
    # (0.382 here, O(1) per the docstring), while reduction-order flip noise measured 0.09-0.12
    # across host states on the same commit. The router-grad cosine check above independently
    # pins the routing direction, so this cannot mask a structural bug.
    checks["loss_matches_reference"] = delta < LOSS_TOL or (delta < 2 * LOSS_TOL and delta < 0.5 * control_shift)

    metrics["peak_gb"] = gpu_peak_mem_gb()
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
