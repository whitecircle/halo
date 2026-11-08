#!/usr/bin/env python
"""Packed micro-batches on the online trainer, on real ranks: TRL's whole-round old-policy pass and the loss forward run
the same bins, and the window trains as the per-row window does.

On the trainer's real scoring path, plan and TRL's real buffer and loss (Qwen3-0.6B under FSDP2, FlashAttention-2):

1. the old-policy pass planned and scored bin by bin hands TRL the per-row pass's log-probs in batch order, and every
   rank ran the world's bin count of scoring calls;
2. the packed window's loss equals the per-row window's;
3. its gradients, through the FSDP2 reduce-scatter and the leading no-op slots, stray from the per-row window in fp32
   (the same rows on sdpa) no further than twice as far as the per-row bf16 window does, and point the same way;
4. negative control: without the row-mean rescale the gradient is off by the rows-per-bin factor.

    torchrun --nproc_per_node=2 tests/gpu/trainers/grpo/test_online_packed_microbatches.py
"""

import torch
from transformers import AutoModelForCausalLM

from halo.distributed.fsdp import apply_fsdp2_for_dp
from tests.common.harness import gpu_test_main, record_check
from tests.common.length_deal import (
    PACKED_DENSE_BUDGET,
    PACKED_DENSE_ROW_TOKENS,
    PACKED_SLOTS,
    world_accelerator,
)
from tests.common.models import QWEN3_0_6B
from tests.common.online_rounds import old_policy_pass, online_host, train_online_window
from tests.common.packed_grad_floor import build_fp32_reference, judge_packed_grads
from tests.common.rank_log import log_all
from tests.common.rank_rounds import reversed_order_round

MODEL, ATTN = QWEN3_0_6B, "flash_attention_2"
SLOTS = PACKED_SLOTS
ROW_TOKENS, BUDGET = PACKED_DENSE_ROW_TOKENS, PACKED_DENSE_BUDGET
VOCAB = 32_000
SHUFFLE_SEED, FLOOR_SEED = 7, 8
LOGP_MEAN_ABS_MAX = 1e-2


def _host(model, *, packed: bool, scaled: bool = True, loss_type: str = "grpo"):
    host = online_host(
        model, slots=SLOTS, budget=BUDGET, accelerator=world_accelerator(), packed=packed, loss_type=loss_type
    )
    if not scaled:
        host._scale_packed_loss = lambda loss, inputs: loss
    return host


def _window(model, batch: dict, *, packed: bool, seed: int = SHUFFLE_SEED, scaled: bool = True):
    model.zero_grad(set_to_none=True)
    window = train_online_window(_host(model, packed=packed, scaled=scaled), batch, seed)
    model.zero_grad(set_to_none=True)
    return window


@gpu_test_main(exact_world_size=2, prefix="online_packed_microbatches")
def run(ctx) -> dict:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation=ATTN).to(ctx.device)
    reference_model = build_fp32_reference(model, world_size=ctx.world_size)
    apply_fsdp2_for_dp(
        model, ctx.world_size, compute_dtype=torch.bfloat16, fp32_reduce=True, reshard_after_forward=True
    )
    model.train()
    batch = reversed_order_round(ctx, ROW_TOKENS, VOCAB)
    checks, metrics = {}, {}

    def old_pass_matches_per_row():
        per_row = old_policy_pass(_host(model, packed=False), batch)
        packed_host = _host(model, packed=True)
        with packed_host._scoring_round_in_bins():
            packed = old_policy_pass(packed_host, batch)
        plan = packed_host._round_plan
        assert 1 < plan.num_bins < SLOTS, f"the budget should leave both bins and no-op slots, got {plan.num_bins}"
        real = batch["completion_mask"].bool()
        diff = (packed - per_row).abs()[real].mean().item()
        metrics["old_pass_mean_abs_diff"] = diff
        log_all(f"  old pass: mean |packed - per-row| {diff:.3e} over {plan.num_bins} bins")
        assert diff <= LOGP_MEAN_ABS_MAX, f"packed old-policy pass strays {diff:.3e} nats from the per-row one"

    record_check(checks, "old_pass_matches_per_row", old_pass_matches_per_row)

    loss_s, grads_s = _window(model, batch, packed=False)
    loss_f, grads_f = _window(model, batch, packed=False, seed=FLOOR_SEED)
    loss_p, grads_p = _window(model, batch, packed=True)
    unscaled = _window(model, batch, packed=True, scaled=False)[1]
    grads_ref = _window(reference_model, batch, packed=False)[1]
    verdict = judge_packed_grads(grads_s, grads_f, grads_p, unscaled, grads_ref)
    metrics.update(loss_per_row=loss_s, loss_order_floor=loss_f, loss_packed=loss_p, **verdict.metrics)
    log_all(
        f"  packed {verdict.packed_err:.3e} vs floor {verdict.floor:.3e} (order floor {verdict.order_floor:.3e}), "
        f"cosine {verdict.cosine:.5f}, unscaled {verdict.unscaled_err:.3e}"
    )
    checks["loss_matches_per_row"] = abs(loss_p - loss_s) <= 5e-3 * max(1.0, abs(loss_s))
    checks["grads_inside_the_floor"] = verdict.inside_the_floor
    checks["grads_point_the_same_way"] = verdict.points_the_same_way
    checks["negative_control_unscaled_grads_are_off"] = verdict.control_bites
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
