#!/usr/bin/env python
"""Packed micro-batches on the online trainer, on real ranks: TRL's whole-round old-policy pass and the loss forward run
the same bins, and the window trains as the per-row window does.

On the trainer's real scoring path, plan and TRL's real buffer and loss (Qwen3-0.6B under FSDP2, FlashAttention-2):

1. the old-policy pass planned and scored bin by bin hands TRL the per-row pass's log-probs in batch order, and every
   rank ran the world's bin count of scoring calls;
2. the packed window's loss equals the per-row window's;
3. its gradients, through the FSDP2 reduce-scatter and the leading no-op slots, stray from the per-row window's no
   further than twice as far as a per-row window in another shuffle does, and point the same way;
4. negative control: without the row-mean rescale the gradient is off by the rows-per-bin factor.

    torchrun --nproc_per_node=2 tests/gpu/trainers/grpo/test_online_packed_microbatches.py
"""

import math

import torch
from transformers import AutoModelForCausalLM

from halo.distributed.fsdp import apply_fsdp2_for_dp
from tests.common.harness import gpu_test_main, record_check
from tests.common.length_deal import (
    PACKED_DENSE_BUDGET,
    PACKED_DENSE_ROW_TOKENS,
    PACKED_SLOTS,
    ragged_round,
    with_world_normalizer,
    world_accelerator,
)
from tests.common.models import QWEN3_0_6B
from tests.common.numerics import cos_sim, relative_l2
from tests.common.online_rounds import old_policy_pass, online_host, train_online_window
from tests.common.rank_log import log_all

MODEL, ATTN = QWEN3_0_6B, "flash_attention_2"
SLOTS = PACKED_SLOTS
ROW_TOKENS, BUDGET = PACKED_DENSE_ROW_TOKENS, PACKED_DENSE_BUDGET
VOCAB = 32_000
SHUFFLE_SEED, FLOOR_SEED = 7, 8
GRAD_FLOOR_FACTOR = 2.0
GRAD_COSINE_MIN = 0.999
LOGP_MEAN_ABS_MAX = 1e-2


def _rank_round(ctx) -> dict:
    """Each rank's rows in the opposite order, so the ranks' bins differ in content and, with an uneven total, in size."""
    lengths = ROW_TOKENS if ctx.rank % 2 == 0 else ROW_TOKENS[::-1]
    halves = [n // 2 for n in lengths]
    completions = [n - h for n, h in zip(lengths, halves, strict=True)]
    return with_world_normalizer(ragged_round(halves, completions, vocab=VOCAB, seed=ctx.rank, device=ctx.device))


def _host(model, *, packed: bool, scaled: bool = True, loss_type: str = "grpo"):
    host = online_host(
        model, slots=SLOTS, budget=BUDGET, accelerator=world_accelerator(), packed=packed, loss_type=loss_type
    )
    if not scaled:
        host._scale_packed_loss = lambda loss, inputs: loss
    return host


def _flat(grads: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([grad.flatten() for grad in grads])


def _window(model, batch: dict, *, packed: bool, seed: int = SHUFFLE_SEED, scaled: bool = True):
    model.zero_grad(set_to_none=True)
    return train_online_window(_host(model, packed=packed, scaled=scaled), batch, seed)


@gpu_test_main(exact_world_size=2, prefix="online_packed_microbatches")
def run(ctx) -> dict:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation=ATTN).to(ctx.device)
    apply_fsdp2_for_dp(
        model, ctx.world_size, compute_dtype=torch.bfloat16, fp32_reduce=True, reshard_after_forward=True
    )
    model.train()
    batch = _rank_round(ctx)
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
    model.zero_grad(set_to_none=True)
    floor = relative_l2(dict(enumerate(grads_f)), dict(enumerate(grads_s)))
    packed = relative_l2(dict(enumerate(grads_p)), dict(enumerate(grads_s)))
    cosine = cos_sim(_flat(grads_p), _flat(grads_s), label="packed vs per-row gradient")
    off = relative_l2(dict(enumerate(unscaled)), dict(enumerate(grads_s)))
    metrics.update(loss_per_row=loss_s, loss_floor=loss_f, loss_packed=loss_p, grad_floor=floor, grad_packed=packed)
    metrics.update(grad_cosine=cosine, grad_unscaled=off)
    log_all(f"  packed {packed:.3e} vs floor {floor:.3e}, cosine {cosine:.5f}, unscaled {off:.3e}")
    checks["loss_matches_per_row"] = abs(loss_p - loss_s) <= 5e-3 * max(1.0, abs(loss_s))
    checks["grads_inside_the_floor"] = math.isfinite(packed) and floor > 0.0 and packed <= GRAD_FLOOR_FACTOR * floor
    checks["grads_point_the_same_way"] = cosine >= GRAD_COSINE_MIN
    checks["negative_control_unscaled_grads_are_off"] = off > 10 * max(floor, packed)
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
