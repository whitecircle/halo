#!/usr/bin/env python
"""Rank balance on the online trainer, on real ranks: the deal over NCCL evens the ranks' attended tokens, and the
window trains as the undealt one does.

On the trainer's real scoring path and TRL's real buffer and loss (Qwen3-0.6B under FSDP2, FlashAttention-2), with rank 0
holding long rows and rank 1 short ones:

1. the dealt round has the same rows per rank and a smaller max-over-mean token spread;
2. the world's summed loss equals the undealt window's;
3. its gradients, through the FSDP2 reduce-scatter, stray from the undealt window's no further than twice as far as an
   undealt window in another shuffle does, and point the same way;
4. negative control: advantages left behind while the tokens move change the loss.

    torchrun --nproc_per_node=2 tests/gpu/trainers/grpo/test_online_rank_balance.py
"""

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

from halo.distributed.fsdp import apply_fsdp2_for_dp
from halo.trainers.grpo.rounds.rank_balance import deal_rows_across_ranks
from tests.common.harness import gpu_test_main
from tests.common.length_deal import ragged_round, with_world_normalizer, world_accelerator
from tests.common.models import QWEN3_0_6B
from tests.common.numerics import cos_sim, relative_l2
from tests.common.online_rounds import online_host, train_online_window
from tests.common.rank_log import log_all

MODEL, ATTN = QWEN3_0_6B, "flash_attention_2"
ROWS = 8
VOCAB = 32_000
SHUFFLE_SEED, FLOOR_SEED = 7, 8
GRAD_FLOOR_FACTOR = 2.0
GRAD_COSINE_MIN = 0.999
PAD_TOKEN_ID = 0


def _rank_round(ctx) -> dict:
    scale = 4 if ctx.rank == 0 else 1
    gen = torch.Generator().manual_seed(11 + ctx.rank)
    prompts = (torch.randint(64, 128, (ROWS,), generator=gen) * scale).tolist()
    completions = (torch.randint(64, 256, (ROWS,), generator=gen) * scale).tolist()
    batch = ragged_round(prompts, completions, vocab=VOCAB, seed=ctx.rank, device=ctx.device)
    batch.pop("row_id")
    return with_world_normalizer(batch)


def _window(model, batch: dict, seed: int = SHUFFLE_SEED) -> tuple[float, list[torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    host = online_host(model, slots=ROWS, budget=0, accelerator=world_accelerator(), packed=False, loss_type="grpo")
    loss, grads = train_online_window(host, batch, seed)
    total = torch.tensor(loss, device=batch["advantages"].device)
    dist.all_reduce(total)
    return total.item(), grads


@gpu_test_main(exact_world_size=2, prefix="online_rank_balance")
def run(ctx) -> dict:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation=ATTN).to(ctx.device)
    apply_fsdp2_for_dp(
        model, ctx.world_size, compute_dtype=torch.bfloat16, fp32_reduce=True, reshard_after_forward=True
    )
    model.train()
    batch = _rank_round(ctx)
    dealt, spread = deal_rows_across_ranks(batch, pad_token_id=PAD_TOKEN_ID)
    checks, metrics = {}, {}
    log_all(f"  spread {spread.before:.3f} -> {spread.after:.3f}")
    checks["rows_per_rank_kept"] = dealt["advantages"].numel() == ROWS
    checks["spread_shrinks"] = spread.before > 1.3 and spread.after < 1.1

    loss_plain, grads_plain = _window(model, batch)
    loss_floor, grads_floor = _window(model, batch, FLOOR_SEED)
    loss_dealt, grads_dealt = _window(model, dealt)
    loss_broken = _window(model, dict(dealt) | {"advantages": batch["advantages"]})[0]
    model.zero_grad(set_to_none=True)
    floor = relative_l2(dict(enumerate(grads_floor)), dict(enumerate(grads_plain)))
    moved = relative_l2(dict(enumerate(grads_dealt)), dict(enumerate(grads_plain)))
    flat = lambda grads: torch.cat([g.flatten() for g in grads])  # noqa: E731
    cosine = cos_sim(flat(grads_dealt), flat(grads_plain), label="dealt vs undealt gradient")
    metrics.update(loss_plain=loss_plain, loss_floor=loss_floor, loss_dealt=loss_dealt, grad_floor=floor)
    metrics.update(grad_dealt=moved, grad_cosine=cosine)
    log_all(f"  dealt {moved:.3e} vs floor {floor:.3e}, cosine {cosine:.5f}")
    checks["loss_matches_undealt"] = abs(loss_dealt - loss_plain) <= 5e-3 * max(1.0, abs(loss_plain))
    checks["grads_inside_the_floor"] = floor > 0.0 and moved <= GRAD_FLOOR_FACTOR * floor
    checks["grads_point_the_same_way"] = cosine >= GRAD_COSINE_MIN
    checks["negative_control_misplaced_advantages_change_the_loss"] = abs(loss_broken - loss_plain) > 10 * abs(
        loss_dealt - loss_plain
    )
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
