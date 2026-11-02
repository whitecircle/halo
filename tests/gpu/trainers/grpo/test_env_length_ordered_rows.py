#!/usr/bin/env python
"""The environmental trainer's length-ordered deal on real ranks: FSDP2 ZeRO-3, rows forwarded one per call.

Each rank holds the same row lengths in the opposite batch order, so in batch order every rank's row runs in
lock-step with an unrelated peer row of another length, the cost the deal removes. On the trainer's real scoring
path and TRL's real buffer and loss:

1. the dealt no-grad recompute returns bitwise the batch order's per-row log-probs on every rank;
2. every rank forwards a row of the same length rank at each position;
3. TRL's loss over the dealt micro-batches equals the shuffled run's, and the loss forward's per-row log-probs are
   bitwise;
4. the dealt gradients, through the FSDP2 reduce-scatter, stray from the shuffled run's no further than twice as
   far as a third run in another shuffle's micro-batch order does: bf16 gradient accumulation rounds differently
   in every order, so the bound is that measured floor, not a fixed one;
5. with ``--timed``, the dealt no-grad pass is faster than the batch-order one (``_MIN_SPEEDUP``), a wall-clock
   bound kept out of the core row.

    torchrun --nproc_per_node=2 tests/gpu/trainers/grpo/test_env_length_ordered_rows.py [--timed]
"""

import argparse
import math

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

from halo.distributed.fsdp import apply_fsdp2_for_dp
from tests.common.harness import gpu_test_main, record_check
from tests.common.length_deal import (
    loss_host,
    ragged_round,
    scoring_host,
    stacked,
    timed_no_grad_pass,
    train_round,
    with_world_normalizer,
    world_accelerator,
)
from tests.common.models import QWEN3_0_6B
from tests.common.numerics import relative_l2, worst_relative_l2
from tests.common.rank_log import log, log_all

# The stage recipes' attention. No gradient checkpointing: what the deal changes is the row order alone.
MODEL, ATTN = QWEN3_0_6B, "sdpa"
# 16 rows dealt into 4 micro-batches; prompt and completion each half a row of 512 .. 15872 tokens.
ROWS, CHUNKS = 16, 4
ROW_TOKENS = [512 + 1024 * i for i in range(ROWS)]
# TRL's buffer shuffle seeds: the shuffled run's, and the floor run's other micro-batch order of the same round.
SHUFFLE_SEED, FLOOR_SEED = 7, 8
# The dealt gradients may differ from the shuffled run's this many times as far as the floor run's do.
FLOOR_GRAD_FACTOR = 2.0
# Another order's bf16 accumulation moves these gradients by about 2e-2; a floor past this ceiling is no longer
# rounding, and would license any dealt gradient under it.
FLOOR_GRAD_CEILING = 5e-2
# Batch order pairs each row with the peer's row of the mirrored length rank: lock-step ~1.5x the dealt cost
# where compute dominates; the bar leaves room for the per-row gathers both orders pay alike.
_MIN_SPEEDUP = 1.15
_TIMED_REPEATS = 3

_PARSER = argparse.ArgumentParser(description=__doc__)
_PARSER.add_argument("--timed", action="store_true", help="also bound the dealt no-grad pass's speedup")
ARGS = _PARSER.parse_args()


def _rank_round(ctx) -> dict:
    """This rank's round: the shared lengths, rank 1's in reverse, the world's loss-token count as the normalizer."""
    lengths = ROW_TOKENS if ctx.rank % 2 == 0 else ROW_TOKENS[::-1]
    halves = [n // 2 for n in lengths]
    completions = [n - h for n, h in zip(lengths, halves, strict=True)]
    return with_world_normalizer(ragged_round(halves, completions, vocab=32_000, seed=ctx.rank, device=ctx.device))


def _local_grads(model) -> dict[str, torch.Tensor]:
    return {
        name: (p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad).float().clone()
        for name, p in model.named_parameters()
    }


@gpu_test_main(exact_world_size=2, prefix="env_length_ordered_rows")
def run(ctx) -> dict:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation=ATTN).to(ctx.device)
    apply_fsdp2_for_dp(
        model, ctx.world_size, compute_dtype=torch.bfloat16, fp32_reduce=True, reshard_after_forward=True
    )
    model.train()
    accelerator = world_accelerator()
    batch = _rank_round(ctx)
    ids, mask, keep = stacked(batch)
    host = scoring_host(model, None, chunks=CHUNKS, accelerator=accelerator)
    order = host._no_grad_row_order(mask, "train")
    checks, metrics = {}, {}

    def recompute_is_bitwise():
        with torch.no_grad():
            plain, _ = host._recompute_logps_and_routing_masks(ids, mask, keep, None, "train")
            dealt, _ = host._recompute_logps_and_routing_masks(ids, mask, keep, None, "train", order)
        diff = (dealt - plain).abs().max().item()
        metrics["recompute_max_abs_diff"] = diff
        assert torch.equal(dealt, plain), f"dealt recompute differs from batch order by up to {diff}"

    def ranks_pair_length_ranks():
        lengths = mask.sum(1)
        ranks = torch.empty_like(lengths)
        ranks[torch.sort(lengths, descending=True, stable=True).indices] = torch.arange(ROWS, device=ctx.device)
        at_positions = ranks[order]
        world = [torch.empty_like(at_positions) for _ in range(ctx.world_size)]
        dist.all_gather(world, at_positions)
        assert all(torch.equal(w, world[0]) for w in world), "ranks forward rows of different length ranks together"

    def dealt_pass_is_faster():
        natural = timed_no_grad_pass(host, batch, None, _TIMED_REPEATS)
        dealt = timed_no_grad_pass(host, batch, order, _TIMED_REPEATS)
        metrics["no_grad_s_batch_order"], metrics["no_grad_s_dealt"] = natural, dealt
        metrics["no_grad_speedup"] = natural / dealt
        log(f"  no-grad pass: batch order {natural:.3f}s, dealt {dealt:.3f}s ({natural / dealt:.2f}x)")
        assert natural / dealt >= _MIN_SPEEDUP, f"dealt pass only {natural / dealt:.2f}x faster"

    record_check(checks, "recompute_is_bitwise", recompute_is_bitwise)
    record_check(checks, "ranks_pair_length_ranks", ranks_pair_length_ranks)

    def train(deal: bool, seed: int = SHUFFLE_SEED):
        """One round from zeroed gradients: (summed loss, row id -> loss-forward log-probs in forward order, this
        rank's gradients). The deal ignores ``seed``."""
        model.zero_grad(set_to_none=True)
        total, row_logps = train_round(
            loss_host(model, chunks=CHUNKS, accelerator=accelerator), batch, deal=deal, seed=seed
        )
        return total, row_logps, _local_grads(model)

    loss_s, logps_s, grads_s = train(False)
    loss_f, logps_f, grads_f = train(False, FLOOR_SEED)
    loss_d, logps_d, grads_d = train(True)
    model.zero_grad(set_to_none=True)

    def loss_matches_shuffled():
        metrics["loss_shuffled"], metrics["loss_floor"], metrics["loss_dealt"] = loss_s, loss_f, loss_d
        moved = [r for r in range(ROWS) if not torch.equal(logps_d[r], logps_s[r])]
        assert not moved, f"loss-forward log-probs of rows {moved} moved under the deal"
        assert abs(loss_d - loss_s) <= 1e-5 * max(1.0, abs(loss_s)), (loss_d, loss_s)

    def grads_inside_the_floor():
        # Without another order neither comparison measures anything.
        assert list(logps_f) != list(logps_s), "the floor run forwarded the rows in the shuffled run's order"
        assert list(logps_d) != list(logps_s), "the dealt run forwarded the rows in the shuffled run's order"
        floor, dealt = relative_l2(grads_f, grads_s), relative_l2(grads_d, grads_s)
        # Per tensor, recorded: an error confined to a small tensor barely moves the whole-model ratio.
        worst_floor, _ = worst_relative_l2(grads_f, grads_s)
        worst_dealt, worst_name = worst_relative_l2(grads_d, grads_s)
        metrics |= {"grad_rel_diff_floor": floor, "grad_rel_diff": dealt}
        metrics |= {"grad_worst_tensor_rel_diff_floor": worst_floor, "grad_worst_tensor_rel_diff": worst_dealt}
        log_all(
            f"  gradients vs the shuffled run: floor {floor:.3e}, dealt {dealt:.3e} "
            f"(worst tensor: floor {worst_floor:.3e}, dealt {worst_dealt:.3e} at {worst_name})"
        )
        assert math.isfinite(floor) and math.isfinite(dealt), (floor, dealt)
        assert 0.0 < floor <= FLOOR_GRAD_CEILING, f"floor {floor:.3e} outside (0, {FLOOR_GRAD_CEILING}]"
        assert dealt <= FLOOR_GRAD_FACTOR * floor, f"dealt {dealt:.3e} past {FLOOR_GRAD_FACTOR}x the floor {floor:.3e}"

    record_check(checks, "loss_matches_shuffled", loss_matches_shuffled)
    record_check(checks, "grads_inside_the_floor", grads_inside_the_floor)
    if ARGS.timed:
        record_check(checks, "dealt_pass_is_faster", dealt_pass_is_faster)
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
