#!/usr/bin/env python
"""The environmental trainer's length-ordered deal on real ranks, rows forwarded one per call.

Each rank holds the same row lengths in the opposite batch order, so in batch order every rank's row runs in
lock-step with an unrelated peer row of another length, the cost the deal removes. On the trainer's real scoring
path and TRL's real buffer and loss, the dense arm (Qwen3-0.6B, FSDP2 ZeRO-3) checks:

1. the dealt no-grad recompute returns bitwise the batch order's per-row log-probs on every rank;
2. every rank forwards a row of the same length rank at each position;
3. TRL's loss over the dealt micro-batches equals the shuffled run's, and the loss forward's per-row log-probs are
   bitwise;
4. the dealt gradients, through the FSDP2 reduce-scatter, stray from the shuffled run's no further than twice as
   far as a third run in another shuffle's micro-batch order does: bf16 gradient accumulation rounds differently
   in every order, so the bound is that measured floor, not a fixed one;
5. with ``--timed``, the dealt no-grad pass is faster than the batch-order one by ``_MIN_SPEEDUP``, while two
   batch-order passes time closer than that, a wall-clock bound kept out of the core row.

``--moe`` runs a tiny Qwen3-MoE at ``ep_size = 2`` (DeepEP dispatch, FSDP2 over the dense parameters), where a
dealt row's routing has to travel with the row, under the trainer's routing-replay injector:

6. R2: the dealt recompute captures each row's routing into that row's batch slot: the real positions exactly, the
   expert sets up to the near-tie flips a peer row's dispatch composition can round apart (``R2_FLIP_MAX``, or
   twice the flips of a floor run in a random row order, whichever is larger);
7. R3: the dealt recompute replays each forwarded row its own engine routing, and so does the gradient forward of
   every dealt micro-batch TRL's buffer hands the loss, each armed mask consumed whole;
8. every rank forwards a row of the same length rank at each position, as in 2.

The engine routing is random per (token, layer), so a row replaying another row's routing differs almost
everywhere; the replayed selections are read off the forwards in forward order, not through the injector's own
row-order remap.

    torchrun --nproc_per_node=2 tests/gpu/trainers/grpo/test_env_length_ordered_rows.py [--timed]
    torchrun --nproc_per_node=2 tests/gpu/trainers/grpo/test_env_length_ordered_rows.py --moe
"""

import argparse
import math

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.fsdp import apply_fsdp2_for_dp
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.trainers.grpo.environmental.routing_replay import RoutingReplayInjector, build_routing_replay_injector
from halo.trainers.grpo.environmental.step_batch import ROUTING_ROWS_KEY
from halo.trainers.grpo.logprob_scoring import dense_row_spans
from halo.trainers.grpo.row_order import length_deal_order
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
from tests.common.routing import dense_rows, distinct_expert_picks, ragged_routing
from tests.common.tiny_models import TINY_MOE_FAMILIES, shared_tiny_family_checkpoint, word_level_tokenizer

# The stage recipes' attention. No gradient checkpointing: what the deal changes is the row order alone.
MODEL, ATTN = QWEN3_0_6B, "sdpa"
# 16 rows dealt into 4 micro-batches; prompt and completion each half a row of 512 .. 15872 tokens.
ROWS, CHUNKS = 16, 4
ROW_TOKENS = [512 + 1024 * i for i in range(ROWS)]
DENSE_VOCAB = 32_000
# TRL's buffer shuffle seeds: the shuffled run's, and the floor run's other micro-batch order of the same round.
SHUFFLE_SEED, FLOOR_SEED = 7, 8
FLOOR_GRAD_FACTOR = 2.0
# Another order's bf16 accumulation moves these gradients by about 2e-2; a floor past this is no longer rounding and
# would license any dealt gradient under it.
FLOOR_GRAD_CEILING = 5e-2
# Batch order pairs each row with the peer's row of the mirrored length rank. Measured on 2x B300: 0.983 s against the
# dealt 0.861 s (1.14x), short of the token counts' 1.5x because both orders pay per-row cost. The bar keeps half that
# gain.
_MIN_SPEEDUP = 1.07
_TIMED_REPEATS = 3
# Tiny Qwen3-MoE (hidden 256, the DeepEP wire width) on rows of 64 .. 424 tokens, inside its 512 positions.
MOE_FAMILY = "qwen3_moe"
MOE_ROW_TOKENS = [64 + 24 * i for i in range(ROWS)]
MOE_WORDS = 500
MOE_SEED = 0
# Real (token, layer) selections a dealt R2 capture may route apart from batch order, or twice the floor's flips where
# that is larger. A row whose routing lands in another row's slot moves most of its selections.
R2_FLIP_MAX = 1e-3
R2_FLOOR_FACTOR = 2.0
R2_FLOOR_CEILING = 1e-2

_PARSER = argparse.ArgumentParser(description=__doc__)
_PARSER.add_argument("--timed", action="store_true", help="also bound the dealt no-grad pass's speedup")
_PARSER.add_argument("--moe", action="store_true", help="the EP-2 MoE arm under routing replay")
ARGS = _PARSER.parse_args()
if ARGS.timed and ARGS.moe:
    _PARSER.error("--timed bounds the dense arm's speedup; the --moe arm checks routing, not wall time")


def _rank_round(ctx, row_tokens: list[int], vocab: int) -> dict:
    lengths = row_tokens if ctx.rank % 2 == 0 else row_tokens[::-1]
    halves = [n // 2 for n in lengths]
    completions = [n - h for n, h in zip(lengths, halves, strict=True)]
    return with_world_normalizer(ragged_round(halves, completions, vocab=vocab, seed=ctx.rank, device=ctx.device))


def _local_grads(model) -> dict[str, torch.Tensor]:
    return {
        name: (p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad).float().clone()
        for name, p in model.named_parameters()
    }


def _ranks_pair_length_ranks(ctx, mask: torch.Tensor, order: torch.Tensor) -> None:
    """COLLECTIVE. Every rank forwards, at each position of ``order``, its row of the same length rank."""
    lengths = mask.sum(1)
    ranks = torch.empty_like(lengths)
    ranks[torch.sort(lengths, descending=True, stable=True).indices] = torch.arange(ROWS, device=ctx.device)
    at_positions = ranks[order]
    world = [torch.empty_like(at_positions) for _ in range(ctx.world_size)]
    dist.all_gather(world, at_positions)
    assert all(torch.equal(w, world[0]) for w in world), "ranks forward rows of different length ranks together"


def _selection_flips(captured: torch.Tensor, expected: torch.Tensor) -> tuple[bool, float]:
    real = captured[..., 0] >= 0
    differ = (captured.sort(-1).values != expected.sort(-1).values).any(-1) & real
    return torch.equal(real, expected[..., 0] >= 0), differ.sum().item() / max(real.sum().item(), 1)


def _engine_masks(mask: torch.Tensor, injector: RoutingReplayInjector, seed: int) -> torch.Tensor:
    """An engine's routing as ``build_rollout_routing`` places a fully covered one: distinct random experts, ``-1`` on
    padding."""
    rows, seq = mask.shape
    generator = torch.Generator().manual_seed(seed)
    picks = distinct_expert_picks((rows, seq), injector.num_ep_layers, injector.top_k, injector.num_experts, generator)
    return picks.masked_fill(~mask.bool().cpu()[:, :, None, None], -1).to(mask.device)


def run_dense(ctx) -> dict:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation=ATTN).to(ctx.device)
    apply_fsdp2_for_dp(
        model, ctx.world_size, compute_dtype=torch.bfloat16, fp32_reduce=True, reshard_after_forward=True
    )
    model.train()
    accelerator = world_accelerator()
    batch = _rank_round(ctx, ROW_TOKENS, DENSE_VOCAB)
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

    def dealt_pass_is_faster():
        first = timed_no_grad_pass(host, batch, None, _TIMED_REPEATS)
        dealt = timed_no_grad_pass(host, batch, order, _TIMED_REPEATS)
        # The control: the un-dealt order timed again, whose gap to the first pass is timing noise alone.
        again = timed_no_grad_pass(host, batch, None, _TIMED_REPEATS)
        natural, undealt = min(first, again), max(first, again) / min(first, again)
        metrics["no_grad_s_batch_order"], metrics["no_grad_s_dealt"] = natural, dealt
        metrics["no_grad_speedup"], metrics["no_grad_undealt_ratio"] = natural / dealt, undealt
        log(
            f"  no-grad pass: batch order {first:.3f}s / {again:.3f}s, dealt {dealt:.3f}s "
            f"({natural / dealt:.2f}x; batch order against itself {undealt:.3f}x)"
        )
        assert undealt < _MIN_SPEEDUP, f"two batch-order passes differ by {undealt:.3f}x: the bar is inside the noise"
        assert natural / dealt >= _MIN_SPEEDUP, f"dealt pass only {natural / dealt:.2f}x faster"

    record_check(checks, "recompute_is_bitwise", recompute_is_bitwise)
    record_check(checks, "ranks_pair_length_ranks", lambda: _ranks_pair_length_ranks(ctx, mask, order))

    def train(deal: bool, seed: int = SHUFFLE_SEED):
        """One round from zeroed gradients. The deal ignores ``seed``."""
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
        # Per tensor: an error confined to a small tensor barely moves the whole-model ratio.
        worst_floor, _ = worst_relative_l2(grads_f, grads_s)
        worst_dealt, worst_name = worst_relative_l2(grads_d, grads_s)
        metrics.update(grad_rel_diff_floor=floor, grad_rel_diff=dealt)
        metrics.update(grad_worst_tensor_rel_diff_floor=worst_floor, grad_worst_tensor_rel_diff=worst_dealt)
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


def _load_moe_policy(ctx):
    """COLLECTIVE. The tiny MoE through the production loader at ``ep_size = world``, wrapped as the EP mode wraps
    it."""
    tokenizer = word_level_tokenizer([f"w{i}" for i in range(MOE_WORDS)])
    family = TINY_MOE_FAMILIES[MOE_FAMILY]
    checkpoint = shared_tiny_family_checkpoint(ctx, family, "length_deal_qwen3_moe", tokenizer, MOE_SEED)
    model, _ = load_distributed_model(
        checkpoint,
        ParallelismConfig(ep_size=ctx.world_size),
        torch.bfloat16,
        trust_remote_code=False,
        attn_implementation=ATTN,
    )
    ep_params = [param for _, layer in named_ep_layers(model) for param in layer.parameters()]
    apply_fsdp2_for_dp(model, ctx.world_size, compute_dtype=torch.bfloat16, fp32_reduce=True, ignored_params=ep_params)
    return model.train(), len(tokenizer)


def run_moe(ctx) -> dict:
    model, vocab = _load_moe_policy(ctx)
    injector = build_routing_replay_injector(model)
    batch = _rank_round(ctx, MOE_ROW_TOKENS, vocab)
    ids, mask, keep = stacked(batch)
    engine = _engine_masks(mask, injector, MOE_SEED + ctx.rank)
    routing = ragged_routing(engine)
    host = loss_host(model, chunks=CHUNKS, accelerator=world_accelerator())
    host._routing_injector = injector
    host.use_liger_kernel = False
    order = host._no_grad_row_order(mask, "train")
    floor_order = torch.randperm(ROWS, generator=torch.Generator().manual_seed(FLOOR_SEED + ctx.rank))
    floor_order = floor_order.to(ctx.device)
    checks, metrics = {}, {}
    log(f"  {MOE_FAMILY}: {injector.num_ep_layers} EP layers, top-{injector.top_k} of {injector.num_experts}")

    def r2_capture_follows_each_row():
        host._rollout_routing_replay = False
        with torch.no_grad():
            plain_logps, plain = host._recompute_logps_and_routing_masks(ids, mask, keep, None, "train")
            dealt_logps, dealt = host._recompute_logps_and_routing_masks(ids, mask, keep, None, "train", order)
            _, floor = host._recompute_logps_and_routing_masks(ids, mask, keep, None, "train", floor_order)
        plain, dealt, floor = (dense_rows(r, ctx.device) for r in (plain, dealt, floor))
        dealt_layout, dealt_flips = _selection_flips(dealt, plain)
        floor_layout, floor_flips = _selection_flips(floor, plain)
        bar = max(R2_FLIP_MAX, R2_FLOOR_FACTOR * floor_flips)
        metrics.update(r2_flips_dealt=dealt_flips, r2_flips_floor=floor_flips)
        metrics["recompute_max_abs_diff"] = (dealt_logps - plain_logps).abs().max().item()
        log_all(f"  R2 capture vs batch order: dealt {dealt_flips:.2e}, floor {floor_flips:.2e}, bar {bar:.2e}")
        # Without another order neither comparison measures anything.
        assert not torch.equal(order, torch.arange(ROWS, device=ctx.device)), "the deal kept the batch order"
        assert not torch.equal(floor_order, order), "the floor run forwarded the rows in the dealt order"
        assert dealt_layout and floor_layout, "a captured row's real positions landed in another row's slot"
        assert floor_flips <= R2_FLOOR_CEILING, f"floor {floor_flips:.2e} past {R2_FLOOR_CEILING}"
        assert dealt_flips <= bar, f"dealt capture flips {dealt_flips:.2e} of the selections, past {bar:.2e}"

    def r3_recompute_replays_each_rows_engine_routing():
        host._rollout_routing_replay = True
        spans, _ = dense_row_spans(mask[order])
        with torch.no_grad(), injector.capture(ROWS, mask.size(1), row_spans=spans) as replayed:
            logps, _ = host._recompute_logps_and_routing_masks(ids, mask, keep, routing, "train", order)
        captured = dense_rows(replayed[0], ctx.device)
        wrong = [p for p in range(ROWS) if not torch.equal(captured[p], engine[order[p]])]
        assert bool(torch.isfinite(logps).all()), "the replayed recompute returned non-finite log-probs"
        assert not wrong, f"dealt positions {wrong} replayed routing other than their row's engine routing"

    def r3_gradient_forward_replays_each_rows_engine_routing():
        host._rollout_routing_replay = True
        host._orders_rows_by_length = True
        host._step, host._buffered_inputs = 0, None
        host._rollout_routing = routing
        host._generate_and_score_completions = lambda _inputs: batch | {
            ROUTING_ROWS_KEY: torch.arange(ROWS, device=ctx.device)
        }
        forwarded, wrong, losses = [], [], []
        model.zero_grad(set_to_none=True)
        for _ in range(CHUNKS):
            inputs = host._prepare_inputs({})
            host._step += 1
            micro_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], 1)
            spans, _ = dense_row_spans(micro_mask)
            with injector.capture(micro_mask.size(0), micro_mask.size(1), row_spans=spans) as replayed:
                loss = host.compute_loss(model, inputs)
            loss.backward()
            # As the trainer's training_step does after the backward: raises on a mask not consumed whole.
            injector.disarm()
            losses.append(loss.item())
            forwarded.append(inputs["row_id"].cpu())
            captured = dense_rows(replayed[0], ctx.device)
            wrong += [int(r) for i, r in enumerate(forwarded[-1]) if not torch.equal(captured[i], engine[r])]
        model.zero_grad(set_to_none=True)
        metrics["replayed_loss"] = sum(losses)
        assert all(map(math.isfinite, losses)), f"replayed micro-batch losses {losses}"
        dealt = length_deal_order(mask.sum(1), CHUNKS)
        assert torch.equal(torch.cat(forwarded), dealt), "TRL's buffer did not hand the loss the dealt rows"
        assert not wrong, f"rows {wrong} replayed routing other than their own engine routing in the loss forward"

    record_check(checks, "r2_capture_follows_each_row", r2_capture_follows_each_row)
    record_check(
        checks, "r3_recompute_replays_each_rows_engine_routing", r3_recompute_replays_each_rows_engine_routing
    )
    record_check(
        checks,
        "r3_gradient_forward_replays_each_rows_engine_routing",
        r3_gradient_forward_replays_each_rows_engine_routing,
    )
    record_check(checks, "ranks_pair_length_ranks", lambda: _ranks_pair_length_ranks(ctx, mask, order))
    return {"checks": checks, "metrics": metrics}


@gpu_test_main(exact_world_size=2, prefix="env_length_ordered_rows_moe" if ARGS.moe else "env_length_ordered_rows")
def run(ctx) -> dict:
    return run_moe(ctx) if ARGS.moe else run_dense(ctx)


if __name__ == "__main__":
    run()
