#!/usr/bin/env python
"""Packed GRPO micro-batches on real ranks: a window of bins trains as the per-row window does.

On the trainer's real scoring path, plan and TRL's real buffer and loss (Qwen3-0.6B under FSDP2, FlashAttention-2):

1. the packed no-grad recompute scores every row as the per-row recompute does, and every rank ran the world's bin
   count of scoring calls;
2. the packed window's loss equals the per-row window's;
3. its gradients, through the FSDP2 reduce-scatter and the leading no-op slots, stray from the per-row window's no
   further than twice as far as a per-row window in another shuffle does (bf16 accumulation rounds differently in
   every order, so the bound is that measured floor), and point the same way;
4. negative control: without the row-mean rescale the gradient is off by the rows-per-bin factor.

``--moe`` runs a tiny Qwen3-MoE at ``ep_size = 2`` (DeepEP dispatch, one dispatch per bin), where the bins must run
on every rank in equal number, under R2 capture and R3 replay:

5. the packed recompute captures each row's routing into its own batch slot, up to the near-tie flips a different
   dispatch composition rounds apart;
6. the packed recompute replays each row its own engine routing, and the window's gradient forwards consume each armed
   mask whole.

    torchrun --nproc_per_node=2 tests/gpu/trainers/grpo/test_env_packed_microbatches.py [--moe]
"""

import argparse
import math

import torch
from transformers import AutoModelForCausalLM

from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.fsdp import apply_fsdp2_for_dp
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.trainers.grpo.environmental.routing_replay import build_routing_replay_injector
from halo.trainers.grpo.environmental.step_batch import ROUTING_MASKS_KEY
from halo.trainers.grpo.microbatch_plan import agree_microbatch_geometry
from tests.common.harness import gpu_test_main, record_check
from tests.common.length_deal import (
    PACKED_DENSE_BUDGET,
    PACKED_DENSE_ROW_TOKENS,
    PACKED_MOE_BUDGET,
    PACKED_MOE_ROW_TOKENS,
    PACKED_ROWS,
    PACKED_SLOTS,
    packed_host,
    ragged_round,
    stacked,
    train_window,
    with_world_normalizer,
    world_accelerator,
)
from tests.common.models import QWEN3_0_6B
from tests.common.numerics import cos_sim, relative_l2
from tests.common.rank_log import log, log_all
from tests.common.tiny_models import TINY_MOE_FAMILIES, shared_tiny_family_checkpoint, word_level_tokenizer

MODEL, ATTN = QWEN3_0_6B, "flash_attention_2"
ROWS, SLOTS = PACKED_ROWS, PACKED_SLOTS
ROW_TOKENS, BUDGET = PACKED_DENSE_ROW_TOKENS, PACKED_DENSE_BUDGET
DENSE_VOCAB = 32_000
SHUFFLE_SEED, FLOOR_SEED = 7, 8
GRAD_FLOOR_FACTOR = 2.0
GRAD_COSINE_MIN = 0.999
LOGP_MEAN_ABS_MAX = 1e-2
MOE_FAMILY, MOE_WORDS, MOE_SEED = "qwen3_moe", 500, 0
MOE_ROW_TOKENS, MOE_BUDGET = PACKED_MOE_ROW_TOKENS, PACKED_MOE_BUDGET
R2_FLIP_MAX = 1e-3

_PARSER = argparse.ArgumentParser(description=__doc__)
_PARSER.add_argument("--moe", action="store_true", help="the EP-2 MoE arm under routing replay")
ARGS = _PARSER.parse_args()


def _rank_round(ctx, row_tokens: list[int], vocab: int) -> dict:
    """Each rank's rows in the opposite order, so the ranks' bins differ in content and, with an uneven total, in size."""
    lengths = row_tokens if ctx.rank % 2 == 0 else row_tokens[::-1]
    halves = [n // 2 for n in lengths]
    completions = [n - h for n, h in zip(lengths, halves, strict=True)]
    return with_world_normalizer(ragged_round(halves, completions, vocab=vocab, seed=ctx.rank, device=ctx.device))


def _plan(batch: dict, budget: int):
    """COLLECTIVE."""
    lengths = (batch["prompt_mask"].sum(1) + batch["completion_mask"].sum(1)).clamp(min=2).tolist()
    return agree_microbatch_geometry(lengths, budget, slots=SLOTS, ceiling=None, round_rows=ROWS)


def _flat(grads: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([grad.flatten() for grad in grads])


def _window(model, batch: dict, *, plan=None, seed: int = SHUFFLE_SEED, scaled: bool = True):
    host = packed_host(model, slots=SLOTS, accelerator=world_accelerator())
    host.use_liger_kernel = False
    host._packs_microbatches = plan is not None
    if not scaled:
        host._scale_packed_loss = lambda loss, inputs: loss
    model.zero_grad(set_to_none=True)
    return train_window(host, batch, plan, seed)


def _check_window(ctx, model, batch: dict, budget: int, checks: dict, metrics: dict) -> None:
    plan = _plan(batch, budget)
    assert 1 < plan.num_bins < SLOTS, f"the budget should leave both bins and no-op slots, got {plan.num_bins}"
    loss_s, grads_s = _window(model, batch)
    loss_f, grads_f = _window(model, batch, seed=FLOOR_SEED)
    loss_p, grads_p = _window(model, batch, plan=plan)
    unscaled = _window(model, batch, plan=plan, scaled=False)[1]
    model.zero_grad(set_to_none=True)
    floor = relative_l2(dict(enumerate(grads_f)), dict(enumerate(grads_s)))
    packed = relative_l2(dict(enumerate(grads_p)), dict(enumerate(grads_s)))
    cosine = cos_sim(_flat(grads_p), _flat(grads_s), label="packed vs per-row gradient")
    off = relative_l2(dict(enumerate(unscaled)), dict(enumerate(grads_s)))
    metrics.update(loss_per_row=loss_s, loss_floor=loss_f, loss_packed=loss_p, grad_floor=floor, grad_packed=packed)
    metrics.update(grad_cosine=cosine, grad_unscaled=off, bins=plan.num_bins, rows_per_bin=plan.rows_per_bin)
    log_all(f"  packed {packed:.3e} vs floor {floor:.3e}, cosine {cosine:.5f}, unscaled {off:.3e}")
    checks["loss_matches_per_row"] = abs(loss_p - loss_s) <= 5e-3 * max(1.0, abs(loss_s))
    checks["grads_inside_the_floor"] = math.isfinite(packed) and floor > 0.0 and packed <= GRAD_FLOOR_FACTOR * floor
    checks["grads_point_the_same_way"] = cosine >= GRAD_COSINE_MIN
    checks["negative_control_unscaled_grads_are_off"] = off > 10 * max(floor, packed)


def run_dense(ctx) -> dict:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation=ATTN).to(ctx.device)
    apply_fsdp2_for_dp(
        model, ctx.world_size, compute_dtype=torch.bfloat16, fp32_reduce=True, reshard_after_forward=True
    )
    model.train()
    batch = _rank_round(ctx, ROW_TOKENS, DENSE_VOCAB)
    ids, mask, keep = stacked(batch)
    checks, metrics = {}, {}

    def recompute_matches_per_row():
        plan = _plan(batch, BUDGET)
        order = torch.tensor(plan.order, device=ctx.device)
        host = packed_host(model, slots=SLOTS, accelerator=world_accelerator())
        host.use_liger_kernel = False
        with torch.no_grad():
            host._packs_microbatches = False
            per_row = host._scored_rows(model, ids, mask, keep, None)
            host._packs_microbatches = True
            packed = host._scored_rows(model, ids[order], mask[order], keep, order, plan)
        real = mask[:, -keep:].bool()
        diff = (packed - per_row).abs()[real].mean().item()
        metrics["recompute_mean_abs_diff"] = diff
        log_all(f"  recompute: mean |packed - per-row| {diff:.3e} over {plan.num_bins} bins")
        assert diff <= LOGP_MEAN_ABS_MAX, f"packed recompute strays {diff:.3e} nats from the per-row one"

    record_check(checks, "recompute_matches_per_row", recompute_matches_per_row)
    _check_window(ctx, model, batch, BUDGET, checks, metrics)
    return {"checks": checks, "metrics": metrics}


def _load_moe_policy(ctx):
    tokenizer = word_level_tokenizer([f"w{i}" for i in range(MOE_WORDS)])
    family = TINY_MOE_FAMILIES[MOE_FAMILY]
    checkpoint = shared_tiny_family_checkpoint(ctx, family, "packed_microbatch_qwen3_moe", tokenizer, MOE_SEED)
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
    checks, metrics = {}, {}
    plan = _plan(batch, MOE_BUDGET)
    order = torch.tensor(plan.order, device=ctx.device)
    log(f"  {MOE_FAMILY}: {plan.num_bins} bins of up to {plan.rows_per_bin} rows, {injector.num_ep_layers} EP layers")

    def host():
        built = packed_host(model, slots=SLOTS, accelerator=world_accelerator())
        built.use_liger_kernel = False
        built._routing_injector, built._replay_row_spans = injector, True
        return built

    def r2_capture_follows_each_row():
        per_row, packed = host(), host()
        per_row._packs_microbatches = False
        with torch.no_grad():
            _, plain = per_row._recompute_logps_and_routing_masks(ids, mask, keep, None, "train")
            _, captured = packed._recompute_logps_and_routing_masks(ids, mask, keep, None, "train", order, plan)
        real = captured[..., 0] >= 0
        assert torch.equal(real, plain[..., 0] >= 0), "a captured row's real positions landed in another row's slot"
        flips = ((captured.sort(-1).values != plain.sort(-1).values).any(-1) & real).sum().item() / real.sum().item()
        metrics["r2_flips"] = flips
        assert flips <= R2_FLIP_MAX, f"packed capture flips {flips:.2e} of the selections"

    def r3_recompute_and_loss_forwards_consume_the_armed_masks():
        engine = torch.randint(0, injector.num_experts, (ROWS, ids.size(1), injector.num_ep_layers, injector.top_k))
        engine = engine.to(torch.int16).masked_fill(~mask.bool().cpu()[:, :, None, None], -1).to(ctx.device)
        built = host()
        built._rollout_routing_replay = True
        with torch.no_grad():
            logps, _ = built._recompute_logps_and_routing_masks(ids, mask, keep, engine, "train", order, plan)
        assert bool(torch.isfinite(logps).all()), "the replayed packed recompute returned non-finite log-probs"
        round_batch = batch | {ROUTING_MASKS_KEY: engine}
        built._step, built._buffered_inputs = 0, None
        built._packs_microbatches = True
        built._generate_and_score_completions = lambda _inputs: built._lay_out_packed_round(dict(round_batch), plan)
        model.zero_grad(set_to_none=True)
        for _ in range(SLOTS):
            inputs = built._prepare_inputs({})
            built._step += 1
            loss = type(built).compute_loss(built, model, inputs)
            loss.backward()
            injector.disarm()
        model.zero_grad(set_to_none=True)

    record_check(checks, "r2_capture_follows_each_row", r2_capture_follows_each_row)
    record_check(checks, "r3_replay_is_consumed_whole", r3_recompute_and_loss_forwards_consume_the_armed_masks)
    _check_window(ctx, model, batch, MOE_BUDGET, checks, metrics)
    return {"checks": checks, "metrics": metrics}


@gpu_test_main(exact_world_size=2, prefix="env_packed_microbatches_moe" if ARGS.moe else "env_packed_microbatches")
def run(ctx) -> dict:
    return run_moe(ctx) if ARGS.moe else run_dense(ctx)


if __name__ == "__main__":
    run()
