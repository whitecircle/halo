#!/usr/bin/env python
"""Packed GRPO micro-batches on real ranks: a window of bins trains as the per-row window does.

On the trainer's real scoring path, plan and TRL's real buffer and loss (Qwen3-0.6B under FSDP2, FlashAttention-2):

1. the packed no-grad recompute scores every row as the per-row recompute does, and every rank ran the world's bin
   count of scoring calls;
2. the packed window's loss equals the per-row window's;
3. its gradients, through the FSDP2 reduce-scatter and the leading no-op slots, stray from the per-row window in fp32
   (the same rows on sdpa) no further than twice as far as the per-row bf16 window does, and point the same way;
4. negative control: without the row-mean rescale the gradient is off by the rows-per-bin factor.

``--moe`` runs a tiny Qwen3-MoE at ``ep_size = 2`` (DeepEP dispatch, one dispatch per bin), where the bins must run
on every rank in equal number, under R2 capture and R3 replay (its gradient floor stays a per-row window in another
shuffle: no fp32 copy of an EP model):

5. the packed recompute captures each row's routing into its own batch slot: a selection that differs from the
   per-row run's has a near-tie margin there (packed bf16 hidden states differ slightly), wide-margin selections
   match exactly, and a packed forward over tokens shifted one position is caught. The flip count does not depend on
   the bin plan: varlen attention and the router read each row alone, so a row's gate input is the same in any bin
   and only the varlen-versus-dense numerics per row can flip a selection;
6. the packed recompute replays each row its own engine routing, and the window's gradient forwards consume each armed
   mask whole.

    torchrun --nproc_per_node=2 tests/gpu/trainers/grpo/test_env_packed_microbatches.py [--moe]
"""

import argparse

import torch
from transformers import AutoModelForCausalLM

from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.fsdp import apply_fsdp2_for_dp
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.trainers.grpo.environmental.routing_replay import build_routing_replay_injector
from halo.trainers.grpo.environmental.step_batch import ROUTING_ROWS_KEY
from halo.trainers.grpo.rounds.microbatch_plan import agree_microbatch_geometry
from tests.common.harness import gpu_test_main, record_check
from tests.common.length_deal import (
    PACKED_DENSE_BUDGET,
    PACKED_DENSE_ROW_TOKENS,
    PACKED_MOE_BUDGET,
    PACKED_MOE_ROW_TOKENS,
    PACKED_ROWS,
    PACKED_SLOTS,
    packed_host,
    stacked,
    train_window,
    world_accelerator,
)
from tests.common.models import QWEN3_0_6B
from tests.common.packed_grad_floor import build_fp32_reference, judge_packed_grads
from tests.common.rank_log import log, log_all
from tests.common.rank_rounds import reversed_order_round
from tests.common.routing import (
    check_flips_are_near_ties,
    dense_rows,
    distinct_expert_picks,
    ragged_routing,
    router_margins,
)
from tests.common.tiny_models import TINY_MOE_FAMILIES, shared_tiny_family_checkpoint, word_level_tokenizer

MODEL, ATTN = QWEN3_0_6B, "flash_attention_2"
ROWS, SLOTS = PACKED_ROWS, PACKED_SLOTS
ROW_TOKENS, BUDGET = PACKED_DENSE_ROW_TOKENS, PACKED_DENSE_BUDGET
DENSE_VOCAB = 32_000
SHUFFLE_SEED, FLOOR_SEED = 7, 8
ROW_MEAN_LOSS = "grpo"
LOGP_MEAN_ABS_MAX = 1e-2
MOE_FAMILY, MOE_WORDS, MOE_SEED = "qwen3_moe", 500, 0
MOE_ROW_TOKENS, MOE_BUDGET = PACKED_MOE_ROW_TOKENS, PACKED_MOE_BUDGET

_PARSER = argparse.ArgumentParser(description=__doc__)
_PARSER.add_argument("--moe", action="store_true", help="the EP-2 MoE arm under routing replay")
ARGS = _PARSER.parse_args()


def _plan(batch: dict, budget: int):
    """COLLECTIVE."""
    lengths = (batch["prompt_mask"].sum(1) + batch["completion_mask"].sum(1)).clamp(min=2).tolist()
    return agree_microbatch_geometry(lengths, budget, slots=SLOTS, ceiling=None, round_rows=ROWS)


def _window(model, batch: dict, *, plan=None, seed: int = SHUFFLE_SEED, scaled: bool = True, loss_type: str = "dapo"):
    host = packed_host(model, slots=SLOTS, accelerator=world_accelerator(), loss_type=loss_type)
    host.use_liger_kernel = False
    host._packs_microbatches = plan is not None
    if not scaled:
        host._scale_packed_loss = lambda loss, inputs: loss
    model.zero_grad(set_to_none=True)
    return train_window(host, batch, plan, seed)


def _check_window(model, reference_model, batch: dict, budget: int, checks: dict, metrics: dict) -> None:
    """``reference_model`` is the fp32 copy the gradient floor is measured against, or ``None`` (the MoE arm)."""
    plan = _plan(batch, budget)
    assert 1 < plan.num_bins < SLOTS, f"the budget should leave both bins and no-op slots, got {plan.num_bins}"
    loss_s, grads_s = _window(model, batch)
    loss_f, grads_f = _window(model, batch, seed=FLOOR_SEED)
    loss_p, grads_p = _window(model, batch, plan=plan)
    unscaled = _window(model, batch, plan=plan, scaled=False)[1]
    repeat = _window(model, batch, plan=plan)[1]
    grads_ref = None if reference_model is None else _window(reference_model, batch)[1]
    verdict = judge_packed_grads(grads_s, grads_f, grads_p, unscaled, grads_ref, repeat=repeat)
    metrics.update(loss_per_row=loss_s, loss_order_floor=loss_f, loss_packed=loss_p, **verdict.metrics)
    metrics.update(bins=plan.num_bins, rows_per_bin=plan.rows_per_bin)
    log_all(
        f"  packed {verdict.packed_err:.3e} vs floor {verdict.floor:.3e} (order floor {verdict.order_floor:.3e}), "
        f"cosine {verdict.cosine:.5f}, unscaled {verdict.unscaled_err:.3e}"
    )
    checks["loss_matches_per_row"] = abs(loss_p - loss_s) <= 5e-3 * max(1.0, abs(loss_s))
    checks["grads_inside_the_floor"] = verdict.inside_the_floor
    checks["grads_point_the_same_way"] = verdict.points_the_same_way
    checks["token_global_loss_ignores_the_rescale"] = verdict.rescale_changes_nothing
    _check_row_mean_rescale(model, reference_model, batch, plan, checks, metrics)


def _check_row_mean_rescale(model, reference_model, batch: dict, plan, checks: dict, metrics: dict) -> None:
    """The default dapo normalizer is one round-global token count, so the per-bin rescale is 1 there and dropping it
    changes nothing. A row-mean loss divides each slot by its own row count: there the rescale carries the gradient and
    dropping it (the negative control) moves it by rows_per_bin * slots / round_rows."""
    grads_s = _window(model, batch, loss_type=ROW_MEAN_LOSS)[1]
    grads_f = _window(model, batch, seed=FLOOR_SEED, loss_type=ROW_MEAN_LOSS)[1]
    grads_p = _window(model, batch, plan=plan, loss_type=ROW_MEAN_LOSS)[1]
    unscaled = _window(model, batch, plan=plan, scaled=False, loss_type=ROW_MEAN_LOSS)[1]
    grads_ref = None if reference_model is None else _window(reference_model, batch, loss_type=ROW_MEAN_LOSS)[1]
    verdict = judge_packed_grads(grads_s, grads_f, grads_p, unscaled, grads_ref)
    metrics.update({f"row_mean_{key}": value for key, value in verdict.metrics.items()})
    log_all(
        f"  row-mean {ROW_MEAN_LOSS}: packed {verdict.packed_err:.3e} vs floor {verdict.floor:.3e} "
        f"(order floor {verdict.order_floor:.3e}), unscaled {verdict.unscaled_err:.3e}"
    )
    checks["row_mean_grads_inside_the_floor"] = verdict.inside_the_floor
    checks["negative_control_unscaled_row_mean_grads_are_off"] = verdict.control_bites


def run_dense(ctx) -> dict:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation=ATTN).to(ctx.device)
    reference_model = build_fp32_reference(model, world_size=ctx.world_size)
    apply_fsdp2_for_dp(
        model, ctx.world_size, compute_dtype=torch.bfloat16, fp32_reduce=True, reshard_after_forward=True
    )
    model.train()
    batch = reversed_order_round(ctx, ROW_TOKENS, DENSE_VOCAB)
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
    _check_window(model, reference_model, batch, BUDGET, checks, metrics)
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
    batch = reversed_order_round(ctx, MOE_ROW_TOKENS, vocab)
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
        gate_logits = [[] for _ in range(injector.num_ep_layers)]
        hooks = [
            layer.gate.register_forward_hook(
                lambda _gate, _args, out, chunks=chunks: chunks.append(
                    (out[0] if isinstance(out, tuple) else out).detach()
                )
            )
            for chunks, (_, layer) in zip(gate_logits, named_ep_layers(model), strict=True)
        ]
        with torch.no_grad():
            try:
                _, plain = per_row._recompute_logps_and_routing(ids, mask, keep, None, "train")
            finally:
                for hook in hooks:
                    hook.remove()
            _, captured = packed._recompute_logps_and_routing(ids, mask, keep, None, "train", order, plan)
            shifted = ids.roll(1, dims=1)
            _, misrouted = packed._recompute_logps_and_routing(shifted, mask, keep, None, "train", order, plan)
        plain, captured, misrouted = (dense_rows(r, ctx.device) for r in (plain, captured, misrouted))
        real = plain[..., 0] >= 0
        margins = torch.zeros(plain.shape[:3], device=plain.device)
        for layer_index, chunks in enumerate(gate_logits):
            margins[..., layer_index][real[..., layer_index]] = router_margins(torch.cat(chunks), injector.top_k).to(
                plain.device
            )
        stats = check_flips_are_near_ties(captured, plain, margins)
        metrics.update({f"r2_{key}": value for key, value in stats.items()})
        try:
            check_flips_are_near_ties(misrouted, plain, margins)
        except AssertionError as err:
            metrics["r2_negative_control"] = str(err)
        else:
            raise AssertionError("the check passed a packed forward over tokens shifted one position")

    def r3_recompute_and_loss_forwards_consume_the_armed_masks():
        engine = distinct_expert_picks(
            (ROWS, ids.size(1)), injector.num_ep_layers, injector.top_k, injector.num_experts
        )
        engine = engine.masked_fill(~mask.bool().cpu()[:, :, None, None], -1)
        routing = ragged_routing(engine)
        built = host()
        built._rollout_routing_replay = True
        built._rollout_routing = routing
        with torch.no_grad():
            logps, _ = built._recompute_logps_and_routing(ids, mask, keep, routing, "train", order, plan)
        assert bool(torch.isfinite(logps).all()), "the replayed packed recompute returned non-finite log-probs"
        round_batch = batch | {ROUTING_ROWS_KEY: torch.arange(ROWS, device=ctx.device)}
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
    _check_window(model, None, batch, MOE_BUDGET, checks, metrics)
    return {"checks": checks, "metrics": metrics}


@gpu_test_main(exact_world_size=2, prefix="env_packed_microbatches_moe" if ARGS.moe else "env_packed_microbatches")
def run(ctx) -> dict:
    return run_moe(ctx) if ARGS.moe else run_dense(ctx)


if __name__ == "__main__":
    run()
