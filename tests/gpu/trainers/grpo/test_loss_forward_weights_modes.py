#!/usr/bin/env python
"""The importance weight and KL clamp built in the loss forward, on real ranks: a window trains as the no-grad pass's does.

On the trainer's real scoring path and TRL's real buffer and loss (Qwen3-0.6B under FSDP2, FlashAttention-2), with rows
of unequal length on the two ranks:

1. the fused window's loss equals the pass window's, and its gradients stray from the pass window's no further than twice
   as far as a pass window in another shuffle does (bf16 accumulation rounds differently in every order), and point the
   same way;
2. the largest difference between the pass's log-probs and the loss forward's is reported, and bounded;
3. a fused build runs no backbone forward, a pass build runs one per row;
4. negative control: a weight built from the live log-probs (no ``detach()``) leaves that floor.

``--packed`` trains both windows on packed bins. ``--moe`` runs a tiny Qwen3-MoE at ``ep_size = 2`` (DeepEP dispatch)
and adds R3: the loss forward replays the engine's masks with no pass to arm them.

    torchrun --nproc_per_node=2 tests/gpu/trainers/grpo/test_loss_forward_weights_modes.py [--packed | --moe]
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
from halo.trainers.grpo.environmental.step_batch import ROUTING_ROWS_KEY
from halo.trainers.grpo.logprob_scoring import ProtectedTokenEntropyMixin
from halo.trainers.grpo.microbatch_plan import agree_microbatch_geometry
from halo.trainers.grpo.objective.logratio import clamp_ref_logps, compute_is_ratio
from halo.trainers.grpo.objective.loss_inputs import (
    IS_CORRECTED_MASK_KEY,
    IS_SAMPLING_LOGPS_KEY,
    NEGATIVE_ONLY_ROWS_KEY,
    UNCLAMPED_REF_LOGPS_KEY,
)
from halo.trainers.world_metrics import WorldMetrics
from tests.common.harness import gpu_test_main, record_check
from tests.common.length_deal import (
    packed_host,
    ragged_round,
    stacked,
    train_window,
    with_world_normalizer,
    world_accelerator,
)
from tests.common.models import QWEN3_0_6B
from tests.common.numerics import cos_sim, relative_l2
from tests.common.rank_log import log_all
from tests.common.routing import distinct_expert_picks, ragged_routing
from tests.common.tiny_models import TINY_MOE_FAMILIES, shared_tiny_family_checkpoint, word_level_tokenizer

MODEL, ATTN = QWEN3_0_6B, "flash_attention_2"
ROWS, SLOTS, CLIP, BETA = 16, 4, 3.0, 0.04
ROW_TOKENS = [256 + 192 * i for i in range(ROWS)]
BUDGET = 6144
DENSE_VOCAB = 32_000
SHUFFLE_SEED, FLOOR_SEED, NOISE_SEED = 7, 8, 3
GRAD_FLOOR_FACTOR = 2.0
GRAD_COSINE_MIN = 0.999
LOGP_MAX_ABS = 0.25
MOE_FAMILY, MOE_WORDS, MOE_SEED = "qwen3_moe", 500, 0
MOE_ROW_TOKENS = [64 + 24 * i for i in range(ROWS)]

_PARSER = argparse.ArgumentParser(description=__doc__)
_GROUP = _PARSER.add_mutually_exclusive_group()
_GROUP.add_argument("--packed", action="store_true", help="train both windows on packed bins")
_GROUP.add_argument("--moe", action="store_true", help="the EP-2 MoE arm, with R3")
ARGS = _PARSER.parse_args()


def _rank_round(ctx, row_tokens: list[int], vocab: int) -> dict:
    """Each rank's rows in the opposite order, so the ranks' micro-batches differ in content and size."""
    lengths = row_tokens if ctx.rank % 2 == 0 else row_tokens[::-1]
    halves = [n // 2 for n in lengths]
    completions = [n - h for n, h in zip(lengths, halves, strict=True)]
    return with_world_normalizer(ragged_round(halves, completions, vocab=vocab, seed=ctx.rank, device=ctx.device))


def _host(model, *, fused: bool, plan=None, injector=None):
    host = packed_host(model, slots=SLOTS, accelerator=world_accelerator())
    host.use_liger_kernel = False
    host._packs_microbatches = plan is not None
    host.beta, host.vllm_importance_sampling_clip_max, host._is_correction = BETA, CLIP, True
    host.args.use_bias_correction_kl = False
    host._skips_logprob_pass = fused
    host._loss_forward_metrics = WorldMetrics() if fused else None
    if injector is not None:
        host._routing_injector, host._replay_row_spans = injector, True
    return host


def _live_weight_scorer(self, model, input_ids, attention_mask, logits_to_keep, *args, **kwargs):
    """The hook with the ``detach()`` taken out: the weight carries the loss forward's graph."""
    scored = ProtectedTokenEntropyMixin._get_per_token_logps_and_entropies(
        self, model, input_ids, attention_mask, logits_to_keep, *args, **kwargs
    )
    seam = self._loss_forward_seam
    if seam is not None and torch.is_grad_enabled() and kwargs.get("compute_entropy"):
        seam.fired += 1
        self._weigh_from_loss_forward(seam, scored[0])
    return scored


def _inputs(model, batch: dict) -> tuple[dict, dict, float]:
    """``(pass batch, fused batch, max |pass logp - loss-forward logp|)`` of one round. The loss forward's log-probs are
    taken row by row with the graph on, as a micro-batch of one row scores them."""
    ids, mask, keep = stacked(batch)
    host = _host(model, fused=False)
    with torch.no_grad():
        recompute = host._scored_rows(model, ids, mask, keep, None)
    loss_forward = torch.cat(
        [
            host._get_per_token_logps_and_entropies(
                model, ids[i : i + 1], mask[i : i + 1], keep, compute_entropy=True
            )[0].detach()
            for i in range(ids.size(0))
        ]
    )
    gap = (recompute - loss_forward).abs()[mask[:, -keep:].bool()].max().item()
    device = recompute.device
    gen = torch.Generator(device=device).manual_seed(NOISE_SEED)
    has_sampling = torch.ones(recompute.size(0), dtype=torch.bool, device=device)
    sampling = recompute - torch.randn(recompute.shape, generator=gen, device=device) * 0.5
    ratio, _, corrected = compute_is_ratio(recompute, sampling, batch["completion_mask"].bool(), has_sampling, CLIP)
    ref = recompute + (torch.rand(recompute.shape, generator=gen, device=device) - 0.5) * 16
    shared = {k: v for k, v in batch.items() if k != "importance_sampling_ratio"}
    in_pass = shared | {"importance_sampling_ratio": ratio, "ref_per_token_logps": clamp_ref_logps(ref, recompute)[0]}
    fused = shared | {
        IS_SAMPLING_LOGPS_KEY: sampling,
        IS_CORRECTED_MASK_KEY: corrected,
        NEGATIVE_ONLY_ROWS_KEY: torch.zeros(recompute.size(0), dtype=torch.bool, device=device),
        UNCLAMPED_REF_LOGPS_KEY: ref,
    }
    return in_pass, fused, gap


def _plan(batch: dict, budget: int):
    """COLLECTIVE."""
    lengths = (batch["prompt_mask"].sum(1) + batch["completion_mask"].sum(1)).clamp(min=2).tolist()
    return agree_microbatch_geometry(lengths, budget, slots=SLOTS, ceiling=None, round_rows=ROWS)


def _flat(grads: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([grad.flatten() for grad in grads])


def _window(model, batch: dict, *, fused: bool, plan=None, seed: int = SHUFFLE_SEED, live_weight: bool = False):
    host = _host(model, fused=fused, plan=plan)
    if live_weight:
        host._get_per_token_logps_and_entropies = _live_weight_scorer.__get__(host)
    model.zero_grad(set_to_none=True)
    return train_window(host, batch, plan, seed)


def _check_windows(model, batch, plan, checks: dict, metrics: dict) -> None:
    in_pass, fused, gap = _inputs(model, batch)
    metrics["max_abs_logp_gap"] = gap
    log_all(f"  max |pass logp - loss-forward logp| {gap:.3e}")
    checks["logp_gap_is_bounded"] = math.isfinite(gap) and gap <= LOGP_MAX_ABS
    loss_pass, grads_pass = _window(model, in_pass, fused=False, plan=plan)
    loss_floor, grads_floor = _window(model, in_pass, fused=False, plan=plan, seed=FLOOR_SEED)
    loss_fused, grads_fused = _window(model, fused, fused=True, plan=plan)
    grads_live = _window(model, fused, fused=True, plan=plan, live_weight=True)[1]
    model.zero_grad(set_to_none=True)
    floor = relative_l2(dict(enumerate(grads_floor)), dict(enumerate(grads_pass)))
    stray = relative_l2(dict(enumerate(grads_fused)), dict(enumerate(grads_pass)))
    cosine = cos_sim(_flat(grads_fused), _flat(grads_pass), label="fused vs pass gradient")
    live = relative_l2(dict(enumerate(grads_live)), dict(enumerate(grads_pass)))
    metrics.update(loss_pass=loss_pass, loss_floor=loss_floor, loss_fused=loss_fused, grad_floor=floor)
    metrics.update(grad_fused=stray, grad_cosine=cosine, grad_live_weight=live)
    log_all(f"  fused {stray:.3e} vs floor {floor:.3e}, cosine {cosine:.6f}, live weight {live:.3e}")
    checks["loss_matches_the_pass"] = abs(loss_fused - loss_pass) <= 5e-3 * max(1.0, abs(loss_pass))
    checks["grads_inside_the_floor"] = math.isfinite(stray) and floor > 0.0 and stray <= GRAD_FLOOR_FACTOR * floor
    checks["grads_point_the_same_way"] = cosine >= GRAD_COSINE_MIN
    checks["negative_control_live_weight_leaves_the_floor"] = live > 10 * max(floor, stray)


def _check_pass_slot(model, batch, checks: dict) -> None:
    """A fused build runs no backbone forward in the pass's slot; a pass build runs one per row."""
    ids, mask, keep = stacked(batch)
    counts = {}
    for fused in (True, False):
        host = _host(model, fused=fused)
        calls = []
        wrapped = host._get_last_hidden_state
        host._get_last_hidden_state = lambda *a, _wrapped=wrapped, **kw: (calls.append(1), _wrapped(*a, **kw))[1]
        with torch.no_grad():
            host._recompute_logps_and_routing_masks(ids, mask, keep, None, "train")
        counts[fused] = len(calls)
    log_all(f"  backbone forwards in the pass slot: fused {counts[True]}, pass {counts[False]}")
    checks["fused_build_runs_no_backbone_forward"] = counts[True] == 0
    checks["pass_build_runs_one_per_row"] = counts[False] == ids.size(0)


def run_dense(ctx) -> dict:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation=ATTN).to(ctx.device)
    apply_fsdp2_for_dp(
        model, ctx.world_size, compute_dtype=torch.bfloat16, fp32_reduce=True, reshard_after_forward=True
    )
    model.train()
    batch = _rank_round(ctx, ROW_TOKENS, DENSE_VOCAB)
    plan = _plan(batch, BUDGET) if ARGS.packed else None
    checks, metrics = {}, {}
    _check_pass_slot(model, batch, checks)
    _check_windows(model, batch, plan, checks, metrics)
    return {"checks": checks, "metrics": metrics}


def run_moe(ctx) -> dict:
    tokenizer = word_level_tokenizer([f"w{i}" for i in range(MOE_WORDS)])
    checkpoint = shared_tiny_family_checkpoint(
        ctx, TINY_MOE_FAMILIES[MOE_FAMILY], "loss_forward_weights_qwen3_moe", tokenizer, MOE_SEED
    )
    model, _ = load_distributed_model(
        checkpoint,
        ParallelismConfig(ep_size=ctx.world_size),
        torch.bfloat16,
        trust_remote_code=False,
        attn_implementation=ATTN,
    )
    ep_params = [param for _, layer in named_ep_layers(model) for param in layer.parameters()]
    apply_fsdp2_for_dp(model, ctx.world_size, compute_dtype=torch.bfloat16, fp32_reduce=True, ignored_params=ep_params)
    model.train()
    injector = build_routing_replay_injector(model)
    batch = _rank_round(ctx, MOE_ROW_TOKENS, len(tokenizer))
    checks, metrics = {}, {}
    _check_pass_slot(model, batch, checks)
    _check_windows(model, batch, None, checks, metrics)

    def r3_loss_forward_replays_engine_masks():
        """Fused, with the engine's masks in the batch: the loss forwards arm them and consume each whole."""
        ids, mask, keep = stacked(batch)
        engine = distinct_expert_picks(
            (ROWS, ids.size(1)), injector.num_ep_layers, injector.top_k, injector.num_experts
        )
        engine = engine.masked_fill(~mask.bool().cpu()[:, :, None, None], -1).to(ctx.device)
        _, fused, _ = _inputs(model, batch)
        built = _host(model, fused=True, injector=injector)
        built._rollout_routing_replay = True
        built._step, built._buffered_inputs = 0, None
        built._rollout_routing = ragged_routing(engine)
        round_batch = fused | {ROUTING_ROWS_KEY: torch.arange(engine.size(0), device=ctx.device)}
        built._generate_and_score_completions = lambda _inputs: dict(round_batch)
        model.zero_grad(set_to_none=True)
        for _ in range(SLOTS):
            inputs = built._prepare_inputs({})
            built._step += 1
            loss = type(built).compute_loss(built, model, inputs)
            loss.backward()
            injector.disarm()
        model.zero_grad(set_to_none=True)

    record_check(checks, "r3_loss_forward_replays_engine_masks", r3_loss_forward_replays_engine_masks)
    return {"checks": checks, "metrics": metrics}


_PREFIX = (
    "loss_forward_weights_moe"
    if ARGS.moe
    else "loss_forward_weights_packed"
    if ARGS.packed
    else "loss_forward_weights"
)


@gpu_test_main(exact_world_size=2, prefix=_PREFIX)
def run(ctx) -> dict:
    return run_moe(ctx) if ARGS.moe else run_dense(ctx)


if __name__ == "__main__":
    run()
