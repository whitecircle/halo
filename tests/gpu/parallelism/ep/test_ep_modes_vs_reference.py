#!/usr/bin/env python
"""A real MoE checkpoint under EP, EP+TP or pure ETP against the same checkpoint run undistributed on rank 0: loss,
router gradient in scale and direction, and a rotated-expert control those comparisons must reject.

Modes: ``ep`` (ep2), ``ep_tp`` (ep2 x tp2 on the same two ranks) and ``etp`` (pure ETP, each expert FFN split over
both ranks) on two GPUs; ``etp_dp2`` is pure ETP on four, two DP replicas of an expert-TP pair. Under ETP every
rank's expert-TP group must also be an ``expert_tp_size``-rank group its members agree on, holding each shard
index once and one DP rank: the comparisons read the identical batch's loss and gradients, not which ranks pair up
or which batch each would read. With several EP groups (``etp_dp2``) no in-backward hook syncs the router or the
experts; the trainer's deferred sweep runs after the backward, must hand every expert gradient of the identical
batch back unchanged, and the router comparisons read the router after it.

    torchrun --nproc_per_node=2 tests/gpu/parallelism/ep/test_ep_modes_vs_reference.py --model gpt_oss_20b --mode ep
    torchrun --nproc_per_node=4 tests/gpu/parallelism/ep/test_ep_modes_vs_reference.py --model gpt_oss_20b \\
        --mode etp_dp2
"""

import argparse
from dataclasses import dataclass, field
from functools import partial

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from halo.distributed.mesh import model_tp_mesh
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from tests.common.distributed import ensure_model_downloaded, world_spread
from tests.common.ep_reference import (
    broadcast_reference,
    compare_grad,
    dense_reference,
    ep_layers,
    find_router_weight,
    fixed_chat_batch,
    full_grad,
    named_router_weights,
    roll_expert_banks,
    sweep_ep_grads,
)
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory, gpu_peak_mem_gb
from tests.common.models import GPT_OSS_20B, QWEN3_30B_A3B
from tests.common.numerics import cos_sim, worst_relative_l2
from tests.common.parallel_shape import expert_tp_group_problems, expert_tp_group_views
from tests.common.rank_log import log, log_all
from tests.common.rosters import ep_layer_class_by_model_type
from tests.common.tolerances import TOL

# EP reorders the bf16 expert sum, and later routers score that reordered output, flipping near-tied picks with no bug.
# gpt-oss (top-4-of-32): reference 5.607, EP 5.589, rotated control 5.224. Qwen3 (top-8-of-128): 3.037, 3.005, 6.271.
LOSS_TOL = TOL.router_pick_flip_loss_abs
LOGIT_COSINE_MIN = 0.95
LIVE_GRAD_FRACTION_MIN = 0.5
# Every replica backpropagates one batch, so the deferred sweep's sum divided by their count hands each expert its own
# gradient back; a wrong divisor or a missing replica moves it by its own size.
SWEEP_IDENTITY_REL_TOL = 1e-2


@dataclass(frozen=True)
class Checkpoint:
    name: str
    seq_len: int
    attn_implementation: str
    load_kwargs: dict = field(default_factory=dict)


CHECKPOINTS = {
    # Eager is the one backend keeping gpt-oss's live sinks, and reset_sinks=False keeps the pretrained ones (the
    # fine-tuning reset alone moves this batch's loss by 2.65).
    "gpt_oss_20b": Checkpoint(GPT_OSS_20B, 128, "eager", {"reset_sinks": False}),
    "qwen3_30b_a3b": Checkpoint(QWEN3_30B_A3B, 256, "flash_attention_2", {"use_liger_kernel": False}),
}


@dataclass(frozen=True)
class Mode:
    axes: dict
    world_size: int = 2


MODES = {
    "ep": Mode({"ep_size": 2}),
    "ep_tp": Mode({"ep_size": 2, "tp_size": 2}),
    "etp": Mode({"ep_size": 1, "expert_tp_size": 2}),
    "etp_dp2": Mode({"ep_size": 1, "expert_tp_size": 2}, world_size=4),
}


def _router_grads(model) -> dict[str, torch.Tensor | None]:
    return {
        name: None if p.grad is None else p.grad.detach().float().clone() for name, p in named_router_weights(model)
    }


def _no_ep_baseline(checkpoint: Checkpoint, batch) -> tuple[float, torch.Tensor, dict]:
    """COLLECTIVE (the load). Nothing distributed and grouped GEMM off, which keeps the family's own MoE modules."""
    model, _ = load_distributed_model(
        model_name_or_path=checkpoint.name,
        parallelism_config=ParallelismConfig(use_grouped_gemm=False),
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=checkpoint.attn_implementation,
        **checkpoint.load_kwargs,
    )
    model.train()
    out = model(**batch, use_cache=False)
    out.loss.backward()
    baseline = out.loss.item(), out.logits.detach().clone(), _router_grads(model)
    del model, out
    cleanup_memory()
    return baseline


def _layout_checks(model, pc: ParallelismConfig) -> dict[str, bool]:
    """The model is split the way ``pc`` asks: otherwise every comparison holds for a dense model too."""
    layers = ep_layers(model)
    family_class = ep_layer_class_by_model_type()[model.config.model_type]
    checks = {"ep_layers_are_the_family_class": bool(layers) and all(type(layer) is family_class for layer in layers)}
    if not layers:
        return checks
    first = layers[0]
    log_all(f"  rank owns experts [{first.expert_start}, {first.expert_end}) of {first.num_experts}")
    if pc.ep_size > 1:
        checks["expert_bank_split_ep_way"] = (
            first.experts_per_rank == first.num_experts // pc.ep_size
            and first.expert_end - first.expert_start == first.experts_per_rank
            and first.experts_per_rank < first.num_experts
        )
    if pc.expert_tp_size > 1:
        checks["expert_ffn_sharded_etp_way"] = all(layer.expert_tp_size == pc.expert_tp_size for layer in layers)
        views = expert_tp_group_views(first, pc.get_data_parallel_rank())
        problems = expert_tp_group_problems(views, pc.expert_tp_size)
        for problem in problems:
            log(f"  expert-TP group: {problem}")
        checks["expert_tp_groups_tile_the_world"] = not problems
    if pc.tp_size > 1:
        # A sharded projection is a DTensor whose local shape is a slice of its global one.
        tp_mesh = model_tp_mesh(model)
        sharded = [
            name
            for name, param in model.named_parameters()
            if ".self_attn." in name and hasattr(param, "to_local") and param.to_local().shape != param.shape
        ]
        checks["tp_mesh_sized_tp_way"] = tp_mesh is not None and tp_mesh.size() == pc.tp_size
        checks["attention_sharded_tp_way"] = bool(sharded)
    return checks


def _no_ep_checks(baseline, ep_logits, ep_router_grads, ref_loss, metrics) -> dict[str, bool]:
    """Both losses are bounded against the reference only: the gap between two bounded deviations is not bounded."""
    baseline_loss, baseline_logits, baseline_grads = baseline
    metrics["baseline_loss"] = baseline_loss
    logit_cos = cos_sim(baseline_logits, ep_logits, label="logits")
    metrics["logit_cosine"] = logit_cos
    metrics["min_position_logit_cosine"] = (
        F.cosine_similarity(baseline_logits.float(), ep_logits.float(), -1).min().item()
    )
    metrics["top1_agreement"] = (baseline_logits.argmax(-1) == ep_logits.argmax(-1)).float().mean().item()
    checks = {
        "baseline_loss_matches_reference": abs(baseline_loss - ref_loss) < LOSS_TOL,
        "baseline_losses_agree_across_ranks": TOL.identical_batch_ranks_agree(world_spread(baseline_loss)),
        "logits_match_no_ep": logit_cos >= LOGIT_COSINE_MIN,
    }
    names = sorted(baseline_grads)
    severed = [name for name in names if baseline_grads[name] is None or ep_router_grads.get(name) is None]
    if not names or severed or baseline_grads.keys() != ep_router_grads.keys():
        log_all(
            f"  router grads severed {severed}, keys differ {sorted(baseline_grads.keys() ^ ep_router_grads.keys())}"
        )
        return checks | {"router_grads_reach_every_layer": False}
    baseline_flat = torch.cat([baseline_grads[name].flatten() for name in names])
    ep_flat = torch.cat([ep_router_grads[name].flatten() for name in names])
    ratio, cosine = compare_grad(ep_flat, baseline_flat, "router grads, all layers")
    log(f"  all router grads vs no-EP: norm_ratio={ratio:.4f} cosine={cosine:.4f} logits cosine={logit_cos:.4f}")
    return checks | {
        "all_router_grads_scale_matches_no_ep": 1 / TOL.grad_norm_ratio_max < ratio < TOL.grad_norm_ratio_max,
        "all_router_grads_direction_matches_no_ep": cosine > TOL.grad_direction_cosine_min,
        "router_grads_alive": min((flat.abs() > 1e-8).float().mean().item() for flat in (baseline_flat, ep_flat))
        > LIVE_GRAD_FRACTION_MIN,
    }


def _reference(ctx, checkpoint: Checkpoint, batch, device: str) -> tuple[float, torch.Tensor]:
    """COLLECTIVE (the broadcast). The undistributed loss and first router gradient rank 0 computes, on every rank."""
    loss, grad = 0.0, None
    if ctx.rank == 0:
        loss, grad = dense_reference(
            checkpoint.name, *batch.values(), device, attn_implementation=checkpoint.attn_implementation
        )
    return broadcast_reference(loss, grad, device, ctx.rank)


def _deferred_sweep_checks(model, metrics: dict) -> dict[str, bool]:
    """COLLECTIVE where the sweep reduces. With several EP groups only the trainer's deferred sweep syncs routers and
    experts. On the identical batch every replica holds, it must leave each expert gradient as it was."""
    layers = ep_layers(model)
    if not layers or not layers[0].ep_config.sweep_reduces_across_replicas:
        return {}
    before = {
        f"{i}.{name}": param.grad.detach().clone()
        for i, layer in enumerate(layers)
        for name, param in layer.expert_named_params()
        if param.grad is not None
    }
    sweep_ep_grads(model)
    after = {
        f"{i}.{name}": param.grad.detach()
        for i, layer in enumerate(layers)
        for name, param in layer.expert_named_params()
        if param.grad is not None
    }
    worst, name = worst_relative_l2(after, before)
    log_all(f"  deferred sweep: worst expert-gradient change {worst:.3e} ({name})")
    metrics["sweep_expert_grad_max_rel_change"] = worst
    return {"deferred_sweep_keeps_replicated_expert_grads": worst <= SWEEP_IDENTITY_REL_TOL}


def _parallel_checks(
    model, batch, ref_loss: float, ref_grad: torch.Tensor, baseline, metrics: dict
) -> dict[str, bool]:
    """COLLECTIVE (the forward)."""
    model.train()
    outputs = model(**batch, use_cache=False)
    loss = outputs.loss.item()
    log_all(f"  parallel loss={loss:.6f} |Δreference|={abs(loss - ref_loss):.3e}")
    metrics |= {"parallel_loss": loss, "loss_abs_err": abs(loss - ref_loss), "rank_loss_spread": world_spread(loss)}
    outputs.loss.backward()
    checks = _deferred_sweep_checks(model, metrics)
    router_name, router_weight = find_router_weight(model)
    ratio, cosine = compare_grad(full_grad(router_weight), ref_grad, router_name)
    log_all(f"  router grad ({router_name}) vs reference: norm_ratio={ratio:.4f} cosine={cosine:.4f}")
    metrics |= {"router_grad_norm_ratio": ratio, "router_grad_cosine": cosine}
    expert_grads = [p.grad for layer in ep_layers(model) for _, p in layer.expert_named_params()]
    checks |= {
        "losses_agree_across_ranks": TOL.identical_batch_ranks_agree(metrics["rank_loss_spread"]),
        "router_grad_scale_matches_reference": 1 / TOL.grad_norm_ratio_max < ratio < TOL.grad_norm_ratio_max,
        "router_grad_direction_matches_reference": cosine > TOL.grad_direction_cosine_min,
        "expert_grads_present_and_finite": bool(expert_grads)
        and all(g is not None and torch.isfinite(g).all() for g in expert_grads),
    }
    if baseline is not None:
        checks |= _no_ep_checks(baseline, outputs.logits.detach(), _router_grads(model), ref_loss, metrics)
    return checks


def _control_checks(model, batch, ref_loss: float, ref_grad: torch.Tensor, metrics: dict) -> dict[str, bool]:
    """COLLECTIVE (the forward). Negative control: a rotated expert bank must break both comparisons."""
    rotated = roll_expert_banks(model)
    model.zero_grad(set_to_none=True)
    control = model(**batch, use_cache=False)
    control.loss.backward()
    shift = abs(control.loss.item() - ref_loss)
    router_name, router_weight = find_router_weight(model)
    _, cosine = compare_grad(full_grad(router_weight), ref_grad, f"{router_name} (rotated experts)")
    log_all(f"  control: loss={control.loss.item():.6f} |Δreference|={shift:.3e} router grad cosine={cosine:.4f}")
    metrics |= {"control_loss_shift": shift, "control_router_grad_cosine": cosine}
    return {
        "control_engaged": rotated > 0,
        "control_wrong_expert_breaks_loss_match": shift > TOL.control_min_loss_shift(LOSS_TOL),
        "control_wrong_expert_breaks_grad_direction": cosine < TOL.grad_direction_cosine_min,
    }


def run_mode(ctx, checkpoint_key: str, mode_key: str, *, logits: bool = False) -> dict:
    """COLLECTIVE. ``logits`` also loads it undistributed on every rank, to compare per-token logits and router
    gradients."""
    checkpoint = CHECKPOINTS[checkpoint_key]
    device = f"cuda:{ctx.local_rank}"
    log(f"{checkpoint.name} --mode {mode_key} (logits={logits}), attn={checkpoint.attn_implementation}")
    ensure_model_downloaded(checkpoint.name, ctx.rank)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # The per-rank loss is then the undistributed one, and a spread a desync.
    batch = dict(
        zip(
            ("input_ids", "attention_mask", "labels"),
            fixed_chat_batch(tokenizer, checkpoint.seq_len, device, broadcast=True),
            strict=True,
        )
    )

    ref_loss, ref_grad = _reference(ctx, checkpoint, batch, device)
    log(f"  reference: loss={ref_loss:.6f} router|g|={ref_grad.norm():.6e}")
    metrics = {"reference_loss": ref_loss, "reference_router_grad_norm": ref_grad.norm().item()}
    # A zero loss or router gradient would let every comparison below pass on a model that computes nothing.
    checks = {
        "reference_loss_nontrivial": 0.5 < ref_loss < 20.0,
        "reference_router_grad_nonzero": metrics["reference_router_grad_norm"] > 1e-8,
    }
    baseline = _no_ep_baseline(checkpoint, batch) if logits else None

    pc = ParallelismConfig(**MODES[mode_key].axes)
    log(f"  {pc.summary()}  ep_group_size={pc.ep_group_size}")
    model, _ = load_distributed_model(
        model_name_or_path=checkpoint.name,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=checkpoint.attn_implementation,
        **checkpoint.load_kwargs,
    )
    checks |= _layout_checks(model, pc)
    if pc.ep_size > 1 and pc.tp_size == 1:
        # Pure EP repeats bit for bit, since the grouped forward never reads DeepEP's receive order. A difference means
        # state survived between calls: a reused arena buffer, a mutated routing cache.
        model.eval()
        with torch.no_grad():
            repeats = [model(**batch, use_cache=False).loss.item() for _ in range(2)]
        log_all(f"  repeated forward: {repeats[0]:.6f} vs {repeats[1]:.6f}")
        checks["repeated_forward_is_deterministic"] = repeats[0] == repeats[1]
    checks |= _parallel_checks(model, batch, ref_loss, ref_grad, baseline, metrics)
    checks |= _control_checks(model, batch, ref_loss, ref_grad, metrics)

    # TP's reduction order moves with NCCL topology state and flips near-tied picks: the loss may pass the flip bound
    # while staying well separated from the corrupted control, whose regime is O(1).
    delta = metrics["loss_abs_err"]
    near_flips = pc.tp_size > 1 and delta < 2 * LOSS_TOL and delta < 0.5 * metrics["control_loss_shift"]
    checks["loss_matches_reference"] = delta < LOSS_TOL or near_flips
    metrics["peak_gb"] = gpu_peak_mem_gb()
    return {"checks": checks, "metrics": metrics}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(CHECKPOINTS), required=True)
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    return parser


if __name__ == "__main__":
    # Parsed here, not at import: test_ep_vs_no_ep.py imports run_mode and launches with no arguments.
    ARGS = _parser().parse_args()
    gpu_test_main(exact_world_size=MODES[ARGS.mode].world_size, prefix="ep_modes_vs_reference")(
        partial(run_mode, checkpoint_key=ARGS.model, mode_key=ARGS.mode)
    )()
