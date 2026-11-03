#!/usr/bin/env python
"""Fused last-stage head+loss has the logits path's gradients at a lower memory peak.

A repeat of the unfused path shows the fixed tolerance sits above identical code's own noise. Another runs fused
eval-before-train, since shape inference latches output form and recv buffers on the first call.
"""

import gc
import math
import sys

import torch
import torch.distributed as dist
from transformers import Qwen3Config, Qwen3ForCausalLM

from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.distributed.parallelism_config import ParallelismConfig
from halo.distributed.pipeline_parallel.stage_losses import FusedHeadLoss, TokenLoss
from halo.kernels import logprobs as logprob_kernels
from tests.common import pp_equivalence as ppeq
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.rank_log import log_all

# The logits plane must dominate so the memory difference is unambiguous, and the head GEMM must be wide enough that
# slicing it is a real test: 65536 x 1024 tokens = 256 MB per fp32 plane, against a ~2 MB backbone.
FUSED_LOSS_CONFIG = {
    **TINY_QWEN3_CONFIG,
    "vocab_size": 65536,
    "hidden_size": 256,
    "num_hidden_layers": 4,
    "max_position_embeddings": 2048,
}
SEQ_LEN = 1024
# One row per microbatch, the production shape at long context. The count follows the launch (4 GPUs -> pp2, 8 -> pp4,
# which Schedule1F1B needs a full pipeline for).
MIN_MICROBATCHES = 2
# 4 CE chunks per microbatch at V=65536 (16M / 65536 = 256 rows of 1024).
CE_CHUNK_ELEMENTS = 16 * 1024 * 1024

# fp32 end to end, so reduction order limits the comparison, not bf16 rounding.
LOSS_TOLERANCE = 1e-6
GRAD_TOLERANCE = 1e-5

# --gc: non-reentrant activation checkpointing on the decoder layers, which must nest with the fused loss's own
# segments.
USE_GC = False


def build_model(device):
    torch.manual_seed(1234)
    config = Qwen3Config(**FUSED_LOSS_CONFIG, attn_implementation="eager")
    return Qwen3ForCausalLM(config).to(device=device, dtype=torch.float32)


def run_once(ctx, config, groups, input_ids, labels, n_microbatches, *, fused, eval_first=False):
    """Rebuilt per run: the stage latches its output metadata at the first step, a scalar or the logits plane."""
    device = torch.device("cuda", ctx.local_rank)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Proves the checkpoint path ran: the silent failure is HF's training/no-grad gate skipping it.
    stage_module, runner, gc_calls = ppeq.build_runner_stage(
        build_model(device),
        config,
        groups,
        device,
        n_microbatches,
        FusedHeadLoss() if fused else TokenLoss(),
        gc=USE_GC,
    )
    assert (stage_module.fused_loss_fn is not None) == (fused and config.is_last_pp_stage)

    # HF's order (``evaluate()`` at step 0), the only one that reaches the fused runner's workaround: ``forward_only``
    # injects a zero label placeholder so the stage latches its scalar form. Without it the first ``step`` returns a
    # scalar against latched logits metadata, and backward-less recv buffers leave later steps no input gradient to
    # send upstream.
    if eval_first:
        pre_eval_loss = runner.eval_loss(input_ids, labels)
        pre_logits = runner.forward_only(input_ids)

    loss = runner.step(input_ids, labels)
    # Before gathering gradients: full_tensor() and the CPU copies would blur the number under measurement.
    peak = torch.cuda.max_memory_allocated()

    # Forward-only scoring must give back real logits (the trainer's compute_metrics path depends on it).
    eval_loss = runner.eval_loss(input_ids, labels)
    logits = runner.forward_only(input_ids)
    if eval_first:
        # Nothing trained in between, so a difference means the latched form decided the answer.
        assert pre_logits is None or torch.equal(pre_logits, logits)
        eval_loss = pre_eval_loss

    # Forward-only passes leave FSDP2's transient unsharded params registered, which carry no ``.grad``. Reading
    # ``named_parameters()`` without resharding collects nothing, which ``all_params_compared`` below catches.
    reshard_fsdp2_modules(stage_module)

    result = {
        "loss": float(loss) if loss is not None else 0.0,
        "eval_loss": float(eval_loss) if eval_loss is not None else 0.0,
        "grads": ppeq.gather_stage_grads(stage_module, to_cpu=True),
        "n_params": len(list(stage_module.named_parameters())),
        "peak": float(peak),
        "gc_calls": gc_calls[0],
        "n_layers": len(stage_module.model.layers),
        "logits_shape": tuple(logits.shape) if logits is not None else (),
        # A digest: holding 256 MB per run would distort the later runs' peaks.
        "logits_digest": logits.sum(dtype=torch.float64).item() if logits is not None else 0.0,
        "logits_slice": logits[0, ::128, ::4096].double().cpu() if logits is not None else None,
    }
    del runner, stage_module, loss, eval_loss, logits
    gc.collect()
    torch.cuda.empty_cache()
    return result


def worst_grad_error(reference: dict, other: dict) -> tuple[float, str | None, int]:
    """A parameter present in only one run is an infinite error: a silently narrowed comparison is the failure here."""
    absent = next((name for name in reference if name not in other), None)
    if absent is not None:
        return math.inf, absent, 0
    errors, worst_name = ppeq.max_abs_grad_errors(other, reference)
    return errors.get(worst_name, 0.0), worst_name, len(errors)


@gpu_test_main(min_world_size=4, prefix="pp_fused_head_loss")
def run(ctx):
    device = torch.device("cuda", ctx.local_rank)
    # Both paths chunk on the same boundaries, so only the head differs.
    logprob_kernels.LOGIT_CHUNK_ELEMENTS = CE_CHUNK_ELEMENTS

    pp_size = ctx.world_size // 2
    n_microbatches = max(MIN_MICROBATCHES, pp_size)
    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=pp_size)
    groups = ppeq.build_stage_groups(config)

    input_ids, labels = ppeq.build_unmasked_batch(FUSED_LOSS_CONFIG["vocab_size"], n_microbatches, SEQ_LEN, device)
    # The chunk loop must skip these targets without renormalizing. Mid-sequence, so it straddles a chunk boundary.
    labels[:, SEQ_LEN // 2 - 8 : SEQ_LEN // 2 + 8] = -100
    unfused = run_once(ctx, config, groups, input_ids, labels, n_microbatches, fused=False)
    control = run_once(ctx, config, groups, input_ids, labels, n_microbatches, fused=False)
    fused = run_once(ctx, config, groups, input_ids, labels, n_microbatches, fused=True)
    eval_first = run_once(ctx, config, groups, input_ids, labels, n_microbatches, fused=True, eval_first=True)

    checks, metrics = {}, {}

    expected_shape = (n_microbatches, SEQ_LEN, FUSED_LOSS_CONFIG["vocab_size"])
    last_stage = torch.zeros(6, device=device)
    if config.is_last_pp_stage:
        last_stage[0] = abs(fused["loss"] - unfused["loss"]) / max(abs(unfused["loss"]), 1e-12)
        last_stage[1] = abs(fused["eval_loss"] - unfused["eval_loss"]) / max(abs(unfused["eval_loss"]), 1e-12)
        slices = (fused["logits_slice"], unfused["logits_slice"])
        logits_ok = (
            fused["logits_shape"] == expected_shape == unfused["logits_shape"]
            and fused["logits_digest"] == unfused["logits_digest"]
            and all(s is not None for s in slices)
            and torch.equal(*slices)
        )
        last_stage[2] = 1.0 if logits_ok else 0.0
        last_stage[3] = abs(unfused["loss"])  # A zero loss would make the ratios above vacuous.
        # The eval-first run scored before its first step, so its eval loss is the untrained model's, as is the
        # reference's.
        last_stage[4] = abs(eval_first["eval_loss"] - unfused["eval_loss"]) / max(abs(unfused["eval_loss"]), 1e-12)
        last_stage[5] = (
            1.0
            if eval_first["logits_shape"] == expected_shape and eval_first["logits_digest"] == unfused["logits_digest"]
            else 0.0
        )
        log_all(
            f"[last stage] loss fused={fused['loss']:.10f} unfused={unfused['loss']:.10f} "
            f"control={control['loss']:.10f} | eval fused={fused['eval_loss']:.10f} "
            f"unfused={unfused['eval_loss']:.10f} | forward_only shape={fused['logits_shape']} "
            f"digest fused={fused['logits_digest']:.6f} unfused={unfused['logits_digest']:.6f} "
            f"| eval-first eval_loss={eval_first['eval_loss']:.10f} shape={eval_first['logits_shape']}"
        )
    dist.all_reduce(last_stage, op=dist.ReduceOp.MAX)
    checks["loss_matches_unfused"] = last_stage[0].item() < LOSS_TOLERANCE
    checks["eval_loss_matches_unfused"] = last_stage[1].item() < LOSS_TOLERANCE
    checks["forward_only_returns_real_logits"] = last_stage[2].item() == 1.0
    checks["loss_is_nonzero"] = last_stage[3].item() > 1e-3
    checks["eval_before_train_loss_matches_unfused"] = last_stage[4].item() < LOSS_TOLERANCE
    checks["eval_before_train_returns_real_logits"] = last_stage[5].item() == 1.0
    metrics["loss_rel_err"] = last_stage[0].item()
    metrics["eval_loss_rel_err"] = last_stage[1].item()
    metrics["eval_before_train_loss_rel_err"] = last_stage[4].item()

    fused_err, fused_worst, n_compared = worst_grad_error(unfused["grads"], fused["grads"])
    noise_err, noise_worst, _ = worst_grad_error(unfused["grads"], control["grads"])
    eval_first_err, eval_first_worst, n_eval_first = worst_grad_error(unfused["grads"], eval_first["grads"])
    log_all(
        f"[rank {ctx.rank} stage {config.pp_rank}] compared={n_compared} "
        f"fused_vs_unfused={fused_err:.3e} @ {fused_worst} | noise_floor={noise_err:.3e} @ {noise_worst} "
        f"| eval_first={eval_first_err:.3e} @ {eval_first_worst}"
    )
    errs = torch.tensor([fused_err, noise_err, eval_first_err], device=device)
    dist.all_reduce(errs, op=dist.ReduceOp.MAX)
    counts = torch.tensor([float(n_compared)], device=device)
    dist.all_reduce(counts, op=dist.ReduceOp.MIN)

    # A vacuous pass (nothing compared, or a parameter left without a gradient) must fail.
    checks["all_params_compared"] = n_compared == unfused["n_params"] == fused["n_params"] and counts.item() > 0
    # Asserted: if the floor reached the tolerance, a passing ``grads_match_unfused`` says only that the runs agree as
    # poorly as one run agrees with itself.
    checks["noise_floor_below_tolerance"] = errs[1].item() < GRAD_TOLERANCE
    checks["grads_match_unfused"] = errs[0].item() < GRAD_TOLERANCE
    checks["eval_first_grads_match_unfused"] = errs[2].item() < GRAD_TOLERANCE and n_eval_first == n_compared
    metrics["grad_rel_err_fused"] = errs[0].item()
    metrics["grad_rel_err_noise_floor"] = errs[1].item()
    metrics["grad_rel_err_eval_first"] = errs[2].item()

    if USE_GC:
        # One call per layer per microbatch forward, plus torch's one-time shape sweep; a shortfall means GC silently
        # skipped.
        floor = fused["n_layers"] * n_microbatches
        log_all(f"[rank {ctx.rank}] gc_calls fused={fused['gc_calls']} unfused={unfused['gc_calls']} floor={floor}")
        engaged = torch.tensor([float(min(fused["gc_calls"], unfused["gc_calls"]))], device=device)
        dist.all_reduce(engaged, op=dist.ReduceOp.MIN)
        checks["gc_engaged_on_both_paths"] = engaged.item() >= floor

    peaks = torch.zeros(2, device=device)
    if config.is_last_pp_stage:
        peaks[0], peaks[1] = unfused["peak"], fused["peak"]
        log_all(
            f"[last stage] peak unfused={unfused['peak'] / 2**20:.1f} MiB "
            f"fused={fused['peak'] / 2**20:.1f} MiB "
            f"saved={(unfused['peak'] - fused['peak']) / 2**20:.1f} MiB"
        )
    dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
    checks["fused_peak_memory_lower"] = peaks[1].item() < peaks[0].item()
    metrics["peak_unfused_mib"] = peaks[0].item() / 2**20
    metrics["peak_fused_mib"] = peaks[1].item() / 2**20
    metrics["peak_saved_mib"] = (peaks[0].item() - peaks[1].item()) / 2**20

    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    if "--gc" in sys.argv:
        USE_GC = True
        sys.argv.remove("--gc")
    sys.exit(run())
