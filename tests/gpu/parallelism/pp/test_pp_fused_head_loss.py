#!/usr/bin/env python
"""Fused last-stage head+loss gate: same gradients as the logits path, at a lower memory peak.

Under PP the last stage normally returns ``[mb, S, V]`` logits for the schedule's ``loss_fn`` to
consume, so the plane and its gradient must both exist. ``fused_head_loss`` instead computes head +
cross-entropy inside the stage's forward, one token chunk at a time under a checkpoint, and returns
the summed loss. This test runs the same model, seed and batch both ways and compares them.

What it would catch:
  * a chunk boundary that drops or double-counts tokens (loss and gradients diverge)
  * the step normalizer applied per microbatch instead of per step (gradients scale by 1/n)
  * the head's gradient not accumulating across chunks, or FSDP2 reducing it more than once
  * ``forward_only`` silently returning the fused scalar instead of real logits
  * a fused path that saves no memory
  * a fused stage whose output form or recv buffers were latched by an eval-first call order — the
    order HF actually runs, and the one the zero-label placeholder in ``forward_only`` exists for

A third run repeats the unfused path to measure the run-to-run noise floor of this hardware
(embedding/attention backward atomics). It is asserted, not just logged: the gradient verdict below
is a fixed tolerance, and that tolerance only means something while identical code reproduces itself
well inside it.

A fourth run is the fused path driven eval-before-train. Shape inference latches the stage's output
form and the backward-capability of its recv buffers on whichever call comes first, so this is a
different code path from the other three, not a repeat of them.

The CE chunk budget is shrunk from its production value so a 1024-token microbatch spans several
chunks — at ``LOGIT_CHUNK_ELEMENTS`` a tiny test would run one chunk and never test a boundary. Both
paths read the same budget, so the CE reduction tree is identical between them and the only
remaining difference is where the head projection is sliced.

Topology: world=4 as 2 simulated nodes of 2 GPUs, pp2 x dp2 — same placement as
``test_pp_vs_single_gpu``, which owns the PP-vs-unsplit equivalence this test builds on.

Run: torchrun --nproc_per_node=4 tests/gpu/parallelism/pp/test_pp_fused_head_loss.py [--gc]
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

# A large vocabulary over a tiny backbone is the whole point: the logits plane must dominate the
# model so the memory difference is unambiguous, and the head GEMM must be wide enough that slicing
# it is a real test. 65536 x 1024 tokens = 256 MB per fp32 plane, against a ~2 MB backbone.
FUSED_LOSS_CONFIG = {
    **TINY_QWEN3_CONFIG,
    "vocab_size": 65536,
    "hidden_size": 256,
    "num_hidden_layers": 4,
    "max_position_embeddings": 2048,
}
SEQ_LEN = 1024
# One row per microbatch — the production shape at long context, where a microbatch is a single
# sequence and the logits plane it would build is the whole memory problem. The microbatch count
# follows the launch (4 GPUs -> pp2, 8 -> pp4, which Schedule1F1B needs a full pipeline for).
MIN_MICROBATCHES = 2
# 4 CE chunks per microbatch at V=65536 (16M / 65536 = 256 rows of 1024).
CE_CHUNK_ELEMENTS = 16 * 1024 * 1024

# fp32 end to end, so the comparison is limited by reduction order rather than by bf16 rounding.
LOSS_TOLERANCE = 1e-6
GRAD_TOLERANCE = 1e-5

# --gc: repeat with non-reentrant activation checkpointing on the decoder layers. The fused loss adds
# its own checkpoint segments around head+CE, and this proves the two nest without disturbing either.
USE_GC = False


def build_model(device):
    """Identical weights on every rank and on every run — the two paths must start equal."""
    torch.manual_seed(1234)
    config = Qwen3Config(**FUSED_LOSS_CONFIG, attn_implementation="eager")
    return Qwen3ForCausalLM(config).to(device=device, dtype=torch.float32)


def run_once(ctx, config, groups, input_ids, labels, n_microbatches, *, fused, eval_first=False):
    """One full step through a freshly built pipeline; returns loss, grads (on CPU), and peak memory.

    Everything is rebuilt per run because the stage latches its output metadata at the first step —
    a fused stage records a scalar, an unfused one records the logits plane — so one stage cannot be
    flipped between the two forms after the fact.
    """
    device = torch.device("cuda", ctx.local_rank)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # The GC counter proves the checkpoint path actually ran: the silent failure is HF's training/no-grad
    # gate skipping it.
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

    # ``eval_first`` puts the scoring passes before any training step, which is the order HF runs
    # (``evaluate()`` at step 0) and the only one that reaches the fused runner's shape-inference
    # workaround: ``forward_only`` carries no labels of its own, so it injects a zero placeholder to
    # make the stage latch its fused scalar output form. Without that, the first real ``step`` would
    # return a scalar against latched logits metadata — and the recv buffers would have been latched
    # by a backward-less pass, leaving every later step with no input gradient to send upstream.
    if eval_first:
        pre_eval_loss = runner.eval_loss(input_ids, labels)
        pre_logits = runner.forward_only(input_ids)

    loss = runner.step(input_ids, labels)
    # Read the peak before gathering gradients: full_tensor() and the CPU copies below are the
    # test's own bookkeeping, identical for both paths, and would blur the number under measurement.
    peak = torch.cuda.max_memory_allocated()

    # Loss-only eval and forward-only scoring both run through the fused stage; the second must give
    # back real logits (the trainer's compute_metrics path depends on it).
    eval_loss = runner.eval_loss(input_ids, labels)
    logits = runner.forward_only(input_ids)
    if eval_first:
        # The pre-step passes must agree with the post-step ones: nothing was trained in between, so
        # a difference means the latched form, not the weights, decided the answer.
        assert pre_logits is None or torch.equal(pre_logits, logits)
        eval_loss = pre_eval_loss

    # The scoring passes above ran forward-only, which leaves FSDP2's transient unsharded params
    # registered — and those carry no ``.grad``, because the gradients live on the sharded DTensors
    # the step reduced into. Reading ``named_parameters()`` without resharding first collects
    # nothing at all, which ``all_params_compared`` below is what catches. Production does the same
    # thing at every state-dict boundary, for the same reason.
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
        # A digest, not the plane: holding 256 MB per run would distort the later runs' peaks. The
        # two forward-only passes execute identical code, so their digests must agree exactly.
        "logits_digest": logits.sum(dtype=torch.float64).item() if logits is not None else 0.0,
        "logits_slice": logits[0, ::128, ::4096].double().cpu() if logits is not None else None,
    }
    del runner, stage_module, loss, eval_loss, logits
    gc.collect()
    torch.cuda.empty_cache()
    return result


def worst_grad_error(reference: dict, other: dict) -> tuple[float, str | None, int]:
    """Worst relative max-abs gradient error between two runs, its parameter, and the count compared.

    A parameter present in one run and not the other is reported as an infinite error rather than
    skipped — a silently narrowed comparison set is the failure this gate exists to catch.
    """
    absent = next((name for name in reference if name not in other), None)
    if absent is not None:
        return math.inf, absent, 0
    errors, worst_name = ppeq.max_abs_grad_errors(other, reference)
    return errors.get(worst_name, 0.0), worst_name, len(errors)


@gpu_test_main(min_world_size=4, prefix="pp_fused_head_loss")
def run(ctx):
    device = torch.device("cuda", ctx.local_rank)
    # Both paths must chunk on the same boundaries, so the only difference under test is the head.
    logprob_kernels.LOGIT_CHUNK_ELEMENTS = CE_CHUNK_ELEMENTS

    pp_size = ctx.world_size // 2
    n_microbatches = max(MIN_MICROBATCHES, pp_size)
    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=pp_size)
    groups = ppeq.build_stage_groups(config)

    input_ids, labels = ppeq.build_unmasked_batch(FUSED_LOSS_CONFIG["vocab_size"], n_microbatches, SEQ_LEN, device)
    # An ignored span in every row: the chunk loop must skip those targets, not renormalize around
    # them. Placed mid-sequence so it straddles a chunk boundary rather than sitting inside one.
    labels[:, SEQ_LEN // 2 - 8 : SEQ_LEN // 2 + 8] = -100
    unfused = run_once(ctx, config, groups, input_ids, labels, n_microbatches, fused=False)
    control = run_once(ctx, config, groups, input_ids, labels, n_microbatches, fused=False)
    fused = run_once(ctx, config, groups, input_ids, labels, n_microbatches, fused=True)
    eval_first = run_once(ctx, config, groups, input_ids, labels, n_microbatches, fused=True, eval_first=True)

    checks, metrics = {}, {}

    # --- Loss, eval loss and forward-only logits: last stage only, asserted on every rank. ---
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
        last_stage[3] = abs(unfused["loss"])  # a zero loss would make the ratios above vacuous
        # The eval-first run scored before its first step, so its eval loss is the untrained model's
        # — which is what the unfused reference measures too. A stage that latched the wrong output
        # form would not get this far; one that latched a non-differentiable recv buffer produces no
        # gradient for the ``eval_first_grads_match_unfused`` comparison above.
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

    # --- Gradients: the real gate. Every stage compares its own parameters. ---
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
    # The control run is the same path twice, so its error is pure reduction-order noise — and it is
    # a check, not a logged number: if the floor ever reached the tolerance, a passing
    # ``grads_match_unfused`` would only be saying the two runs agree as poorly as one run agrees
    # with itself. Asserting it is what makes the constant below mean something.
    checks["noise_floor_below_tolerance"] = errs[1].item() < GRAD_TOLERANCE
    checks["grads_match_unfused"] = errs[0].item() < GRAD_TOLERANCE
    checks["eval_first_grads_match_unfused"] = errs[2].item() < GRAD_TOLERANCE and n_eval_first == n_compared
    metrics["grad_rel_err_fused"] = errs[0].item()
    metrics["grad_rel_err_noise_floor"] = errs[1].item()
    metrics["grad_rel_err_eval_first"] = errs[2].item()

    if USE_GC:
        # One checkpointed call per layer per microbatch forward, plus torch's one-time shape
        # inference sweep; a shortfall means GC silently skipped and this degraded to a plain re-run.
        floor = fused["n_layers"] * n_microbatches
        log_all(f"[rank {ctx.rank}] gc_calls fused={fused['gc_calls']} unfused={unfused['gc_calls']} floor={floor}")
        engaged = torch.tensor([float(min(fused["gc_calls"], unfused["gc_calls"]))], device=device)
        dist.all_reduce(engaged, op=dist.ReduceOp.MIN)
        checks["gc_engaged_on_both_paths"] = engaged.item() >= floor

    # --- Memory: the fused path must lower the peak. ---
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
