#!/usr/bin/env python
"""PP+EP with and without activation checkpointing must produce the same gradients (pp2 × ep2, 4 GPUs).

The two runs differ only in ``gradient_checkpointing``: identical weights, identical batch, the same
real ``DistributedSFTTrainer`` and pipeline schedule. Under checkpointing the stage's decoder layers
run their body a second time inside the schedule-driven backward, and that recompute must replay the
checkpoint frame's DeepEP dispatch/combine — a second dispatch reuses the same ``ElasticBuffer`` and
invalidates the handle the original forward's backward node still holds, which corrupts every
gradient in the stage without raising anything.

Three legs make the comparison honest:

  * leg A — no GC, batch A: the reference gradients;
  * leg B — GC, batch A: must match leg A to :data:`GRAD_REL_TOL` (bf16 recompute is not bitwise),
    with the checkpoint-engagement counter proving the GC gate actually ran;
  * leg C — no GC, batch B: the anti-vacuity control. A different batch must fail the same
    tolerance by a wide margin, so a tolerance that would pass anything is a failing test.

Peak per-rank memory is reported for legs A and B (at this model size the frame caches dominate;
the production memory win is a full-model measurement, not a tiny-model assertion).

Run: torchrun --nproc_per_node=4 tests/gpu/parallelism/pp/test_pp_ep_gc_equivalence.py
"""

import math
import sys

import torch
from torch.distributed.tensor import DTensor

from src.distributed.parallelism_config import ParallelismConfig
from tests.common import pp_equivalence as ppeq
from tests.common.harness import gpu_test_main
from tests.common.tolerances import TOL
from tests.common.utils import cleanup_memory, log

PP_SIZE = 2
EP_SIZE = 2
GLOBAL_BATCH = 4
PER_DEVICE_BATCH = 2
N_MICROBATCHES = 2

# bf16 recompute reproduces the forward to the last bits only; measured worst param 1.1e-3 here.
GRAD_REL_TOL = 5e-3
# The control must miss by far more than the tolerance, or the tolerance proves nothing.
CONTROL_MIN_REL_ERR = 0.1


def run_leg(ctx, config, rows, labels, *, gradient_checkpointing):
    """One PP+EP training step; returns (loss, {global name: grad}, clip norm, peak GB, gc calls)."""
    model = ppeq.build_patched_moe_model(config)
    trainer = ppeq.build_pp_sft_trainer(
        ctx,
        config,
        model,
        rows,
        labels,
        per_device_batch=PER_DEVICE_BATCH,
        n_microbatches=N_MICROBATCHES,
        gradient_checkpointing=gradient_checkpointing,
    )
    stage = trainer._top_level_model()
    calls = ppeq.count_checkpoint_calls(stage) if gradient_checkpointing else [0]

    batch = ppeq.dp_slice_batch(rows, labels, config.get_data_parallel_rank(), PER_DEVICE_BATCH)
    torch.cuda.reset_peak_memory_stats()
    loss = float(trainer.training_step(trainer.model, dict(batch)))
    clip_norm = float(trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), 1e6))
    peak = torch.cuda.max_memory_allocated() / 1e9

    grads = {}
    for local_name, param in stage.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad
        # full_tensor() is collective over the mesh — every rank walks the params in the same order.
        full = grad.full_tensor() if isinstance(grad, DTensor) else grad
        grads[stage.global_parameter_name(local_name)] = full.detach().float().clone()

    trainer.cleanup_ep()
    del trainer, model, stage
    cleanup_memory()
    return loss, grads, clip_norm, peak, calls[0]


def worst_diff(test: dict, reference: dict) -> tuple[float, str, int]:
    """Worst per-parameter relative gradient error and how many parameters were compared."""
    worst, name, compared = 0.0, "", 0
    for key, ref in reference.items():
        if ref.norm() == 0:
            continue
        compared += 1
        err = ppeq.rel_err(test[key], ref)
        # A NaN compares false against every bound, so a non-finite error counts as infinite.
        err = err if math.isfinite(err) else math.inf
        if err > worst:
            worst, name = err, key
    return worst, name, compared


@gpu_test_main(exact_world_size=4, prefix="pp_ep_gc_equivalence")
def run(ctx):
    torch.set_float32_matmul_precision("highest")
    checks, metrics = {}, {}

    config = ParallelismConfig(
        world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE, ep_size=EP_SIZE
    )
    rows_a, labels_a = ppeq.build_ragged_batch(GLOBAL_BATCH)
    # Different tokens, same shapes — the pipeline's P2P buffers stay valid. Reordering the rows
    # would not do: the compared gradients are DP-synced, so a permuted global batch reproduces them.
    rows_b, labels_b = ppeq.build_ragged_batch(GLOBAL_BATCH, seed=99)

    loss_a, grads_a, clip_a, peak_a, _ = run_leg(ctx, config, rows_a, labels_a, gradient_checkpointing=False)
    loss_b, grads_b, clip_b, peak_b, gc_calls = run_leg(ctx, config, rows_a, labels_a, gradient_checkpointing=True)
    loss_c, grads_c, _, _, _ = run_leg(ctx, config, rows_b, labels_b, gradient_checkpointing=False)

    worst, worst_name, compared = worst_diff(grads_b, grads_a)
    control, control_name, _ = worst_diff(grads_c, grads_a)
    log(f"[rank {ctx.rank}] gc-vs-nogc worst {worst:.3e} @{worst_name}; control {control:.3e} @{control_name}")

    checks["gc_engaged"] = gc_calls > 0
    checks["grads_match_without_gc"] = worst < GRAD_REL_TOL
    checks["every_param_compared"] = compared == len(grads_a) and compared > 0
    checks["control_different_batch_exceeds_tol"] = control > CONTROL_MIN_REL_ERR
    checks["loss_matches_without_gc"] = abs(loss_b - loss_a) < TOL.gc_recompute_loss_rel * max(abs(loss_a), 1.0)
    checks["clip_norm_matches_without_gc"] = abs(clip_b - clip_a) < GRAD_REL_TOL * clip_a
    checks["control_loss_differs"] = abs(loss_c - loss_a) > TOL.gc_recompute_loss_rel * abs(loss_a)

    metrics.update(
        worst_grad_rel_err_gc_vs_nogc=worst,
        control_worst_grad_rel_err=control,
        params_compared=compared,
        gc_calls=gc_calls,
        loss_no_gc=loss_a,
        loss_gc=loss_b,
        clip_norm_no_gc=clip_a,
        clip_norm_gc=clip_b,
        peak_gb_no_gc=peak_a,
        peak_gb_gc=peak_b,
    )
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    sys.exit(run())
