#!/usr/bin/env python
"""Pipeline parallelism equivalence gate: PP + FSDP2 must reproduce single-GPU gradients.

The base PP correctness gate. It splits a tiny dense Qwen3 across 2 pipeline stages with FSDP2
data parallelism inside each stage, and compares both the loss and every parameter gradient against
an unsplit single-process baseline over the same global batch.

It fails when any of the load-bearing pieces breaks:
  * layer partition / stage slicing puts a layer on the wrong stage
  * the final norm is applied mid-network (non-last stages must neutralize it)
  * loss normalization or ``scale_grads`` mis-scales gradients by ``n_microbatches``
  * the FSDP2 per-step grad reduction inside torch's schedule stops running
  * boundary activations lose their gradient path

Topology: world=4 run as 2 simulated nodes of 2 GPUs (``gpus_per_node``/NVLink domain = 2) with the
pipeline crossing the simulated node boundary — the placement PP is designed for, so the stage
groups exercise the cross-domain path rather than a single-node shortcut.

Run: torchrun --nproc_per_node=4 tests/gpu/parallelism/pp/test_pp_vs_single_gpu.py
"""

import sys

import torch

from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.parallelism.grad_clipping import pp_clip_grad_norm
from tests.common import pp_equivalence as ppeq
from tests.common.distributed import world_min
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.tolerances import TOL
from tests.common.utils import log

# Depth scales with the launch: 4 GPUs -> pp2 (first+last only), 8 GPUs -> pp4, whose two middle
# stages (no embedding, no loss, pure hidden->hidden) and 3-hop chains nothing else exercises.
GLOBAL_BATCH = 8
GRAD_TOLERANCE = 2e-3

# --gc: run the split model with non-reentrant activation checkpointing enabled. Checkpointing is
# numerically exact, so the same golden gradients must reproduce; an engagement counter proves the
# checkpoint path actually ran (the silent failure mode is HF's training/no-grad gate skipping it).
USE_GC = False
# --pp_split lo,hi...: manual per-stage layer counts — proves the non-uniform partition machinery
# (global naming, boundary activations, normalizer) is partition-agnostic, not just 4/4-correct.
PP_SPLIT = None
# --schedule gpipe: same equivalence under ScheduleGPipe, which no other GPU test runs.
SCHEDULE = "1f1b"


@gpu_test_main(min_world_size=4, prefix="pp_vs_single_gpu")
def run(ctx):
    device = torch.device("cuda", ctx.local_rank)

    # world/2 simulated nodes of 2 GPUs; the pipeline crosses every simulated node boundary.
    pp_size = ctx.world_size // 2
    n_microbatches = max(2, pp_size)  # Schedule1F1B needs a full pipeline
    config = ParallelismConfig(
        world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=pp_size, pp_schedule=SCHEDULE
    )
    assert config.stage_world_size == 2, config.stage_world_size
    assert config.pp_rank == ctx.rank // 2, (config.pp_rank, ctx.rank)

    input_ids, labels = ppeq.build_unmasked_batch(TINY_QWEN3_CONFIG["vocab_size"], GLOBAL_BATCH, device=device)
    model = ppeq.build_dense_model(attn_implementation="eager").to(device)
    ref_loss, _, ref_grads = ppeq.baseline_grads(model, input_ids, labels)
    log(f"[baseline] loss={ref_loss.item():.8f} over {len(ref_grads)} params")

    groups = ppeq.build_stage_groups(config)
    model = ppeq.build_dense_model(attn_implementation="eager").to(device)
    stage_module, runner, gc_calls = ppeq.build_runner_stage(
        model, config, groups, device, n_microbatches, gc=USE_GC, pp_split=PP_SPLIT
    )
    if PP_SPLIT is not None:
        lo, hi = stage_module.layer_range
        assert hi - lo == PP_SPLIT[config.pp_rank], (PP_SPLIT, config.pp_rank, (lo, hi))

    # This DP replica's slice of the global batch; every rank of a chain gets the same slice.
    dp_rank = config.get_data_parallel_rank()
    local_bs = GLOBAL_BATCH // config.data_parallel_size
    sl = slice(dp_rank * local_bs, (dp_rank + 1) * local_bs)
    loss = runner.step(input_ids[sl], labels[sl])

    # Average the last stage's per-replica losses over DP -> the full-batch loss.
    checks, metrics = ppeq.check_loss_vs_baseline(loss, ref_loss, config, groups, tol=1e-4)

    # Gradient check: every stage-local parameter against its counterpart in the unsplit model, through
    # the production name remap, so a regression in it fails this test too.
    errors, worst_name = ppeq.max_abs_grad_errors(ppeq.gather_stage_grads(stage_module), ref_grads)
    stage_names = {stage_module.global_parameter_name(n) for n, _ in stage_module.named_parameters()}
    missing = sorted(stage_names - errors.keys())
    worst_err = errors.get(worst_name, 0.0)
    log(
        f"[rank {ctx.rank} stage {config.pp_rank}] checked={len(errors)} "
        f"worst_rel_err={worst_err:.3e} @ {worst_name} missing={missing}"
    )

    # A vacuous pass (nothing compared) must fail: it is the failure mode of a broken FQN mapping.
    checks["all_stage_params_compared"] = bool(errors) and not missing
    checks["grads_match_baseline"] = bool(errors) and worst_err < GRAD_TOLERANCE
    metrics["worst_grad_rel_err"] = worst_err
    metrics["params_compared"] = len(errors)

    # Every stage must have contributed some parameters, or the split silently produced empty stages.
    checks["every_stage_has_params"] = world_min(len(errors), device) > 0

    # The clip must actually clip (every other call site uses max_norm=1e6, so a wrong coefficient
    # or a missed plain-vs-DTensor scale would never fail a test): clip to half the current norm,
    # verify the achieved norm halves; then max_norm=0 must disable (not zero the gradients).
    # Runs after the grad comparisons — it scales the gradients.
    params = list(stage_module.parameters())
    norm_full = pp_clip_grad_norm(params, 1e9)
    pp_clip_grad_norm(params, float(norm_full) / 2)
    norm_after_zero_disable = pp_clip_grad_norm(params, 0)  # must not scale
    norm_final = pp_clip_grad_norm(params, 1e9)
    halved_gap = abs(float(norm_final) - float(norm_full) / 2)
    checks["clip_halves_the_norm"] = halved_gap < TOL.clip_achieved_norm_rel * float(norm_full)
    checks["clip_zero_max_norm_disables"] = abs(float(norm_after_zero_disable) - float(norm_final)) < 1e-6 * max(
        1.0, float(norm_final)
    )

    if USE_GC:
        c, m = ppeq.check_gc_engaged(stage_module, gc_calls, n_microbatches)
        checks.update(c)
        metrics.update(m)

    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    if "--gc" in sys.argv:
        USE_GC = True
        sys.argv.remove("--gc")
    if "--schedule" in sys.argv:
        i = sys.argv.index("--schedule")
        SCHEDULE = sys.argv[i + 1]
        del sys.argv[i : i + 2]
    if "--pp_split" in sys.argv:
        i = sys.argv.index("--pp_split")
        PP_SPLIT = [int(part) for part in sys.argv[i + 1].split(",")]
        del sys.argv[i : i + 2]
    sys.exit(run())
