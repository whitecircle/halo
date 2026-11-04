#!/usr/bin/env python
"""PP + FSDP2 reproduces single-GPU loss and every gradient on a tiny dense Qwen3, over simulated nodes the pipeline
crosses."""

import sys

import torch

from halo.distributed.parallelism_config import ParallelismConfig
from halo.trainers.parallelism.grad_clipping import pp_clip_grad_norm
from tests.common import pp_equivalence as ppeq
from tests.common.distributed import world_min
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.rank_log import log
from tests.common.tolerances import TOL

# 4 GPUs -> pp2, 8 GPUs -> pp4, whose two middle stages (no embedding, no loss) and 3-hop chains nothing else
# exercises.
GLOBAL_BATCH = 8
GRAD_TOLERANCE = 2e-3

# --gc: non-reentrant activation checkpointing, numerically exact; an engagement counter proves it ran, since HF's
# training/no-grad gate can silently skip it.
USE_GC = False
# --pp_split lo,hi...: manual per-stage layer counts, proving the partition machinery isn't just 4/4-correct.
PP_SPLIT = None
# --schedule gpipe: same equivalence under ScheduleGPipe, which no other GPU test runs.
SCHEDULE = "1f1b"


@gpu_test_main(min_world_size=4, prefix="pp_vs_single_gpu")
def run(ctx):
    device = torch.device("cuda", ctx.local_rank)

    # The pipeline crosses every simulated node boundary.
    pp_size = ctx.world_size // 2
    n_microbatches = max(2, pp_size)  # Schedule1F1B needs a full pipeline.
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

    # Every rank of a chain gets the same slice.
    dp_rank = config.get_data_parallel_rank()
    local_bs = GLOBAL_BATCH // config.data_parallel_size
    sl = slice(dp_rank * local_bs, (dp_rank + 1) * local_bs)
    loss = runner.step(input_ids[sl], labels[sl])

    checks, metrics = ppeq.check_loss_vs_baseline(loss, ref_loss, config, groups, tol=1e-4)

    # Through the production name remap, so a regression in it fails this test too.
    errors, worst_name = ppeq.max_abs_grad_errors(ppeq.gather_stage_grads(stage_module), ref_grads)
    stage_names = {stage_module.global_parameter_name(n) for n, _ in stage_module.named_parameters()}
    missing = sorted(stage_names - errors.keys())
    worst_err = errors.get(worst_name, 0.0)
    log(
        f"[rank {ctx.rank} stage {config.pp_rank}] checked={len(errors)} "
        f"worst_rel_err={worst_err:.3e} @ {worst_name} missing={missing}"
    )

    # A vacuous pass is the failure mode of a broken FQN mapping.
    checks["all_stage_params_compared"] = bool(errors) and not missing
    checks["grads_match_baseline"] = bool(errors) and worst_err < GRAD_TOLERANCE
    metrics["worst_grad_rel_err"] = worst_err
    metrics["params_compared"] = len(errors)

    # Or the split silently produced empty stages.
    checks["every_stage_has_params"] = world_min(len(errors), device) > 0

    # Every other call site uses max_norm=1e6, so a wrong coefficient never fails a test: clip to half, then max_norm=0
    # must disable. After the grad comparisons, since it scales the gradients.
    params = list(stage_module.parameters())
    norm_full = pp_clip_grad_norm(params, 1e9)
    pp_clip_grad_norm(params, float(norm_full) / 2)
    norm_after_zero_disable = pp_clip_grad_norm(params, 0)  # Must not scale.
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
