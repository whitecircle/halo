#!/usr/bin/env python
"""PP's forward-only pass returns correct logits without damaging the stage's backward state.

torch latches shape inference and recv buffers on the schedule's first call, taking ``requires_grad`` from
``has_backward`` then, so a forward-only pass that runs first can freeze the boundary activations non-differentiable.
"""

import math
import sys

import torch
import torch.distributed as dist

from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.distributed.parallelism_config import ParallelismConfig
from tests.common import pp_equivalence as ppeq
from tests.common.distributed import world_all
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.numerics import max_abs_rel_err, max_or_nan
from tests.common.rank_log import log

PP_SIZE = 2
GLOBAL_BATCH = 8
SEQ_LEN = 32
N_MICROBATCHES = 2
LOGITS_TOLERANCE = 1e-4
GRAD_TOLERANCE = 2e-3


def worst_over_ranks(value: float, device) -> float:
    t = torch.tensor([value], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


@gpu_test_main(exact_world_size=4, prefix="pp_forward_only")
def run(ctx):
    device = torch.device("cuda", ctx.local_rank)

    # The pipeline crosses the simulated node boundary.
    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE)

    input_ids, labels = ppeq.build_unmasked_batch(TINY_QWEN3_CONFIG["vocab_size"], GLOBAL_BATCH, SEQ_LEN, device)
    _, ref_logits, ref_grads = ppeq.baseline_grads(
        ppeq.build_dense_model(attn_implementation="eager").to(device), input_ids, labels
    )

    model = ppeq.build_dense_model(attn_implementation="eager").to(device)
    stage_module, runner, _ = ppeq.build_runner_stage(
        model, config, ppeq.build_stage_groups(config), device, N_MICROBATCHES
    )

    # Every rank of a chain gets the same slice.
    dp_rank = config.get_data_parallel_rank()
    local_bs = GLOBAL_BATCH // config.data_parallel_size
    sl = slice(dp_rank * local_bs, (dp_rank + 1) * local_bs)

    checks, metrics = {}, {}

    # Before any step: the ordering that corrupts a stage holding stale state.
    logits = runner.forward_only(input_ids[sl])
    # A forward-only drive leaves the FSDP2 stage gathered, and every read below is by identity; the trainer does the
    # same at the end of an eval loop and before building the optimizer.
    reshard_fsdp2_modules(stage_module)

    is_last = config.is_last_pp_stage
    expected_shape = (local_bs, SEQ_LEN, TINY_QWEN3_CONFIG["vocab_size"])
    has_logits = logits is not None and tuple(logits.shape) == expected_shape
    # Vacuously true on other ranks, and reduced over the world so it can't vanish on the rank the harness reports
    # from.
    checks["forward_only_returns_logits_on_last_stage"] = world_all(has_logits or not is_last, device)
    checks["forward_only_returns_none_off_last_stage"] = world_all(is_last or logits is None, device)

    logits_err = max_abs_rel_err(logits, ref_logits[sl]) if is_last and has_logits else 0.0
    if is_last:
        log(f"[rank {ctx.rank}] forward_only logits rel_err={logits_err:.3e}")
    metrics["logits_rel_err"] = worst_over_ranks(logits_err, device)
    checks["forward_only_logits_match_baseline"] = (
        checks["forward_only_returns_logits_on_last_stage"] and metrics["logits_rel_err"] < LOGITS_TOLERANCE
    )

    checks["forward_only_leaves_no_grads"] = world_all(all(p.grad is None for p in stage_module.parameters()), device)

    # State first: without it step() deadlocks (the last stage raises on None input grads while the previous waits on
    # the grad recv).
    backward_capable = config.is_first_pp_stage or all(
        info.buffer.requires_grad for info in runner.stage.args_recv_info[0]
    )
    checks["recv_buffers_stay_backward_capable"] = world_all(backward_capable, device)

    if checks["recv_buffers_stay_backward_capable"]:
        runner.step(input_ids[sl], labels[sl])

        grads = ppeq.gather_stage_grads(stage_module)
        expected = {stage_module.global_parameter_name(n) for n, _ in stage_module.named_parameters()}
        # A vacuous pass is the failure mode of a broken FQN mapping.
        missing = sorted((expected - grads.keys()) | (grads.keys() - ref_grads.keys()))
        errors, worst_name = ppeq.max_abs_grad_errors(grads, ref_grads)
        worst_err = errors.get(worst_name, 0.0)
        log(
            f"[rank {ctx.rank} stage {config.pp_rank}] after forward_only: checked={len(grads)} "
            f"worst_rel_err={worst_err:.3e} @ {worst_name} missing={missing}"
        )
        checks["all_stage_params_compared"] = world_all(len(grads) > 0 and not missing, device)
        metrics["grad_rel_err_after_forward_only"] = worst_over_ranks(worst_err, device)
        checks["grads_match_baseline_after_forward_only"] = metrics["grad_rel_err_after_forward_only"] < GRAD_TOLERANCE

        # The pending gradients live on the shards; a drive that ran FSDP's post_backward or the grad scaling would
        # reduce or halve them.
        runner.forward_only(input_ids[sl])
        reshard_fsdp2_modules(stage_module)
        after = ppeq.gather_stage_grads(stage_module)
        drift = max_or_nan((after[fqn] - grads[fqn]).abs().max().item() for fqn in grads)
        drift = drift if math.isfinite(drift) else math.inf
        metrics["grad_drift_from_forward_only"] = worst_over_ranks(drift, device)
        checks["forward_only_after_step_preserves_grads"] = metrics["grad_drift_from_forward_only"] == 0.0
    else:
        log("stage is not backward-capable after forward_only — skipping step() (it would deadlock)")
        checks["all_stage_params_compared"] = False
        checks["grads_match_baseline_after_forward_only"] = False
        checks["forward_only_after_step_preserves_grads"] = False

    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    sys.exit(run())
