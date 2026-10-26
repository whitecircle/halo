#!/usr/bin/env python
"""Pipeline forward-only pass: correct logits, and no damage to the stage's backward state.

``PipelineRunner.forward_only`` is the scoring path (RL reference log-probs, eval metrics) — no
rank holds the whole model, so it must drive the schedule instead of calling ``model(...)``. torch
defers the stage's shape inference and recv-buffer allocation to the schedule's first call and
latches it, taking ``requires_grad`` from ``has_backward`` at that moment, so a forward-only pass
that runs first can permanently freeze the boundary activations non-differentiable.

It fails when:
  * the forward-only pass raises (torch computes the loss for a target that does not exist)
  * the merged last-stage logits stop matching the unsplit model
  * a forward-only pass leaves the stage unable to produce input gradients — the later ``step()``
    then trains on wrong (or no) gradients
  * a forward-only pass populates or perturbs the shards' ``.grad`` (the runner suppresses torch's
    trailing FSDP post_backward on forward-only drives, leaving the stage gathered; the shards are
    re-registered with ``reshard_fsdp2_modules`` before anything reads parameters by identity)

Topology mirrors ``test_pp_vs_single_gpu.py``: world=4 as 2 simulated nodes of 2 GPUs (NVLink
domain = 2), pipeline crossing the simulated node boundary, FSDP2 inside each stage.

Run: torchrun --nproc_per_node=4 tests/gpu/parallelism/pp/test_pp_forward_only.py
"""

import math
import sys

import torch
import torch.distributed as dist

from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.parallelism_config import ParallelismConfig
from tests.common import pp_equivalence as ppeq
from tests.common.distributed import world_all
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.utils import log, max_abs_rel_err, max_or_nan

PP_SIZE = 2
GLOBAL_BATCH = 8
SEQ_LEN = 32
N_MICROBATCHES = 2
LOGITS_TOLERANCE = 1e-4
GRAD_TOLERANCE = 2e-3


def worst_over_ranks(value: float, device) -> float:
    """The largest error any rank saw (ranks with nothing to compare contribute 0.0)."""
    t = torch.tensor([value], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


@gpu_test_main(exact_world_size=4, prefix="pp_forward_only")
def run(ctx):
    device = torch.device("cuda", ctx.local_rank)

    # 2 simulated nodes of 2 GPUs; the pipeline crosses the simulated node boundary.
    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE)

    input_ids, labels = ppeq.build_unmasked_batch(TINY_QWEN3_CONFIG["vocab_size"], GLOBAL_BATCH, SEQ_LEN, device)
    _, ref_logits, ref_grads = ppeq.baseline_grads(
        ppeq.build_dense_model(attn_implementation="eager").to(device), input_ids, labels
    )

    model = ppeq.build_dense_model(attn_implementation="eager").to(device)
    stage_module, runner, _ = ppeq.build_runner_stage(
        model, config, ppeq.build_stage_groups(config), device, N_MICROBATCHES
    )

    # This DP replica's slice of the global batch; every rank of a chain gets the same slice.
    dp_rank = config.get_data_parallel_rank()
    local_bs = GLOBAL_BATCH // config.data_parallel_size
    sl = slice(dp_rank * local_bs, (dp_rank + 1) * local_bs)

    checks, metrics = {}, {}

    # ── (a) Forward-only before any step: the ordering that corrupts a stage holding stale state ─
    logits = runner.forward_only(input_ids[sl])
    # A forward-only drive leaves the FSDP2 stage gathered (transient params registered); every
    # parameter read below is by identity, so re-register the shards first — the trainer does the
    # same at the end of an eval loop and before building the optimizer.
    reshard_fsdp2_modules(stage_module)

    is_last = config.is_last_pp_stage
    expected_shape = (local_bs, SEQ_LEN, TINY_QWEN3_CONFIG["vocab_size"])
    has_logits = logits is not None and tuple(logits.shape) == expected_shape
    # Each check is vacuously true on the ranks it does not apply to, and reduced over the world so
    # it cannot silently vanish on the rank the harness reports from.
    checks["forward_only_returns_logits_on_last_stage"] = world_all(has_logits or not is_last, device)
    checks["forward_only_returns_none_off_last_stage"] = world_all(is_last or logits is None, device)

    logits_err = max_abs_rel_err(logits, ref_logits[sl]) if is_last and has_logits else 0.0
    if is_last:
        log(f"[rank {ctx.rank}] forward_only logits rel_err={logits_err:.3e}")
    metrics["logits_rel_err"] = worst_over_ranks(logits_err, device)
    checks["forward_only_logits_match_baseline"] = (
        checks["forward_only_returns_logits_on_last_stage"] and metrics["logits_rel_err"] < LOGITS_TOLERANCE
    )

    # ── (c) A scoring pass must not touch gradients ────────────────────────────────────────────
    checks["forward_only_leaves_no_grads"] = world_all(all(p.grad is None for p in stage_module.parameters()), device)

    # ── (b) The stage must still be able to run backward ───────────────────────────────────────
    # Checked as state first: without it the step() below deadlocks (the last stage raises on its
    # None input grads while the previous stage waits forever on the grad recv) instead of failing.
    backward_capable = config.is_first_pp_stage or all(
        info.buffer.requires_grad for info in runner.stage.args_recv_info[0]
    )
    checks["recv_buffers_stay_backward_capable"] = world_all(backward_capable, device)

    if checks["recv_buffers_stay_backward_capable"]:
        runner.step(input_ids[sl], labels[sl])

        grads = ppeq.gather_stage_grads(stage_module)
        expected = {stage_module.global_parameter_name(n) for n, _ in stage_module.named_parameters()}
        # A vacuous pass (nothing compared, or a stage-local name that does not map back into the
        # unsplit model) must fail — it is the failure mode of a broken FQN mapping.
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

        # ── (d) A scoring pass interleaved with training must not disturb pending gradients ────
        # The pending gradients live on the shards; a forward-only drive that ran FSDP's post_backward
        # or the grad scaling would reduce or halve them before the optimizer steps.
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
