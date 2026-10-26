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
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.pipeline_parallel.groups import build_pipeline_group
from src.distributed.pipeline_parallel.runner import PipelineRunner
from src.distributed.pipeline_parallel.stage import build_pipeline_stage
from src.distributed.pipeline_parallel.stage_losses import TokenLoss
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.utils import log, max_or_nan

PP_SIZE = 2
GLOBAL_BATCH = 8
SEQ_LEN = 32
N_MICROBATCHES = 2
LOGITS_TOLERANCE = 1e-4
GRAD_TOLERANCE = 2e-3


def build_model(device):
    """Identical weights on every rank — the baseline and the split model must start equal."""
    torch.manual_seed(1234)
    config = Qwen3Config(**TINY_QWEN3_CONFIG, attn_implementation="eager")
    return Qwen3ForCausalLM(config).to(device=device, dtype=torch.float32)


def make_batch(device, vocab_size):
    g = torch.Generator(device="cpu").manual_seed(99)
    input_ids = torch.randint(0, vocab_size, (GLOBAL_BATCH, SEQ_LEN), generator=g).to(device)
    return input_ids, input_ids.clone()


def causal_lm_loss(logits, labels):
    """The same shifted, token-count-normalized CE the pipeline runner computes."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    n_tokens = (shift_labels != -100).sum().clamp(min=1)
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss / n_tokens


def baseline(device, input_ids, labels):
    """Unsplit model, full global batch: the reference logits and per-FQN gradients."""
    model = build_model(device)
    logits = model(input_ids=input_ids).logits
    causal_lm_loss(logits, labels).backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    logits = logits.detach().clone()
    del model
    torch.cuda.empty_cache()
    return logits, grads


def stage_grads(stage_module):
    """This stage's gradients as plain tensors, keyed by their FQN in the unsplit model."""
    grads = {}
    for name, param in stage_module.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad
        grads[stage_module.global_parameter_name(name)] = (
            grad.full_tensor() if isinstance(grad, DTensor) else grad
        ).clone()
    return grads


def all_ranks(flag: bool, device) -> bool:
    """True only when ``flag`` holds on every rank — a per-rank check must not vanish on rank 0."""
    t = torch.tensor([1.0 if flag else 0.0], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return t.item() > 0


def worst_over_ranks(value: float, device) -> float:
    """The largest error any rank saw (ranks with nothing to compare contribute 0.0)."""
    t = torch.tensor([value], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def rel_err(actual, ref) -> float:
    return ((actual - ref).abs().max() / ref.abs().max().clamp_min(1e-12)).item()


@gpu_test_main(exact_world_size=4, prefix="pp_forward_only")
def run(ctx):
    device = torch.device("cuda", ctx.local_rank)

    # 2 simulated nodes of 2 GPUs; the pipeline crosses the simulated node boundary.
    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE)

    input_ids, labels = make_batch(device, TINY_QWEN3_CONFIG["vocab_size"])
    ref_logits, ref_grads = baseline(device, input_ids, labels)

    mesh = init_device_mesh("cuda", (PP_SIZE, config.stage_world_size), mesh_dim_names=("pp", "dp"))
    dp_mesh = mesh["dp"]
    pp_group = build_pipeline_group(config)

    model = build_model(device)
    stage_module = build_pipeline_stage(model, config.pp_rank, PP_SIZE)
    for layer in stage_module.model.layers:
        fully_shard(layer, mesh=dp_mesh, reshard_after_forward=False)
    fully_shard(stage_module, mesh=dp_mesh, reshard_after_forward=False)

    runner = PipelineRunner(stage_module, config, pp_group, device, N_MICROBATCHES, TokenLoss())

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
    checks["forward_only_returns_logits_on_last_stage"] = all_ranks(has_logits or not is_last, device)
    checks["forward_only_returns_none_off_last_stage"] = all_ranks(is_last or logits is None, device)

    logits_err = rel_err(logits.float(), ref_logits[sl]) if is_last and has_logits else 0.0
    if is_last:
        log(f"[rank {ctx.rank}] forward_only logits rel_err={logits_err:.3e}")
    metrics["logits_rel_err"] = worst_over_ranks(logits_err, device)
    checks["forward_only_logits_match_baseline"] = (
        checks["forward_only_returns_logits_on_last_stage"] and metrics["logits_rel_err"] < LOGITS_TOLERANCE
    )

    # ── (c) A scoring pass must not touch gradients ────────────────────────────────────────────
    checks["forward_only_leaves_no_grads"] = all_ranks(all(p.grad is None for p in stage_module.parameters()), device)

    # ── (b) The stage must still be able to run backward ───────────────────────────────────────
    # Checked as state first: without it the step() below deadlocks (the last stage raises on its
    # None input grads while the previous stage waits forever on the grad recv) instead of failing.
    backward_capable = config.is_first_pp_stage or all(
        info.buffer.requires_grad for info in runner.stage.args_recv_info[0]
    )
    checks["recv_buffers_stay_backward_capable"] = all_ranks(backward_capable, device)

    if checks["recv_buffers_stay_backward_capable"]:
        runner.step(input_ids[sl], labels[sl])

        grads = stage_grads(stage_module)
        expected = {stage_module.global_parameter_name(n) for n, _ in stage_module.named_parameters()}
        # A vacuous pass (nothing compared, or a stage-local name that does not map back into the
        # unsplit model) must fail — it is the failure mode of a broken FQN mapping.
        missing = sorted((expected - grads.keys()) | (grads.keys() - ref_grads.keys()))
        worst_name, worst_err = None, 0.0
        for fqn, grad in grads.items():
            if fqn not in ref_grads:
                continue
            err = rel_err(grad, ref_grads[fqn])
            # A NaN compares false against every bound, so a non-finite error counts as infinite.
            err = err if math.isfinite(err) else math.inf
            if err > worst_err:
                worst_name, worst_err = fqn, err
        log(
            f"[rank {ctx.rank} stage {config.pp_rank}] after forward_only: checked={len(grads)} "
            f"worst_rel_err={worst_err:.3e} @ {worst_name} missing={missing}"
        )
        checks["all_stage_params_compared"] = all_ranks(len(grads) > 0 and not missing, device)
        metrics["grad_rel_err_after_forward_only"] = worst_over_ranks(worst_err, device)
        checks["grads_match_baseline_after_forward_only"] = metrics["grad_rel_err_after_forward_only"] < GRAD_TOLERANCE

        # ── (d) A scoring pass interleaved with training must not disturb pending gradients ────
        # The pending gradients live on the shards; a forward-only drive that ran FSDP's post_backward
        # or the grad scaling would reduce or halve them before the optimizer steps.
        runner.forward_only(input_ids[sl])
        reshard_fsdp2_modules(stage_module)
        after = stage_grads(stage_module)
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
