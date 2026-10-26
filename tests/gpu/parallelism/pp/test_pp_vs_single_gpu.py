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

import math
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.pipeline_parallel.groups import build_pipeline_group
from src.distributed.pipeline_parallel.runner import PipelineRunner
from src.distributed.pipeline_parallel.stage import build_pipeline_stage
from src.distributed.pipeline_parallel.stage_losses import TokenLoss
from src.trainers.mixins.pipeline import PipelineTrainerMixin
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.tolerances import TOL
from tests.common.utils import log

# Depth scales with the launch: 4 GPUs -> pp2 (first+last only), 8 GPUs -> pp4, whose two middle
# stages (no embedding, no loss, pure hidden->hidden) and 3-hop chains nothing else exercises.
GLOBAL_BATCH = 8
SEQ_LEN = 32
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


def build_model(device):
    """Identical weights on every rank — the baseline and the split model must start equal."""
    torch.manual_seed(1234)
    config = Qwen3Config(**TINY_QWEN3_CONFIG, attn_implementation="eager")
    return Qwen3ForCausalLM(config).to(device=device, dtype=torch.float32)


def make_batch(device, vocab_size):
    g = torch.Generator(device="cpu").manual_seed(99)
    input_ids = torch.randint(0, vocab_size, (GLOBAL_BATCH, SEQ_LEN), generator=g).to(device)
    # No ignore_index anywhere: every DP replica then holds the same token count, so the
    # per-replica normalization the runner applies averages to the exact full-batch loss.
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


def baseline_grads(device, input_ids, labels):
    """Unsplit model, full global batch: the reference loss and per-FQN gradients."""
    model = build_model(device)
    loss = causal_lm_loss(model(input_ids=input_ids).logits, labels)
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    del model
    torch.cuda.empty_cache()
    return loss.detach(), grads


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

    input_ids, labels = make_batch(device, TINY_QWEN3_CONFIG["vocab_size"])
    ref_loss, ref_grads = baseline_grads(device, input_ids, labels)
    log(f"[baseline] loss={ref_loss.item():.8f} over {len(ref_grads)} params")

    mesh = init_device_mesh("cuda", (pp_size, config.stage_world_size), mesh_dim_names=("pp", "dp"))
    dp_mesh = mesh["dp"]
    pp_group = build_pipeline_group(config)

    model = build_model(device)
    stage_module = build_pipeline_stage(model, config.pp_rank, pp_size, pp_split=PP_SPLIT)
    if PP_SPLIT is not None:
        lo, hi = stage_module.layer_range
        assert hi - lo == PP_SPLIT[config.pp_rank], (PP_SPLIT, config.pp_rank, (lo, hi))

    gc_calls = [0]
    if USE_GC:
        # Same order as production: TRL enables GC in __init__ before the stage FSDP wrap.
        stage_module.gradient_checkpointing_enable()
        for layer in stage_module.model.layers:
            inner_fn = layer._gradient_checkpointing_func

            def counting_fn(*args, _inner=inner_fn, **kw):
                gc_calls[0] += 1
                return _inner(*args, **kw)

            layer._gradient_checkpointing_func = counting_fn

    for layer in stage_module.model.layers:
        fully_shard(layer, mesh=dp_mesh, reshard_after_forward=False)
    fully_shard(stage_module, mesh=dp_mesh, reshard_after_forward=False)

    runner = PipelineRunner(stage_module, config, pp_group, device, n_microbatches, TokenLoss())

    # This DP replica's slice of the global batch; every rank of a chain gets the same slice.
    dp_rank = config.get_data_parallel_rank()
    local_bs = GLOBAL_BATCH // config.data_parallel_size
    sl = slice(dp_rank * local_bs, (dp_rank + 1) * local_bs)
    loss = runner.step(input_ids[sl], labels[sl])

    # Loss check: average the last stage's per-replica losses over DP -> the full-batch loss. Only
    # last-stage ranks hold a loss, so the result is reduced over the world (non-last ranks
    # contribute 0) and every rank asserts it — otherwise the check silently vanishes on the rank
    # the harness reports from.
    checks, metrics = {}, {}
    loss_err_t = torch.zeros(1, device=device)
    if config.is_last_pp_stage:
        mean_loss = loss.detach().clone()
        dist.all_reduce(mean_loss, op=dist.ReduceOp.AVG, group=dp_mesh.get_group())
        loss_err_t[0] = abs(mean_loss.item() - ref_loss.item()) / abs(ref_loss.item())
        log(f"[pp] loss={mean_loss.item():.8f} baseline={ref_loss.item():.8f} rel_err={loss_err_t.item():.3e}")
    dist.all_reduce(loss_err_t, op=dist.ReduceOp.MAX)
    checks["loss_matches_baseline"] = loss_err_t.item() < 1e-4
    metrics["loss_rel_err"] = loss_err_t.item()

    # Gradient check: every stage-local parameter against its counterpart in the unsplit model.
    worst_name, worst_err, n_checked, missing = None, 0.0, 0, []
    for name, param in stage_module.named_parameters():
        # Uses the production remap, so a regression in it fails this test too.
        fqn = stage_module.global_parameter_name(name)
        if fqn not in ref_grads:
            missing.append(f"{name} -> {fqn}")
            continue
        if param.grad is None:
            missing.append(f"{name} (grad is None)")
            continue
        grad = param.grad
        grad = grad.full_tensor() if isinstance(grad, torch.distributed.tensor.DTensor) else grad
        ref = ref_grads[fqn]
        err = ((grad - ref).abs().max() / ref.abs().max().clamp_min(1e-12)).item()
        n_checked += 1
        # A NaN compares false against every bound, so a non-finite error counts as infinite.
        err = err if math.isfinite(err) else math.inf
        if err > worst_err:
            worst_name, worst_err = fqn, err

    log(
        f"[rank {ctx.rank} stage {config.pp_rank}] checked={n_checked} "
        f"worst_rel_err={worst_err:.3e} @ {worst_name} missing={missing}"
    )

    # A vacuous pass (nothing compared) must fail: it is the failure mode of a broken FQN mapping.
    expected = len(list(stage_module.named_parameters()))
    checks["all_stage_params_compared"] = n_checked == expected and not missing
    checks["grads_match_baseline"] = n_checked > 0 and worst_err < GRAD_TOLERANCE
    metrics["worst_grad_rel_err"] = worst_err
    metrics["params_compared"] = n_checked

    # Every stage must have contributed some parameters, or the split silently produced empty stages.
    stage_param_count = torch.tensor([float(n_checked)], device=device)
    dist.all_reduce(stage_param_count, op=dist.ReduceOp.MIN)
    checks["every_stage_has_params"] = stage_param_count.item() > 0

    # The clip must actually clip (every other call site uses max_norm=1e6, so a wrong coefficient
    # or a missed plain-vs-DTensor scale would never fail a test): clip to half the current norm,
    # verify the achieved norm halves; then max_norm=0 must disable (not zero the gradients).
    # Runs after the grad comparisons — it scales the gradients. The unbound-call binding exercises
    # the production method itself.
    clip = PipelineTrainerMixin._pp_clip_grad_norm
    holder = object.__new__(PipelineTrainerMixin)
    params = list(stage_module.parameters())
    norm_full = clip(holder, params, 1e9)
    clip(holder, params, float(norm_full) / 2)
    norm_after_zero_disable = clip(holder, params, 0)  # must not scale
    norm_final = clip(holder, params, 1e9)
    halved_gap = abs(float(norm_final) - float(norm_full) / 2)
    checks["clip_halves_the_norm"] = halved_gap < TOL.clip_achieved_norm_rel * float(norm_full)
    checks["clip_zero_max_norm_disables"] = abs(float(norm_after_zero_disable) - float(norm_final)) < 1e-6 * max(
        1.0, float(norm_final)
    )

    if USE_GC:
        # At least one checkpointed call per stage layer per microbatch forward (torch's one-time
        # shape-inference pass adds one more sweep of the layers); 0 or a shortfall means the GC
        # gate silently skipped (not-training / no-grad) and this run degraded to a plain re-test.
        n_layers = len(stage_module.model.layers)
        expected_min = n_layers * n_microbatches
        log(f"[rank {ctx.rank}] gc_calls={gc_calls[0]} expected>={expected_min}")
        checks["gc_engaged_every_layer_every_microbatch"] = (
            expected_min <= gc_calls[0] <= n_layers * (n_microbatches + 1)
        )

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
