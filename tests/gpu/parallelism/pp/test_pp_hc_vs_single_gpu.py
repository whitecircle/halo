#!/usr/bin/env python
"""Hyper-connection PP equivalence gate: DeepSeek-V4 / GLM-5 stages must reproduce single-GPU grads.

These families' stage boundary is the ``hc_mult``-widened ``[B, S, hc_mult, D]`` stream — carried
over torch pipelining's P2P by :class:`Glm5NextPPSpec`/:class:`DeepSeekV4PPSpec` (``hc_head``
neutralized off non-last stages, the mirrored mid-chain forward bound on non-first stages). This
runs the real schedule (recv buffers, shape inference, backward P2P for the 4-D activation) against
an unsplit single-process baseline over the same global batch, so it fails when:
  * the boundary stream is collapsed/re-widened (lossy) instead of carried whole
  * ``hc_head`` (or the final norm) runs mid-network, or is dropped from the last stage
  * the mid-chain forward drifts from the family's real forward (masks, rotary, indexer chain)
  * the widened activation loses its gradient path through the P2P boundary

DeepSeek-V4 additionally covers the hash-router (tid2eid) layer pinned to stage 0 with live
``input_ids``; GLM-5 runs the real KDA/DSA interleave with KDA layers on both stages (the CPU suite
can only run all-DSA — the installed ``causal_conv1d`` package is CUDA-only).

Run: torchrun --nproc_per_node=4 tests/gpu/parallelism/pp/test_pp_hc_vs_single_gpu.py --family deepseek_v4
"""

import math
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from transformers import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    Glm5NextConfig,
    Glm5NextForConditionalGeneration,
)

from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.pipeline_parallel.groups import create_pipeline_group
from src.distributed.pipeline_parallel.runtime import PipelineRuntime
from src.distributed.pipeline_parallel.stage import build_pipeline_stage
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_DSV4_CONFIG, TINY_GLM5_CONFIG, TINY_GLM5_VISION_CONFIG
from tests.common.utils import log

FAMILY = "deepseek_v4"
GLOBAL_BATCH = 8
SEQ_LEN = 32
GRAD_TOLERANCE = 2e-3
PP_SIZE = 2
N_MICROBATCHES = 2

# 8 layers put KDA layers on both stages of the [4, 4] split (default interleave: DSA at 3 and 7),
# so the mid-chain forward runs the linear-attention path too, not just the DSA one.
_GLM5_TEXT_8L = {
    **TINY_GLM5_CONFIG,
    "num_hidden_layers": 8,
    "mlp_layer_types": ["dense"] + ["sparse"] * 7,
}

# pp_split pins the boundary: dsv4's default [2, 1] keeps the hash layer (0) on stage 0; glm5's
# [4, 4] lands on the KDA/DSA period so both stages hold both layer kinds.
_FAMILIES = {
    "deepseek_v4": {"pp_split": [2, 1]},
    "glm5_next": {"pp_split": [4, 4]},
}


def build_model(device):
    """Identical weights on every rank — the baseline and the split model must start equal."""
    torch.manual_seed(1234)
    if FAMILY == "deepseek_v4":
        # The roster config keeps the hub's metadata-only MTP field at 1: the liveness gate must
        # accept it (no tail layer is built). Eager is the family's only attention path.
        model = DeepseekV4ForCausalLM(DeepseekV4Config(**TINY_DSV4_CONFIG, attn_implementation="eager"))
        # Random init leaves the hash table all-zero, i.e. top_k copies of expert 0 per token: the
        # normalized weights are then a constant 1/top_k and the gate weight's gradient is pure
        # rounding residue, so the comparison would judge noise. Distinct experts per token id
        # (seeded, identical on every build) give the hash router a real gradient.
        gen = torch.Generator().manual_seed(4321)
        num_experts = model.config.n_routed_experts
        for layer in model.model.layers:
            if layer.mlp.is_hash:
                table = layer.mlp.gate.tid2eid
                perm = torch.rand(table.shape[0], num_experts, generator=gen).argsort(dim=-1)
                table.copy_(perm[:, : table.shape[1]])
    else:
        config = Glm5NextConfig(
            text_config=dict(_GLM5_TEXT_8L),
            vision_config=dict(TINY_GLM5_VISION_CONFIG),
            attn_implementation="sdpa",
        )
        model = Glm5NextForConditionalGeneration(config)
    return model.to(device=device, dtype=torch.float32)


def make_batch(device, vocab_size):
    g = torch.Generator(device="cpu").manual_seed(99)
    input_ids = torch.randint(0, vocab_size, (GLOBAL_BATCH, SEQ_LEN), generator=g).to(device)
    return input_ids, input_ids.clone()


def causal_lm_loss(logits, labels):
    """The same shifted, token-count-normalized CE the pipeline runtime computes."""
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


@gpu_test_main(min_world_size=4, exact_world_size=4, prefix="pp_hc_vs_single_gpu")
def run(ctx):
    device = torch.device("cuda", ctx.local_rank)
    torch.cuda.set_device(device)
    pp_split = _FAMILIES[FAMILY]["pp_split"]

    # Two simulated 2-GPU nodes: each stage is one whole NVLink domain (dp=2 inside it, the
    # smallest width the stage FSDP wrap supports) and the pipeline crosses the node boundary.
    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE)
    assert config.stage_world_size == 2 and config.pp_rank == ctx.rank // 2

    vocab = build_model(device).config.get_text_config().vocab_size
    torch.cuda.empty_cache()
    input_ids, labels = make_batch(device, vocab)
    ref_loss, ref_grads = baseline_grads(device, input_ids, labels)
    log(f"[baseline:{FAMILY}] loss={ref_loss.item():.8f} over {len(ref_grads)} params")

    mesh = init_device_mesh("cuda", (PP_SIZE, config.stage_world_size), mesh_dim_names=("pp", "dp"))
    dp_mesh = mesh["dp"]
    pp_group = create_pipeline_group(config)

    model = build_model(device)
    stage_module = build_pipeline_stage(model, config.pp_rank, PP_SIZE, moe_balancing="none", pp_split=pp_split)
    lo, hi = stage_module.layer_range
    assert hi - lo == pp_split[config.pp_rank], (pp_split, config.pp_rank, (lo, hi))

    for layer in stage_module.model.layers:
        fully_shard(layer, mesh=dp_mesh, reshard_after_forward=False)
    fully_shard(stage_module, mesh=dp_mesh, reshard_after_forward=False)

    runtime = PipelineRuntime(stage_module, config, pp_group, device, N_MICROBATCHES)

    # This DP replica's slice of the global batch; every rank of a chain gets the same slice.
    dp_rank = config.get_data_parallel_rank()
    local_bs = GLOBAL_BATCH // config.data_parallel_size
    sl = slice(dp_rank * local_bs, (dp_rank + 1) * local_bs)
    loss = runtime.step(input_ids[sl], labels[sl])

    checks, metrics = {}, {}
    loss_err_t = torch.zeros(1, device=device)
    if config.is_last_pp_stage:
        mean_loss = loss.detach().clone()
        dist.all_reduce(mean_loss, op=dist.ReduceOp.AVG, group=dp_mesh.get_group())
        loss_err_t[0] = abs(mean_loss.item() - ref_loss.item()) / abs(ref_loss.item())
        log(
            f"[pp:{FAMILY}] loss={mean_loss.item():.8f} baseline={ref_loss.item():.8f} rel_err={loss_err_t.item():.3e}"
        )
    dist.all_reduce(loss_err_t, op=dist.ReduceOp.MAX)
    checks["loss_matches_baseline"] = loss_err_t.item() < 1e-4
    metrics["loss_rel_err"] = loss_err_t.item()

    # Gradient check via the production FQN remap. A routed-out expert legitimately has no gradient
    # in either run (both-None is skipped); one-sided absence is the failure this must catch.
    worst_name, worst_err, n_checked, mismatched = None, 0.0, 0, []
    for name, param in stage_module.named_parameters():
        fqn = stage_module.global_parameter_name(name)
        grad = param.grad
        ref = ref_grads.get(fqn)
        if grad is None and ref is None:
            continue
        if grad is None or ref is None:
            mismatched.append(
                f"{fqn} (stage grad {'None' if grad is None else 'set'}, ref {'None' if ref is None else 'set'})"
            )
            continue
        grad = grad.full_tensor() if isinstance(grad, torch.distributed.tensor.DTensor) else grad
        err = ((grad - ref).abs().max() / ref.abs().max().clamp_min(1e-12)).item()
        n_checked += 1
        # A NaN compares false against every bound, so a non-finite error counts as infinite.
        err = err if math.isfinite(err) else math.inf
        if err > worst_err:
            worst_name, worst_err = fqn, err

    log(
        f"[rank {ctx.rank} stage {config.pp_rank}] checked={n_checked} "
        f"worst_rel_err={worst_err:.3e} @ {worst_name} mismatched={mismatched[:4]}"
    )
    checks["no_one_sided_gradients"] = not mismatched
    checks["grads_match_baseline"] = n_checked > 0 and worst_err < GRAD_TOLERANCE
    metrics["worst_grad_rel_err"] = worst_err
    metrics["params_compared"] = n_checked

    # Every stage must contribute comparisons, or the FQN map silently emptied the check.
    stage_param_count = torch.tensor([float(n_checked)], device=device)
    dist.all_reduce(stage_param_count, op=dist.ReduceOp.MIN)
    checks["every_stage_compared_params"] = stage_param_count.item() > 0

    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    if "--family" in sys.argv:
        i = sys.argv.index("--family")
        FAMILY = sys.argv[i + 1]
        del sys.argv[i : i + 2]
    if FAMILY not in _FAMILIES:
        raise SystemExit(f"--family must be one of {sorted(_FAMILIES)}, got {FAMILY!r}")
    sys.exit(run())
