#!/usr/bin/env python
"""DeepSeek-V4 / GLM-5 PP stages reproduce single-GPU grads, carrying the ``[B, S, hc_mult, D]`` hyper-connection
stream over real P2P.

GLM-5 runs the real KDA/DSA interleave, which the CPU suite can't, since the installed ``causal_conv1d`` package is
CUDA-only.
"""

import sys

import torch
from transformers import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    Glm5NextConfig,
    Glm5NextForConditionalGeneration,
)

from halo.distributed.parallelism_config import ParallelismConfig
from tests.common import pp_equivalence as ppeq
from tests.common.distributed import world_min
from tests.common.fla_autotune import pin_fla_autotune_first_fit
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_DSV4_CONFIG, TINY_GLM5_CONFIG, TINY_GLM5_VISION_CONFIG
from tests.common.rank_log import log

FAMILY = "deepseek_v4"
GLOBAL_BATCH = 8
GRAD_TOLERANCE = 2e-3
PP_SIZE = 2
N_MICROBATCHES = 2

# KDA layers on both stages of the [4, 4] split (DSA at 3 and 7), so the mid-chain forward runs the linear-attention
# path too.
_GLM5_TEXT_8L = {
    **TINY_GLM5_CONFIG,
    "num_hidden_layers": 8,
    "mlp_layer_types": ["dense"] + ["sparse"] * 7,
}

# dsv4's [2, 1] keeps the hash layer (0) on stage 0; glm5's [4, 4] lands on the KDA/DSA period.
_FAMILIES = {
    "deepseek_v4": {"pp_split": [2, 1]},
    "glm5_next": {"pp_split": [4, 4]},
}


def build_model(device):
    """Identical on every rank: the baseline and the split model must start equal."""
    torch.manual_seed(1234)
    if FAMILY == "deepseek_v4":
        # The roster keeps the hub's metadata-only MTP field at 1, which the liveness gate must accept.
        model = DeepseekV4ForCausalLM(DeepseekV4Config(**TINY_DSV4_CONFIG, attn_implementation="eager"))
        # Random init leaves the hash table all-zero, so the gate weight's gradient is pure rounding residue. Distinct
        # experts per token id give the hash router a real gradient.
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


@gpu_test_main(min_world_size=4, exact_world_size=4, prefix="pp_hc_vs_single_gpu")
def run(ctx):
    device = torch.device("cuda", ctx.local_rank)
    torch.cuda.set_device(device)
    pp_split = _FAMILIES[FAMILY]["pp_split"]

    # Each stage is one whole NVLink domain (dp=2, the smallest width the stage FSDP wrap supports).
    config = ParallelismConfig(world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE)
    assert config.stage_world_size == 2 and config.pp_rank == ctx.rank // 2

    vocab = build_model(device).config.get_text_config().vocab_size
    # fla autotunes per batch size, so the full batch and its microbatches would run on separately timed picks.
    pin_fla_autotune_first_fit(ctx)
    torch.cuda.empty_cache()
    input_ids, labels = ppeq.build_unmasked_batch(vocab, GLOBAL_BATCH, device=device)
    ref_loss, _, ref_grads = ppeq.baseline_grads(build_model(device), input_ids, labels)
    log(f"[baseline:{FAMILY}] loss={ref_loss.item():.8f} over {len(ref_grads)} params")

    groups = ppeq.build_stage_groups(config)
    stage_module, runner, _ = ppeq.build_runner_stage(
        build_model(device), config, groups, device, N_MICROBATCHES, moe_balancing="none", pp_split=pp_split
    )
    lo, hi = stage_module.layer_range
    assert hi - lo == pp_split[config.pp_rank], (pp_split, config.pp_rank, (lo, hi))

    # Every rank of a chain gets the same slice.
    dp_rank = config.get_data_parallel_rank()
    local_bs = GLOBAL_BATCH // config.data_parallel_size
    sl = slice(dp_rank * local_bs, (dp_rank + 1) * local_bs)
    loss = runner.step(input_ids[sl], labels[sl])

    checks, metrics = ppeq.check_loss_vs_baseline(loss, ref_loss, config, groups, tol=1e-4)

    # A routed-out expert has no gradient in either run; one-sided absence is the failure this must catch.
    grads = ppeq.gather_stage_grads(stage_module)
    stage_names = {stage_module.global_parameter_name(n) for n, _ in stage_module.named_parameters()}
    mismatched = sorted(grads.keys() ^ (stage_names & ref_grads.keys()))
    errors, worst_name = ppeq.max_abs_grad_errors(grads, ref_grads)
    worst_err = errors.get(worst_name, 0.0)
    log(
        f"[rank {ctx.rank} stage {config.pp_rank}] checked={len(errors)} "
        f"worst_rel_err={worst_err:.3e} @ {worst_name} mismatched={mismatched[:4]}"
    )
    checks["no_one_sided_gradients"] = not mismatched
    checks["grads_match_baseline"] = bool(errors) and worst_err < GRAD_TOLERANCE
    metrics["worst_grad_rel_err"] = worst_err
    metrics["params_compared"] = len(errors)

    # Or the FQN map silently emptied the check.
    checks["every_stage_compared_params"] = world_min(len(errors), device) > 0

    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    if "--family" in sys.argv:
        i = sys.argv.index("--family")
        FAMILY = sys.argv[i + 1]
        del sys.argv[i : i + 2]
    if FAMILY not in _FAMILIES:
        raise SystemExit(f"--family must be one of {sorted(_FAMILIES)}, got {FAMILY!r}")
    sys.exit(run())
