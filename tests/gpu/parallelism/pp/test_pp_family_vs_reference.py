#!/usr/bin/env python
"""A pipeline-registered tiny family split into two stages, against the unsplit model with the same ep1 wrappers.

Expert TP reorders each expert's sum, which in bf16 flips near-tied picks with no sync bug, so ETP rows run fp32.
Gradients score by direction and norm ratio, since a flipped pick moves mass. A gradient under
``TOL.grad_negligible_layer_fraction`` of its layer's largest (Qwen3.5's ``A_log`` / ``dt_bias``) is held to that floor
as absolute error, Nemotron-H's Mamba layers to their own.
"""

import argparse
import copy

import torch

from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from tests.common import pp_equivalence as ppeq
from tests.common.fla_autotune import pin_fla_autotune_first_fit
from tests.common.harness import gpu_test_main
from tests.common.tiny_models import TINY_MOE_FAMILIES, TINY_PP_SPLIT_OVERRIDES, tiny_family_model
from tests.common.tolerances import TOL

SEED = 1234
PP_SIZE = 2
PER_DEVICE_BATCH = 2  # 2 microbatches of 1 per DP replica.
N_MICROBATCHES = 2
# The shared two-stage overrides, DeepSeek-V4's hash_moe layer on stage 0, GLM-5 Next on its KDA/DSA period, Inkling at
# the batch's vocabulary, Nemotron-H at it too with a MoE layer on each stage, its two experts a token as the sum below
# needs.
FAMILIES = {
    "qwen3_5_moe_text": (TINY_PP_SPLIT_OVERRIDES["qwen3_5_moe_text"], [4, 4]),
    "step3p7": ({}, [2, 2]),
    "deepseek_v4": ({}, [2, 1]),
    "glm5_next": ({"num_hidden_layers": 8, "mlp_layer_types": ["dense"] + ["sparse"] * 7}, [4, 4]),
    "qwen3_moe": ({}, [4, 4]),
    "glm4_moe_lite": (TINY_PP_SPLIT_OVERRIDES["glm4_moe_lite"], [2, 2]),
    "laguna": (TINY_PP_SPLIT_OVERRIDES["laguna"], [4, 4]),
    "nemotron_h": ({"vocab_size": ppeq.RAGGED_BATCH_VOCAB_SIZE, "num_experts_per_tok": 2}, [2, 4]),
    "inkling_text": (
        {**TINY_PP_SPLIT_OVERRIDES["inkling_text"], "vocab_size": ppeq.RAGGED_BATCH_VOCAB_SIZE},
        [2, 2],
    ),
}

NEGLIGIBLE_FRACTIONS = {"nemotron_h": TOL.nemotron_mamba_grad_negligible_layer_fraction}

parser = argparse.ArgumentParser()
parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
parser.add_argument("--ep-size", type=int, choices=(1, 2), default=2)
parser.add_argument("--etp-size", type=int, choices=(1, 2), default=1)
ARGS = parser.parse_args()
if ARGS.etp_size > 1 and ARGS.ep_size > 1:
    parser.error("PP+EP+ETP is refused at config time: --etp-size 2 runs at --ep-size 1")
# fp32 under expert TP (the module docstring says why).
FP32 = ARGS.etp_size > 1
DTYPE = torch.float32 if FP32 else torch.bfloat16


@gpu_test_main(exact_world_size=4, prefix="pp_family_vs_reference")
def run(ctx):
    checks, metrics = {}, {}
    overrides, pp_split = FAMILIES[ARGS.family]
    if FP32:
        torch.set_float32_matmul_precision("highest")  # Keeps the fp32 reference an actual fp32 reference.
    config = ParallelismConfig(
        world_size=ctx.world_size,
        gpus_per_node=2,
        nvlink_domain_size=2,
        pp_size=PP_SIZE,
        ep_size=ARGS.ep_size,
        expert_tp_size=ARGS.etp_size,
        pp_split=pp_split,
    )
    dp_size = config.data_parallel_size

    torch.manual_seed(SEED)
    base = tiny_family_model(TINY_MOE_FAMILIES[ARGS.family], overrides=overrides).to(DTYPE)
    rows, labels = ppeq.build_ragged_batch(dp_size * PER_DEVICE_BATCH)

    # Both runs sum each token's two weighted expert outputs alike on the same fla configs, so they route alike.
    ppeq.pin_unfused_expert_sum(ctx)
    pin_fla_autotune_first_fit(ctx)
    twin = ppeq.apply_single_process_ep1(copy.deepcopy(base), ctx.rank, use_grouped_gemm=not FP32).to(ctx.device)
    reference = ppeq.run_unsplit_reference(twin, rows, labels, ctx.device, dp_size)
    del twin

    model = apply_ep_layers(copy.deepcopy(base), config.ep_config())
    trainer = ppeq.build_pp_sft_trainer(
        ctx,
        config,
        model,
        rows,
        labels,
        per_device_batch=PER_DEVICE_BATCH,
        n_microbatches=N_MICROBATCHES,
        moe_balancing="none",
        bf16=not FP32,
    )
    ctx.on_teardown(trainer.cleanup_ep)
    stage = trainer._top_level_model()
    ep_modules = named_ep_layers(stage)
    lo, hi = stage.layer_range
    checks["stage_holds_its_split"] = hi - lo == pp_split[config.pp_rank]
    checks["stage_has_ep_layers"] = bool(ep_modules)
    if ARGS.etp_size > 1:
        # Or an expert_tp_size that collapsed to 1 passes the comparison.
        checks["stage_experts_split_over_the_etp_group"] = all(
            module.expert_tp_size == ARGS.etp_size for _, module in ep_modules
        )

    dp_rank = config.get_data_parallel_rank()
    loss = trainer.training_step(trainer.model, dict(ppeq.dp_slice_batch(rows, labels, dp_rank, PER_DEVICE_BATCH)))
    trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), 1e6)  # Runs the deferred sweep, no scaling.
    c, m = ppeq.check_loss_vs_reference(loss, reference, dp_rank)
    checks.update(c)
    metrics.update(m)

    fraction = NEGLIGIBLE_FRACTIONS.get(ARGS.family, TOL.grad_negligible_layer_fraction)
    ppeq.score_stage_grads_vs_reference(ctx, stage, reference, checks, metrics, negligible_fraction=fraction)
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
