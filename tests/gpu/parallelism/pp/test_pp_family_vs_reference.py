#!/usr/bin/env python
"""A pipeline-registered tiny ``TINY_MOE_FAMILIES`` model split into two stages through the real SFT trainer, its
experts on the ep1 grouped-GEMM EP layers (``--ep-size 1``, pp2 × dp2), distributed (``--ep-size 2``, pp2 × ep2)
or each FFN split by expert TP (``--etp-size 2``, pp2 × etp2, one ETP group per stage), against the unsplit model
carrying the same ep1 wrappers: every DP replica's loss, every non-expert gradient by its unsplit name, and every
expert bank through the export's own gather. GPT-OSS has its own tighter gates (``test_pp_ep_vs_unsplit.py``,
``test_pp_ep1_gmm_vs_unsplit.py``, ``test_pp_etp_vs_unsplit.py``).

Expert TP reorders the sum that produces each expert's output, which in bf16 flips near-tied top-k picks with no
sync bug present, so the ETP rows run both sides in fp32, the reference's experts on the per-expert loop.

Gradients are scored by direction and norm ratio, the EP family suite's bounds: the bf16 run and its twin can flip
a near-tied top-k pick, which moves an expert's gradient mass with no pipeline fault, while a stage holding the
wrong layers, a norm run mid-network or a missed DP divide moves the cosine or the ratio far past them. A gradient
under ``TOL.grad_negligible_layer_fraction`` of its decoder layer's largest (Qwen3.5's GatedDeltaNet ``A_log`` /
``dt_bias`` on a layer whose heads all decay hard) is held to that floor as absolute error instead, while every
parameter kind keeps a layer on the scores.

Topology: 4 GPUs as 2 simulated NVLink domains of 2; each stage is one domain.

    torchrun --nproc_per_node=4 tests/gpu/parallelism/pp/test_pp_family_vs_reference.py --family step3p7 --ep-size 2
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

SEED = 1234
PP_SIZE = 2
PER_DEVICE_BATCH = 2  # 2 microbatches of 1 per DP replica
N_MICROBATCHES = 2
# Each family's text-config overrides and stage split: the shared two-stage overrides (TINY_PP_SPLIT_OVERRIDES),
# DeepSeek-V4's hash_moe layer (0) on stage 0, GLM-5 Next on its KDA/DSA period with KDA layers on both stages,
# Inkling at the batch's vocabulary.
FAMILIES = {
    "qwen3_5_moe_text": (TINY_PP_SPLIT_OVERRIDES["qwen3_5_moe_text"], [4, 4]),
    "step3p7": ({}, [2, 2]),
    "deepseek_v4": ({}, [2, 1]),
    "glm5_next": ({"num_hidden_layers": 8, "mlp_layer_types": ["dense"] + ["sparse"] * 7}, [4, 4]),
    "qwen3_moe": ({}, [4, 4]),
    "glm4_moe_lite": (TINY_PP_SPLIT_OVERRIDES["glm4_moe_lite"], [2, 2]),
    "laguna": (TINY_PP_SPLIT_OVERRIDES["laguna"], [4, 4]),
    "inkling_text": (
        {**TINY_PP_SPLIT_OVERRIDES["inkling_text"], "vocab_size": ppeq.RAGGED_BATCH_VOCAB_SIZE},
        [2, 2],
    ),
}

parser = argparse.ArgumentParser()
parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
parser.add_argument("--ep-size", type=int, choices=(1, 2), default=2)
parser.add_argument("--etp-size", type=int, choices=(1, 2), default=1)
ARGS = parser.parse_args()
if ARGS.etp_size > 1 and ARGS.ep_size > 1:
    parser.error("PP+EP+ETP is refused at config time: --etp-size 2 runs at --ep-size 1")
# fp32 under expert TP (the module docstring says why), the run dtype otherwise.
FP32 = ARGS.etp_size > 1
DTYPE = torch.float32 if FP32 else torch.bfloat16


@gpu_test_main(exact_world_size=4, prefix="pp_family_vs_reference")
def run(ctx):
    checks, metrics = {}, {}
    overrides, pp_split = FAMILIES[ARGS.family]
    if FP32:
        torch.set_float32_matmul_precision("highest")  # keep the fp32 reference an actual fp32 reference
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

    # Both runs sum each token's two weighted expert outputs alike and launch fla's kernels on the same configs, so
    # they route alike.
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
        # The ETP axis live on every stage layer, or an expert_tp_size that collapsed to 1 passes the comparison.
        checks["stage_experts_split_over_the_etp_group"] = all(
            module.expert_tp_size == ARGS.etp_size for _, module in ep_modules
        )

    dp_rank = config.get_data_parallel_rank()
    loss = trainer.training_step(trainer.model, dict(ppeq.dp_slice_batch(rows, labels, dp_rank, PER_DEVICE_BATCH)))
    trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), 1e6)  # runs the deferred sweep, no scaling
    c, m = ppeq.check_loss_vs_reference(loss, reference, dp_rank)
    checks.update(c)
    metrics.update(m)

    ppeq.score_stage_grads_vs_reference(ctx, stage, reference, checks, metrics)
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
