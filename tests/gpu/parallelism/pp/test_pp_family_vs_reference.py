#!/usr/bin/env python
"""A pipeline-registered tiny family split into two stages, against the unsplit model with the same ep1 wrappers.

Expert TP reorders each expert's sum, which in bf16 flips near-tied picks with no sync bug, so ETP rows run fp32.
A hyper-connection family (``hc_mult``) runs fp32 where no DeepEP dispatch does (DeepEP takes bf16 only): its base and
scale gradients sit 4 to 5 orders under their layer's largest, so bf16 accumulation order alone decides them. A bf16 EP
row leaves those parameters unscored and the fp32 ETP row covers them.
Gradients score by direction and norm ratio, since a flipped pick moves mass. A gradient under
``TOL.grad_negligible_layer_fraction`` of its layer's largest (Qwen3.5's ``A_log`` / ``dt_bias``) is held to that floor
as absolute error, Nemotron-H's Mamba layers to their own. ``--text-only`` reads a composite release from disk through
the real loader, as ``text_only_model: true`` does, and scores it against the whole text class that checkpoint loads
as.
"""

import argparse
import copy

import torch
from transformers import AutoModelForCausalLM

from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.finalization import finalize_loaded_model
from halo.model_loading.model_preparation import resolve_text_only_config
from halo.model_loading.parallel.loader import load_distributed_model
from tests.common import pp_equivalence as ppeq
from tests.common.fla_autotune import pin_fla_autotune_first_fit
from tests.common.harness import gpu_test_main
from tests.common.qwen4_exp import stamped_config
from tests.common.rank_log import log
from tests.common.tiny_models import (
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    TINY_PP_SPLIT_OVERRIDES,
    shared_tiny_family_checkpoint,
    tiny_family_model,
    word_level_tokenizer,
)
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

# The same towers in the composite spelling their releases ship, a PLE layer on each of Flash-Next's stages. Its two
# experts a token are the sum below's too: the PLE gate's gradient grows as 1/sqrt|gate|, so the rounding of a third add
# moves a near-zero gate's share of it by 10x.
TEXT_ONLY_FAMILIES = {
    "qwen3_5_moe": FAMILIES["qwen3_5_moe_text"],
    "nemotron_h_omni": FAMILIES["nemotron_h"],
    "qwen4_exp": (
        {
            "num_hidden_layers": 8,
            "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 2,
            "ple_layer_ids": [2, 6],
            "num_experts_per_tok": 2,
        },
        [3, 5],
    ),
}

NEGLIGIBLE_FRACTIONS = {
    "nemotron_h": TOL.nemotron_mamba_grad_negligible_layer_fraction,
    "nemotron_h_omni": TOL.nemotron_mamba_grad_negligible_layer_fraction,
}

parser = argparse.ArgumentParser()
parser.add_argument("--family", choices=sorted({*FAMILIES, *TEXT_ONLY_FAMILIES}), required=True)
parser.add_argument("--ep-size", type=int, choices=(1, 2), default=2)
parser.add_argument("--etp-size", type=int, choices=(1, 2), default=1)
parser.add_argument("--text-only", action="store_true")
ARGS = parser.parse_args()
if ARGS.etp_size > 1 and ARGS.ep_size > 1:
    parser.error("PP+EP+ETP is refused at config time: --etp-size 2 runs at --ep-size 1")
if ARGS.text_only != (ARGS.family in TEXT_ONLY_FAMILIES):
    parser.error(f"--text-only runs exactly {sorted(TEXT_ONLY_FAMILIES)}")


def _run_dtype(config) -> torch.dtype:
    """fp32 under expert TP, or for hyper-connections where no DeepEP dispatch runs (the module docstring says why)."""
    fp32 = ARGS.etp_size > 1 or (ARGS.ep_size == 1 and ppeq.carries_hyper_connections(config))
    return torch.float32 if fp32 else torch.bfloat16


def _load_from_composite_checkpoint(ctx, config: ParallelismConfig, overrides: dict):
    """This rank's stage through the real loader, the whole text class the checkpoint loads as, and the run's dtype.
    The family patches the stage's load applies stamp the config the whole model must be built from too."""
    path = shared_tiny_family_checkpoint(
        ctx, TINY_MOE_VLM_FAMILIES[ARGS.family], f"pp_text_only_{ARGS.family}", None, SEED, overrides=overrides
    )
    if ctx.rank == 0:
        word_level_tokenizer().save_pretrained(path)
    ctx.barrier()
    text_config = resolve_text_only_config(stamped_config(path))
    dtype = _run_dtype(text_config)
    stage, _ = load_distributed_model(
        path, config, dtype, trust_remote_code=False, text_only_model=True, attn_implementation="sdpa"
    )
    whole = AutoModelForCausalLM.from_pretrained(path, config=text_config, dtype=dtype, attn_implementation="sdpa")
    finalize_loaded_model(whole)
    return stage, whole, dtype


@gpu_test_main(exact_world_size=4, prefix="pp_family_vs_reference")
def run(ctx):
    checks, metrics = {}, {}
    overrides, pp_split = TEXT_ONLY_FAMILIES[ARGS.family] if ARGS.text_only else FAMILIES[ARGS.family]
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
    if ARGS.text_only:
        model, base, dtype = _load_from_composite_checkpoint(ctx, config, overrides)
    else:
        model = None
        base = tiny_family_model(TINY_MOE_FAMILIES[ARGS.family], overrides=overrides)
        dtype = _run_dtype(base.config)
        base = base.to(dtype)
    fp32 = dtype == torch.float32
    if fp32:
        torch.set_float32_matmul_precision("highest")  # Keeps the fp32 reference an actual fp32 reference.
    rows, labels = ppeq.build_ragged_batch(dp_size * PER_DEVICE_BATCH)

    # Both runs sum each token's two weighted expert outputs alike on the same fla configs, so they route alike.
    ppeq.pin_unfused_expert_sum(ctx)
    pin_fla_autotune_first_fit(ctx)
    twin = ppeq.apply_single_process_ep1(copy.deepcopy(base), ctx.rank, use_grouped_gemm=not fp32).to(ctx.device)
    reference = ppeq.run_unsplit_reference(twin, rows, labels, ctx.device, dp_size)
    del twin

    if ARGS.text_only:
        # The class test: a loader falling back to the composite would still score alike on its text tower.
        checks["stage_is_the_text_class"] = type(model) is type(base)
    else:
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
        bf16=not fp32,
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
    unscored = frozenset()
    if not fp32 and ppeq.carries_hyper_connections(base.config):
        unscored = frozenset(ppeq.hyper_connection_role_params(stage))
        log(f"[rank {ctx.rank}] bf16 EP row leaves hyper-connection base/scale unscored: {sorted(unscored)}")
    ppeq.score_stage_grads_vs_reference(
        ctx, stage, reference, checks, metrics, negligible_fraction=fraction, unscored=unscored
    )
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
