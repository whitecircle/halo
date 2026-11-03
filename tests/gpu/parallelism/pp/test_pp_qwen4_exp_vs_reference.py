#!/usr/bin/env python
"""Qwen3.8-Flash-Next split into two stages, against the unsplit model with the same ep1 wrappers.

Split ``[3, 5]`` off the ``L,L,L,F`` period, with a PLE layer on each stage. ``etp2`` runs fp32 since the sum reorder
flips picks in bf16; ``packed`` since over ~20-token documents bf16 rounding alone turns the decays' and PLE norms'
small gradients past the bars.
"""

import argparse
import copy
from unittest import mock

import torch
import torch.distributed as dist

from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.distributed.pipeline_parallel import stage as stage_module
from halo.models.segment_markers import segment_marker_kwargs, segment_markers_for
from tests.common import pp_equivalence as ppeq
from tests.common.distributed import shared_output_dir, world_all
from tests.common.fla_autotune import pin_fla_autotune_first_fit
from tests.common.harness import gpu_test_main
from tests.common.qwen4_exp import build_tiny_qwen4_exp_checkpoint, load_host_table_model
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

SEED = 1234
PP_SIZE = 2
PER_DEVICE_BATCH = 2  # 2 microbatches of 1 per DP replica.
N_MICROBATCHES = 2
QWEN4_OVERRIDES = {
    "num_hidden_layers": 8,
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 2,
    "ple_layer_ids": [2, 6],
}
SPLITS = {"qwen4_exp_text": [3, 5], "qwen3_5_moe_text": [4, 4]}
# Fill the row to its fixed length; positions restart at each.
PACKED_DOCUMENTS = (20, 27, 17)
MODES = {
    "ep2": {"ep_size": 2},
    "etp2": {"expert_tp_size": 2},
    "packed": {"expert_tp_size": 2},
    "gc": {"ep_size": 2},
}

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=sorted(MODES), required=True)
parser.add_argument("--family", choices=sorted(SPLITS), default="qwen4_exp_text")
ARGS = parser.parse_args()
if ARGS.family != "qwen4_exp_text" and ARGS.mode != "packed":
    parser.error("--family qwen3_5_moe_text runs the packed mode only; its other rows are test_pp_family_vs_reference")
FP32 = ARGS.mode in ("etp2", "packed")
DTYPE = torch.float32 if FP32 else torch.bfloat16


def _base_model(ctx):
    """Qwen4 through the host-table load from a checkpoint rank 0 writes; Qwen3.5-MoE from its seed with eight
    layers."""
    if ARGS.family == "qwen3_5_moe_text":
        torch.manual_seed(SEED)
        return tiny_family_model(TINY_MOE_FAMILIES[ARGS.family], overrides={"num_hidden_layers": 8}).to(DTYPE)
    directory = f"{shared_output_dir(ctx)}/qwen4_exp_checkpoint"
    if ctx.rank == 0:
        build_tiny_qwen4_exp_checkpoint(directory, seed=SEED, overrides=QWEN4_OVERRIDES)
    dist.barrier()
    return load_host_table_model(directory, dtype=DTYPE)


def _packed_batches(rows: torch.Tensor, labels: torch.Tensor, dp_rank: int, model_config):
    """Positions restart per document, its first target ignored as the packing collator does, no mask; the reference
    reads the collator's markers per row."""
    assert sum(PACKED_DOCUMENTS) == rows.shape[1], "the packed documents fill the row"
    positions = torch.cat([torch.arange(n) for n in PACKED_DOCUMENTS]).unsqueeze(0)
    labels = labels.masked_fill(positions == 0, -100)
    batch = ppeq.dp_slice_batch(rows, labels, dp_rank, PER_DEVICE_BATCH)
    del batch["attention_mask"]
    batch["position_ids"] = positions.expand(PER_DEVICE_BATCH, -1)
    markers = {"position_ids": positions, **segment_marker_kwargs(positions, segment_markers_for(model_config))}
    return labels, batch, [markers] * rows.shape[0]


@gpu_test_main(exact_world_size=4, prefix="pp_qwen4_exp_vs_reference")
def run(ctx):
    checks, metrics = {}, {}
    pp_split = SPLITS[ARGS.family]
    if FP32:
        torch.set_float32_matmul_precision("highest")  # Keeps the fp32 reference an actual fp32 reference.
    config = ParallelismConfig(
        world_size=ctx.world_size,
        gpus_per_node=2,
        nvlink_domain_size=2,
        pp_size=PP_SIZE,
        pp_split=pp_split,
        **MODES[ARGS.mode],
    )
    dp_size, dp_rank = config.data_parallel_size, config.get_data_parallel_rank()

    base = _base_model(ctx)
    rows, labels = ppeq.build_ragged_batch(dp_size * PER_DEVICE_BATCH)
    assert int(rows.max()) < base.config.get_text_config().vocab_size, "the ragged rows fit the tiny vocabulary"
    if ARGS.mode == "packed":
        labels, batch, row_kwargs = _packed_batches(rows, labels, dp_rank, base.config)
    else:
        batch, row_kwargs = ppeq.dp_slice_batch(rows, labels, dp_rank, PER_DEVICE_BATCH), None

    ppeq.pin_unfused_expert_sum(ctx)
    pin_fla_autotune_first_fit(ctx)
    twin = ppeq.apply_single_process_ep1(copy.deepcopy(base), ctx.rank, use_grouped_gemm=not FP32).to(ctx.device)
    reference = ppeq.run_unsplit_reference(twin, rows, labels, ctx.device, dp_size, row_kwargs=row_kwargs)
    del twin

    trainer = ppeq.build_pp_sft_trainer(
        ctx,
        config,
        apply_ep_layers(copy.deepcopy(base), config.ep_config()),
        rows,
        labels,
        per_device_batch=PER_DEVICE_BATCH,
        n_microbatches=N_MICROBATCHES,
        moe_balancing="none",
        bf16=not FP32,
        gradient_checkpointing=ARGS.mode == "gc",
    )
    ctx.on_teardown(trainer.cleanup_ep)
    stage = trainer._top_level_model()
    lo, hi = stage.layer_range
    checks["stage_holds_its_split"] = hi - lo == pp_split[config.pp_rank]
    checks["stage_has_ep_layers"] = bool(named_ep_layers(stage))
    if ARGS.family == "qwen4_exp_text":
        checks["stage_holds_a_ple_layer"] = any(layer.ple is not None for layer in stage.model.layers)
    calls = ppeq.count_checkpoint_calls(stage) if ARGS.mode == "gc" else None

    loss = trainer.training_step(trainer.model, dict(batch))
    trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), 1e6)  # Runs the deferred sweep, no scaling.
    c, m = ppeq.check_loss_vs_reference(loss, reference, dp_rank)
    checks.update(c)
    metrics.update(m)
    ppeq.score_stage_grads_vs_reference(ctx, stage, reference, checks, metrics)
    if calls is not None:
        checks["checkpointing_ran"] = world_all(calls[0] > 0)
        metrics["checkpoint_calls"] = calls[0]

    if ARGS.mode == "packed" and ARGS.family == "qwen4_exp_text":
        # Last: the refused forward leaves the stage's FSDP state mid-forward. It raises before the first layer, so
        # stage 0's two ranks join only its root all-gather.
        refused = True
        if config.is_first_pp_stage:
            microbatch = {key: value[:1].to(ctx.device) for key, value in batch.items() if key != "labels"}
            with mock.patch.object(stage_module, "packed_row_segment_kwargs", lambda position_ids, markers: {}):
                try:
                    stage(microbatch["input_ids"], position_ids=microbatch["position_ids"])
                    refused = False
                except ValueError as exc:
                    refused = "without seq_idx" in str(exc)
        checks["a_stage_without_the_rebuilt_markers_is_refused"] = world_all(refused)
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
