#!/usr/bin/env python
"""Every trainable parameter FSDP2 leaves out of its shard groups is still synced across DP ranks.

FSDP2 leaves out frozen parameters whose dtype no trainable parameter shares, one parameter at a time, so
a trainable child of the module owning one stays in its shard group; at ep1 under
``fsdp_shard_ep1_experts`` nothing else would sync it, and it would train on each rank's own batch and
drift while every loss stays finite. Each row trains two SFT steps on 2 ranks and requires the trainable
parameters to move and end bitwise identical on both (see ``tests/common/pinned_params.py``):

  * ``--case router_only``: ep1 GLM-4 MoE Lite with ``ep_fp32_router`` and only the routers trainable.
    The trainable set is fp32, so every bf16 parameter is excluded, the EP layers owning the routers
    among them. The family's router computes in fp32 whatever its weight's dtype, so an unsynced router
    trains without error.
  * ``--case glm5_next_mixed_uncast``: GLM-5 Next mixed LoRA at ep1 with the load-time run-dtype cast
    bypassed, so the fp32 ``dt_bias``/``A_log`` pins stay frozen beside the forget gate's adapters. The
    per-parameter exclusion alone has to keep those adapters in their shard group.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_sft_fsdp_excluded_params.py --case router_only
"""

import argparse
from unittest import mock

import torch

import src.distributed.loading.model_loading as model_loading
from src.distributed.loading.peft_setup import unfreeze_modules_by_patterns
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.harness import gpu_test_main
from tests.common.pinned_params import (
    RUN_DTYPE,
    base_params_off_run_dtype,
    load_row_model,
    tiny_family_checkpoint,
    train_row,
)
from tests.common.utils import log

CASES = ("router_only", "glm5_next_mixed_uncast")

# The ep1 wrapper keeps the router at ``mlp.gate``.
ROUTER_PATTERNS = ["*.mlp.gate"]


def _router_only(ctx) -> dict:
    base_dir, _ = tiny_family_checkpoint(ctx, "glm4_moe_lite")
    pc = ParallelismConfig(ep_size=1, ep_fp32_router=True)
    model, tokenizer, _ = load_row_model(base_dir, "full", pc)
    unfreeze_modules_by_patterns(model, ROUTER_PATTERNS)
    trainable_dtypes = {p.dtype for p in model.parameters() if p.requires_grad}
    log(f"trainable dtypes: {trainable_dtypes}")
    result = train_row(ctx, model, tokenizer, pc, None, check_sync=True)
    result["checks"]["only_fp32_routers_train"] = trainable_dtypes == {torch.float32}
    return result


def _glm5_next_mixed_uncast(ctx) -> dict:
    base_dir, _ = tiny_family_checkpoint(ctx, "glm5_next")
    pc = ParallelismConfig(ep_size=1)
    with mock.patch.object(model_loading, "cast_parameters_to_run_dtype", lambda *args, **kwargs: None):
        model, tokenizer, peft_config = load_row_model(base_dir, "mixed", pc)
    pins = base_params_off_run_dtype(model)
    log(f"{len(pins)} base params left off {RUN_DTYPE}: {pins[:6]}")
    result = train_row(ctx, model, tokenizer, pc, peft_config, check_sync=True)
    result["checks"]["pins_left_fp32"] = bool(pins)
    return result


@gpu_test_main(exact_world_size=2, prefix="sft_fsdp_excluded_params")
def run(ctx):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    case = parser.parse_args().case
    torch.cuda.set_device(ctx.device)
    return _router_only(ctx) if case == "router_only" else _glm5_next_mixed_uncast(ctx)


if __name__ == "__main__":
    run()
