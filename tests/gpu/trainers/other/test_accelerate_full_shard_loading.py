#!/usr/bin/env python
"""Trainers holding a frozen model, under ``accelerate launch`` with rank-0-only weight loading.

``launcher-configs/accelerate/fsdp2_full_config.yaml`` turns ``fsdp_cpu_ram_efficient_loading`` on, under which
``from_pretrained`` replaces the weights off each node's local rank 0 with a CPU zero copy and leaves them to the
broadcast in ``accelerator.prepare``. A frozen reference or teacher never gets that broadcast, and the policy is that
zero copy until it does, so a reference sweep run at construction fails. At step 1 the policy equals its reference or
teacher, so each case's step-1 value is known: the loss is ln 2 for DPO's sigmoid loss (with a live reference, and
with PEFT and precomputed reference log-probs), 0.5 for KTO and 0 for teacher distillation's KL, and offline GRPO's
logged KL is 0. The model is a pretrained one, since a randomly initialized model's near-uniform logits barely differ
from a zeroed copy's. One manifest row per case; the row runs this file as one process (``launcher="python"``), and
``halo launch`` starts the ranks.

    HALO_TEST_NPROC=2 python tests/gpu/trainers/other/test_accelerate_full_shard_loading.py --case dpo
"""

import argparse
import math
from pathlib import Path

from halo.env import env_int
from tests.common.entry_scripts import (
    CASES,
    checkpoint_dir,
    launch_command,
    logged_steps,
    run_launch,
    write_run_config,
)
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.ports import free_port
from tests.common.rank_log import log
from tests.common.repo import REPO_ROOT

# case -> (method, extra CLI overrides, logged step-1 key (a leading "/" matches every key ending in it), its value
# when the policy equals its reference or teacher)
STEP_1_CASES = {
    "dpo": ("preference/dpo", [], "loss", math.log(2.0)),
    "dpo_peft_precompute": (
        "preference/dpo",
        ["--use_peft=true", "--precompute_ref_log_probs=true"],
        "loss",
        math.log(2.0),
    ),
    "kto": ("preference/kto", [], "loss", 0.5),
    "offline_grpo_kl": ("grpo/offline", ["--kl_beta=0.1"], "/kl_mean", 0.0),
    "teacher_distill": ("distillation/teacher-distill", [], "loss", 0.0),
}
ACCELERATE_CONFIG = REPO_ROOT / "launcher-configs" / "accelerate" / "fsdp2_full_config.yaml"
# Above bf16 noise between the FSDP2-wrapped policy and its frozen copy. Measured on Qwen3-0.6B over 2 GPUs: DPO, with
# and without PEFT and precomputed reference log-probs, 1.9e-9 off ln 2; KTO and teacher distillation exact; offline
# GRPO's KL 3.2e-3. With the weights zeroed off rank 0, DPO missed by 0.37 and KTO by 0.14, and the other three
# cases crashed on rank 1.
STEP_1_ATOL = 1e-2

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case", choices=sorted(STEP_1_CASES), required=True)
ARGS = parser.parse_args()
NPROC = env_int("HALO_TEST_NPROC", 2)


@gpu_test_main(prefix="accelerate_full_shard_loading", partial_state=False, process_group=False)
def run(ctx):
    method, overrides, key, expected = STEP_1_CASES[ARGS.case]
    scratch = Path(ctx.output_dir)
    dataset_dir = scratch / "data"
    CASES[method].rows().save_to_disk(str(dataset_dir))
    config = write_run_config(method, scratch, model=QWEN3_0_6B, dataset_dir=str(dataset_dir))
    argv = launch_command(
        method,
        config,
        nproc=NPROC,
        port=free_port(),
        overrides=["--max_steps=1", "--save_steps=1", *overrides],
        accelerate_config=ACCELERATE_CONFIG,
    )
    log(f"{ARGS.case}: one step on {NPROC} ranks under {ACCELERATE_CONFIG.name}")
    checks = {"launch_exits_cleanly": run_launch(argv).returncode == 0}
    if not checks["launch_exits_cleanly"]:
        return {"checks": checks}

    first = logged_steps(checkpoint_dir(scratch / "output", 1))[0]
    logged = {
        name: value for name, value in first.items() if (name.endswith(key) if key.startswith("/") else name == key)
    }
    log(f"{ARGS.case}: step-1 {logged}, expected {expected:.6f}")
    checks["step_1_value_logged"] = bool(logged)
    checks["step_1_value_matches_an_intact_reference"] = all(abs(v - expected) < STEP_1_ATOL for v in logged.values())
    return {"checks": checks}


if __name__ == "__main__":
    run()
