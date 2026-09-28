#!/usr/bin/env python
"""SFT smoke of Qwen3-30B-A3B-Instruct-2507 (128 experts, 8 active) under EP=2.

Checks, on the model, that the experts were wrapped and split 2-way, then that every step ran with
finite losses, grad norms and final eval loss.

Usage:
    torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_sft_qwen3_moe.py

Requirements: 2x GPUs with >=80GB each, DeepEP, Qwen/Qwen3-30B-A3B-Instruct-2507.
"""

from functools import partial

from tests.common.datasets import VERBOSE_MATH_TEMPLATES, create_single_turn_sft_dataset
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_30B_A3B
from tests.common.sft_modes import SFTMode, SFTSuite, run_sft_suite

SUITE = SFTSuite(
    model_name=QWEN3_30B_A3B,
    attn_implementation="flash_attention_2",
    sft_args={
        "max_steps": 5,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1e-5,
        "warmup_steps": 1,
        "max_length": 4096,
        "remove_unused_columns": False,
    },
    dataset=partial(create_single_turn_sft_dataset, templates=VERBOSE_MATH_TEMPLATES),
)
MODES = {"ep": SFTMode({"ep_size": 2})}


@gpu_test_main(exact_world_size=2, prefix="sft_qwen3_moe_test")
def run(ctx):
    return run_sft_suite(ctx, SUITE, MODES, default_mode="ep")


if __name__ == "__main__":
    run()
