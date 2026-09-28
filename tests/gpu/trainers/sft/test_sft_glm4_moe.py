#!/usr/bin/env python
"""SFT smoke of GLM-4.7-Flash (GLM4 MoE Lite) under EP, ``ep_size`` = world size unless
``HALO_TEST_EP`` sets it.

Checks, on the model, that the experts were wrapped (and split EP-way at ``ep_size > 1``), then that
every step ran with finite losses, grad norms and final eval loss.

Usage:
    torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_sft_glm4_moe.py

Requirements: 2x GPUs with >=80GB each, DeepEP, zai-org/GLM-4.7-Flash (``HALO_TEST_GLM4_MODEL`` points
it at another checkpoint).
"""

from functools import partial

from src.env import env_int, env_str
from tests.common.datasets import VERBOSE_MATH_TEMPLATES, create_single_turn_sft_dataset
from tests.common.harness import gpu_test_main
from tests.common.models import GLM4_FLASH
from tests.common.sft_modes import SFTMode, SFTSuite, run_sft_suite

EP_SIZE_OVERRIDE = env_int("HALO_TEST_EP", None)
SUITE = SFTSuite(
    model_name=env_str("HALO_TEST_GLM4_MODEL", GLM4_FLASH),
    attn_implementation="flash_attention_2",
    sft_args={
        "max_steps": 5,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1e-5,
        "warmup_steps": 1,
        "max_length": 4096,
        "remove_unused_columns": False,
        # FLCE-only model: its Liger forward returns no logits, and with the flag off TRL's metric
        # path slices the missing logits. The loader already applied the kernels, so none are reapplied.
        "use_liger_kernel": True,
    },
    dataset=partial(create_single_turn_sft_dataset, templates=VERBOSE_MATH_TEMPLATES),
)


@gpu_test_main(min_world_size=1, prefix="sft_glm4_moe_test")
def run(ctx):
    ep_size = EP_SIZE_OVERRIDE if EP_SIZE_OVERRIDE is not None else ctx.world_size
    return run_sft_suite(ctx, SUITE, {"ep": SFTMode({"ep_size": ep_size})}, default_mode="ep")


if __name__ == "__main__":
    run()
