#!/usr/bin/env python
"""SFT smoke of Gemma4-26B-A4B under EP, ``ep_size`` = world size unless ``HALO_TEST_EP`` sets it.

Gemma4 is a VLM-wrapped MoE (Gemma4ForConditionalGeneration → Gemma4Model → Gemma4TextModel, whose
decoder layers inline their router and experts). Checks, on the model, that EP wrapped the experts
(EPGemma4MoELayer) and split them EP-way, then that every step ran on text-only inputs, with the
vision and audio towers unused, and finite losses, grad norms and final eval loss. No loss is compared
against a reference.

Usage:
    torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_sft_gemma4_moe.py

Requirements: 2x B200/B300 GPUs, DeepEP, a local checkpoint at
``$HALO_DATA_ROOT/models/gemma-4-26B-A4B-it-patched`` (``HALO_TEST_GEMMA4_MODEL`` points it elsewhere).
"""

from src.env import env_int, env_str
from tests.common.datasets import create_single_turn_sft_dataset
from tests.common.harness import gpu_test_main, skip_unless_local_checkpoint
from tests.common.models import GEMMA4_26B_A4B_PATCHED
from tests.common.sft_modes import SFTMode, SFTSuite, run_sft_suite

MODEL_NAME = env_str("HALO_TEST_GEMMA4_MODEL", GEMMA4_26B_A4B_PATCHED)
EP_SIZE_OVERRIDE = env_int("HALO_TEST_EP", None)
SUITE = SFTSuite(
    model_name=MODEL_NAME,
    # The global heads use head_dim=512, past FA2's 256 limit; sdpa takes any head dim.
    attn_implementation="sdpa",
    sft_args={
        "max_steps": 3,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1e-5,
        "warmup_steps": 1,
        "max_length": 1024,
        "remove_unused_columns": False,
        "gradient_checkpointing": False,
    },
    num_samples=(16, 4),
    dataset=create_single_turn_sft_dataset,
    # Liger has no Gemma4-specific kernels.
    use_liger_kernel=False,
)


@gpu_test_main(min_world_size=1, prefix="sft_gemma4_moe_test")
def run(ctx):
    ep_size = EP_SIZE_OVERRIDE if EP_SIZE_OVERRIDE is not None else ctx.world_size
    return run_sft_suite(ctx, SUITE, {"ep": SFTMode({"ep_size": ep_size})}, default_mode="ep")


if __name__ == "__main__":
    skip_unless_local_checkpoint(MODEL_NAME, "HALO_TEST_GEMMA4_MODEL")
    run()
