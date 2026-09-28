#!/usr/bin/env python
"""SFT smoke of a Qwen3.5/3.6 MoE under EP, ``ep_size`` = world size unless ``HALO_TEST_EP`` sets it.

Checks, on the model, that the experts were wrapped and split EP-way, then that every step ran with
finite losses, grad norms and final eval loss. FA2 crashes on the family's M-RoPE varlen path
(cudaErrorIllegalAddress), so it runs sdpa.

Usage:
    torchrun --nproc_per_node=4 tests/gpu/trainers/sft/test_sft_qwen3_5_ep.py

Requirements: 2-8 B200/B300 GPUs, DeepEP, Qwen/Qwen3.5-35B-A3B (``HALO_TEST_QWEN3_5_MODEL`` points it
at another Qwen3.5/3.6 MoE checkpoint, e.g. a local ``$HALO_DATA_ROOT/models/Qwen3.6-35B-A3B-patched``).
"""

from src.env import env_int, env_str
from tests.common.datasets import create_single_turn_sft_dataset
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_5_MOE_35B
from tests.common.sft_modes import SFTMode, SFTSuite, run_sft_suite

EP_SIZE_OVERRIDE = env_int("HALO_TEST_EP", None)
SUITE = SFTSuite(
    model_name=env_str("HALO_TEST_QWEN3_5_MODEL", QWEN3_5_MOE_35B),
    attn_implementation="sdpa",
    sft_args={
        "max_steps": 3,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1e-5,
        "warmup_steps": 1,
        "max_length": 1024,
        "remove_unused_columns": False,
    },
    num_samples=(16, 4),
    dataset=create_single_turn_sft_dataset,
)


@gpu_test_main(min_world_size=2, prefix="sft_qwen3_5_ep_test")
def run(ctx):
    ep_size = EP_SIZE_OVERRIDE if EP_SIZE_OVERRIDE is not None else ctx.world_size
    return run_sft_suite(ctx, SUITE, {"ep": SFTMode({"ep_size": ep_size})}, default_mode="ep")


if __name__ == "__main__":
    run()
