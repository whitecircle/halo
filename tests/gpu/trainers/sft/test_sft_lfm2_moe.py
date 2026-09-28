#!/usr/bin/env python
"""SFT smoke of LiquidAI/LFM2-24B-A2B under FSDP and EP, one ``--mode`` per launch.

Modes: ``fsdp`` (every expert on every GPU) and ``ep`` (EP=2 over DeepEP). Each checks, on the model,
that its shape took effect, then that every step ran with finite losses, grad norms and final eval
loss.

LFM2-24B-A2B: 40 hybrid conv + full-attention layers, two of them dense, the rest a 64-expert top-4
MoE with sigmoid routing and an expert bias; ~24B parameters, ~2B active per token.

Usage:
    torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_sft_lfm2_moe.py --mode ep

Requirements: 2x GPUs with >=80GB each, DeepEP (EP mode), LiquidAI/LFM2-24B-A2B.
"""

from tests.common.harness import gpu_test_main
from tests.common.models import LFM2_24B_A2B
from tests.common.sft_modes import SFTMode, SFTSuite, run_sft_suite

SUITE = SFTSuite(
    model_name=LFM2_24B_A2B,
    attn_implementation="flash_attention_2",
    sft_args={
        "max_steps": 5,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1e-5,
        "warmup_steps": 1,
        "max_length": 4096,
        "remove_unused_columns": False,
    },
)
MODES = {
    "fsdp": SFTMode(),
    "ep": SFTMode({"ep_size": 2}),
}


@gpu_test_main(exact_world_size=2, prefix="sft_lfm2")
def run(ctx):
    return run_sft_suite(ctx, SUITE, MODES, default_mode="fsdp")


if __name__ == "__main__":
    run()
