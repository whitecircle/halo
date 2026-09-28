#!/usr/bin/env python
"""SFT smoke of Qwen3.5-35B-A3B under each parallel shape, one ``--mode`` per launch.

Modes: ``ep`` (EP=2, grouped GEMM auto-enabled on SM90+), ``ep_no_gmm`` (EP=2 on the per-expert loop),
``tp`` (TP=2 on the full-attention layers) and ``etp`` (the MoE FFN weights split 2-way). Each checks,
on the model, that its axis took effect, then that every step ran with finite losses, grad norms and
final eval loss.

Qwen3.5-35B-A3B interleaves 30 linear-attention (GatedDeltaNet) and 10 full-attention layers, every one
with a 256-expert top-8 MoE MLP. FA2 crashes on its M-RoPE varlen path (cudaErrorIllegalAddress), so it
runs sdpa.

Usage:
    torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_sft_qwen3_5_moe.py --mode ep

Requirements: 2x GPUs with >=80GB each, DeepEP (EP modes), causal-conv1d and flash-linear-attention
(GatedDeltaNet), Qwen/Qwen3.5-35B-A3B.
"""

from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_5_MOE_35B
from tests.common.sft_modes import SFTMode, SFTSuite, run_sft_suite

SUITE = SFTSuite(
    model_name=QWEN3_5_MOE_35B,
    attn_implementation="sdpa",
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
    "ep": SFTMode({"ep_size": 2}),
    "ep_no_gmm": SFTMode({"ep_size": 2, "use_grouped_gemm": False}),
    "tp": SFTMode({"tp_size": 2}),
    "etp": SFTMode({"ep_size": 1, "expert_tp_size": 2}),
}


@gpu_test_main(exact_world_size=2, prefix="sft_qwen3_5")
def run(ctx):
    return run_sft_suite(ctx, SUITE, MODES, default_mode="ep")


if __name__ == "__main__":
    run()
