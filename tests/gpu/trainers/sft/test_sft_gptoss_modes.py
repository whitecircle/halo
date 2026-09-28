#!/usr/bin/env python
"""SFT smoke of GptOss-20B under each parallel shape, one ``--mode`` per launch.

Modes (world size 2): ``fsdp`` (plain FSDP2 with grouped-GEMM experts), ``ep`` (32 experts split
2-way over DeepEP), ``cp`` (Ulysses sequence sharding, experts local), ``tp`` (attention, embedding and
head sharded over DTensor, experts replicated, loaded one rank at a time), ``ep_cp``, ``ep_tp`` and
``ep_etp`` (``ep_size=1`` with each expert FFN split 2-way). Every mode checks, on the model, that its
axis took effect, then that five steps ran with finite losses and grad norms and a last step loss
below the first. No loss is compared against a reference, so a wrong-but-finite loss passes; the
gpt-oss correctness gates live under ``tests/gpu/parallelism/``.

Usage:
    torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_sft_gptoss_modes.py --mode ep

Requirements: 2x GPUs with >=80GB each, DeepEP for the EP modes, unsloth/gpt-oss-20b-BF16.
"""

from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B
from tests.common.sft_modes import SFTMode, SFTSuite, run_sft_suite

SUITE = SFTSuite(
    model_name=GPT_OSS_20B,
    attn_implementation="flex_attention",
    sft_args={"max_steps": 5, "gradient_accumulation_steps": 1, "learning_rate": 2e-5, "max_length": 4096},
    num_samples=(32, 0),
    evaluate=False,
    loss_decreased=True,
)
FA2 = "flash_attention_2"
MODES = {
    "fsdp": SFTMode(),
    "ep": SFTMode({"ep_size": 2}),
    "cp": SFTMode({"cp_size": 2}),
    "tp": SFTMode({"tp_size": 2, "max_concurrent_loading": 1}, attn_implementation=FA2),
    "ep_cp": SFTMode({"ep_size": 2, "cp_size": 2}),
    "ep_tp": SFTMode({"ep_size": 2, "tp_size": 2}),
    "ep_etp": SFTMode({"ep_size": 1, "expert_tp_size": 2}, attn_implementation=FA2),
}


@gpu_test_main(min_world_size=2, prefix="sft_gptoss_modes")
def run(ctx):
    return run_sft_suite(ctx, SUITE, MODES, default_mode="ep")


if __name__ == "__main__":
    run()
