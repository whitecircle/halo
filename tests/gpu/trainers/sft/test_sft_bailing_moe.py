#!/usr/bin/env python
"""SFT smoke of inclusionAI/Ring-mini-linear-2.0 under FSDP and EP, and an EP overfit run.

Modes, one per launch: ``fsdp`` (every expert on every GPU), ``ep`` (EP=2, grouped GEMM auto-enabled
on SM90+), ``ep_no_gmm`` (EP=2 on the per-expert loop), and ``overfit`` (EP=2, 4 samples for 40 steps,
which must reach a peak logged token accuracy of 99%). Each checks, on the model, that its shape took
effect, then that every step ran with finite losses, grad norms and final eval loss.

Ring-mini-linear-2.0 (BailingMoeLinearV2, remote code): 20 layers, 16 linear-attention and 4
full-attention; layer 0 dense, the rest a 256-expert top-8 MoE with one shared expert and sigmoid
group-limited routing (8 groups, top 4); ~16.4B parameters, ~1.6B active per token.

Usage:
    torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_sft_bailing_moe.py --mode ep

Requirements: 2x GPUs with >=40GB each, DeepEP (EP modes), flash-linear-attention,
inclusionAI/Ring-mini-linear-2.0.
"""

from src.models.patches.remote_code_compat import apply_remote_code_compat_shims

apply_remote_code_compat_shims()

from tests.common.harness import gpu_test_main
from tests.common.models import BAILING_MOE_RING_MINI
from tests.common.sft_modes import SFTMode, SFTSuite, run_sft_suite
from tests.common.utils import log

OVERFIT_MIN_PEAK_TOKEN_ACCURACY = 0.99


def peak_token_accuracy_check(trainer) -> dict[str, bool]:
    """The overfit mode's verdict: the peak ``mean_token_accuracy`` the training steps logged.

    Read off the trainer's own per-batch metric, since a separate inference pass would need its own EP
    coordination. The peak proves the model can memorize the set; the tail mean is logged to show
    whether it held.
    """
    accuracies = [
        entry["mean_token_accuracy"] for entry in trainer.state.log_history if "mean_token_accuracy" in entry
    ]
    peak = max(accuracies, default=0.0)
    tail = accuracies[-10:]
    log(f"  Token accuracy: peak {peak:.4%}, mean of the last {len(tail)} {sum(tail) / max(len(tail), 1):.4%}")
    return {"peak_token_accuracy": peak >= OVERFIT_MIN_PEAK_TOKEN_ACCURACY}


SUITE = SFTSuite(
    model_name=BAILING_MOE_RING_MINI,
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
    "fsdp": SFTMode(),
    "ep": SFTMode({"ep_size": 2}),
    "ep_no_gmm": SFTMode({"ep_size": 2, "use_grouped_gemm": False}),
    "overfit": SFTMode(
        {"ep_size": 2},
        sft_args={"max_steps": 40, "gradient_accumulation_steps": 1, "learning_rate": 5e-5},
        num_samples=(4, 4),
        extra_checks=peak_token_accuracy_check,
    ),
}


@gpu_test_main(exact_world_size=2, prefix="sft_bailing")
def run(ctx):
    return run_sft_suite(ctx, SUITE, MODES, default_mode="fsdp")


if __name__ == "__main__":
    run()
