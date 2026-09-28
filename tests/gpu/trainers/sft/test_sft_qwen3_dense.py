#!/usr/bin/env python
"""SFT of Qwen3-0.6B (dense) under FSDP, TP=2 and CP=2, one ``--mode`` per launch.

Each mode checks, on the model, that its axis took effect (TP-sharded parameters, Ulysses attention
layers), then trains 10 steps on synthetic math conversations, evaluating every 5, and checks the
logged metrics: every loss and grad norm finite, the loss falling, the first loss in the range a
pretrained 0.6B model starts at on this data, no grad norm past the ceiling a missing TP reduction
blows through, token accuracy in [0, 1] and rising, and every eval loss finite. No loss is compared
against a reference or across modes. ``--mode all`` runs the three in one process.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_sft_qwen3_dense.py --mode tp
"""

from functools import partial

from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.sft_modes import SFTMode, SFTSuite, run_sft_suite, sft_metric_checks

MAX_STEPS = 10
EVAL_STEPS = 5
# Qwen3-0.6B on the math data starts near 4-5.
FIRST_LOSS_BAND = (1.0, 8.0)
# Healthy norms stay in the tens; a TP run whose partial sums skip their reduction logged ~16,000.
MAX_GRAD_NORM = 500.0

SUITE = SFTSuite(
    model_name=QWEN3_0_6B,
    attn_implementation="flash_attention_2",
    sft_args={
        "max_steps": MAX_STEPS,
        "learning_rate": 2e-5,
        "max_length": 4096,
        "eval_strategy": "steps",
        "eval_steps": EVAL_STEPS,
    },
    num_samples=(64, 16),
    evaluate=False,
    loss_decreased=True,
    extra_checks=partial(
        sft_metric_checks,
        first_loss_band=FIRST_LOSS_BAND,
        max_grad_norm=MAX_GRAD_NORM,
        token_accuracy_rises=True,
        eval_loss_logged=True,
    ),
)
MODES = {
    "fsdp": SFTMode(),
    "tp": SFTMode({"tp_size": 2}),
    "cp": SFTMode({"cp_size": 2}),
}


@gpu_test_main(min_world_size=2, prefix="test_sft_qwen3_dense")
def run(ctx):
    return run_sft_suite(ctx, SUITE, MODES, default_mode="all")


if __name__ == "__main__":
    run()
