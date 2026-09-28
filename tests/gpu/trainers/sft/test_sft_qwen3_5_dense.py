#!/usr/bin/env python
"""SFT of Qwen3.5-2B (dense) under FSDP and TP=2; ``--mode all`` (the default) runs both.

Qwen3.5-2B: 24 decoder layers, 18 linear-attention (GatedDeltaNet) and 6 full-attention, a dense MLP
and a double-width ``q_proj`` (query + sigmoid gate) that the Ulysses wrapper does not take, so there
is no CP mode. FA2 crashes on its M-RoPE varlen path, so it runs sdpa, and Liger has no ``qwen3_5``
kernels.

Each mode checks, on the model, that its axis took effect, then trains 10 steps and checks the logged
metrics: every loss and grad norm finite, the loss falling, the first loss in range, no grad norm past
the ceiling a missing TP reduction blows through, and token accuracy in [0, 1].

Usage:
    torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_sft_qwen3_5_dense.py --mode tp

Requirements: 2x GPUs, causal-conv1d and flash-linear-attention (GatedDeltaNet), Qwen/Qwen3.5-2B.
"""

from functools import partial

from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_5_2B
from tests.common.sft_modes import SFTMode, SFTSuite, run_sft_suite, sft_metric_checks

FIRST_LOSS_BAND = (0.5, 12.0)
MAX_GRAD_NORM = 500.0

SUITE = SFTSuite(
    model_name=QWEN3_5_2B,
    attn_implementation="sdpa",
    sft_args={"max_steps": 10, "learning_rate": 2e-5, "max_length": 2048},
    num_samples=(64, 0),
    use_liger_kernel=False,
    evaluate=False,
    loss_decreased=True,
    extra_checks=partial(sft_metric_checks, first_loss_band=FIRST_LOSS_BAND, max_grad_norm=MAX_GRAD_NORM),
)
MODES = {
    "fsdp": SFTMode(),
    "tp": SFTMode({"tp_size": 2}),
}


@gpu_test_main(min_world_size=2, prefix="test_sft_qwen3_5_dense")
def run(ctx):
    return run_sft_suite(ctx, SUITE, MODES, default_mode="all")


if __name__ == "__main__":
    run()
