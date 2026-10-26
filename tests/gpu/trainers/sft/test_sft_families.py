#!/usr/bin/env python
"""SFT of one real checkpoint per family under its parallel shapes, one ``--suite`` and ``--mode`` per launch.

Each mode checks, on the model, that its shape took effect (experts wrapped and split EP-way, TP-sharded
parameters, Ulysses attention layers), then trains and checks the finished run
(:func:`tests.common.sft_modes.run_sft_suite`): every step ran with finite losses and grad norms, and the
final eval loss is finite. No loss is compared against a reference. ``--mode all`` runs every mode of a
suite in one process.

    qwen3_dense     Qwen3-0.6B under fsdp, tp (TP=2) and cp (CP=2); 10 steps evaluated every 5, then the
                    logged metrics: the loss falling, the first loss in the range a pretrained 0.6B model
                    starts at on this data, no grad norm past the ceiling a missing TP reduction blows
                    through, token accuracy in [0, 1] and rising, every eval loss finite.
    qwen3_5_dense   Qwen3.5-2B (18 GatedDeltaNet and 6 full-attention layers, a double-width ``q_proj`` the
                    Ulysses wrapper does not take, so no CP) under fsdp and tp; the same metric checks
                    over 10 steps, without the eval ones.
    qwen3_moe       Qwen3-30B-A3B-Instruct-2507 (128 experts, 8 active) at ep2.
    qwen3_5_moe     Qwen3.5-35B-A3B (30 GatedDeltaNet and 10 full-attention layers, each with a 256-expert
                    top-8 MoE) under ep, ep_no_gmm (the per-expert loop) and etp. Attention TP leaves every
                    expert replicated with no data-parallel axis to shard them over, which two GPUs cannot
                    hold at this size; the tiny-model precompute-resume sweep runs Qwen3.5 under TP.
    qwen3_5_ep      a Qwen3.5/3.6 MoE at ``ep_size`` = world size (``HALO_TEST_QWEN3_5_MODEL`` points it at
                    another checkpoint, e.g. a local ``$HALO_DATA_ROOT/models/Qwen3.6-35B-A3B-patched``).
    glm4_moe        GLM-4.7-Flash (GLM4 MoE Lite) at ``ep_size`` = world size (``HALO_TEST_GLM4_MODEL``).
    gemma4_moe      Gemma4-26B-A4B at ``ep_size`` = world size: a VLM-wrapped MoE whose decoder layers inline
                    their router and experts (EPGemma4MoELayer), trained on text-only inputs with the vision
                    and audio towers unused, from a local checkpoint at
                    ``$HALO_DATA_ROOT/models/gemma-4-26B-A4B-it-patched`` (``HALO_TEST_GEMMA4_MODEL``).
    lfm2_moe        LFM2-24B-A2B (40 hybrid conv + full-attention layers, two dense, the rest a 64-expert
                    top-4 MoE with sigmoid routing and an expert bias) under fsdp and ep.
    bailing_moe     Ring-mini-linear-2.0 (BailingMoeLinearV2, remote code: 16 linear- and 4 full-attention
                    layers, layer 0 dense, the rest a 256-expert top-8 MoE with one shared expert and sigmoid
                    group-limited routing) under fsdp, ep and ep_no_gmm, and overfit (EP=2, 4 samples for
                    40 steps, which must reach a peak logged token accuracy of 99%).

``HALO_TEST_EP`` sets ``ep_size`` on the suites that default it to the world size. FA2 crashes on the
Qwen3.5 family's M-RoPE varlen path (cudaErrorIllegalAddress), so those suites run sdpa.

Requirements: DeepEP for the EP modes; causal-conv1d and flash-linear-attention for GatedDeltaNet and
Bailing's linear attention.

Usage:
    torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_sft_families.py --suite qwen3_dense --mode tp
"""

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial

from halo.env import env_int, env_str
from halo.model_loading.remote_code import apply_remote_code_compat_shims
from tests.common.datasets import VERBOSE_MATH_TEMPLATES, create_single_turn_sft_dataset
from tests.common.harness import gpu_test_main, skip_unless_local_checkpoint
from tests.common.models import (
    BAILING_MOE_RING_MINI,
    GEMMA4_26B_A4B_PATCHED,
    GLM4_FLASH,
    LFM2_24B_A2B,
    QWEN3_0_6B,
    QWEN3_5_2B,
    QWEN3_5_MOE_35B,
    QWEN3_30B_A3B,
)
from tests.common.rank_log import log
from tests.common.sft_modes import SFTMode, SFTSuite, run_sft_suite, sft_metric_checks

EP_SIZE_OVERRIDE = env_int("HALO_TEST_EP", None)
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


@dataclass(frozen=True)
class FamilySuite:
    """One ``--suite``: the shared run, its modes (``None``: one ``ep`` mode at ``ep_size`` = the world unless
    ``HALO_TEST_EP`` sets it), the mode a bare launch runs, the world sizes it takes, and the knob of a local
    checkpoint whose absence makes the launch a skip."""

    suite: SFTSuite
    modes: Mapping[str, SFTMode] | None
    default_mode: str
    min_world_size: int = 2
    exact_world_size: int | None = 2
    local_checkpoint_knob: str | None = None


# The five-step EP smoke the large MoE suites share.
SMOKE_ARGS = {
    "max_steps": 5,
    "gradient_accumulation_steps": 2,
    "learning_rate": 1e-5,
    "warmup_steps": 1,
    "max_length": 4096,
    "remove_unused_columns": False,
}
# The three-step smoke of the suites sized to the world, on short single-turn samples.
SHORT_SMOKE_ARGS = {**SMOKE_ARGS, "max_steps": 3, "max_length": 1024}
VERBOSE_MATH = partial(create_single_turn_sft_dataset, templates=VERBOSE_MATH_TEMPLATES)

SUITES = {
    "qwen3_dense": FamilySuite(
        SFTSuite(
            model_name=QWEN3_0_6B,
            attn_implementation="flash_attention_2",
            sft_args={
                "max_steps": 10,
                "learning_rate": 2e-5,
                "max_length": 4096,
                "eval_strategy": "steps",
                "eval_steps": 5,
            },
            num_samples=(64, 16),
            evaluate=False,
            loss_decreased=True,
            # Qwen3-0.6B on the math data starts near 4-5; healthy norms stay in the tens, and a TP run
            # whose partial sums skip their reduction logged ~16,000.
            extra_checks=partial(
                sft_metric_checks,
                first_loss_band=(1.0, 8.0),
                max_grad_norm=500.0,
                token_accuracy_rises=True,
                eval_loss_logged=True,
            ),
        ),
        {"fsdp": SFTMode(), "tp": SFTMode({"tp_size": 2}), "cp": SFTMode({"cp_size": 2})},
        default_mode="all",
        exact_world_size=None,
    ),
    "qwen3_5_dense": FamilySuite(
        SFTSuite(
            model_name=QWEN3_5_2B,
            attn_implementation="sdpa",
            sft_args={"max_steps": 10, "learning_rate": 2e-5, "max_length": 2048},
            num_samples=(64, 0),
            # Liger has no qwen3_5 kernels.
            use_liger_kernel=False,
            evaluate=False,
            loss_decreased=True,
            # One batch of the TP run spikes the pre-clip norm to 190-540 depending on the rounding
            # stream; a missing TP reduction lands far above the ceiling.
            extra_checks=partial(sft_metric_checks, first_loss_band=(0.5, 12.0), max_grad_norm=1000.0),
        ),
        {"fsdp": SFTMode(), "tp": SFTMode({"tp_size": 2})},
        default_mode="all",
        exact_world_size=None,
    ),
    "qwen3_moe": FamilySuite(
        SFTSuite(
            model_name=QWEN3_30B_A3B,
            attn_implementation="flash_attention_2",
            sft_args=SMOKE_ARGS,
            dataset=VERBOSE_MATH,
        ),
        {"ep": SFTMode({"ep_size": 2})},
        default_mode="ep",
    ),
    "qwen3_5_moe": FamilySuite(
        SFTSuite(model_name=QWEN3_5_MOE_35B, attn_implementation="sdpa", sft_args=SMOKE_ARGS),
        {
            "ep": SFTMode({"ep_size": 2}),
            "ep_no_gmm": SFTMode({"ep_size": 2, "use_grouped_gemm": False}),
            "etp": SFTMode({"ep_size": 1, "expert_tp_size": 2}),
        },
        default_mode="ep",
    ),
    "qwen3_5_ep": FamilySuite(
        SFTSuite(
            model_name=env_str("HALO_TEST_QWEN3_5_MODEL", QWEN3_5_MOE_35B),
            attn_implementation="sdpa",
            sft_args=SHORT_SMOKE_ARGS,
            num_samples=(16, 4),
            dataset=create_single_turn_sft_dataset,
        ),
        None,
        default_mode="ep",
        exact_world_size=None,
    ),
    "glm4_moe": FamilySuite(
        SFTSuite(
            model_name=env_str("HALO_TEST_GLM4_MODEL", GLM4_FLASH),
            attn_implementation="flash_attention_2",
            # FLCE-only model: its Liger forward returns no logits, and with the flag off TRL's metric path
            # slices the missing logits. The loader already applied the kernels, so none are reapplied.
            sft_args={**SMOKE_ARGS, "use_liger_kernel": True},
            dataset=VERBOSE_MATH,
        ),
        None,
        default_mode="ep",
        min_world_size=1,
        exact_world_size=None,
    ),
    "gemma4_moe": FamilySuite(
        SFTSuite(
            model_name=env_str("HALO_TEST_GEMMA4_MODEL", GEMMA4_26B_A4B_PATCHED),
            # The global heads use head_dim=512, past FA2's 256 limit; sdpa takes any head dim.
            attn_implementation="sdpa",
            sft_args={**SHORT_SMOKE_ARGS, "gradient_checkpointing": False},
            num_samples=(16, 4),
            dataset=create_single_turn_sft_dataset,
            # Liger has no Gemma4-specific kernels.
            use_liger_kernel=False,
        ),
        None,
        default_mode="ep",
        min_world_size=1,
        exact_world_size=None,
        local_checkpoint_knob="HALO_TEST_GEMMA4_MODEL",
    ),
    "lfm2_moe": FamilySuite(
        SFTSuite(model_name=LFM2_24B_A2B, attn_implementation="flash_attention_2", sft_args=SMOKE_ARGS),
        {"fsdp": SFTMode(), "ep": SFTMode({"ep_size": 2})},
        default_mode="fsdp",
    ),
    "bailing_moe": FamilySuite(
        SFTSuite(model_name=BAILING_MOE_RING_MINI, attn_implementation="sdpa", sft_args=SMOKE_ARGS),
        {
            "fsdp": SFTMode(),
            "ep": SFTMode({"ep_size": 2}),
            "ep_no_gmm": SFTMode({"ep_size": 2, "use_grouped_gemm": False}),
            "overfit": SFTMode(
                {"ep_size": 2},
                sft_args={"max_steps": 40, "gradient_accumulation_steps": 1, "learning_rate": 5e-5},
                num_samples=(4, 4),
                extra_checks=peak_token_accuracy_check,
            ),
        },
        default_mode="fsdp",
    ),
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--suite", choices=sorted(SUITES), required=True)
parser.add_argument("--mode", help="one of the suite's modes, or all (default: the suite's own)")
ARGS = parser.parse_args()
FAMILY = SUITES[ARGS.suite]
MODE = ARGS.mode or FAMILY.default_mode
if MODE not in (*(("ep",) if FAMILY.modes is None else FAMILY.modes), "all"):
    parser.error(f"--suite {ARGS.suite} has no mode {MODE!r}")
# Before the tokenizer load: remote modeling modules (Ring-mini-linear's) are repaired as they load.
apply_remote_code_compat_shims()


@gpu_test_main(
    min_world_size=FAMILY.min_world_size, exact_world_size=FAMILY.exact_world_size, prefix=f"sft_{ARGS.suite}"
)
def run(ctx):
    ep_size = ctx.world_size if EP_SIZE_OVERRIDE is None else EP_SIZE_OVERRIDE
    modes = FAMILY.modes or {"ep": SFTMode({"ep_size": ep_size})}
    return run_sft_suite(ctx, FAMILY.suite, modes, MODE)


if __name__ == "__main__":
    if FAMILY.local_checkpoint_knob is not None:
        skip_unless_local_checkpoint(FAMILY.suite.model_name, FAMILY.local_checkpoint_knob)
    run()
