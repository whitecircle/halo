#!/usr/bin/env python
"""DPO / KTO ``precompute_ref_log_probs`` resumed on two ranks: the representative rows.

The precompute-resume body (``tests/common/preference_precompute_e2e.py``, whose docstring lists the
four phases) on the tiny Qwen3-MoE under ep2, ep1 (DTensor experts), etp2 and tp2, and on the tiny
dense Qwen3 under FSDP2 DP, TP and an attention-LoRA resume that builds the policy from the base;
KTO with and without its KL term. Every other MoE family runs the same body in
``tests/gpu/trainers/preference/test_preference_precompute_resume_families.py``.

Run: torchrun --nproc_per_node=2 tests/gpu/parallelism/ep/test_ep_preference_precompute_resume.py \\
         --trainer dpo --family qwen3_moe --mode ep2
"""

from tests.common.harness import gpu_test_main
from tests.common.preference_precompute_e2e import DENSE, WORLD_SIZE, precompute_resume_parser, run_precompute_resume

ARGS = precompute_resume_parser((DENSE, "qwen3_moe")).parse_args()


@gpu_test_main(exact_world_size=WORLD_SIZE, prefix=f"pref_precompute_resume_{ARGS.trainer}_{ARGS.family}_{ARGS.mode}")
def run(ctx):
    return run_precompute_resume(
        ctx, trainer=ARGS.trainer, family=ARGS.family, mode=ARGS.mode, peft=ARGS.peft, kto_loss=ARGS.kto_loss
    )


if __name__ == "__main__":
    run()
