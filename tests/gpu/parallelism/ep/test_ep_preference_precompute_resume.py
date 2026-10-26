#!/usr/bin/env python
"""DPO / KTO ``precompute_ref_log_probs`` resumed, on every EP MoE family and the dense model.

The precompute-resume body (``tests/common/preference_precompute_e2e.py``, whose docstring lists the four
phases). The core rows: the tiny Qwen3-MoE under ep2, ep1 (DTensor experts), etp2 and tp2, and the tiny
dense Qwen3 under FSDP2 DP, TP and an attention-LoRA resume that builds the policy from the base; KTO with
and without its KL term. The full rows: every other family of
:data:`~tests.common.tiny_models.TINY_MOE_FAMILIES` under each two-rank layout (TP and EP+TP where its
attention has a TP plan), and every family at ep2etp2, an expert group of four ranks, each expert on one of
two dispatch ranks with its FFN split two ways. ``tests/cpu/conventions/test_tiny_family_roster.py`` holds
the rows to the registry.

Run: torchrun --nproc_per_node=2 tests/gpu/parallelism/ep/test_ep_preference_precompute_resume.py \\
         --trainer dpo --family qwen3_moe --mode ep2
"""

from tests.common.harness import gpu_test_main
from tests.common.preference_precompute_e2e import precompute_resume_parser, run_precompute_resume, world_size

ARGS = precompute_resume_parser().parse_args()


@gpu_test_main(
    exact_world_size=world_size(ARGS.mode), prefix=f"pref_precompute_resume_{ARGS.trainer}_{ARGS.family}_{ARGS.mode}"
)
def run(ctx):
    return run_precompute_resume(
        ctx, trainer=ARGS.trainer, family=ARGS.family, mode=ARGS.mode, peft=ARGS.peft, kto_loss=ARGS.kto_loss
    )


if __name__ == "__main__":
    run()
