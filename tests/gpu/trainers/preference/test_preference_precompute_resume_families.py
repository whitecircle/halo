#!/usr/bin/env python
"""DPO / KTO ``precompute_ref_log_probs`` resumed on two ranks, over every EP MoE family.

The precompute-resume body (``tests/common/preference_precompute_e2e.py``) on each family of
:data:`~tests.common.tiny_models.TINY_MOE_FAMILIES` (one per EP layer class;
``tests/cpu/conventions/test_tiny_family_roster.py`` holds the roster and these rows to the registry)
under ep2 and ep1, for DPO and KTO. The per-family sweep behind the representative rows of
``tests/gpu/parallelism/ep/test_ep_preference_precompute_resume.py``.

Run: torchrun --nproc_per_node=2 tests/gpu/trainers/preference/test_preference_precompute_resume_families.py \\
         --trainer dpo --family laguna --mode ep2
"""

from tests.common.harness import gpu_test_main
from tests.common.preference_precompute_e2e import precompute_resume_parser, run_precompute_resume, world_size
from tests.common.tiny_models import TINY_MOE_FAMILIES

ARGS = precompute_resume_parser(TINY_MOE_FAMILIES).parse_args()


@gpu_test_main(
    exact_world_size=world_size(ARGS.mode), prefix=f"pref_precompute_resume_{ARGS.trainer}_{ARGS.family}_{ARGS.mode}"
)
def run(ctx):
    return run_precompute_resume(
        ctx, trainer=ARGS.trainer, family=ARGS.family, mode=ARGS.mode, peft=ARGS.peft, kto_loss=ARGS.kto_loss
    )


if __name__ == "__main__":
    run()
