#!/usr/bin/env python
"""DPO / KTO ``precompute_ref_log_probs`` resumed under EP+ETP on four ranks, over every EP MoE family.

The precompute-resume body (``tests/common/preference_precompute_e2e.py``) at ep2etp2: each expert lives
on one of two dispatch ranks with its FFN split two ways, an expert group of four that the two-rank sweep
(``test_preference_precompute_resume_families.py``) cannot hold. Each family's expert-TP split under its
own dispatch trains through the real trainer, saves gathered, and resumes from that save.

Run: torchrun --nproc_per_node=4 tests/gpu/trainers/preference/test_preference_precompute_resume_ep_etp.py \\
         --trainer dpo --family laguna
"""

from tests.common.harness import gpu_test_main
from tests.common.preference_precompute_e2e import precompute_resume_parser, run_precompute_resume, world_size
from tests.common.tiny_models import TINY_MOE_FAMILIES

ARGS = precompute_resume_parser(TINY_MOE_FAMILIES, world=4).parse_args()


@gpu_test_main(
    exact_world_size=world_size(ARGS.mode), prefix=f"pref_precompute_resume_{ARGS.trainer}_{ARGS.family}_{ARGS.mode}"
)
def run(ctx):
    return run_precompute_resume(
        ctx, trainer=ARGS.trainer, family=ARGS.family, mode=ARGS.mode, peft=ARGS.peft, kto_loss=ARGS.kto_loss
    )


if __name__ == "__main__":
    run()
