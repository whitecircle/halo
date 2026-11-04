#!/usr/bin/env python
"""DPO / KTO ``precompute_ref_log_probs`` resumed on every EP MoE family and the dense model; body and phases in
``tests/common/preference_precompute_e2e.py``.

``tests/cpu/conventions/test_tiny_family_roster.py`` holds the rows to the registry.
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
