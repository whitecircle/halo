#!/usr/bin/env python
"""A ``merge_expert_lora_on_save`` checkpoint serves and resumes exactly, on each tiny MoE family; body and phases in
``tests/common/merged_resume_e2e.py``."""

from tests.common.harness import gpu_test_main
from tests.common.merged_resume_e2e import WORLD_SIZE, merged_resume_parser, run_merged_resume

ARGS = merged_resume_parser().parse_args()


@gpu_test_main(
    exact_world_size=WORLD_SIZE,
    prefix=f"lora_merged_resume_{ARGS.family}_{ARGS.adapters}_ep{ARGS.ep_size}_cp{ARGS.cp_size}",
)
def run(ctx):
    return run_merged_resume(
        ctx,
        family=ARGS.family,
        adapters=ARGS.adapters,
        ep_size=ARGS.ep_size,
        cp_size=ARGS.cp_size,
        fp32_masters=ARGS.fp32_masters,
    )


if __name__ == "__main__":
    run()
