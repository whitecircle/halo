#!/usr/bin/env python
"""Test: a ``merge_expert_lora_on_save`` checkpoint serves AND resumes exactly (tiny MoE, 2 ranks).

The representative rows of the merged-resume body (``tests/common/merged_resume_e2e.py``, whose
docstring lists the four phases): the per-expert unfused (Qwen3-MoE) and interleaved fused
(GptOss, with attention sinks) expert layouts, expert-only and mixed adapters, at ``--ep-size 2``,
at ``--ep-size 1`` (DTensor experts) and under EP+CP (``--cp-size 2``). Every other family runs the
same body in ``test_lora_merged_save_resume_families.py``.

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_lora_merged_save_resume.py --family qwen3_moe --adapters mixed --ep-size 2
"""

from tests.common.harness import gpu_test_main
from tests.common.merged_resume_e2e import WORLD_SIZE, merged_resume_parser, run_merged_resume

ARGS = merged_resume_parser(("gpt_oss", "qwen3_moe")).parse_args()


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
