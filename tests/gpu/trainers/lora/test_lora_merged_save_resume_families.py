#!/usr/bin/env python
"""Test: the merge-on-save serve-and-resume body over every EP MoE family (tiny models, 2 ranks).

``tests/common/merged_resume_e2e.py`` on each family of :data:`~tests.common.tiny_models.TINY_MOE_FAMILIES`
(one per EP layer class; ``tests/cpu/conventions/test_tiny_family_roster.py`` holds the roster and
these rows to the registry), expert-only and mixed adapters, at ``--ep-size 2``, ``--ep-size 1``
(DTensor experts) and EP+CP (``--cp-size 2``). The per-family sweep behind the representative rows
of ``test_lora_merged_save_resume.py``.

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_lora_merged_save_resume_families.py --family laguna --adapters expert
"""

from tests.common.harness import gpu_test_main
from tests.common.merged_resume_e2e import WORLD_SIZE, merged_resume_parser, run_merged_resume
from tests.common.tiny_models import TINY_MOE_FAMILIES

ARGS, _ = merged_resume_parser(TINY_MOE_FAMILIES).parse_known_args()


@gpu_test_main(
    exact_world_size=WORLD_SIZE,
    prefix=f"lora_merged_resume_{ARGS.family}_{ARGS.adapters}_ep{ARGS.ep_size}_cp{ARGS.cp_size}",
)
def run(ctx):
    return run_merged_resume(
        ctx, family=ARGS.family, adapters=ARGS.adapters, ep_size=ARGS.ep_size, cp_size=ARGS.cp_size
    )


if __name__ == "__main__":
    run()
