#!/usr/bin/env python
"""Under ``full_determinism``, an EP backward repeats bit for bit, over every EP MoE family.

The backward-replay body (``tests/common/ep_determinism.py``) under ep2 on each family of
:data:`~tests.common.tiny_models.TINY_MOE_FAMILIES` beyond the representative rows of
``test_ep_deterministic_expert_grads.py``: each family's own expert kernels (bias gathers, clamped
GLUs, separate halves, external routers) sit between the deterministic dispatch and the expert weight
gradients.

    torchrun --nproc_per_node=2 \\
        tests/gpu/parallelism/ep/test_ep_deterministic_expert_grads_families.py --family laguna
"""

from tests.common.ep_determinism import REPRESENTATIVE_FAMILIES, WORLD_SIZE, determinism_parser, run_backward_replay
from tests.common.harness import gpu_test_main
from tests.common.tiny_models import TINY_MOE_FAMILIES

ARGS = determinism_parser(set(TINY_MOE_FAMILIES) - set(REPRESENTATIVE_FAMILIES)).parse_args()


@gpu_test_main(exact_world_size=WORLD_SIZE, prefix=f"ep_deterministic_grads_{ARGS.family}_{ARGS.mode}")
def run(ctx):
    return run_backward_replay(ctx, family=ARGS.family, mode=ARGS.mode)


if __name__ == "__main__":
    run()
