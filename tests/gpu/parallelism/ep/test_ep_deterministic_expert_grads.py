#!/usr/bin/env python
"""Under ``full_determinism``, an EP backward repeats bit for bit, expert weight gradients included.

The backward-replay body (``tests/common/ep_determinism.py``) on GPT-OSS under every two-rank layout
and on Qwen3-MoE under ep2. DeepEP's default dispatch claims receive slots with atomics, so without
the deterministic buffer every expert weight gradient is summed in a different order on each pass;
the per-family sweep is ``test_ep_deterministic_expert_grads_families.py``.

    torchrun --nproc_per_node=2 \\
        tests/gpu/parallelism/ep/test_ep_deterministic_expert_grads.py --family gpt_oss --mode ep2
"""

from tests.common.ep_determinism import REPRESENTATIVE_FAMILIES, WORLD_SIZE, determinism_parser, run_backward_replay
from tests.common.harness import gpu_test_main

ARGS = determinism_parser(REPRESENTATIVE_FAMILIES).parse_args()


@gpu_test_main(exact_world_size=WORLD_SIZE, prefix=f"ep_deterministic_grads_{ARGS.family}_{ARGS.mode}")
def run(ctx):
    return run_backward_replay(ctx, family=ARGS.family, mode=ARGS.mode)


if __name__ == "__main__":
    run()
