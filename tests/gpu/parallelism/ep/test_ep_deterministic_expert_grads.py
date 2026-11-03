#!/usr/bin/env python
"""Under ``full_determinism``, an EP backward repeats bit for bit, expert weight gradients included; body in
``tests/common/ep_determinism.py``.

DeepEP's default dispatch claims receive slots with atomics, so without the deterministic buffer every expert gradient
sums in a different order.
"""

from tests.common.ep_determinism import WORLD_SIZE, determinism_parser, run_backward_replay
from tests.common.harness import gpu_test_main

ARGS = determinism_parser().parse_args()


@gpu_test_main(exact_world_size=WORLD_SIZE, prefix=f"ep_deterministic_grads_{ARGS.family}_{ARGS.mode}")
def run(ctx):
    return run_backward_replay(ctx, family=ARGS.family, mode=ARGS.mode)


if __name__ == "__main__":
    run()
