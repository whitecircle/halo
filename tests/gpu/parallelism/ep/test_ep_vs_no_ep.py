#!/usr/bin/env python
"""gpt-oss-20b under EP=2 also against its no-EP load: per-token logits, top-1 and every router gradient. The
``full``-tier row of ``test_ep_modes_vs_reference.py``, a script of its own because a manifest entry has one tier.

    torchrun --nproc_per_node=2 tests/gpu/parallelism/ep/test_ep_vs_no_ep.py
"""

from functools import partial

from tests.common.harness import gpu_test_main
from tests.gpu.parallelism.ep.test_ep_modes_vs_reference import run_mode

run = gpu_test_main(exact_world_size=2, prefix="ep_vs_no_ep")(
    partial(run_mode, checkpoint_key="gpt_oss_20b", mode_key="ep", logits=True)
)


if __name__ == "__main__":
    run()
