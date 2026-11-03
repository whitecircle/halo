#!/usr/bin/env python
"""gpt-oss-20b at EP=2 against its no-EP load: the ``full``-tier row of ``test_ep_modes_vs_reference.py``, its own
script because a manifest entry has one tier."""

from functools import partial

from tests.common.harness import gpu_test_main
from tests.gpu.parallelism.ep.test_ep_modes_vs_reference import run_mode

run = gpu_test_main(exact_world_size=2, prefix="ep_vs_no_ep")(
    partial(run_mode, checkpoint_key="gpt_oss_20b", mode_key="ep", logits=True)
)


if __name__ == "__main__":
    run()
