#!/usr/bin/env python
"""Inkling's expert FFNs expert-TP sharded 2-way against the stock model (ep2 x etp2 on 4 GPUs, pure ETP on 2): the
family body of ``test_ep_family_vs_reference.py``, apart because a manifest entry has one world size.

    torchrun --nproc_per_node=4 tests/gpu/parallelism/combined/test_ep_etp_inkling.py
"""

from functools import partial

from tests.common.harness import gpu_test_main
from tests.common.tolerances import TOL
from tests.gpu.parallelism.ep.test_ep_family_vs_reference import run_family

run = gpu_test_main(min_world_size=2, prefix="ep_etp_inkling")(
    partial(run_family, family="inkling_text", cos_min=TOL.ep_grad_cosine_min, expert_tp_size=2)
)


if __name__ == "__main__":
    run()
