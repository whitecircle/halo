#!/usr/bin/env python
"""Actual FSDP2/EP2 DPO scoring with FP32 masters, declared compute dtype and a full FP32-head oracle."""

import argparse

from tests.common.dpo_scoring import run_dpo_scoring_equivalence
from tests.common.harness import gpu_test_main

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--family", choices=("dense", "qwen3_moe"), required=True)
PARSER.add_argument("--dtype", choices=("float32", "bfloat16"), required=True)
ARGS = PARSER.parse_args()


@gpu_test_main(exact_world_size=2, prefix=f"dpo_chunked_equivalence_{ARGS.family}_{ARGS.dtype}")
def run(ctx):
    return run_dpo_scoring_equivalence(ctx, family=ARGS.family, dtype_name=ARGS.dtype)


if __name__ == "__main__":
    run()
