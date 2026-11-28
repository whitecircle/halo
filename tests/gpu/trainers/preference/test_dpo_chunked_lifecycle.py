#!/usr/bin/env python
"""Native bounded DPO train, eval, save, Path-B resume and stock HF export on dense DP2 or Qwen3-MoE EP2."""

import argparse

from tests.common.harness import gpu_test_main
from tests.common.preference_precompute_e2e import run_precompute_resume

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--family", choices=("dense", "qwen3_moe"), required=True)
ARGS = PARSER.parse_args()


@gpu_test_main(exact_world_size=2, prefix=f"dpo_chunked_lifecycle_{ARGS.family}")
def run(ctx):
    return run_precompute_resume(
        ctx,
        trainer="dpo",
        family=ARGS.family,
        mode="dp2" if ARGS.family == "dense" else "ep2",
        peft=False,
        kto_loss="kto",
        chunked_dpo=True,
    )


if __name__ == "__main__":
    run()
