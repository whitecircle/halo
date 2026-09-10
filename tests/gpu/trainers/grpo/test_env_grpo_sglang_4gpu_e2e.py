#!/usr/bin/env python
"""Environmental GRPO against a live SGLang server on FOUR ranks: the shapes two GPUs cannot form.

The SGLang counterpart of ``test_env_grpo_vllm_4gpu_e2e.py``: EP+ETP (a rank owns a slice of half
the experts, so the gather reassembles both splits), EP+TP (DTensor attention beside FSDP-ignored
plain experts) and EP=4 (one dispatch group spanning the domain), each pushed into an engine whose
loader takes every tensor as it arrives and assembles nothing itself.

Prerequisites: the SGLang container serving the SAME checkpoint on a GPU outside
``CUDA_VISIBLE_DEVICES``, from this repo's ``Dockerfile.sglang`` image.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_IB_DISABLE=1 NCCL_NET=Socket \\
        torchrun --nproc_per_node=4 \\
        tests/gpu/trainers/grpo/test_env_grpo_sglang_4gpu_e2e.py --ep-size 2 --etp-size 2
"""

import argparse

from src.env import env_int, env_str
from tests.common.env_grpo_e2e import run_env_grpo_e2e
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B

SERVER_URL = env_str("SGLANG_SERVER_URL") or "http://localhost:30000"
GROUP_PORT = env_int("HALO_TEST_SGLANG_GROUP_PORT", 51240)
# The 2-GPU SGLang wrapper's knob: same server, same default family.
MODEL_NAME = env_str("HALO_TEST_ENV_GRPO_SGLANG_MODEL", GPT_OSS_20B)


@gpu_test_main(exact_world_size=4, prefix="env_grpo_sglang_4gpu_e2e")
def run(ctx):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep-size", type=int, choices=(2, 4), default=2)
    parser.add_argument("--tp-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--etp-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--peft", choices=("lora", "expert_lora"), default=None)
    args = parser.parse_args()
    return run_env_grpo_e2e(
        ctx,
        backend="sglang",
        server_url=SERVER_URL,
        group_port=GROUP_PORT,
        ep_size=args.ep_size,
        tp_size=args.tp_size,
        expert_tp_size=args.etp_size,
        peft=args.peft,
        model_name=MODEL_NAME,
    )


if __name__ == "__main__":
    run()
