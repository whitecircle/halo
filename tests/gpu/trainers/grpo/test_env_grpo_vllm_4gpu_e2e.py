#!/usr/bin/env python
"""Environmental GRPO against a live vLLM server on FOUR ranks: the shapes two GPUs cannot form.

The 2-GPU file covers one axis at a time. Four ranks are the smallest world that can hold TWO at
once, and each combination reaches the engine's loader by a different route:

    --ep-size 2 --etp-size 2   a rank owns a SLICE of half the experts, so the gather has to
                               reassemble both splits before a tensor is the engine's
    --ep-size 2 --tp-size 2    attention sharded as DTensor while the experts stay FSDP-ignored plain
                               tensors — opposite paths on one model, plus gpt-oss's hand-sliced sinks
    --ep-size 4                one dispatch group spanning the whole domain, legal here because
                               ``ep_group_size == nvlink_domain_size == 4`` (a job this size cannot
                               form the multi-group shape the racy-EP gate refuses)

gpt-oss by default: it carries the most engine-specific machinery on this path (live frozen
pretrained sinks in the sync stream, the layerwise-reload skip list that keeps the server from
reverting them, the fused expert layout).

Prerequisites: the vLLM container serving the SAME checkpoint on a GPU outside
``CUDA_VISIBLE_DEVICES`` (a rank cannot NCCL broadcast to itself), with ``--moe-backend triton``.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,4,5 NCCL_IB_DISABLE=1 NCCL_NET=Socket \
        torchrun --nproc_per_node=4 \
        tests/gpu/trainers/grpo/test_env_grpo_vllm_4gpu_e2e.py --ep-size 2 --etp-size 2
"""

import argparse

from src.env import env_int, env_str
from tests.common.env_grpo_e2e import run_env_grpo_e2e
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B

MODEL_NAME = env_str("HALO_TEST_ENV_GRPO_4GPU_MODEL", GPT_OSS_20B)
SERVER_URL = env_str("VLLM_SERVER_URL") or "http://localhost:8000"
GROUP_PORT = env_int("HALO_TEST_VLLM_GROUP_PORT", 51240)


@gpu_test_main(exact_world_size=4, prefix="env_grpo_vllm_4gpu_e2e")
def run(ctx):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep-size", type=int, choices=(2, 4), default=2)
    parser.add_argument("--tp-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--etp-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--peft", choices=("lora", "expert_lora"), default=None)
    args = parser.parse_args()
    return run_env_grpo_e2e(
        ctx,
        backend="vllm",
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
