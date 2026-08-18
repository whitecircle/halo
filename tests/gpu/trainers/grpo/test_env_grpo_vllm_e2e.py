#!/usr/bin/env python
"""Environmental GRPO end-to-end against a live vLLM server, under FSDP2 and under EP.

``--ep-size 1`` trains the MoE with its experts inside the FSDP2 shard (DTensor), ``--ep-size 2``
with them FSDP-ignored and routed through DeepEP. Both gather into the same vLLM loader, and only
the served policy can tell a correct gather from one the engine accepted and reverted.

The shared body, and what these assert beyond the existing tier, is in
:mod:`tests.common.env_grpo_e2e`.

Prerequisites (``make test-gpu-vllm`` sets these up):
    VLLM_CUDA_DEVICES=7 VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
        docker compose -f docker-compose.vllm.yml up -d vllm-server
    # --moe-backend triton is required for MoE: the FLASHINFER/CUTLASS paths repack expert weights
    # and corrupt an update.

Usage (trainer on GPUs the server does NOT own — a rank cannot NCCL broadcast to itself):
    CUDA_VISIBLE_DEVICES=0,1 NCCL_IB_DISABLE=1 NCCL_NET=Socket \
        torchrun --nproc_per_node=2 \
        tests/gpu/trainers/grpo/test_env_grpo_vllm_e2e.py --ep-size 2
"""

import argparse

from src.env import env_int, env_str
from tests.common.env_grpo_e2e import run_env_grpo_e2e
from tests.common.harness import gpu_test_main

SERVER_URL = env_str("VLLM_SERVER_URL") or "http://localhost:8000"
GROUP_PORT = env_int("HALO_TEST_VLLM_GROUP_PORT", 51216)


@gpu_test_main(exact_world_size=2, prefix="env_grpo_vllm_e2e")
def run(ctx):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--tp-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--etp-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--peft", choices=("lora", "expert_lora"), default=None)
    parser.add_argument("--resume", action="store_true")
    # The engine enforces the CoT cap, so this row needs a server booted with a reasoning parser and
    # VLLM_USE_V2_MODEL_RUNNER=0 (see tests/common/thinking_budget.py); it reports the server's own
    # 400 when it was not.
    parser.add_argument("--thinking-budget", type=int, default=None)
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
        resume=args.resume,
        thinking_budget=args.thinking_budget,
    )


if __name__ == "__main__":
    run()
