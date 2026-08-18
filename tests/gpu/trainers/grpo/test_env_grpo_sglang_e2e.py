#!/usr/bin/env python
"""Environmental GRPO end-to-end against a live SGLang server, under plain FSDP2 (ep1 only —
any expert distribution is refused for the SGLang backend at construction).

The SGLang counterpart of ``test_env_grpo_vllm_e2e.py``, and the only coverage of MoE expert weights
going through SGLang's own ``load_weights``. vLLM needs a server-side patch for its expert sync to
land — an expert layer missing from the layerwise-reload skip list silently reverts every update —
and whether SGLang needs the equivalent is only answerable against a live engine. A no-op sync leaves
the served logprobs bit-identical, which is exactly what this asserts against.

The shared body, and what these assert beyond the existing tier, is in
:mod:`tests.common.env_grpo_e2e`.

gpt-oss, not the vLLM counterpart's Qwen3 MoE: SGLang loads MoE experts in the checkpoint-FUSED
layout, and ``EPGptOssMoELayer`` is the only layer implementing ``gather_fused_expert_state_dict``.
Every other MoE family is refused for this backend at construction (0.5.17's ``qwen3_moe`` loader
maps per-expert names only and drops fused keys), so pointing this test at one could assert nothing
but that refusal.

Prerequisites (``make test-gpu-sglang`` sets these up):
    SGLANG_CUDA_DEVICES=7 SGLANG_MODEL=unsloth/gpt-oss-20b-BF16 \
        docker compose -f docker-compose.sglang.yml up -d
    # the NCCL-aligned sglang-server image — upstream's NCCL is two minors behind the training image
    # and the weight-sync group will not form against it

Usage (trainer on GPUs the server does NOT own — a rank cannot NCCL broadcast to itself):
    CUDA_VISIBLE_DEVICES=0,1 NCCL_IB_DISABLE=1 NCCL_NET=Socket NCCL_NET_PLUGIN=none \
        NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \
        torchrun --nproc_per_node=2 \
        tests/gpu/trainers/grpo/test_env_grpo_sglang_e2e.py
"""

import argparse

from src.env import env_int, env_str
from tests.common.env_grpo_e2e import run_env_grpo_e2e
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B

# ``or`` (not an env_str default): an exported-but-empty SGLANG_SERVER_URL passes the conftest gate,
# which reads it the same way, so the client must fall back to the same URL rather than to "".
SERVER_URL = env_str("SGLANG_SERVER_URL") or "http://localhost:30000"
GROUP_PORT = env_int("HALO_TEST_SGLANG_GROUP_PORT", 51216)


@gpu_test_main(exact_world_size=2, prefix="env_grpo_sglang_e2e")
def run(ctx):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        # ep1 only: any expert distribution is refused for this backend at trainer
        # construction, so 2 could only ever produce a ValueError.
        "--ep-size",
        type=int,
        choices=(1,),
        default=1,
    )
    parser.add_argument("--tp-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--peft", choices=("lora",), default=None)
    parser.add_argument("--resume", action="store_true")
    # R3: the engine returns the experts it routed each token through, and the trainer replays them
    # instead of its own router. Needs the server on --enable-return-routed-experts and
    # --moe-runner-backend triton (the fused runners bypass the capture hook).
    parser.add_argument("--routing-replay", choices=("none", "rollout"), default="none")
    args = parser.parse_args()
    return run_env_grpo_e2e(
        ctx,
        backend="sglang",
        server_url=SERVER_URL,
        group_port=GROUP_PORT,
        ep_size=args.ep_size,
        tp_size=args.tp_size,
        peft=args.peft,
        resume=args.resume,
        routing_replay=args.routing_replay,
        model_name=GPT_OSS_20B,
    )


if __name__ == "__main__":
    run()
