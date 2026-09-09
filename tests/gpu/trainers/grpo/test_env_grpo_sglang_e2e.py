#!/usr/bin/env python
"""Environmental GRPO end-to-end against a live SGLang server, under plain FSDP2 and under EP.

The SGLang counterpart of ``test_env_grpo_vllm_e2e.py``, and the only coverage of MoE expert weights
going through SGLang's own ``load_weights``. vLLM needs a server-side patch for its expert sync to
land — an expert layer missing from the layerwise-reload skip list silently reverts every update —
and whether SGLang needs the equivalent is only answerable against a live engine. A no-op sync leaves
the served logprobs bit-identical, which is exactly what this asserts against.

The shared body, and what these assert beyond the existing tier, is in
:mod:`tests.common.env_grpo_e2e`.

gpt-oss by default; ``HALO_TEST_ENV_GRPO_SGLANG_MODEL`` points it at any family the SGLang client
serves (the server must run the same checkpoint). What each family's sync puts on the wire is whatever
its ``gather_expert_state_dict`` emits — gpt-oss's interleaved fused pair, Qwen3's per-expert
tensors — and only a live engine shows the loader consumed it.

Prerequisites (``make test-gpu-sglang`` sets these up):
    SGLANG_CUDA_DEVICES=7 SGLANG_MODEL=unsloth/gpt-oss-20b-BF16 \
        docker compose -f docker-compose.sglang.yml up -d
    # the NCCL-aligned sglang-server image — upstream's NCCL is two minors behind the training image
    # and the weight-sync group will not form against it

Usage (trainer on GPUs the server does NOT own — a rank cannot NCCL broadcast to itself; the
server side needs cuMem parity, which docker-compose.sglang.yml sets):
    CUDA_VISIBLE_DEVICES=0,1 NCCL_IB_DISABLE=1 NCCL_NET=Socket \
        torchrun --nproc_per_node=2 \
        tests/gpu/trainers/grpo/test_env_grpo_sglang_e2e.py --ep-size 2
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
# Its own knob: the server and the default family differ from the vLLM leg's.
MODEL_NAME = env_str("HALO_TEST_ENV_GRPO_SGLANG_MODEL", GPT_OSS_20B)


@gpu_test_main(exact_world_size=2, prefix="env_grpo_sglang_e2e")
def run(ctx):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep-size", type=int, choices=(1, 2), default=1)
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
        model_name=MODEL_NAME,
    )


if __name__ == "__main__":
    run()
