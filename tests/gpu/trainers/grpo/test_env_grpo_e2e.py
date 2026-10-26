#!/usr/bin/env python
"""Environmental GRPO end-to-end against a live vLLM or SGLang server, on two and on four ranks.

The shared body, and what these assert beyond the existing tier, is in :mod:`tests.common.env_grpo_e2e`.
Two ranks take one axis at a time: ``--ep-size 1`` trains the MoE with its experts inside the FSDP2 shard
(DTensor), ``--ep-size 2`` with them FSDP-ignored and routed through DeepEP. Both gather into the engine's
loader, and only the served policy can tell a correct gather from one the engine accepted and reverted.
Four ranks are the smallest world that holds two axes at once, each reaching the loader by its own route:

    --ep-size 2 --etp-size 2   a rank owns a SLICE of half the experts, so the gather has to
                               reassemble both splits before a tensor is the engine's
    --ep-size 2 --tp-size 2    attention sharded as DTensor while the experts stay FSDP-ignored plain
                               tensors — opposite paths on one model, plus gpt-oss's hand-sliced sinks
    --ep-size 4                one dispatch group spanning the whole domain, legal here because
                               ``ep_group_size == nvlink_domain_size == 4`` (a job this size cannot
                               form the multi-group shape the racy-EP gate refuses)

On SGLang every expert weight goes through the engine's own ``load_weights``, which takes each tensor as
it arrives and assembles nothing itself, and ``--ep-size 2`` runs DeepEP beside the SGLang group in one
process, which the server's cuMem parity (``docker-compose.sglang.yml``) makes possible.

The checkpoint, which the server must already serve: on vLLM ``HALO_TEST_ENV_GRPO_MODEL`` (the body's
default) on two ranks and ``HALO_TEST_ENV_GRPO_4GPU_MODEL`` (gpt-oss, which carries the most
engine-specific machinery: live frozen sinks in the sync stream, the layerwise-reload skip list, the fused
expert layout) on four; on SGLang ``HALO_TEST_ENV_GRPO_SGLANG_MODEL`` (gpt-oss). What a family's sync
puts on the wire is whatever its ``gather_expert_state_dict`` emits, and only a live engine shows the
loader consumed it.

Prerequisites (``make test-gpu-vllm`` / ``make test-gpu-sglang`` set these up):
    VLLM_CUDA_DEVICES=7 VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
        docker compose -f docker-compose.vllm.yml up -d vllm-server
    # --moe-backend triton is required for MoE: the FLASHINFER/CUTLASS paths repack expert weights
    # and corrupt an update.
    SGLANG_CUDA_DEVICES=7 SGLANG_MODEL=unsloth/gpt-oss-20b-BF16 \
        docker compose -f docker-compose.sglang.yml up -d
    # the NCCL-aligned halo:sglang image: upstream's NCCL differs from the training image's and the
    # weight-sync group will not form against it

Usage (trainer on GPUs the server does NOT own — a rank cannot NCCL broadcast to itself):
    CUDA_VISIBLE_DEVICES=0,1 NCCL_IB_DISABLE=1 NCCL_NET=Socket \
        torchrun --nproc_per_node=2 \
        tests/gpu/trainers/grpo/test_env_grpo_e2e.py --backend vllm --ep-size 2
"""

import argparse

from halo.env import env_str
from tests.common.env_grpo_e2e import MODEL_NAME, run_env_grpo_e2e
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B
from tests.common.weight_sync import SGLANG_SERVER_URL, weight_transfer_port

# Per backend: the server's URL and the knob pinning the trainer's group port.
SERVERS = {
    "vllm": (env_str("VLLM_SERVER_URL") or "http://localhost:8000", "HALO_TEST_VLLM_GROUP_PORT"),
    "sglang": (SGLANG_SERVER_URL, "HALO_TEST_SGLANG_GROUP_PORT"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=sorted(SERVERS), required=True)
parser.add_argument("--ep-size", type=int, choices=(1, 2, 4), default=1)
parser.add_argument("--tp-size", type=int, choices=(1, 2), default=1)
parser.add_argument("--etp-size", type=int, choices=(1, 2), default=1)
parser.add_argument("--peft", choices=("lora", "expert_lora"), default=None)
parser.add_argument("--resume", action="store_true")
# The engine enforces the CoT cap, so this row needs a vLLM server booted with a reasoning parser (see
# tests/common/thinking_budget.py); it reports the server's own 400 when it was not.
parser.add_argument("--thinking-budget", type=int, default=None)
# R3: the engine returns the experts it routed each token through, and the trainer replays them instead
# of its own router. Needs the server on --enable-return-routed-experts with the triton MoE backend
# (vLLM: VLLM_ENABLE_R3=1; SGLang: --moe-runner-backend triton, since the fused runners bypass the
# capture hook).
parser.add_argument("--routing-replay", choices=("none", "rollout"), default="none")
ARGS = parser.parse_args()
# The expert group, or attention TP over the dispatch group, and never fewer than two ranks.
WORLD_SIZE = max(2, ARGS.ep_size * max(ARGS.tp_size, ARGS.etp_size))
# The checkpoint the server must already serve.
if ARGS.backend == "sglang":
    MODEL = env_str("HALO_TEST_ENV_GRPO_SGLANG_MODEL", GPT_OSS_20B)
elif WORLD_SIZE == 4:
    MODEL = env_str("HALO_TEST_ENV_GRPO_4GPU_MODEL", GPT_OSS_20B)
else:
    MODEL = MODEL_NAME


@gpu_test_main(exact_world_size=WORLD_SIZE, prefix=f"env_grpo_{ARGS.backend}_e2e")
def run(ctx):
    server_url, group_port_knob = SERVERS[ARGS.backend]
    return run_env_grpo_e2e(
        ctx,
        backend=ARGS.backend,
        server_url=server_url,
        group_port=weight_transfer_port(group_port_knob),
        ep_size=ARGS.ep_size,
        tp_size=ARGS.tp_size,
        expert_tp_size=ARGS.etp_size,
        peft=ARGS.peft,
        resume=ARGS.resume,
        thinking_budget=ARGS.thinking_budget,
        routing_replay=ARGS.routing_replay,
        model_name=MODEL,
    )


if __name__ == "__main__":
    run()
