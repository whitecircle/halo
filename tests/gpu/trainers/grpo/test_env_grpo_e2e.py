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

The data-parallel half of the layout rides on top. ``--zero3`` runs FSDP2 FULL_SHARD and
``--unsharded-ep1-experts`` keeps ep1 experts whole and FSDP-ignored (both ep1 only, the shapes the shipped
full-ep1 recipes train); ``--world 4`` adds DP replicas past the expert group (``--tp-size 2 --world 4`` is
TP x DP, ``--ep-size 2 --world 4`` two EP groups, ``--gpus-per-node 2`` puts each in its own simulated NVLink
domain, ``--hsdp`` a 2-D HSDP mesh over two such domains); ``--fp32-router`` / ``--fp32-experts`` keep those
masters fp32; ``--servers N`` pushes through a WeightSyncPool over the first N of
``HALO_TEST_VLLM_SERVER_URLS``, each serving the row's checkpoint. The body reads each of them off the trained
policy (``record_dp_layout``) before the served-policy rounds.

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
    VLLM_CUDA_DEVICES=7 VLLM_MODEL=unsloth/gpt-oss-20b-BF16 <the Makefile's VLLM_GPTOSS_PARSERS> \
        docker compose -f docker-compose.vllm.yml up -d vllm-server
    # the four-rank rows' server: the compose default tool parser (hermes) leaves gpt-oss's calls as
    # text; another family takes its parser from agent-docs/infrastructure/rollout-servers.md
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
import sys

from halo.env import env_str
from tests.common.env_grpo_e2e import MODEL_NAME, run_env_grpo_e2e
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B
from tests.common.on_policy_e2e import PolicyLayout
from tests.common.rank_log import log
from tests.common.weight_sync import SGLANG_SERVER_URL, weight_transfer_port

VLLM_SERVER_URL = env_str("VLLM_SERVER_URL") or "http://localhost:8000"
# Per backend: the server's URL and the knob pinning the trainer's group port.
SERVERS = {
    "vllm": (VLLM_SERVER_URL, "HALO_TEST_VLLM_GROUP_PORT"),
    "sglang": (SGLANG_SERVER_URL, "HALO_TEST_SGLANG_GROUP_PORT"),
}
# A --servers row's pool, each URL serving the row's checkpoint on a GPU of its own.
POOL_URLS = [
    url.strip() for url in (env_str("HALO_TEST_VLLM_SERVER_URLS") or VLLM_SERVER_URL).split(",") if url.strip()
]

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
# enable_prefetch: true at this depth on the one server: each round rolls out while the steps before it train,
# across the push (vLLM freezes it, SGLang aborts and re-issues its in-flight turns).
parser.add_argument("--prefetch-depth", type=int, default=None)
# Chunked log-probs at one row per micro-batch, two a round: each row is its own forward, the shape the length
# deal reorders. --length-ordered false is the arm that keeps TRL's shuffle, for the replay flip-rate comparison.
parser.add_argument("--chunked", action="store_true")
parser.add_argument("--length-ordered", choices=("unset", "false"), default="unset")
parser.add_argument("--zero3", action="store_true", help="fsdp_reshard_after_forward=True (FULL_SHARD)")
parser.add_argument("--unsharded-ep1-experts", action="store_true", help="fsdp_shard_ep1_experts=False")
parser.add_argument("--world", type=int, default=None, help="ranks, a multiple of the expert group and of TP")
parser.add_argument("--hsdp", action="store_true", help="use_hsdp=True over two simulated NVLink domains")
parser.add_argument("--gpus-per-node", type=int, default=0, help="simulated NVLink domain width (0: the launch's)")
parser.add_argument("--fp32-router", action="store_true", help="ep_fp32_router=True")
parser.add_argument("--fp32-experts", action="store_true", help="ep_fp32_experts=True")
parser.add_argument("--servers", type=int, default=1, help="vLLM servers the WeightSyncPool pushes to")
ARGS = parser.parse_args()
if ARGS.length_ordered != "unset" and not ARGS.chunked:
    parser.error("--length-ordered sets the deal of a --chunked row; a padded row never deals")
# The expert group, or attention TP over the dispatch group, and never fewer than two ranks.
MIN_WORLD = max(2, ARGS.ep_size * max(ARGS.tp_size, ARGS.etp_size))
WORLD_SIZE = ARGS.world or MIN_WORLD
if WORLD_SIZE % MIN_WORLD:
    parser.error(f"--world {WORLD_SIZE} must be a multiple of the expert group and TP width ({MIN_WORLD})")
replicated = ARGS.ep_size == 1 and ARGS.tp_size == 1 and ARGS.etp_size == 1
if (ARGS.zero3 or ARGS.unsharded_ep1_experts or ARGS.hsdp) and not replicated:
    # ParallelismConfig refuses FULL_SHARD and HSDP beside an expert group or TP, and the ep1 expert knob
    # does nothing past ep1.
    parser.error("--zero3, --unsharded-ep1-experts and --hsdp run at --ep-size 1 without --tp-size / --etp-size")
if ARGS.hsdp and WORLD_SIZE < 4:
    parser.error("--hsdp needs --world 4 or more: two simulated NVLink domains of at least two ranks")
if ARGS.servers > 1 and ARGS.backend != "vllm":
    parser.error("--servers pools vLLM endpoints (HALO_TEST_VLLM_SERVER_URLS)")
LAYOUT = PolicyLayout(
    ep_size=ARGS.ep_size,
    tp_size=ARGS.tp_size,
    expert_tp_size=ARGS.etp_size,
    reshard_after_forward=ARGS.zero3,
    shard_ep1_experts=not ARGS.unsharded_ep1_experts,
    use_hsdp=ARGS.hsdp,
    gpus_per_node=ARGS.gpus_per_node or (WORLD_SIZE // 2 if ARGS.hsdp else 0),
    fp32_router=ARGS.fp32_router,
    fp32_experts=ARGS.fp32_experts,
)
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
    pool = tuple(POOL_URLS[: ARGS.servers]) if ARGS.servers > 1 else ()
    return run_env_grpo_e2e(
        ctx,
        backend=ARGS.backend,
        server_url=pool[0] if pool else server_url,
        group_port=weight_transfer_port(group_port_knob),
        layout=LAYOUT,
        peft=ARGS.peft,
        resume=ARGS.resume,
        thinking_budget=ARGS.thinking_budget,
        routing_replay=ARGS.routing_replay,
        pool_urls=pool,
        prefetch_depth=ARGS.prefetch_depth,
        chunked=ARGS.chunked,
        length_ordered=None if ARGS.length_ordered == "unset" else False,
        model_name=MODEL,
    )


if __name__ == "__main__":
    if len(POOL_URLS) < ARGS.servers:
        log(f"SKIP: --servers {ARGS.servers} needs {ARGS.servers} URLs in HALO_TEST_VLLM_SERVER_URLS, got {POOL_URLS}")
        sys.exit(0)
    run()
