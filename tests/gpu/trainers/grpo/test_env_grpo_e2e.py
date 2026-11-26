#!/usr/bin/env python
"""Environmental GRPO against a live vLLM or SGLang server, on two and four ranks; body in
:mod:`tests.common.env_grpo_e2e`.

Only the served policy can tell a correct gather from one the engine accepted and reverted. ``--ep-size 4`` is legal
because ``ep_group_size == nvlink_domain_size``. Server setup is ``make test-gpu-vllm`` / ``make test-gpu-sglang``. MoE
needs ``--moe-backend triton``, since FLASHINFER and CUTLASS repack expert weights and corrupt an update; hermes leaves
gpt-oss's calls as text; upstream SGLang's NCCL differs, so the group won't form without the halo:sglang image.
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
# The server's URL and the knob pinning the trainer's group port.
SERVERS = {
    "vllm": (VLLM_SERVER_URL, "HALO_TEST_VLLM_GROUP_PORT"),
    "sglang": (SGLANG_SERVER_URL, "HALO_TEST_SGLANG_GROUP_PORT"),
}
# Each URL serves the row's checkpoint on a GPU of its own.
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
# The engine enforces the CoT cap, so this row needs a vLLM server with a reasoning parser
# (tests/common/thinking_budget.py); it reports the server's own 400 otherwise.
parser.add_argument("--thinking-budget", type=int, default=None)
# R3: the trainer replays the experts the engine routed each token through. Needs --enable-return-routed-experts with
# the triton MoE backend (vLLM: VLLM_ENABLE_R3=1; SGLang: --moe-runner-backend triton, as fused runners bypass the
# capture hook).
parser.add_argument("--routing-replay", choices=("none", "recompute", "rollout"), default="none")
# Each round rolls out while the steps before it train, across the push (vLLM freezes it, SGLang aborts and re-issues).
parser.add_argument("--prefetch-depth", type=int, default=None)
# One row per micro-batch, two a round: the shape the length deal reorders. --length-ordered false keeps TRL's shuffle,
# for the replay flip-rate comparison.
parser.add_argument("--chunked", action="store_true")
parser.add_argument("--length-ordered", choices=("unset", "false"), default="unset")
# Logs phase_time/* and checks them: every phase present, the slowest rank never under the mean, and what no phase
# holds a small share of the step.
parser.add_argument("--phase-times", action="store_true")
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
    # ParallelismConfig refuses FULL_SHARD and HSDP beside an expert group or TP; the ep1 expert knob does nothing past
    # ep1.
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
        phase_times=ARGS.phase_times,
        model_name=MODEL,
    )


if __name__ == "__main__":
    if len(POOL_URLS) < ARGS.servers:
        log(f"SKIP: --servers {ARGS.servers} needs {ARGS.servers} URLs in HALO_TEST_VLLM_SERVER_URLS, got {POOL_URLS}")
        sys.exit(0)
    run()
