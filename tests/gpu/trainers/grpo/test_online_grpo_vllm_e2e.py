#!/usr/bin/env python
"""Online GRPO / SDPG (and, in one process, Environmental GRPO) against live vLLM servers.

Every row asserts on the SERVER's behaviour, not on the trainer's tensors: the gather-side weight-sync
tests check what ``gather_and_send_weights`` produces, never that a real engine applied it, and a server
that accepts an update and silently reverts it makes every sync a no-op — the served policy frozen for a
whole RL run while the training loss moves normally. ``--mode`` picks the policy, and with it the server:

    MoE (Qwen3-30B-A3B, ``VLLM_SERVER_URL``, two ranks, 2 GPUs of >=140 GB):
    full_ep2            full fine-tune at ``ep_size=2`` — the EP expert gather into the engine
    lora_ep2            attention PEFT under EP — the sync's fold → strip
    expert_lora_ep2     native grouped expert LoRA — the ``merge_lora=True`` fold in the expert gather,
                        the only route an expert adapter has into the engine
    lora_etp2           attention PEFT at ``ep_size=1, expert_tp_size=2``, plus the config-time refusal
                        of native expert LoRA under expert TP

    dense (Qwen3-0.6B, ``HALO_TEST_VLLM_DENSE_SERVER_URL``, two ranks):
    full_tp2            full fine-tune at ``tp_size=2`` — the sync gathers DTensor shards
    lora_fsdp           attention PEFT under plain FSDP2 DP — the merge path over DTensor adapters
    lora_tp2_rejected   ``tp_size=2`` + attention LoRA must raise at construction, naming TP
    full_fsdp_kl        full fine-tune under plain FSDP2 DP at ``beta=0.01`` — TRL scores the KL against the
                        frozen reference the trainer hands it

    one process, no FSDP wrap (Qwen3-0.6B, the first ``--servers`` URLs of ``HALO_TEST_VLLM_SERVER_URLS``):
    full_world1, lora_world1   ``--trainer environmental`` runs the ReAct math environment through
                        :mod:`tests.common.env_grpo_e2e`, pushing through a WeightSyncPool; each row
                        first probes every server it drives

``--resume`` (two-rank modes only) drives the two-phase resume invariant: the checkpoint's weights must generate
the first resumed rollout. Each server must serve the SAME checkpoint the row trains, with ``--moe-backend triton``
for MoE, on a GPU outside ``CUDA_VISIBLE_DEVICES`` (a rank cannot NCCL broadcast to itself).

Usage:
    VLLM_SERVER_URL=http://localhost:8000 NCCL_IB_DISABLE=1 NCCL_NET=Socket \
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
        tests/gpu/trainers/grpo/test_online_grpo_vllm_e2e.py --trainer online --mode lora_ep2
"""

import argparse
import sys

import requests

from halo.env import env_str
from tests.common.env_grpo_e2e import run_env_grpo_e2e
from tests.common.harness import gpu_test_main, record_check
from tests.common.models import QWEN3_0_6B, QWEN3_30B_A3B
from tests.common.online_grpo_e2e import MODES, modes_for, run_online_grpo_e2e
from tests.common.ports import free_port
from tests.common.rank_log import log
from tests.common.weight_sync import VLLM_DENSE_SERVER_URL, weight_transfer_port

MOE_MODEL = env_str("HALO_TEST_ONLINE_GRPO_MOE_MODEL", QWEN3_30B_A3B)
DENSE_MODEL = env_str("HALO_TEST_ONLINE_GRPO_DENSE_MODEL", QWEN3_0_6B)
MOE_SERVER_URL = env_str("VLLM_SERVER_URL") or "http://localhost:8000"
# Comma-separated, each serving DENSE_MODEL on a GPU of its own; the first is the one every one-process row
# drives.
SERVER_URLS = [
    url.strip() for url in (env_str("HALO_TEST_VLLM_SERVER_URLS") or VLLM_DENSE_SERVER_URL).split(",") if url.strip()
]
WEIGHT_TRANSFER_ROUTES = ("/init_weight_transfer_engine", "/update_weights", "/get_world_size")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--trainer", choices=("online", "sdpg", "environmental"), required=True)
parser.add_argument("--mode", choices=sorted(MODES), required=True)
parser.add_argument("--resume", action="store_true", help="drive the two-phase resume invariant")
parser.add_argument("--servers", type=int, default=1, help="environmental: the WeightSyncPool's server count")
ARGS = parser.parse_args()
POLICY = MODES[ARGS.mode].family
if POLICY != "world1" and (ARGS.trainer == "environmental" or ARGS.servers != 1):
    parser.error("--trainer environmental and --servers run in one process: a *_world1 mode")
if POLICY == "world1" and ARGS.resume:
    parser.error("--resume drives a two-rank mode; the one-process body has no resume row")


def _row_index() -> int:
    """This row's place in its policy's matrix. A pinned group port is offset by it, so back-to-back rows of
    one pass never bind the same port; a resume row rebinds its own for phase 2, which phase 1's
    close_communicator makes available again on both ends."""
    rows = [(t, m, r) for t in ("online", "sdpg") for m in modes_for(POLICY) for r in (False, True)]
    return rows.index((ARGS.trainer, ARGS.mode, ARGS.resume))


def probe_server(url: str) -> None:
    """The server is healthy, exposes the native weight-transfer routes, and answers a chat request."""
    requests.get(f"{url}/health", timeout=10).raise_for_status()
    routes = requests.get(f"{url}/openapi.json", timeout=10).json()["paths"]
    missing = [route for route in WEIGHT_TRANSFER_ROUTES if route not in routes]
    assert not missing, f"{url} lacks the weight-transfer routes {missing}"
    reply = requests.post(
        f"{url}/v1/chat/completions",
        json={"model": DENSE_MODEL, "messages": [{"role": "user", "content": "What is 1+1?"}], "max_tokens": 64},
        timeout=30,
    ).json()["choices"][0]["message"]
    # A reasoning parser routes a plain reply into ``reasoning``, leaving ``content`` empty.
    assert reply.get("content") or reply.get("reasoning") or reply.get("reasoning_content"), f"{url}: {reply}"


def _run_one_process(ctx) -> dict:
    servers = SERVER_URLS[: ARGS.servers]
    checks: dict[str, bool] = {}
    for url in servers:
        record_check(checks, f"server_ready:{url}", lambda url=url: probe_server(url))
    if ARGS.trainer == "environmental":
        result = run_env_grpo_e2e(
            ctx,
            backend="vllm",
            server_url=servers[0],
            group_port=free_port(),
            ep_size=1,
            peft=MODES[ARGS.mode].adapter,
            model_name=DENSE_MODEL,
            environment_type="react_math",
            pool_urls=tuple(servers),
        )
    else:
        result = run_online_grpo_e2e(
            ctx,
            trainer_kind=ARGS.trainer,
            mode=ARGS.mode,
            resume=False,
            server_url=servers[0],
            group_port=free_port(),
            model_name=DENSE_MODEL,
        )
    return {**result, "checks": checks | result["checks"]}


@gpu_test_main(exact_world_size=1 if POLICY == "world1" else 2, prefix=f"online_grpo_vllm_{POLICY}_e2e")
def run(ctx) -> dict:
    if POLICY == "world1":
        return _run_one_process(ctx)
    server_url, model_name = (MOE_SERVER_URL, MOE_MODEL) if POLICY == "moe" else (VLLM_DENSE_SERVER_URL, DENSE_MODEL)
    return run_online_grpo_e2e(
        ctx,
        trainer_kind=ARGS.trainer,
        mode=ARGS.mode,
        resume=ARGS.resume,
        server_url=server_url,
        group_port=weight_transfer_port("HALO_TEST_VLLM_GROUP_PORT", offset=_row_index()),
        model_name=model_name,
    )


if __name__ == "__main__":
    if len(SERVER_URLS) < ARGS.servers:
        log(
            f"SKIP: --servers {ARGS.servers} needs {ARGS.servers} URLs in HALO_TEST_VLLM_SERVER_URLS, got {SERVER_URLS}"
        )
        sys.exit(0)
    run()
