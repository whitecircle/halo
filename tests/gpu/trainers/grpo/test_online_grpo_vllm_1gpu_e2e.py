#!/usr/bin/env python
"""Online GRPO, SDPG and Environmental GRPO in one process (no FSDP wrap) against live vLLM servers of Qwen3-0.6B.

``--trainer {online,sdpg}`` rows run :mod:`tests.common.online_grpo_e2e`, ``--trainer environmental`` the ReAct math
environment through :mod:`tests.common.env_grpo_e2e`, pushing through a WeightSyncPool over the first ``--servers``
URLs of ``HALO_TEST_VLLM_SERVER_URLS``. Each row first probes every server it drives.
"""

import argparse
import sys

import requests

from src.env import env_str
from tests.common.env_grpo_e2e import run_env_grpo_e2e
from tests.common.harness import gpu_test_main, record_check
from tests.common.models import QWEN3_0_6B
from tests.common.online_grpo_e2e import MODES, modes_for, run_online_grpo_e2e
from tests.common.ports import free_port
from tests.common.utils import log
from tests.common.weight_sync import VLLM_DENSE_SERVER_URL

MODEL_NAME = env_str("HALO_TEST_ONLINE_GRPO_DENSE_MODEL", QWEN3_0_6B)
# Comma-separated, each serving MODEL_NAME on a GPU of its own; the first is the one every row drives.
SERVER_URLS = [
    url.strip() for url in (env_str("HALO_TEST_VLLM_SERVER_URLS") or VLLM_DENSE_SERVER_URL).split(",") if url.strip()
]
WEIGHT_TRANSFER_ROUTES = ("/init_weight_transfer_engine", "/update_weights", "/get_world_size")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer", choices=("online", "sdpg", "environmental"), required=True)
    parser.add_argument("--mode", choices=modes_for("world1"), required=True)
    parser.add_argument("--servers", type=int, default=1, help="environmental: the WeightSyncPool's server count")
    return parser.parse_args()


def probe_server(url: str) -> None:
    """The server is healthy, exposes the native weight-transfer routes, and answers a chat request."""
    requests.get(f"{url}/health", timeout=10).raise_for_status()
    routes = requests.get(f"{url}/openapi.json", timeout=10).json()["paths"]
    missing = [route for route in WEIGHT_TRANSFER_ROUTES if route not in routes]
    assert not missing, f"{url} lacks the weight-transfer routes {missing}"
    reply = requests.post(
        f"{url}/v1/chat/completions",
        json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "What is 1+1?"}], "max_tokens": 64},
        timeout=30,
    ).json()["choices"][0]["message"]
    # A reasoning parser routes a plain reply into ``reasoning``, leaving ``content`` empty.
    assert reply.get("content") or reply.get("reasoning") or reply.get("reasoning_content"), f"{url}: {reply}"


def run(ctx) -> dict:
    args = _parse_args()
    servers = SERVER_URLS[: args.servers]
    checks: dict[str, bool] = {}
    for url in servers:
        record_check(checks, f"server_ready:{url}", lambda url=url: probe_server(url))
    if args.trainer == "environmental":
        result = run_env_grpo_e2e(
            ctx,
            backend="vllm",
            server_url=servers[0],
            group_port=free_port(),
            ep_size=1,
            peft=MODES[args.mode].adapter,
            model_name=MODEL_NAME,
            environment_type="react_math",
            pool_urls=tuple(servers),
        )
    else:
        result = run_online_grpo_e2e(
            ctx,
            trainer_kind=args.trainer,
            mode=args.mode,
            resume=False,
            server_url=servers[0],
            group_port=free_port(),
            model_name=MODEL_NAME,
        )
    return {**result, "checks": checks | result["checks"]}


main = gpu_test_main(exact_world_size=1, prefix="online_grpo_vllm_1gpu_e2e")(run)

if __name__ == "__main__":
    pool = _parse_args().servers
    if len(SERVER_URLS) < pool:
        log(f"SKIP: --servers {pool} needs {pool} URLs in HALO_TEST_VLLM_SERVER_URLS, got {SERVER_URLS}")
        sys.exit(0)
    main()
