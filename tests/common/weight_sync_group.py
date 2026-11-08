"""The live-server side of the one-group GPU suites: which servers a row drives, one group over them, and a timed push."""

import argparse
import time

import torch

from halo.env import env_str
from halo.weight_sync.clients.base import WeightSyncClientBase
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient
from halo.weight_sync.group_layout import SyncTopologySetting
from halo.weight_sync.pool import WeightSyncPool
from tests.common.weight_sync import SGLANG_SERVER_URL, VLLM_DENSE_SERVER_URL
from tests.common.weight_sync_reinit import CONNECTION_TIMEOUT_S

# Client, the knob listing a row's servers, their default, and the knob pinning the trainer-side group port.
BACKENDS = {
    "vllm": (VLLMWeightSyncClient, "HALO_TEST_VLLM_SERVER_URLS", VLLM_DENSE_SERVER_URL, "HALO_TEST_VLLM_GROUP_PORT"),
    "sglang": (
        SGLangWeightSyncClient,
        "HALO_TEST_SGLANG_SERVER_URLS",
        SGLANG_SERVER_URL,
        "HALO_TEST_SGLANG_GROUP_PORT",
    ),
}


def parse_server_args(*, default_servers: int | None = None) -> tuple[argparse.Namespace, list[str]]:
    """The row's ``--backend`` and ``--servers``, and every URL its backend's knob lists."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    parser.add_argument("--servers", type=int, default=default_servers, required=default_servers is None)
    args = parser.parse_args()
    _, knob, fallback, _ = BACKENDS[args.backend]
    return args, [url.strip() for url in (env_str(knob) or fallback).split(",") if url.strip()]


def form_group(
    client_cls: type[WeightSyncClientBase],
    urls: list[str],
    port: int,
    device,
    topology: SyncTopologySetting = "single_group",
) -> WeightSyncPool:
    pool = WeightSyncPool(
        [{"url": url} for url in urls],
        connection_timeout=CONNECTION_TIMEOUT_S,
        client_cls=client_cls,
        base_group_port=port,
        topology=topology,
    )
    pool.init_communicator(device)
    return pool


def timed_push(pool: WeightSyncPool, params: list[tuple[str, torch.Tensor]]) -> float:
    """Seconds of one sync of ``params``; an interrupted one is aborted first."""
    started = time.monotonic()
    try:
        for name, tensor in params:
            pool.update_named_param(name, tensor)
        pool.finish_weight_update()
    except BaseException:
        pool.abort_weight_update()
        raise
    return time.monotonic() - started
