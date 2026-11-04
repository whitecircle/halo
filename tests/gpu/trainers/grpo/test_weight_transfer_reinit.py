#!/usr/bin/env python
"""Repeated trainer connect, sync, disconnect cycles against one live vLLM or SGLang server.

A close leaving either end's half alive fails a later cycle while ``/health`` stays 200. On vLLM each cycle would
strand a communicator until ``ncclCommInitRank`` fails. SGLang refuses a second join under a held group name, so each
close must release the engine's half (``/destroy_weights_update_group``), the c10d registration and the rendezvous
port.
"""

import argparse

from halo.env import env_int, env_str
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.weight_sync import SGLANG_SERVER_URL, VLLM_DENSE_SERVER_URL
from tests.common.weight_sync_reinit import VLLM_REINIT_MODEL, run_reinit_cycles

# A per-cycle leak is unmistakable next to the growth budget, under a minute of the row.
DEFAULT_CYCLES = 12
# The client and run_reinit_cycles' keyword arguments.
BACKENDS = {
    "vllm": (
        VLLMWeightSyncClient,
        {
            "server_url": VLLM_DENSE_SERVER_URL,
            "model_name": VLLM_REINIT_MODEL,
            "server_gpu": env_str("HALO_TEST_VLLM_SERVER_GPU"),
            "cycles": env_int("HALO_TEST_VLLM_REINIT_CYCLES", DEFAULT_CYCLES),
            "group_port_knob": "HALO_TEST_VLLM_GROUP_PORT",
        },
    ),
    "sglang": (
        SGLangWeightSyncClient,
        {
            "server_url": SGLANG_SERVER_URL,
            "model_name": env_str("HALO_TEST_SGLANG_REINIT_MODEL", QWEN3_0_6B),
            "server_gpu": env_str("HALO_TEST_SGLANG_SERVER_GPU"),
            "cycles": env_int("HALO_TEST_SGLANG_REINIT_CYCLES", DEFAULT_CYCLES),
            "group_port_knob": "HALO_TEST_SGLANG_GROUP_PORT",
        },
    ),
}


def run(ctx) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    client_class, kwargs = BACKENDS[parser.parse_args().backend]
    return run_reinit_cycles(ctx, client_class, **kwargs)


main = gpu_test_main(exact_world_size=1, prefix="weight_transfer_reinit", partial_state=False)(run)

if __name__ == "__main__":
    main()
