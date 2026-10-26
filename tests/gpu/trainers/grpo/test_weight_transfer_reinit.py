#!/usr/bin/env python
"""Repeated trainer connect → sync → disconnect cycles against ONE live vLLM or SGLang server.

A long-lived server outlives its trainers: a crashed run resumed from a checkpoint, or a sequence of short
runs, reconnects to the same engine, and a close that leaves either end's half of the group alive fails a
later cycle while ``/health`` keeps answering 200. The device-memory checks catch a stranded
communicator; the served-policy checks prove each re-formed group still lands its sync.

* ``--backend vllm``: neither end of the weight-transfer group destroys its ``PyNcclCommunicator`` on its
  own (the class has no ``__del__``, and stock vLLM 0.32.0's ``init_transfer_engine`` only dereferences
  the previous one), so without ``docker/vllm/patches/vllm_weight_transfer_reinit_patch.py`` (server half)
  and ``VLLMWeightSyncClient.close_communicator`` (trainer half) every cycle strands a live NCCL
  communicator until ``ncclCommInitRank`` fails outright.
* ``--backend sglang``: every cycle forms a fresh c10d group under the same per-server group name on the
  same group port, and the engine refuses a second join under a name it holds, so each close has to
  release the engine's half (``/destroy_weights_update_group``, sent while the local destroy runs), the
  trainer's c10d registration and its rendezvous port; an engine that drops the name but not the
  communicator grows its device memory every cycle.

Requires the server serving the SAME dense checkpoint on a GPU outside ``CUDA_VISIBLE_DEVICES`` (a rank
cannot NCCL broadcast to itself; ``make test-gpu-vllm`` / ``make test-gpu-sglang`` set it up). The
``HALO_TEST_<ENGINE>_SERVER_GPU`` knob names the server's GPU as ``nvidia-smi`` indexes it; unset, the
tier's shape (the trainer owns every GPU but the server's) makes it every GPU this process does not own.

Usage:
    HALO_TEST_VLLM_DENSE_SERVER_URL=http://localhost:8010 NCCL_IB_DISABLE=1 NCCL_NET=Socket \
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
        tests/gpu/trainers/grpo/test_weight_transfer_reinit.py --backend vllm
"""

import argparse

from halo.env import env_int, env_str
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.weight_sync import SGLANG_SERVER_URL, VLLM_DENSE_SERVER_URL
from tests.common.weight_sync_reinit import VLLM_REINIT_MODEL, run_reinit_cycles

# Enough cycles that a per-cycle leak is unmistakable next to the growth budget, and few enough to stay
# under a minute of the row.
DEFAULT_CYCLES = 12
# Per backend: the client and run_reinit_cycles' keyword arguments.
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
