#!/usr/bin/env python
"""Repeated trainer connect → sync → disconnect cycles against ONE live vLLM server.

A long-lived server outlives its trainers: a crashed run resumed from a checkpoint, or a sequence of
short runs, reconnects to the same engine. Neither end of that weight-transfer group destroys its
``PyNcclCommunicator`` on its own — the class has no ``__del__``, and stock vLLM 0.32.0's
``init_transfer_engine`` only dereferences the previous one — so without
``docker/vllm/patches/vllm_weight_transfer_reinit_patch.py`` (server half) and
``VLLMWeightSyncClient.close_communicator`` (trainer half) every cycle strands a live NCCL
communicator until ``ncclCommInitRank`` fails outright, while ``/health`` keeps answering 200. The
device-memory checks below prove both halves; the served-policy checks prove the destroy does not
break re-init itself.

Requires the vLLM container serving the SAME dense checkpoint on a GPU outside
``CUDA_VISIBLE_DEVICES`` (a rank cannot NCCL broadcast to itself).

Usage:
    HALO_TEST_VLLM_DENSE_SERVER_URL=http://localhost:8010 NCCL_IB_DISABLE=1 NCCL_NET=Socket \
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
        tests/gpu/trainers/grpo/test_vllm_weight_transfer_reinit.py
"""

from src.env import env_int, env_str
from src.weight_sync.clients.vllm import VLLMWeightSyncClient
from tests.common.harness import gpu_test_main
from tests.common.weight_sync import VLLM_DENSE_SERVER_URL
from tests.common.weight_sync_reinit import VLLM_REINIT_MODEL, run_reinit_cycles

# The server's GPU as `nvidia-smi` indexes it. Unset (the tier's shape: the trainer owns every GPU
# but the server's) means "every GPU this process does not own", which is that one GPU.
SERVER_GPU = env_str("HALO_TEST_VLLM_SERVER_GPU")
# Enough cycles that a per-cycle leak is unmistakable next to the growth budget, and few enough to
# stay under a minute of the row.
CYCLES = env_int("HALO_TEST_VLLM_REINIT_CYCLES", 12)


def run(ctx) -> dict:
    return run_reinit_cycles(
        ctx,
        VLLMWeightSyncClient,
        server_url=VLLM_DENSE_SERVER_URL,
        model_name=VLLM_REINIT_MODEL,
        server_gpu=SERVER_GPU,
        cycles=CYCLES,
        group_port_knob="HALO_TEST_VLLM_GROUP_PORT",
    )


main = gpu_test_main(exact_world_size=1, prefix="vllm_weight_transfer_reinit", partial_state=False)(run)

if __name__ == "__main__":
    main()
