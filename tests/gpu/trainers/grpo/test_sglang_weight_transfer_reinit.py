#!/usr/bin/env python
"""Repeated trainer connect → sync → disconnect cycles against ONE live SGLang server.

Every cycle forms a fresh c10d weight-update group under the same per-server group name on the same
group port. The engine registers a group by name and refuses a second join under it, so each close has
to release the engine's half (``/destroy_weights_update_group``, sent while the local destroy runs),
the trainer's c10d registration and its rendezvous port; a close that misses one fails the next cycle,
and an engine that drops the name but not the communicator grows its device memory every cycle while
``/health`` keeps answering 200. The device-memory checks catch the leak; the served-policy checks prove
each re-formed group still lands its sync.

Requires the SGLang container serving the SAME dense checkpoint on a GPU outside
``CUDA_VISIBLE_DEVICES`` (a rank cannot NCCL broadcast to itself; ``make test-gpu-sglang`` sets it up).

Usage:
    SGLANG_SERVER_URL=http://localhost:30000 NCCL_IB_DISABLE=1 NCCL_NET=Socket \
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
        tests/gpu/trainers/grpo/test_sglang_weight_transfer_reinit.py
"""

from src.env import env_int, env_str
from src.weight_sync.clients.sglang import SGLangWeightSyncClient
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.weight_sync import SGLANG_SERVER_URL
from tests.common.weight_sync_reinit import run_reinit_cycles

MODEL_NAME = env_str("HALO_TEST_SGLANG_REINIT_MODEL", QWEN3_0_6B)
# The server's GPU as `nvidia-smi` indexes it; unset means every GPU this process does not own.
SERVER_GPU = env_str("HALO_TEST_SGLANG_SERVER_GPU")
CYCLES = env_int("HALO_TEST_SGLANG_REINIT_CYCLES", 12)


def run(ctx) -> dict:
    return run_reinit_cycles(
        ctx,
        SGLangWeightSyncClient,
        server_url=SGLANG_SERVER_URL,
        model_name=MODEL_NAME,
        server_gpu=SERVER_GPU,
        cycles=CYCLES,
        group_port_knob="HALO_TEST_SGLANG_GROUP_PORT",
    )


main = gpu_test_main(exact_world_size=1, prefix="sglang_weight_transfer_reinit", partial_state=False)(run)

if __name__ == "__main__":
    main()
