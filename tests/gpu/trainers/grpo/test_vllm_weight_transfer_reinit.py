#!/usr/bin/env python
"""Repeated trainer connect → sync → disconnect cycles against ONE live vLLM server.

A long-lived server outlives its trainers: a crashed run resumed from a checkpoint, or a sequence of
short runs, reconnects to the same engine. Both ends of that weight-transfer group used to strand
their ``PyNcclCommunicator`` on every cycle — the class has no ``__del__``, stock vLLM 0.26.0's
``init_transfer_engine`` only dereferences the previous one, and this client's ``close_communicator``
did the same — so each cycle left a live NCCL communicator behind until ``ncclCommInitRank`` failed
outright, while ``/health`` kept answering 200. The server half is fixed by
``docker/vllm/patches/vllm_weight_transfer_reinit_patch.py``, the trainer half by
``VLLMWeightSyncClient.close_communicator``; the device-memory checks below are what prove both, and
the served-policy checks prove the destroy did not break re-init itself.

Requires the vLLM container serving the SAME dense checkpoint on a GPU outside
``CUDA_VISIBLE_DEVICES`` (a rank cannot NCCL broadcast to itself).

Usage:
    HALO_TEST_VLLM_DENSE_SERVER_URL=http://localhost:8010 NCCL_IB_DISABLE=1 NCCL_NET=Socket \
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
        tests/gpu/trainers/grpo/test_vllm_weight_transfer_reinit.py
"""

import subprocess

import torch
from transformers import AutoModelForCausalLM

from src.distributed.nccl.clients.vllm import VLLMWeightSyncClient
from src.env import env_int, env_str
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.on_policy_e2e import probe_top_logprobs
from tests.common.utils import log

MODEL_NAME = env_str("HALO_TEST_VLLM_REINIT_MODEL", QWEN3_0_6B)
# The dense endpoint of the vLLM tier: this file asserts on logprobs only a 0.6B server can produce.
SERVER_URL = env_str("HALO_TEST_VLLM_DENSE_SERVER_URL") or env_str("VLLM_SERVER_URL") or "http://localhost:8010"
# Trainer-side weight-transfer port, rebound by every cycle — itself part of the invariant, since
# close_communicator must release it on both ends or the next cycle cannot form its group.
GROUP_PORT = env_int("HALO_TEST_VLLM_GROUP_PORT", 51228)
# The server's GPU as `nvidia-smi` indexes it. Unset (the tier's shape: the trainer owns every GPU
# but the server's) means "every GPU this process does not own", which is that one GPU.
SERVER_GPU = env_str("HALO_TEST_VLLM_SERVER_GPU")
# Enough cycles that a per-cycle leak is unmistakable next to the tolerance below, and few enough to
# stay under a minute of the row.
CYCLES = env_int("HALO_TEST_VLLM_REINIT_CYCLES", 12)

# Measured: an unpatched pair strands ~633 MiB of NCCL communicator per cycle on EACH side
# (a vLLM 0.26.0 server grew 89,166 -> 248,078 MiB over 251 cycles), so the CYCLES above overshoot
# this budget ~7x while an idle engine's own footprint does not move at all.
MAX_MEMORY_GROWTH_MIB = 1024

# The final RMSNorm scales every logit, so a cycle's push shifts the whole top-k without reordering
# it into a degenerate distribution the next cycle could no longer move.
PERTURBED_PARAM = "model.norm.weight"
PERTURBATION = 1.02

CONNECTION_TIMEOUT_S = 120.0


def _server_gpu_used_mib() -> int:
    """Used device memory on the server's GPU(s), read out of process.

    The server holds a GPU outside ``CUDA_VISIBLE_DEVICES`` (it must — a rank cannot broadcast to
    itself), so torch cannot see it here; ``nvidia-smi`` enumerates every GPU regardless, the way
    ``src/distributed/nvlink.py`` reads the fabric state.
    """
    query = ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"]
    if SERVER_GPU:
        query += ["-i", SERVER_GPU]
    rows = subprocess.run(query, capture_output=True, text=True, timeout=30, check=True).stdout
    owned = {str(torch.cuda.get_device_properties(i).uuid) for i in range(torch.cuda.device_count())}
    used = 0
    for row in rows.splitlines():
        _, uuid, memory = (field.strip() for field in row.split(","))
        if SERVER_GPU or uuid.removeprefix("GPU-") not in owned:
            used += int(memory)
    return used


def _trainer_gpu_used_mib(device: torch.device) -> int:
    """Used device memory on this rank's GPU, driver-level — NCCL's allocations sit outside torch's."""
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) // 2**20


def _sync_one_param(name: str, weight: torch.Tensor, device: torch.device) -> None:
    """One whole trainer lifetime against the server: connect, one quiesced update, disconnect."""
    client = VLLMWeightSyncClient(base_url=SERVER_URL, group_port=GROUP_PORT, connection_timeout=CONNECTION_TIMEOUT_S)
    client.init_communicator(device=device)
    try:
        client.update_named_param(name, weight)
        client.reset_prefix_cache()
    finally:
        client.close_communicator()


def _probe_or_none() -> dict[str, float] | None:
    """The served policy's fingerprint, or ``None`` if the server can no longer produce one."""
    try:
        return probe_top_logprobs(SERVER_URL, MODEL_NAME)
    except Exception as e:  # noqa: BLE001 — the verdict is "unusable", whatever the engine failed with
        log(f"served-policy probe failed: {type(e).__name__}: {e}")
        return None


def run(ctx) -> dict:
    log(f"vLLM weight-transfer re-init: {CYCLES} connect/sync/disconnect cycles against {SERVER_URL}")
    checks: dict[str, bool] = {}

    baseline = probe_top_logprobs(SERVER_URL, MODEL_NAME)
    # Anti-vacuity: the probe is greedy against an idle server, so a later difference is the sync
    # landing and not sampling noise.
    checks["probe_reproducible_before_any_sync"] = probe_top_logprobs(SERVER_URL, MODEL_NAME) == baseline

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(ctx.device)
    param = dict(model.named_parameters())[PERTURBED_PARAM]
    original = param.detach().clone()

    probes = [baseline]
    errors: list[str] = []
    # Both baselines are taken after the first cycle: the first sync loads the NCCL library, sizes
    # the packed staging buffers and warms both engines' allocators, and only what a SECOND and
    # later cycle adds on top of that is a leak.
    server_before = trainer_before = 0
    for cycle in range(1, CYCLES + 1):
        with torch.no_grad():
            param.mul_(PERTURBATION)
        try:
            _sync_one_param(PERTURBED_PARAM, param.detach(), ctx.device)
        except Exception as e:  # noqa: BLE001 — a refused re-init is the defect under test, not an error
            errors.append(f"cycle {cycle}: {type(e).__name__}: {e}")
            log(errors[-1])
            break
        if cycle == 1:
            server_before, trainer_before = _server_gpu_used_mib(), _trainer_gpu_used_mib(ctx.device)
        probes.append(probe_top_logprobs(SERVER_URL, MODEL_NAME))

    server_growth = _server_gpu_used_mib() - server_before
    trainer_growth = _trainer_gpu_used_mib(ctx.device) - trainer_before

    # The restore is a cycle of its own, and its exact-equality verdict is what proves the engine
    # reloaded the pushed tensor rather than drifting under repeated layerwise reloads.
    restored: dict[str, float] | None = None
    if not errors:
        try:
            _sync_one_param(PERTURBED_PARAM, original, ctx.device)
        except Exception as e:  # noqa: BLE001 — same verdict as a failed cycle
            errors.append(f"restore cycle: {type(e).__name__}: {e}")
            log(errors[-1])
        else:
            restored = probe_top_logprobs(SERVER_URL, MODEL_NAME)

    completed = len(probes) - 1
    checks["every_cycle_reconnected_and_synced"] = not errors
    checks["every_cycle_moved_the_served_policy"] = completed == CYCLES and all(
        probes[i] != probes[i - 1] for i in range(1, len(probes))
    )
    checks["served_policy_restored_by_a_final_sync"] = restored == baseline
    checks["server_frees_its_communicator_each_cycle"] = completed == CYCLES and server_growth <= MAX_MEMORY_GROWTH_MIB
    checks["trainer_frees_its_communicator_each_cycle"] = (
        completed == CYCLES and trainer_growth <= MAX_MEMORY_GROWTH_MIB
    )
    checks["server_still_generates"] = _probe_or_none() is not None

    log(f"{completed}/{CYCLES} cycles; server +{server_growth} MiB, trainer +{trainer_growth} MiB")
    return {
        "checks": ctx.broadcast_checks(checks),
        "metrics": {
            "cycles_completed": completed,
            "server_growth_mib": server_growth,
            "trainer_growth_mib": trainer_growth,
            "first_error": errors[0] if errors else "",
        },
    }


main = gpu_test_main(exact_world_size=1, prefix="vllm_weight_transfer_reinit", partial_state=False)(run)

if __name__ == "__main__":
    main()
