#!/usr/bin/env python
"""A weight-sync pool whose client fails to open an update before any chunk went out reconnects once and
lands the whole sync through the rebuilt client.

``InferenceClientManager`` retries a failed client step once, on a fresh client, only while nothing of
the sync reached the engine, since re-sending the snapshots it kept then puts the whole update there.
The pool's client here fails its first update open, as against a server that dropped the request; the
pool must retire it (closing its communicator and freeing the group port), form a new client on the
same port and re-flush, after which the served policy is the one an undisturbed sync of the same
weights serves.

Requires the vLLM container serving the SAME dense checkpoint on a GPU outside
``CUDA_VISIBLE_DEVICES``, as the re-init suite does.

Usage:
    HALO_TEST_VLLM_DENSE_SERVER_URL=http://localhost:8010 NCCL_IB_DISABLE=1 NCCL_NET=Socket \
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
        tests/gpu/trainers/grpo/test_weight_sync_pool_reconnect.py
"""

from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM

from src.distributed.nccl.clients.vllm import VLLMWeightSyncClient
from src.distributed.nccl.transport.pynccl import PyNcclCommunicator
from src.trainers.grpo.rollout.weight_sync_clients import InferenceClientManager
from tests.common.harness import gpu_test_main
from tests.common.on_policy_e2e import probe_top_logprobs, record_served_baseline, served_policy_moved
from tests.common.utils import log
from tests.common.weight_sync import VLLM_DENSE_SERVER_URL, weight_transfer_port
from tests.common.weight_sync_reinit import CONNECTION_TIMEOUT_S, PERTURBATION, PERTURBED_PARAM, VLLM_REINIT_MODEL


def _push(manager: InferenceClientManager, weight: torch.Tensor) -> None:
    manager.update_named_param(PERTURBED_PARAM, weight)
    manager.reset_prefix_cache()


def _restore(manager: InferenceClientManager, original: torch.Tensor) -> dict[str, float] | None:
    """Push the checkpoint's own weight back and probe it, or ``None`` if that failed. Runs on every
    path: the tier's next suite asserts against the unperturbed server."""
    try:
        _push(manager, original)
        return probe_top_logprobs(VLLM_DENSE_SERVER_URL, VLLM_REINIT_MODEL)
    except Exception as e:  # noqa: BLE001 — a failed restore fails its own check
        log(f"restoring sync failed: {type(e).__name__}: {e}")
        return None


def run(ctx) -> dict:
    log(f"weight-sync pool reconnect against {VLLM_DENSE_SERVER_URL}")
    checks: dict[str, bool] = {}
    baseline = record_served_baseline(VLLM_DENSE_SERVER_URL, VLLM_REINIT_MODEL, checks)
    model = AutoModelForCausalLM.from_pretrained(VLLM_REINIT_MODEL, dtype=torch.bfloat16).to(ctx.device)
    param = dict(model.named_parameters())[PERTURBED_PARAM]
    original = param.detach().clone()
    with torch.no_grad():
        param.mul_(PERTURBATION)

    manager = InferenceClientManager(
        [{"url": VLLM_DENSE_SERVER_URL}],
        connection_timeout=CONNECTION_TIMEOUT_S,
        client_cls=VLLMWeightSyncClient,
        base_group_port=weight_transfer_port("HALO_TEST_VLLM_GROUP_PORT"),
    )
    begin = VLLMWeightSyncClient.begin_weight_update
    opened_by: list[VLLMWeightSyncClient] = []
    # Each opening client's communicator, kept here because its close drops the client's own reference.
    communicators: list[PyNcclCommunicator] = []

    def first_open_dropped(client):
        opened_by.append(client)
        communicators.append(client.communicator)
        if len(opened_by) == 1:
            raise ConnectionError("the engine dropped the update's opening request")
        return begin(client)

    reconnect = InferenceClientManager.reconnect_client
    rebuilt: list[VLLMWeightSyncClient] = []

    def counted_reconnect(pool, index):
        rebuilt.append(reconnect(pool, index))
        return rebuilt[-1]

    manager.init_communicators(ctx.device)
    try:
        with (
            patch.object(VLLMWeightSyncClient, "begin_weight_update", first_open_dropped),
            patch.object(InferenceClientManager, "reconnect_client", counted_reconnect),
        ):
            _push(manager, param.detach())
        recovered = probe_top_logprobs(VLLM_DENSE_SERVER_URL, VLLM_REINIT_MODEL)
        _push(manager, param.detach())
        undisturbed = probe_top_logprobs(VLLM_DENSE_SERVER_URL, VLLM_REINIT_MODEL)
    finally:
        restored = _restore(manager, original)
        manager.close_communicators()

    retry_client = opened_by[1] if len(opened_by) == 2 else None
    checks["one_reconnect"] = len(rebuilt) == 1
    checks["the_retry_opened_on_the_rebuilt_client"] = retry_client is not None and rebuilt == [retry_client]
    checks["the_retired_client_aborted_its_communicator"] = (
        retry_client is not None
        and opened_by[0] is not retry_client
        and communicators[0] is not None
        and communicators[0].aborted
        and opened_by[0].communicator is None
    )
    checks["the_recovered_sync_moved_the_served_policy"] = served_policy_moved(recovered, baseline)
    checks["the_recovered_sync_landed_what_an_undisturbed_one_lands"] = recovered == undisturbed
    checks["served_policy_restored_by_a_final_sync"] = restored == baseline
    return {"checks": ctx.broadcast_checks(checks), "metrics": {"reconnects": len(rebuilt)}}


main = gpu_test_main(exact_world_size=1, prefix="weight_sync_pool_reconnect", partial_state=False)(run)

if __name__ == "__main__":
    main()
