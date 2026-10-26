#!/usr/bin/env python
"""A weight-sync pool whose client fails to open an update before any chunk went out reconnects once and
lands the whole sync through the rebuilt client.

``WeightSyncPool`` retries a failed client step once, on a fresh client, only while nothing of
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

from halo.weight_sync.clients.vllm import VLLMWeightSyncClient
from halo.weight_sync.pool import WeightSyncPool
from halo.weight_sync.transport.pynccl import PyNcclCommunicator
from tests.common.harness import gpu_test_main
from tests.common.on_policy_e2e import probe_top_logprobs, record_served_baseline, served_policy_moved
from tests.common.rank_log import log
from tests.common.weight_sync import VLLM_DENSE_SERVER_URL, weight_transfer_port
from tests.common.weight_sync_reinit import (
    CONNECTION_TIMEOUT_S,
    PERTURBATION,
    PERTURBED_PARAM,
    VLLM_REINIT_MODEL,
    serve_the_checkpoint,
)


def _push(pool: WeightSyncPool, weight: torch.Tensor) -> None:
    pool.update_named_param(PERTURBED_PARAM, weight)
    pool.finish_weight_update()


def _restore(pool: WeightSyncPool, original: torch.Tensor) -> dict[str, float] | None:
    """Push the checkpoint's own weight back and probe it, or ``None`` if that failed. Runs on every
    path: the tier's next suite asserts against the unperturbed server."""
    try:
        _push(pool, original)
        return probe_top_logprobs(VLLM_DENSE_SERVER_URL, VLLM_REINIT_MODEL)
    except Exception as e:  # noqa: BLE001 — a failed restore fails its own check
        log(f"restoring sync failed: {type(e).__name__}: {e}")
        return None


def run(ctx) -> dict:
    log(f"weight-sync pool reconnect against {VLLM_DENSE_SERVER_URL}")
    checks: dict[str, bool] = {}
    model = AutoModelForCausalLM.from_pretrained(VLLM_REINIT_MODEL, dtype=torch.bfloat16).to(ctx.device)
    base_group_port = weight_transfer_port("HALO_TEST_VLLM_GROUP_PORT")
    serve_the_checkpoint(VLLMWeightSyncClient, VLLM_DENSE_SERVER_URL, model, ctx.device, base_group_port)
    baseline = record_served_baseline(VLLM_DENSE_SERVER_URL, VLLM_REINIT_MODEL, checks)
    param = dict(model.named_parameters())[PERTURBED_PARAM]
    original = param.detach().clone()
    with torch.no_grad():
        param.mul_(PERTURBATION)

    pool = WeightSyncPool(
        [{"url": VLLM_DENSE_SERVER_URL}],
        connection_timeout=CONNECTION_TIMEOUT_S,
        client_cls=VLLMWeightSyncClient,
        base_group_port=base_group_port,
    )
    begin = VLLMWeightSyncClient._start_engine_update
    opened_by: list[VLLMWeightSyncClient] = []
    # Each opening client's communicator, kept here because its close drops the client's own reference.
    communicators: list[PyNcclCommunicator] = []

    def first_open_dropped(client):
        opened_by.append(client)
        communicators.append(client._transport)
        if len(opened_by) == 1:
            raise ConnectionError("the engine dropped the update's opening request")
        return begin(client)

    reconnect = WeightSyncPool._reconnect
    rebuilt: list[VLLMWeightSyncClient] = []

    def counted_reconnect(pool, index):
        reconnect(pool, index)
        rebuilt.append(pool._clients[index])

    pool.init_communicator(ctx.device)
    try:
        with (
            patch.object(VLLMWeightSyncClient, "_start_engine_update", first_open_dropped),
            patch.object(WeightSyncPool, "_reconnect", counted_reconnect),
        ):
            _push(pool, param.detach())
        recovered = probe_top_logprobs(VLLM_DENSE_SERVER_URL, VLLM_REINIT_MODEL)
        _push(pool, param.detach())
        undisturbed = probe_top_logprobs(VLLM_DENSE_SERVER_URL, VLLM_REINIT_MODEL)
    finally:
        restored = _restore(pool, original)
        pool.close_communicator()

    retry_client = opened_by[1] if len(opened_by) == 2 else None
    checks["one_reconnect"] = len(rebuilt) == 1
    checks["the_retry_opened_on_the_rebuilt_client"] = retry_client is not None and rebuilt == [retry_client]
    checks["the_retired_client_aborted_its_communicator"] = (
        retry_client is not None
        and opened_by[0] is not retry_client
        and communicators[0] is not None
        and communicators[0].aborted
        and opened_by[0]._transport is None
    )
    checks["the_recovered_sync_moved_the_served_policy"] = served_policy_moved(recovered, baseline)
    checks["the_recovered_sync_landed_what_an_undisturbed_one_lands"] = recovered == undisturbed
    checks["served_policy_restored_by_a_final_sync"] = restored == baseline
    return {"checks": ctx.broadcast_checks(checks), "metrics": {"reconnects": len(rebuilt)}}


main = gpu_test_main(exact_world_size=1, prefix="weight_sync_pool_reconnect", partial_state=False)(run)

if __name__ == "__main__":
    main()
